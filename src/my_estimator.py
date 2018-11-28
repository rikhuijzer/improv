import time
from enum import Enum, auto
from pathlib import Path
from typing import Set, List

import tensorflow as tf

from src.config import HParams
from src.data_reader import get_filtered_messages
from src.modeling import BertConfig
from src.my_classifier import get_tokenizer
from src.run_classifier import input_fn_builder, convert_examples_to_features, model_fn_builder, InputExample
from src.tokenization import convert_to_unicode
"""Based on https://medium.com/tensorflow/how-to-write-a-custom-estimator-model-for-the-cloud-tpu-7d8bd9068c26."""


class SetType(Enum):
    train = auto()
    dev = auto()
    test = auto()


def get_examples(filename: Path, set_type: SetType) -> List:
    if set_type == SetType.train:
        messages = get_filtered_messages(filename, training=True)
    else:
        tf.logging.warning('There currently is no difference between dev and test set.')
        messages = get_filtered_messages(filename, training=False)
    examples = []
    for (i, message) in enumerate(messages):
        guid = "%s-%s" % (str(set_type.name), i)
        text_a = convert_to_unicode(message.text)
        label = convert_to_unicode(message.data['intent'])
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def get_intents(filename: Path) -> List[str]:
    return list(map(lambda m: m.data['intent'], get_filtered_messages(filename, training=True)))


def get_unique_intents(filename: Path) -> Set[str]:
    return set(map(lambda m: m.data['intent'], get_filtered_messages(filename, training=True)))


def serving_input_fn():
    # Note: only handles one image at a time
    feature_placeholders = {'image_bytes':
                                tf.placeholder(tf.string, shape=())}
    features = {
        'image': 'empty'
    }
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


def load_global_step_from_checkpoint_dir(checkpoint_dir):
    try:
        checkpoint_reader = tf.train.NewCheckpointReader(
            tf.train.latest_checkpoint(checkpoint_dir))
        return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
    except:  # pylint: disable=bare-except
        return 0


def train_and_evaluate(hparams: HParams):
    tf.logging.set_verbosity(tf.logging.INFO)

    data_filename = hparams.data_dir / (hparams.task_name + '.tsv')
    train_examples = get_examples(data_filename, SetType.train)
    num_train_steps = int(len(train_examples) / hparams.train_batch_size * hparams.num_train_epochs)
    steps_per_epoch = len(train_examples) // hparams.train_batch_size
    max_steps = hparams.num_train_epochs * steps_per_epoch
    tf.logging.info('train_batch_size=%d  eval_batch_size=%d  max_steps=%d',
                    hparams.train_batch_size,
                    hparams.eval_batch_size,
                    max_steps)

    # TPU change 3
    if hparams.use_tpu:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            hparams.tpu_name,
            zone=hparams.tpu_zone,
            project=hparams.gcp_project)
        config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=hparams.output_dir,  # according to my_classifier
            save_checkpoints_steps=hparams.save_checkpoints_steps,
            # save_summary_steps=hparams.save_checkpoints_steps,  # source: https://stackoverflow.com/questions/51965950
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=hparams.save_checkpoints_steps,
                per_host_input_for_training=True))
    else:
        config = tf.contrib.tpu.RunConfig()

    model_fn = model_fn_builder(
        bert_config=BertConfig.from_json_file(str(hparams.bert_config_file)),
        num_labels=len(get_unique_intents(data_filename)),
        init_checkpoint=str(hparams.init_checkpoint),
        learning_rate=hparams.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=int(num_train_steps * hparams.warmup_proportion),
        use_tpu=hparams.use_tpu,
        use_one_hot_embeddings=True)

    estimator = tf.contrib.tpu.TPUEstimator(  # TPU change 4
        model_fn=model_fn,
        config=config,
        # params=hparams,
        # model_dir=hparams.data_dir,
        train_batch_size=hparams.train_batch_size,
        eval_batch_size=hparams.eval_batch_size,
        use_tpu=hparams.use_tpu
    )

    train_features = convert_examples_to_features(
        train_examples, get_unique_intents(data_filename), hparams.max_seq_length, get_tokenizer(hparams))
    tf.logging.info("  Num steps = %d", num_train_steps)

    train_input_fn = input_fn_builder(
        features=train_features,
        seq_length=hparams.max_seq_length,
        is_training=True,
        drop_remainder=True,
        use_tpu=hparams.use_tpu)

    eval_examples = get_examples(data_filename, SetType.dev)
    eval_features = convert_examples_to_features(
        eval_examples, get_unique_intents(data_filename), hparams.max_seq_length, get_tokenizer(hparams))

    # Eval will be slightly WRONG on the TPU because it will truncate
    # the last batch.
    eval_steps = int(len(eval_examples) / hparams.eval_batch_size)
    eval_input_fn = input_fn_builder(
        features=eval_features,
        seq_length=hparams.max_seq_length,
        is_training=False,
        drop_remainder=True,
        use_tpu=hparams.use_tpu)

    # set up training and evaluation in a loop
    # def input_fn_builder(features, seq_length, is_training, drop_remainder, use_tpu):

    # load last checkpoint and start from there
    current_step = load_global_step_from_checkpoint_dir(hparams.output_dir)
    tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                    ' step %d.',
                    max_steps,
                    max_steps / steps_per_epoch,
                    current_step)

    start_timestamp = time.time()  # This time will include compilation time

    while current_step < max_steps:
        # Train for up to steps_per_eval number of steps.
        # At the end of training, a checkpoint will be written to --model_dir.
        next_checkpoint = min(current_step + hparams.iterations_per_loop,
                              max_steps)  # possibly need to save checkpoints

        if hparams.do_train:
            estimator.train(input_fn=train_input_fn, max_steps=next_checkpoint)

        current_step = next_checkpoint
        tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                        next_checkpoint, int(time.time() - start_timestamp))

        # Evaluate the model on the most recent model in --model_dir.
        # Since evaluation happens in batches of --eval_batch_size, some images
        # may be excluded modulo the batch size. As long as the batch size is
        # consistent, the evaluated images are also consistent.
        tf.logging.info('Starting to evaluate at step %d', next_checkpoint)
        eval_results = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        tf.logging.info('Eval results at step %d: %s', next_checkpoint, eval_results)

    elapsed_time = int(time.time() - start_timestamp)
    tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                    max_steps, elapsed_time)

    # export similar to Cloud ML Engine convention
    # tf.logging.info('Starting to export model.')
    # estimator.export_savedmodel(
    #    export_dir_base=os.path.join(hparams.output_dir, 'export/exporter'),
    #    serving_input_receiver_fn=serving_input_fn)


def predict(hparams: HParams) -> List[str]:
    from src.my_classifier import get_model_fn_and_estimator, file_based_input_fn_builder
    from src.run_classifier import file_based_convert_examples_to_features
    import os
    from typing import Iterable
    import numpy as np
    from src.utils import convert_result_pred, get_rounded_f1

    data_filename = hparams.data_dir / (hparams.task_name + '.tsv')
    params = hparams._replace(use_tpu=False)  # BERT code warns against using TPU for predictions.
    model_fn, estimator = get_model_fn_and_estimator(params)

    predict_examples = get_examples(data_filename, SetType.test)
    predict_file = os.path.join(params.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, get_unique_intents(data_filename),
                                            params.max_seq_length, get_tokenizer(params),
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", params.predict_batch_size)

    predict_drop_remainder = params.use_tpu
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=params.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result: Iterable[np.ndarray] = estimator.predict(input_fn=predict_input_fn)
    label_list = get_intents(data_filename)  # used for label_list[max_class] this might be wrong
    y_pred = convert_result_pred(result, label_list)
    print('f1 score: {}'.format(get_rounded_f1(params.data_dir / 'askubuntu.tsv', y_pred, average='micro')))
    return y_pred


if __name__ == '__main__':
    from src.config import get_debug_hparams

    # train_and_evaluate(get_debug_hparams())

    predict(get_debug_hparams())
