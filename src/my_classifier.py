import os
from copy import copy
from datetime import datetime
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import numpy as np
import tensorflow as tf

from src.config import HParams
from src.data_reader import get_messages, get_filtered_messages
from src.modeling import BertConfig
from src.run_classifier import InputExample
from src.run_classifier import (
    input_fn_builder, convert_examples_to_features, model_fn_builder,
    file_based_convert_examples_to_features, file_based_input_fn_builder
)
from src.tokenization import FullTokenizer, convert_to_unicode
from src.utils import convert_result_pred, get_rounded_f1, print_eval_results


@lru_cache(maxsize=1)
def get_tokenizer(params: HParams) -> FullTokenizer:
    return FullTokenizer(vocab_file=str(params.vocab_file), do_lower_case=params.do_lower_case)


class SetType(Enum):
    train = auto()
    dev = auto()
    test = auto()


def get_examples(filename: Path, set_type: SetType) -> List[InputExample]:
    """Returns input examples as specified in run_classifier.py"""
    if set_type == SetType.train:
        messages = get_filtered_messages(filename, training=True)
    else:
        tf.logging.info('There currently is no difference between dev and test set.')
        messages = get_filtered_messages(filename, training=False)
    examples = []
    for (i, message) in enumerate(messages):
        guid = "%s-%s" % (str(set_type.name), i)
        text_a = convert_to_unicode(message.text)
        label = convert_to_unicode(message.data['intent'])
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


@lru_cache(maxsize=3)
def get_intents(filename: Path, training: bool) -> List[str]:
    """Get all the intents from some file which has a training column."""
    return list(map(lambda m: m.data['intent'], get_filtered_messages(filename, training=training)))


@lru_cache(maxsize=1)
def get_unique_intents(filename: Path) -> List[str]:
    """Returning list to make sure the order does not change."""
    return list(set(map(lambda m: m.data['intent'], get_messages(filename))))


@lru_cache(maxsize=1)
def get_model_fn_and_estimator(hparams: HParams):
    from src.my_estimator import get_unique_intents

    data_filename = hparams.data_dir / (hparams.task_name + '.tsv')
    num_train_steps = hparams.num_train_steps
    num_warmup_steps = int(num_train_steps * hparams.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=BertConfig.from_json_file(str(hparams.bert_config_file)),
        num_labels=len(get_unique_intents(data_filename)),
        init_checkpoint=str(hparams.init_checkpoint),
        learning_rate=hparams.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=hparams.use_tpu,
        use_one_hot_embeddings=True)

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(hparams.tpu_name)  # used to be TPU_ADDRESS
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=hparams.output_dir,
        save_checkpoints_steps=hparams.save_checkpoints_steps,
        save_summary_steps=hparams.save_summary_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=hparams.iterations_per_loop,
            num_shards=hparams.num_tpu_cores,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    # using normal Estimator to enable tf.summary
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=hparams.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=hparams.train_batch_size,
        eval_batch_size=hparams.eval_batch_size
    )

    return model_fn, estimator


def train(hparams: HParams, estimator, max_steps: int):
    training_start_time = datetime.now()
    tf.logging.info('***** Started training at {} *****'.format(training_start_time))
    data_filename = hparams.data_dir / (hparams.task_name + '.tsv')
    train_examples = get_examples(data_filename, SetType.train)
    num_train_steps = hparams.num_train_steps
    train_features = convert_examples_to_features(
        train_examples, get_unique_intents(data_filename), hparams.max_seq_length, get_tokenizer(hparams))
    tf.logging.info('  Num examples = {}'.format(len(train_examples)))
    tf.logging.info('  Batch size = {}'.format(hparams.train_batch_size))
    tf.logging.info("  Num steps = %d", num_train_steps)

    # see run_classifier.convert_single_example for feature creation
    train_input_fn = input_fn_builder(
        features=train_features,
        seq_length=hparams.max_seq_length,
        is_training=True,
        drop_remainder=True
    )

    estimator.train(input_fn=train_input_fn, max_steps=max_steps)
    tf.logging.info('Training took {}.'.format(datetime.now() - training_start_time))


def evaluate(params: HParams, estimator):
    hparams = copy(params)
    hparams = hparams._replace(use_tpu=False)  # evaluation is quicker on CPU
    data_filename = hparams.data_dir / (hparams.task_name + '.tsv')
    eval_examples = get_examples(data_filename, SetType.dev)
    eval_features = convert_examples_to_features(
        eval_examples, get_unique_intents(data_filename), hparams.max_seq_length, get_tokenizer(hparams))
    tf.logging.info('  Num examples = {}'.format(len(eval_examples)))
    tf.logging.info('  Batch size = {}'.format(hparams.eval_batch_size))
    # Eval will be slightly WRONG on the TPU because it will truncate
    # the last batch.
    eval_steps = int(len(eval_examples) / hparams.eval_batch_size)
    eval_input_fn = input_fn_builder(
        features=eval_features,
        seq_length=hparams.max_seq_length,
        is_training=False,
        drop_remainder=True,
    )

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    tf.logging.info(result)
    tf.logging.info('***** Finished evaluation at {} *****'.format(datetime.now()))
    output_eval_file = hparams.output_dir + '/eval_results.txt'
    with tf.gfile.GFile(str(output_eval_file), "w") as writer:
        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            # eval_accuracy = 0.90625
            # eval_loss = 0.76804435
            # global_step = 90
            # loss = 1.7459234
            tf.logging.info('  {} = {}'.format(key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return result


def train_evaluate(hparams: HParams, estimator):
    max_steps = hparams.save_summary_steps  # TPU just plows on, we need to manually enforce save_summary_steps
    step = 0
    while step < hparams.num_train_steps:
        train(hparams, estimator, max_steps)
        results = evaluate(hparams, estimator)
        print('Step {}: {}'.format(step, results))
        step += max_steps


def predict(params: HParams) -> List[str]:
    hparams = copy(params)
    hparams = hparams._replace(use_tpu=False)
    model_fn, estimator = get_model_fn_and_estimator(hparams)

    data_filename = hparams.data_dir / (hparams.task_name + '.tsv')
    predict_examples = get_examples(data_filename, SetType.test)
    predict_file = os.path.join(hparams.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, get_unique_intents(data_filename),
                                            hparams.max_seq_length, get_tokenizer(hparams),
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", hparams.predict_batch_size)

    predict_drop_remainder = hparams.use_tpu
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=hparams.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result: Iterable[np.ndarray] = estimator.predict(input_fn=predict_input_fn)
    label_list = get_unique_intents(data_filename)  # used for label_list[max_class] this might be wrong
    y_pred = convert_result_pred(result, label_list)
    tf.logging.info('f1 score: {}'.format(get_rounded_f1(hparams.data_dir / (hparams.task_name + '.tsv'), y_pred)))
    return y_pred
