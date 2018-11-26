import time

import tensorflow as tf

from src.my_classifier import get_model_fn_and_estimator, get_processor, get_tokenizer
from src.run_classifier import input_fn_builder, convert_examples_to_features
from src.config import HParams
import os
"""Based on https://medium.com/tensorflow/how-to-write-a-custom-estimator-model-for-the-cloud-tpu-7d8bd9068c26."""


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
    # inefficient but it works
    processor = get_processor(hparams)
    train_examples = processor.get_train_examples(hparams.data_dir)
    num_train_steps = int(len(train_examples) / hparams.train_batch_size * hparams.num_train_epochs)
    steps_per_epoch = len(train_examples) // hparams.train_batch_size
    max_steps = hparams.num_train_epochs * steps_per_epoch

    tf.logging.info('train_batch_size=%d  eval_batch_size=%d  max_steps=%d',
                    hparams.train_batch_size,
                    hparams.eval_batch_size,
                    max_steps)

    '''
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
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=hparams.save_checkpoints_steps,
                per_host_input_for_training=True))
    else:
        config = tf.contrib.tpu.RunConfig()
    '''

    model_fn, estimator = get_model_fn_and_estimator(hparams)

    '''
    estimator = tf.contrib.tpu.TPUEstimator(  # TPU change 4
        model_fn=model_fn,
        config=config,
        # params=hparams,
        # model_dir=hparams.data_dir,
        train_batch_size=hparams.train_batch_size,
        eval_batch_size=hparams.eval_batch_size,
        use_tpu=hparams.use_tpu
    )
    '''

    train_examples = processor.get_train_examples(hparams.data_dir)
    train_features = convert_examples_to_features(
        train_examples, processor.get_labels(), hparams.max_seq_length, get_tokenizer(hparams))
    tf.logging.info("  Num steps = %d", num_train_steps)

    train_input_fn = input_fn_builder(
        features=train_features,
        seq_length=hparams.max_seq_length,
        is_training=True,
        drop_remainder=True,
        use_tpu=hparams.use_tpu)

    eval_examples = processor.get_dev_examples(hparams.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, processor.get_labels(), hparams.max_seq_length, get_tokenizer(hparams))

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

    while current_step < num_train_steps:
        # Train for up to steps_per_eval number of steps.
        # At the end of training, a checkpoint will be written to --model_dir.
        next_checkpoint = min(current_step + hparams.save_checkpoints_steps, max_steps)

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


if __name__ == '__main__':
    from src.config import get_debug_hparams

    train_and_evaluate(get_debug_hparams())
