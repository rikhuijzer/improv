import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs

from src import tokenization
from src.config import Params
from src.modeling import BertConfig
from src.run_classifier import DataProcessor, InputExample
from src.run_classifier import (
    input_fn_builder, convert_examples_to_features, model_fn_builder,
    file_based_convert_examples_to_features, file_based_input_fn_builder
)
from src.utils import convert_result_pred, get_rounded_f1
from src.tokenization import FullTokenizer
from functools import lru_cache


@lru_cache(maxsize=1)
def get_tokenizer(params: Params) -> FullTokenizer:
    return FullTokenizer(vocab_file=str(params.vocab_file), do_lower_case=params.do_lower_case)


@lru_cache(maxsize=1)
def get_processor(params: Params) -> DataProcessor:
    processor = IntentProcessor()
    processor.data_dir = params.data_dir
    return processor


def get_model_and_estimator(params: Params):
    processor = get_processor(params)
    train_examples = processor.get_train_examples(params.data_dir)
    num_train_steps = int(len(train_examples) / params.train_batch_size * params.num_train_epochs)
    num_warmup_steps = int(num_train_steps * params.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=BertConfig.from_json_file(str(params.bert_config_file)),
        num_labels=len(processor.get_labels()),
        init_checkpoint=str(params.init_checkpoint),
        learning_rate=params.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=params.use_tpu,
        use_one_hot_embeddings=True)

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(params.tpu_name)  # used to be TPU_ADDRESS
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=params.output_dir,
        save_checkpoints_steps=params.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=params.iterations_per_loop,
            num_shards=params.num_tpu_cores,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=params.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=params.train_batch_size,
        eval_batch_size=params.eval_batch_size)

    return model_fn, estimator


def train(params: Params, estimator, hook=None):
    summary_hook = SummarySaverHook(
        1,  # save every n steps
        output_dir='/tmp/tf',
        summary_op=tf.summary.merge_all())

    processor = get_processor(params)
    train_examples = processor.get_train_examples(params.data_dir)
    num_train_steps = int(len(train_examples) / params.train_batch_size * params.num_train_epochs)
    train_features = convert_examples_to_features(
        train_examples, processor.get_labels(), params.max_seq_length, get_tokenizer(params))
    print('***** Started training at {} *****'.format(datetime.now()))
    print('  Num examples = {}'.format(len(train_examples)))
    print('  Batch size = {}'.format(params.train_batch_size))
    tf.logging.info("  Num steps = %d", num_train_steps)

    # see run_classifier.convert_single_example for feature creation
    train_input_fn = input_fn_builder(
        features=train_features,
        seq_length=params.max_seq_length,
        is_training=True,
        drop_remainder=True)

    if not hook:
        hook = MetadataHook(save_steps=1, output_dir=params.output_dir)

    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[hook])
    print('***** Finished training at {} *****'.format(datetime.now()))


def evaluate(params: Params, estimator):
    processor = get_processor(params)
    eval_examples = processor.get_dev_examples(params.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, processor.get_labels(), params.max_seq_length, get_tokenizer(params))
    print('***** Started evaluation at {} *****'.format(datetime.now()))
    print('  Num examples = {}'.format(len(eval_examples)))
    print('  Batch size = {}'.format(params.eval_batch_size))
    # Eval will be slightly WRONG on the TPU because it will truncate
    # the last batch.
    eval_steps = int(len(eval_examples) / params.eval_batch_size)
    eval_input_fn = input_fn_builder(
        features=eval_features,
        seq_length=params.max_seq_length,
        is_training=False,
        drop_remainder=True)
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    print('***** Finished evaluation at {} *****'.format(datetime.now()))
    output_eval_file = params.output_dir + '/eval_results.txt'
    with tf.gfile.GFile(str(output_eval_file), "w") as writer:
        print("***** Eval results *****")
        for key in sorted(result.keys()):
            print('  {} = {}'.format(key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))


def predict(params: Params) -> List[str]:
    params = params._replace(use_tpu=False)  # BERT code warns against using TPU for predictions.
    model_fn, estimator = get_model_and_estimator(params)

    processor = get_processor(params)
    predict_examples = processor.get_test_examples(params.data_dir)
    predict_file = os.path.join(params.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, processor.get_labels(),
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
    label_list = processor.get_labels()  # used for label_list[max_class] this might be wrong
    y_pred = convert_result_pred(result, label_list)
    print('f1 score: {}'.format(get_rounded_f1(params.data_dir / 'test.tsv', y_pred)))
    return y_pred


def get_intents(data_dir: Path) -> Iterable[str]:
    with open(str(data_dir / 'test.tsv'), 'r', encoding='utf8', newline='') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t', lineterminator='\n')

        for row in tsv_reader:
            yield row[1]


class IntentProcessor(DataProcessor):
    """Processor for the intent classification data set."""
    data_dir: Path

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        if not self.data_dir:
            raise AssertionError('Set IntentProcessor.data_dir')

        return list(set(get_intents(self.data_dir)))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class MetadataHook(SessionRunHook):
    """hook, based on ProfilerHook, to have the estimator output the run metadata into the model directory
        source: https://stackoverflow.com/questions/45719176"""
    def __init__(self, save_steps=None, save_secs=None, output_dir=""):
        self._output_tag = "step-{}"
        self._output_dir = output_dir
        self._timer = SecondOrStepTimer(every_secs=save_secs, every_steps=save_steps)

    def begin(self):
        self._next_step = None
        self._global_step_tensor = training_util.get_global_step()
        tf.logging.info('creating file in: {}'.format(self._output_dir))
        self._writer = tf.summary.FileWriter(self._output_dir + '/hook_data', tf.get_default_graph())

        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use ProfilerHook.")

    def before_run(self, run_context):
        self._request_summary = (
                self._next_step is None or
                self._timer.should_trigger_for_step(self._next_step)
        )
        requests = {"global_step": self._global_step_tensor}
        opts = (tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                if self._request_summary else None)
        return SessionRunArgs(requests, options=opts)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results["global_step"]
        global_step = stale_global_step + 1
        if self._request_summary:
            global_step = run_context.session.run(self._global_step_tensor)
            self._writer.add_run_metadata(
                run_values.run_metadata, self._output_tag.format(global_step))
            self._writer.flush()
        self._next_step = global_step + 1

    def end(self, session):
        self._writer.close()
