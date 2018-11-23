import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import tensorflow as tf

from src import tokenization
from src.config import Params
from src.modeling import BertConfig
from src.run_classifier import DataProcessor, InputExample
from src.run_classifier import (
    input_fn_builder, convert_examples_to_features, model_fn_builder,
    file_based_convert_examples_to_features, file_based_input_fn_builder
)
import numpy as np


def get_model_and_estimator(params: Params, processor):
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


def train(params: Params, processor, tokenizer, estimator):
    train_examples = processor.get_train_examples(params.data_dir)
    num_train_steps = int(len(train_examples) / params.train_batch_size * params.num_train_epochs)
    train_features = convert_examples_to_features(
        train_examples, processor.get_labels(), params.max_seq_length, tokenizer)
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
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print('***** Finished training at {} *****'.format(datetime.now()))


def evaluate(params: Params, processor, tokenizer, estimator):
    # Eval the model.
    eval_examples = processor.get_dev_examples(params.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, processor.get_labels(), params.max_seq_length, tokenizer)
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
    output_eval_file = params.output_dir / 'eval_results.txt'
    with tf.gfile.GFile(str(output_eval_file), "w") as writer:
        print("***** Eval results *****")
        for key in sorted(result.keys()):
            print('  {} = {}'.format(key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))


def predict(params: Params, processor, tokenizer, estimator):
    predict_examples = processor.get_test_examples(params.data_dir)
    predict_file = os.path.join(params.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, processor.get_labels(),
                                            params.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", params.predict_batch_size)

    # if USE_TPU:
    # Warning: According to tpu_estimator.py Prediction on TPU is an
    # experimental feature and hence not supported here
    # raise ValueError("Prediction in TPU not supported")

    predict_drop_remainder = params.use_tpu
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=params.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result: Iterable[np.ndarray] = estimator.predict(input_fn=predict_input_fn)

    label_list = processor.get_labels()  # used for label_list[max_class] this might be wrong

    output_predict_file = os.path.join(params.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, 'w') as writer:
        tf.logging.info("***** Predict results *****")
        for prediction in result:
            max_class = prediction.argmax()  # get index for highest confidence prediction
            output_line = str(max_class) + '\t' + label_list[max_class] + '\t' + '\t'.join(
                str(class_probability) for class_probability in prediction)
            writer.write(output_line + '\n')


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
