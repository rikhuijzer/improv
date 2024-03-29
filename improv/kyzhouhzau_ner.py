#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author:zhoukaiyin
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import pickle
from pathlib import Path

import tensorflow as tf

import improv.kyzhouhzau_tf_metrics as tf_metrics
import improv.modeling as modeling
import improv.optimization as optimization
import improv.tokenization as tokenization
from improv.config import HParams
from improv.read_ner import get_ner_lines, get_unique_labels, get_interesting_labels_indexes
from typing import Iterable, List
from numpy import ndarray
from improv.my_types import NERData
from improv.evaluate import print_scores
from improv.my_classifier import get_tokenizer, get_model_fn_and_estimator, train, evaluate, predict


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        from improv.read_ner import get_ner_lines
        return get_ner_lines(Path(input_file))


class NerProcessor(DataProcessor):
    def __init__(self, h_params: HParams):
        self.h_params = h_params

    def get_train_examples(self, data_dir):
        return self._create_example(
            get_ner_lines(Path(data_dir) / 'train.txt'), "train"
        )

    def get_dev_examples(self, data_dir):
        tf.logging.warning('Returned dev examples are from test set.')
        return self._create_example(
            get_ner_lines(Path(data_dir) / 'test.txt'), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            get_ner_lines(Path(data_dir) / 'test.txt'), "test"
        )

    def get_labels(self):
        from improv.read_ner import get_unique_labels
        return get_unique_labels(self.h_params.data_dir)
        # return ["B-MISC", "I-MISC", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples


def write_tokens(tokens, mode, h_params):
    if mode == "test":
        path = os.path.join(h_params.local_dir, "token_" + mode + ".txt")
        wf = open(path, 'a')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode, h_params):
    label_map = {}
    for i, label in enumerate(label_list, 1):
        label_map[label] = i
    with open(os.path.join(h_params.local_dir, 'label2id.pkl'), 'wb') as w:
        pickle.dump(label_map, w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    # ntokens.append("[CLS]")  # useless, should not be appended i think
    # segment_ids.append(0)
    # label_ids.append(label_map["[CLS]"])  # also removed
    # append("O") or append("[CLS]") not sure!
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    write_tokens(ntokens, mode, h_params)
    return feature


def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                             output_file, h_params, mode=None):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode, h_params)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings, h_params):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1,  # -1 means infer from total number of elements
                                     h_params.max_seq_length,
                                     len(get_unique_labels(h_params.data_dir)) + 1])
        # mask = tf.cast(input_mask,tf.float32)
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(indices=labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)

        # TPUEstimatorSpec.predictions must be dict of Tensors.
        predict = tf.argmax(probabilities, axis=-1)
        predict_dict = {'predictions': predict}  # this way it is not shot down by check in TPUEstimatorSpec
        return loss, per_example_loss, logits, predict_dict
        ##########################################################################


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, h_params):
    init_checkpoint = str(init_checkpoint)
    unique_labels = get_unique_labels(h_params.data_dir)
    interesting_labels_indexes = get_interesting_labels_indexes(unique_labels)
    n_labels = len(unique_labels) + 1

    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, predicts) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings, h_params)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                     init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
        output_spec = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, logits):
                # def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                precision = tf_metrics.precision(label_ids, predictions, n_labels, interesting_labels_indexes,
                                                 average="weighted")
                recall = tf_metrics.recall(label_ids, predictions, n_labels, interesting_labels_indexes,
                                           average="weighted")
                f = tf_metrics.f1(label_ids, predictions, n_labels, interesting_labels_indexes, average="weighted")
                #
                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f,
                    # "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)

        else:  # if ModeKeys.PREDICT
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts, scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


def get_ner_model_fn_and_estimator(h_params: HParams, processor):
    tpu_cluster_resolver = None
    if h_params.use_tpu and h_params.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            h_params.tpu_name, zone=h_params.tpu_zone, project=h_params.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=h_params.master,
        model_dir=h_params.output_dir,
        save_checkpoints_steps=h_params.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=h_params.iterations_per_loop,
            num_shards=h_params.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    label_list = processor.get_labels()
    bert_config = modeling.BertConfig.from_json_file(h_params.bert_config_file)

    model_fn = model_fn_builder(  # train and warmup used to become zero when h_params.do_train == False
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=h_params.init_checkpoint,
        learning_rate=h_params.learning_rate,
        num_train_steps=h_params.num_train_steps,
        num_warmup_steps=int(h_params.num_train_steps * h_params.warmup_proportion),
        use_tpu=h_params.use_tpu,
        use_one_hot_embeddings=h_params.use_tpu,
        h_params=h_params)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=h_params.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=h_params.train_batch_size,
        eval_batch_size=h_params.eval_batch_size,
        predict_batch_size=h_params.predict_batch_size)

    return model_fn, estimator


def ner_train(h_params: HParams, estimator, processor):
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if h_params.do_train:
        train_examples = processor.get_train_examples(h_params.data_dir)
        num_train_steps = h_params.num_train_steps
        num_warmup_steps = int(num_train_steps * h_params.warmup_proportion)

    tf.gfile.MakeDirs(h_params.output_dir)
    label_list = processor.get_labels()
    tokenizer = get_tokenizer(h_params)
    if h_params.do_train:
        train_file = os.path.join(h_params.output_dir, "train.tf_record")
        filed_based_convert_examples_to_features(
            train_examples, label_list, h_params.max_seq_length, tokenizer, train_file, h_params)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", h_params.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=h_params.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


def ner_eval(h_params: HParams, estimator, processor):
    label_list = processor.get_labels()
    tokenizer = get_tokenizer(h_params)

    eval_examples = processor.get_dev_examples(h_params.data_dir)
    eval_file = os.path.join(h_params.output_dir, "eval.tf_record")
    filed_based_convert_examples_to_features(
        eval_examples, label_list, h_params.max_seq_length, tokenizer, eval_file, h_params)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(eval_examples))
    tf.logging.info("  Batch size = %d", h_params.eval_batch_size)
    eval_steps = None
    if h_params.use_tpu:
        eval_steps = int(len(eval_examples) / h_params.eval_batch_size)
    eval_drop_remainder = h_params.use_tpu
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=h_params.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    output_eval_file = os.path.join(h_params.local_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def ner_pred(h_params: HParams, estimator, processor):
    h_params = h_params._replace(use_tpu=False)
    label_list = processor.get_labels()
    tokenizer = get_tokenizer(h_params)

    token_path = os.path.join(h_params.local_dir, "token_test.txt")
    with open(os.path.join(h_params.local_dir, 'label2id.pkl'), 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}
    if os.path.exists(token_path):
        os.remove(token_path)
    predict_examples = processor.get_test_examples(h_params.data_dir)

    predict_file = os.path.join(h_params.output_dir, "predict.tf_record")
    filed_based_convert_examples_to_features(predict_examples, label_list,
                                             h_params.max_seq_length, tokenizer,
                                             predict_file, h_params, mode="test")

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", h_params.predict_batch_size)

    if h_params.use_tpu:
        # Warning: According to tpu_estimator.py Prediction on TPU is an
        # experimental feature and hence not supported here
        tf.logging.warning("Prediction in TPU not supported")

    predict_drop_remainder = h_params.use_tpu
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=h_params.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    result = list(result)
    result = [pred['predictions'] for pred in result]
    return result


def run(h_params: HParams):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "ner": NerProcessor
    }

    bert_config = modeling.BertConfig.from_json_file(h_params.bert_config_file)

    if h_params.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (h_params.max_seq_length, bert_config.max_position_embeddings))

    processor = processors['ner'](h_params)

    if h_params.task == 'intent':
        model_fn, estimator = get_model_fn_and_estimator(h_params)
    else:
        model_fn, estimator = get_ner_model_fn_and_estimator(h_params, processor)

    if h_params.do_train:
        if h_params.task == 'intent':
            train(h_params, estimator, max_steps=h_params.num_train_steps)
        else:
            ner_train(h_params, estimator, processor)

    if h_params.do_eval:
        if h_params.task == 'intent':
            evaluate(h_params, estimator)
        else:
            ner_eval(h_params, estimator, processor)

    if h_params.do_predict:
        if h_params.task == 'intent':
            result = predict(h_params)
        else:
            result = ner_pred(h_params, estimator, processor)
        return result


def get_pred_true(h_params: HParams) -> Iterable[NERData]:
    """Get true values for prediction."""
    lines = get_ner_lines(Path(h_params.data_dir) / 'test.txt')

    def helper(item: List[str]) -> NERData:
        sentence = item[1].split(' ')
        true = item[0].split(' ')
        pred = []
        return NERData(sentence, true, pred)
    return map(helper, lines)


def convert_ner_str(ner_data: NERData) -> str:
    """Format as string to look like:
        text: ['theresienstraße', 'to', 'assling']
        true: ['B-StationStart', 'O', 'B-StationDest']
        pred: ['I-StationDest', 'B-Criterion', 'X'])
        <empty line>
    """
    out = 'text: {}\ntrue: {}\npred: {}\n\n'.format(ner_data.text, ner_data.true, ner_data.pred)
    return out


def evaluate_ner_pred_result(h_params: HParams, result: Iterable[ndarray]):
    """Evaluate prediction result
        result is a generator which creates one ndarray of length 128 for each prediction"""
    """Note that evaluation code by kyzhouhzau is available in Github"""

    # label2id.pkl is different for each run (since set in get_unique_labels is not ordered)
    with open(os.path.join(h_params.local_dir, 'label2id.pkl'), 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}

    ner_datas = get_pred_true(h_params)
    updated_ner_datas = []
    for ner_data, prediction in zip(ner_datas, result):
        n = len(ner_data.text)

        pred = list(id2label[pred_id] if pred_id != 0 else '<0>' for i, pred_id in enumerate(prediction) if i < n)
        updated_ner_datas.append(NERData(ner_data.text, ner_data.true, pred))

    for ner_data in updated_ner_datas:
        print(convert_ner_str(ner_data))

    with open(os.path.join(h_params.local_dir, 'results.txt'), 'w') as w:
        w.writelines(map(lambda data: convert_ner_str(data), updated_ner_datas))


def main(h_params: HParams):
    result = run(h_params)
    evaluate_ner_pred_result(h_params, result)
    print_scores(Path(h_params.local_dir) / 'results.txt')


if __name__ == "__main__":
    tf.app.run()
