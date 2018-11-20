import os
import time
from pathlib import Path

import tensorflow as tf

import run_classifier as rc

#
# BEGIN GLOBAL PARAMETERS
#
DEBUG = True
BERT_BASE_DIR = Path('/home/rik/Downloads/bert_multilingual')
BERT_PRETRAINED_DIR = Path('/home/rik/Downloads/uncased_L-12_H-768_A-12')
#
# END GLOBAL PARAMETERS
#


# TODO: Be able to train and classify locally
# TODO: Convert dataset to TSV


def define_model_estimator():
    import modeling
    import run_classifier
    import tokenization


    # Model Hyper Parameters
    TRAIN_BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 0.01 if DEBUG else 3.0
    WARMUP_PROPORTION = 0.1
    MAX_SEQ_LENGTH = 128
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 1000
    ITERATIONS_PER_LOOP = 1000
    NUM_TPU_CORES = 8
    VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
    CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
    INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
    DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

    processors = {
        "cola": run_classifier.ColaProcessor,
        "mnli": run_classifier.MnliProcessor,
        "mrpc": run_classifier.MrpcProcessor,
    }
    processor = processors[TASK.lower()]()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=OUTPUT_DIR,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=ITERATIONS_PER_LOOP,
            num_shards=NUM_TPU_CORES,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

    train_examples = processor.get_train_examples(TASK_DATA_DIR)
    num_train_steps = int(
        len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    model_fn = run_classifier.model_fn_builder(
        bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
        num_labels=len(label_list),
        init_checkpoint=INIT_CHECKPOINT,
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=USE_TPU,
        use_one_hot_embeddings=True)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=USE_TPU,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE)


def classify_sentence():
    rc.FLAGS.task_name = 'MRPC'
    rc.FLAGS.do_train = False
    rc.FLAGS.do_eval = True
    rc.FLAGS.data_dir = str(BERT_BASE_DIR.parent / 'glue_data' / 'MRPC')
    rc.FLAGS.vocab_file = str(BERT_BASE_DIR / 'vocab.txt')
    rc.FLAGS.bert_config_file = str(BERT_BASE_DIR / 'bert_config.json')
    rc.FLAGS.init_checkpoint = str(BERT_BASE_DIR / 'bert_model.ckpt')
    rc.FLAGS.max_seq_length = 128
    rc.FLAGS.train_batch_size = 16  # 32
    rc.FLAGS.learning_rate = 2e-5
    rc.FLAGS.num_train_epochs = 0.01 if DEBUG else 3.0
    rc.FLAGS.output_dir = str(BERT_BASE_DIR.parent / 'bert_output')

    start_time = time.time()
    rc.main('')
    running_time = round(((time.time() - start_time) / 60), 2)
    print('Execution took {} minutes'.format(running_time))


if __name__ == '__main__':
    classify_sentence()
