import tensorflow as tf
from pathlib import Path
from src.utils import get_project_root
import os
import datetime
from src.utils import get_project_root


def main():
    USE_TPU = False
    TASK = 'askubuntu'

    TASK_DATA_DIR = get_project_root() / 'data' / TASK

    # Available pretrained model checkpoints:
    #   uncased_L-12_H-768_A-12: uncased BERT base model
    #   uncased_L-24_H-1024_A-16: uncased BERT large model
    #   cased_L-12_H-768_A-12: cased BERT large model
    BERT_MODEL = 'uncased_L-12_H-768_A-12'
    BERT_PRETRAINED_DIR = Path.home() / 'Downloads' / 'uncased_L-12_H-768_A-12'

    OUTPUT_DIR_NAME = '50_epochs_large'
    # OUTPUT_DIR = 'gs://{}/bert/models/{}/{}'.format(BUCKET, TASK, OUTPUT_DIR_NAME)
    OUTPUT_DIR = get_project_root() / 'generated' / TASK / OUTPUT_DIR_NAME
    tf.gfile.MakeDirs(str(OUTPUT_DIR))
    TPU_ADDRESS = ''
    do_train = False

    '''
    Define model and estimator
    '''
    # Setup task specific model and TPU running config.

    import src.modeling as modeling
    import src.optimization as optimization
    import src.run_classifier as run_classifier
    import src.tokenization as tokenization
    from src.my_classifier import IntentProcessor

    # Model Hyper Parameters
    TRAIN_BATCH_SIZE = 1  # can't go lower
    EVAL_BATCH_SIZE = 1
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 1
    WARMUP_PROPORTION = 0.1
    MAX_SEQ_LENGTH = 128
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 1000
    ITERATIONS_PER_LOOP = 1000
    NUM_TPU_CORES = 8
    VOCAB_FILE = BERT_PRETRAINED_DIR / 'vocab.txt'
    CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
    INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
    DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

    processor = IntentProcessor()
    processor.data_dir = TASK_DATA_DIR

    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=str(VOCAB_FILE), do_lower_case=DO_LOWER_CASE)

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
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
    num_train_steps = int(len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
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

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=USE_TPU,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE)

    '''
    Train the model
    '''
    if do_train:
        # Train the model.
        train_features = run_classifier.convert_examples_to_features(
            train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
        print('***** Started training at {} *****'.format(datetime.datetime.now()))
        print('  Num examples = {}'.format(len(train_examples)))
        print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
        tf.logging.info("  Num steps = %d", num_train_steps)

        # see run_classifier.convert_single_example for feature creation
        train_input_fn = run_classifier.input_fn_builder(
            features=train_features,
            seq_length=MAX_SEQ_LENGTH,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print('***** Finished training at {} *****'.format(datetime.datetime.now()))

    '''
    Eval
    '''
    # Eval the model.
    eval_examples = processor.get_dev_examples(TASK_DATA_DIR)
    eval_features = run_classifier.convert_examples_to_features(
        eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    print('***** Started evaluation at {} *****'.format(datetime.datetime.now()))
    print('  Num examples = {}'.format(len(eval_examples)))
    print('  Batch size = {}'.format(EVAL_BATCH_SIZE))
    # Eval will be slightly WRONG on the TPU because it will truncate
    # the last batch.
    eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
    eval_input_fn = run_classifier.input_fn_builder(
        features=eval_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=True)
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    print('***** Finished evaluation at {} *****'.format(datetime.datetime.now()))
    output_eval_file = OUTPUT_DIR / 'eval_results.txt'
    with tf.gfile.GFile(str(output_eval_file), "w") as writer:
        print("***** Eval results *****")
        for key in sorted(result.keys()):
            print('  {} = {}'.format(key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == '__main__':
    main()
