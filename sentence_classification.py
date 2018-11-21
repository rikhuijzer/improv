import time
from pathlib import Path

import run_classifier as rc
from utils import get_root


#
# BEGIN GLOBAL PARAMETERS
#
DEBUG = False
BERT_BASE_DIR = Path('/home/rik/Downloads/bert_multilingual')
TASK = 'askubuntu'  # askubuntu, ColA, MRPC

#
# END GLOBAL PARAMETERS
#
# TODO: Be able to train and classify intents locally using few epochs.
# TODO: !!! CHANGE DEV TO REAL DEV DATA INSTEAD OF TEST !!!


def classify_sentence():
    rc.FLAGS.task_name = TASK
    rc.FLAGS.do_train = False
    rc.FLAGS.do_eval = True
    rc.FLAGS.data_dir = str(get_root() / 'data' / TASK)
    rc.FLAGS.vocab_file = str(BERT_BASE_DIR / 'vocab.txt')
    rc.FLAGS.bert_config_file = str(BERT_BASE_DIR / 'bert_config.json')
    rc.FLAGS.init_checkpoint = str(BERT_BASE_DIR / 'bert_model.ckpt')
    rc.FLAGS.max_seq_length = 128
    rc.FLAGS.train_batch_size = 16 if DEBUG else 32  # reducing memory usage on local system
    rc.FLAGS.learning_rate = 2e-5
    rc.FLAGS.num_train_epochs = 1e-3 if DEBUG else 3.0
    rc.FLAGS.output_dir = str(get_root() / 'output')

    start_time = time.time()

    rc.main('')
    running_time = round(((time.time() - start_time) / 60), 2)
    print('Execution took {} minutes'.format(running_time))


if __name__ == '__main__':
    classify_sentence()
