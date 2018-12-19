import time
from pathlib import Path

from src import run_classifier as rc
from src.utils import get_project_root


#
# BEGIN GLOBAL PARAMETERS
#
DEBUG = False
BERT_BASE_DIR = Path('/home/rik/Downloads/bert_multilingual')
TASK = 'askubuntu'  # askubuntu, ColA, MRPC

#
# END GLOBAL PARAMETERS


def classify_sentence():
    rc.FLAGS.task_name = TASK
    rc.FLAGS.do_train = True
    rc.FLAGS.do_eval = True
    rc.FLAGS.data_dir = str(get_project_root() / 'data' / TASK)
    rc.FLAGS.vocab_file = str(BERT_BASE_DIR / 'vocab.txt')
    rc.FLAGS.bert_config_file = str(BERT_BASE_DIR / 'bert_config.json')
    rc.FLAGS.init_checkpoint = str(BERT_BASE_DIR / 'bert_model.ckpt')
    rc.FLAGS.max_seq_length = 128
    rc.FLAGS.train_batch_size = 16  # using too much memory at 32 batches
    rc.FLAGS.learning_rate = 2e-5
    rc.FLAGS.num_train_epochs = 1
    rc.FLAGS.output_dir = str(get_project_root() / 'output' / '1_epochs')

    start_time = time.time()

    rc.main('')
    running_time = round(((time.time() - start_time) / 60), 2)
    print('Execution took {} minutes'.format(running_time))


if __name__ == '__main__':
    classify_sentence()
