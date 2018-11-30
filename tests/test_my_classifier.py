from src.data_reader import get_filename
from src.my_classifier import SetType, get_examples
from src.my_types import Corpus
from src.utils import get_project_root
from src.config import get_debug_hparams
import tensorflow as tf
from src.my_classifier import get_model_fn_and_estimator, evaluate, train, predict
from src.utils import clean_folder
tf.logging.set_verbosity(tf.logging.ERROR)
from pathlib import Path


def test_get_examples():
    filename = get_project_root() / get_filename(Corpus.ASKUBUNTU)
    test = get_examples(filename, SetType.test)
    assert 109 == len(test)
    first_test_example = test[0].__dict__
    assert 'What software can I use to view epub documents?' == first_test_example['text_a']
    assert not first_test_example['text_b']
    assert 'Software Recommendation' == first_test_example['label']

    assert 53 == len(get_examples(filename, SetType.train))


def validate_debug_params():
    hparams = get_debug_hparams()
    assert 3 == hparams.num_train_steps
    assert 1 == hparams.train_batch_size
    assert 1 == hparams.save_summary_steps


def test_train():
    validate_debug_params()
    hparams = get_debug_hparams()
    _, estimator = get_model_fn_and_estimator(hparams)
    train(hparams, estimator)

    output_dir = Path(hparams.output_dir)
    if output_dir.is_dir():
        clean_folder(output_dir)

