from src.data_reader import get_filename
from src.my_classifier import SetType, get_examples
from src.my_types import Corpus
from src.utils import get_project_root
import tensorflow as tf


def test_get_examples():
    tf.logging.set_verbosity(tf.logging.WARN)
    filename = get_project_root() / get_filename(Corpus.ASKUBUNTU)
    assert 53 == len(get_examples(filename, SetType.train))
    assert 109 == len(get_examples(filename, SetType.test))
