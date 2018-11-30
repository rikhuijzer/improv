from src.data_reader import get_filename
from src.my_classifier import SetType, get_examples
from src.my_types import Corpus
from src.utils import get_project_root
import tensorflow as tf


def test_get_examples():
    tf.logging.set_verbosity(tf.logging.WARN)
    filename = get_project_root() / get_filename(Corpus.ASKUBUNTU)
    test = get_examples(filename, SetType.test)
    assert 109 == len(test)
    first_test_example = test[0].__dict__
    assert 'What software can I use to view epub documents?' == first_test_example['text_a']
    assert not first_test_example['text_b']
    assert 'Software Recommendation' == first_test_example['label']

    assert 53 == len(get_examples(filename, SetType.train))

