from pathlib import Path

from improv.config import get_debug_hparams
from improv.data_reader import get_filename
from improv.my_classifier import (
    SetType, get_examples, get_model_fn_and_estimator, train, get_intents, get_unique_intents, train_evaluate
)
from improv.my_types import Corpus
from improv.utils import clean_folder, get_project_root, reduce_output


def get_debug_filename() -> Path:
    return get_project_root() / get_filename(Corpus.ASKUBUNTU)


def test_get_examples():
    filename = get_debug_filename()
    test = get_examples(filename, SetType.test)
    assert 109 == len(test)
    first_test_example = test[0].__dict__
    assert 'What software can I use to view epub documents?' == first_test_example['text_a']
    assert not first_test_example['text_b']
    assert 'Software Recommendation' == first_test_example['label']

    assert 53 == len(get_examples(filename, SetType.train))


def test_get_intents():
    intents = get_intents(get_debug_filename(), training=False)
    assert 109 == len(intents)
    assert 'Software Recommendation' == intents[0]
    assert 'Setup Printer' == intents[-1]


def test_get_unique_intents():
    intents = get_unique_intents(get_debug_filename())
    assert 'Shutdown Computer' in intents
    assert 5 == len(intents)


def validate_debug_params():
    hparams = get_debug_hparams()
    assert 2 == hparams.num_train_steps
    assert 1 == hparams.save_checkpoints_steps
    assert 1 == hparams.train_batch_size
    assert 1 == hparams.save_summary_steps


# TODO: Set learning rate very high such that it can correctly predict some sentence if we ask for that sentence.


def test_train():
    reduce_output()
    validate_debug_params()
    hparams = get_debug_hparams()
    hparams = hparams._replace(output_dir=str(get_project_root() / 'tmp' / 'test_my_classifier_train'))
    _, estimator = get_model_fn_and_estimator(hparams)
    train(hparams, estimator, max_steps=2)

    clean_folder(hparams.output_dir)


def test_train_evaluate():
    reduce_output()
    validate_debug_params()
    hparams = get_debug_hparams()
    hparams = hparams._replace(output_dir=str(get_project_root() / 'tmp' / 'test_my_classifier_train_eval'))
    _, estimator = get_model_fn_and_estimator(hparams)
    train_evaluate(hparams, estimator)
