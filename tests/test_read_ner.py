from improv.utils import get_project_root
from improv.read_ner import get_ner_lines, get_unique_labels, get_interesting_labels_indexes


data_dir = get_project_root() / 'data' / 'chatbot'


def test_get_ner_lines():
    lines = get_ner_lines(data_dir / 'test.txt')
    assert len(lines) == 105
    assert lines[0][0] == 'C O O O O B-StationDest'
    assert lines[0][1] == 'FindConnection i want to go marienplatz'


def test_get_unique_labels():
    labels = get_unique_labels(data_dir)
    # order might change for each execution
    expected = ['Criterion', 'Line', 'StationDest', 'StationStart', 'Vehicle']
    for label in expected:
        # it might be that some I-<label> is missing, this could be caused by (small) dataset.
        full_label = 'B-{}'.format(label)
        assert full_label in labels
    assert '[CLS]' in labels
    assert '[SEP]' in labels
    assert 'X' in labels
    assert len(labels) > 16  # changed by the intent examples, two intents occur

    assert 'DepartureTime' in labels  # this is a sentence intent example


def test_get_interesting_labels_indexes():
    unique_labels = get_unique_labels(data_dir)
    indexes = get_interesting_labels_indexes(unique_labels)
    interesting_labels = list(map(lambda index: unique_labels[index], indexes))
    assert len(interesting_labels) == 13

    from improv.read_ner import bert_tokens
    assert not any(map(lambda bert_token: bert_token in interesting_labels, bert_tokens))
