from src.utils import get_project_root
from src.read_ner import get_ner_lines, get_unique_labels


test_filename = get_project_root() / 'data' / 'chatbot'


def test_get_ner_lines():
    lines = get_ner_lines(test_filename / 'test.txt')
    assert len(lines) == 105
    assert lines[0][0] == 'O O O O B-StationDest'
    assert lines[0][1] == 'i want to go marienplatz'


def test_get_unique_labels():
    labels = get_unique_labels(test_filename)
    # order might change for each execution
    expected = ['Criterion', 'Line', 'StationDest', 'StationStart', 'Vehicle']
    for label in expected:
        # it might be that some I-<label> is missing, this could be caused by (small) dataset.
        full_label = 'B-{}'.format(label)
        assert full_label in labels
    assert len(labels) == 13
