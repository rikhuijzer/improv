from src.utils import get_project_root, convert_result_pred, get_y_true, find_tf_events
import numpy as np


def test_convert_result_pred():
    result = np.zeros((2, 2))
    result[0][1] = 1.0
    label_list = ['A', 'B']
    expected = ['B', 'A']  # highest confidence is for B in first row
    assert expected == convert_result_pred(result, label_list)


def test_get_y_true():
    file = get_project_root() / 'data' / 'askubuntu' / 'askubuntu.tsv'
    y_true = get_y_true(file, training=False)
    assert 'Software Recommendation' == y_true[0]
    assert 109 == len(y_true)


def test_find_tf_events():
    folder = get_project_root() / 'tests'
    assert 'events.out.tfevents.000000' == find_tf_events(folder).name
