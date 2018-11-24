from src.utils import get_project_root, convert_result_pred, get_y_true
import numpy as np


def test_convert_result_pred():
    result = np.zeros((2, 2))
    result[0][1] = 1.0
    label_list = ['A', 'B']
    expected = ['B', 'A']  # highest confidence is for B in first row
    assert expected == convert_result_pred(result, label_list)


def test_get_y_true():
    file = get_project_root() / 'data' / 'askubuntu' / 'test.tsv'
    y_true = get_y_true(file)
    assert 'Make Update' == y_true[0]
    assert 33 == len(y_true)
