from src.utils import convert_result_pred
import numpy as np


def test_convert_result_pred():
    result = np.zeros((2, 2))
    result[0][1] = 1.0
    label_list = ['A', 'B']
    expected = ['B', 'A']  # highest confidence is for B in first row
    assert expected == convert_result_pred(result, label_list)
