from pathlib import Path
from typing import List, Iterable
import numpy as np
from sklearn.metrics import f1_score


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def convert_result_pred(result: Iterable[np.ndarray], label_list: List[str]) -> List[str]:
    """Converts result returned by estimator.evaluate or estimator.predict to list of predictions."""
    return list(map(lambda prediction: label_list[prediction.argmax()], result))


def get_y_true(file: Path) -> List[str]:
    """Get gold standard labels from tsv file."""
    y_true = []
    with open(str(file), 'r') as f:
        for row in f:
            row = row.replace('\n', '')
            lines = row.split('\t')
            y_true.append(lines[1])
    return y_true


def get_rounded_f1(file: Path, y_pred: List[str], average='micro') -> float:
    """Returns rounded f1 score"""
    return round(f1_score(get_y_true(file), y_pred, average=average), 3)


def print_eval_results(results: List[dict]):
    for result in results:
        print(result)
