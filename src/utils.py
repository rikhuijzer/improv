from pathlib import Path
from typing import List, Iterable
import numpy as np


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def convert_result_pred(result: Iterable[np.ndarray], label_list: List[str]) -> List[str]:
    """Converts result returned by estimator.evaluate or estimator.predict to list of predictions."""
    return list(map(lambda prediction: label_list[prediction.argmax()], result))
