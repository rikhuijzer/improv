import shutil
from pathlib import Path
from typing import List, Iterable

import numpy as np
from sklearn.metrics import f1_score

from improv.data_reader import get_filtered_messages
import tensorflow as tf
import os
import warnings


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def convert_result_pred(result: Iterable[np.ndarray], label_list: List[str]) -> List[str]:
    """Converts result returned by estimator.evaluate or estimator.predict to list of predictions."""
    return list(map(lambda prediction: label_list[prediction.argmax()], result))


def get_y_true(file: Path, training: bool) -> List[str]:
    """Get gold standard labels from tsv file."""
    return list(map(lambda m: m.data['intent'], get_filtered_messages(file, training=training)))


def get_rounded_f1(file: Path, y_pred: List[str], average='micro') -> float:
    """Returns rounded f1 score"""
    return round(f1_score(get_y_true(file, training=False), y_pred, average=average), 3)


def print_eval_results(results: List[dict]):
    for result in results:
        print(result)


def find_tf_events(folder: Path) -> Path:
    """Returns first file containing 'tfevents' in filename from folder."""
    for filename in folder.glob('./*'):
        if 'tfevents' in str(filename):
            return filename


def get_f1_score(labels, predicted_classes) -> float:
    return -1.0


def clean_folder(folder: Path):
    """Makes sure that folder is an existing empty folder."""
    if folder.is_dir():
        shutil.rmtree(str(folder))
    tf.gfile.MakeDirs(str(folder))


def reduce_output():
    """Reduces the output in terminal. Do not move warnings.simplefilter to module."""
    tf.logging.set_verbosity(tf.logging.ERROR)
    warnings.simplefilter('ignore')  # warnings still show up in the pytest warning summary
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignores CPU instructions warning
