from src.my_estimator import train_and_evaluate
from src.config import get_debug_hparams


def test_train_and_evaluate():
    train_and_evaluate(get_debug_hparams())
