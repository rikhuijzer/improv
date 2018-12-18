from src.config import get_debug_hparams
from src.kyzhouhzau_ner import run, evaluate_pred_result


def test_eval():
    """Smoke test"""
    h_params = get_debug_hparams()._replace(do_train=False, do_eval=True, do_predict=False)
    run(h_params)


def test_pred():
    """Smoke test"""
    h_params = get_debug_hparams()._replace(do_train=False, do_eval=False, do_predict=True)
    run(h_params)


def test_evaluate_result():
    """Evaluate prediction result."""
    h_params = get_debug_hparams()._replace(do_train=False, do_eval=False, do_predict=True)
    result = run(h_params)
    evaluate_pred_result(h_params, result)
