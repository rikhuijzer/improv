from src.config import get_debug_hparams
from src.kyzhouhzau_ner import main


def test_eval():
    """Smoke test"""
    h_params = get_debug_hparams()._replace(do_train=False, do_eval=True, do_predict=False)
    main(h_params)


def test_pred():
    """Smoke test"""
    h_params = get_debug_hparams()._replace(do_train=False, do_eval=False, do_predict=True)
    main(h_params)
