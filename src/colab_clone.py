import tensorflow as tf

from src.config import get_debug_hparams
from src.my_classifier import (
    get_model_fn_and_estimator, evaluate, train, predict
)


def main():
    params = get_debug_hparams()

    tf.gfile.MakeDirs(str(params.output_dir))

    model_fn, estimator = get_model_fn_and_estimator(params)

    if params.do_train:
        train(params, estimator)

    if params.do_eval:
        evaluate(params, estimator)

    # note that predictions are non-deterministic
    if params.do_predict:
        predict(params)


if __name__ == '__main__':
    main()
