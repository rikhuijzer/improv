import tensorflow as tf

from improv.config import get_debug_hparams
from improv.my_classifier import (
    get_model_fn_and_estimator, evaluate, train, predict
)


def main():
    hparams = get_debug_hparams()

    tf.gfile.MakeDirs(str(hparams.output_dir))

    model_fn, estimator = get_model_fn_and_estimator(hparams)

    if hparams.do_train:
        train(hparams, estimator)

    if hparams.do_eval:
        evaluate(hparams, estimator)

    # note that predictions are non-deterministic
    if hparams.do_predict:
        predict(hparams)


if __name__ == '__main__':
    main()
