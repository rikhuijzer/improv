import tensorflow as tf

import src.tokenization as tokenization
from src.config import get_debug_params
from src.my_classifier import (
    get_f1_score, IntentProcessor, get_model_and_estimator, evaluate, train, predict, MetadataHook
)


def main():
    params = get_debug_params()

    tf.gfile.MakeDirs(str(params.output_dir))

    processor = IntentProcessor()
    processor.data_dir = params.data_dir
    tokenizer = tokenization.FullTokenizer(vocab_file=str(params.vocab_file), do_lower_case=params.do_lower_case)
    model_fn, estimator = get_model_and_estimator(params, processor)
    hook = MetadataHook(save_steps=1, output_dir=params.output_dir)

    if params.do_train:
        train(params, processor, tokenizer, estimator, hook)

    if params.do_eval:
        evaluate(params, processor, tokenizer, estimator)

    # note that predictions are non-deterministic
    params = params._replace(use_tpu=False)
    if params.do_predict:
        y_pred = predict(params, processor, tokenizer, estimator)
        print('f1 score: {}'.format(get_f1_score(params, y_pred)))


if __name__ == '__main__':
    main()
