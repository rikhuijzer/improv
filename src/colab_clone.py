import tensorflow as tf

import src.tokenization as tokenization
from src.config import get_debug_params
from src.my_classifier import IntentProcessor, get_model_and_estimator, evaluate, train, predict


def main():
    params = get_debug_params()

    tf.gfile.MakeDirs(str(params.output_dir))

    processor = IntentProcessor()
    processor.data_dir = params.data_dir
    tokenizer = tokenization.FullTokenizer(vocab_file=str(params.vocab_file), do_lower_case=params.do_lower_case)
    model_fn, estimator = get_model_and_estimator(params, processor)

    if params.do_train:
        train(params, processor, tokenizer, estimator)

    if params.do_eval:
        evaluate(params, processor, tokenizer, estimator)

    if params.do_predict:
        predict(params, processor, tokenizer, estimator)


if __name__ == '__main__':
    main()
