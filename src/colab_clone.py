import tensorflow as tf

import src.tokenization as tokenization
from src.config import get_debug_params
from src.my_classifier import IntentProcessor, get_model_and_estimator, evaluate, train, predict
from sklearn.metrics import f1_score


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

    # note that predictions are non-deterministic
    params = params._replace(use_tpu=False)
    if params.do_predict:
        y_pred = predict(params, processor, tokenizer, estimator)
        y_true = []

        file = params.data_dir / "test.tsv"
        with open(str(file), 'r') as f:
            for row in f:
                row = row.replace('\n', '')
                lines = row.split('\t')
                y_true.append(lines[1])

        score = f1_score(y_true, y_pred, average='micro')
        score = round(score, 3)
        print('f1 score: {}'.format(score))


if __name__ == '__main__':
    main()
