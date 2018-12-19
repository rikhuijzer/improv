"""A simple smoke test that runs these examples for 1 training iteration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
from six.moves import StringIO

import improv.iris_estimator.iris_data as iris_data
from improv.iris_estimator.custom_estimator import main
from improv.utils import get_project_root, clean_folder, reduce_output


FOUR_LINES = '\n'.join([
    '1,52.40, 2823,152,2',
    '164, 99.80,176.60,66.20,1',
    '176,2824, 136,3.19,0',
    '2,177.30,66.30, 53.10,1', ])


def four_lines_data():
    text = StringIO(FOUR_LINES)

    df = pd.read_csv(text, names=iris_data.CSV_COLUMN_NAMES)

    xy = (df, df.pop('Species'))
    return xy, xy


'''
def get_tf_event_values(folder: Path):
    for e in tf.train.summary_iterator(str(folder)):
        for v in e.summary.value:
            print(v)
            # if v.tag == 'loss' or v.tag == 'accuracy':
             #    print(v.simple_value)
'''


def test_main():
    reduce_output()
    model_dir = get_project_root() / 'tmp' / 'custom_estimator'
    if model_dir.is_dir():
        clean_folder(model_dir)

    args = [
        None,
        '--train_steps=10',
        '--model_dir={}'.format(model_dir)
    ]
    tf.logging.set_verbosity(tf.logging.WARN)
    main(args)

    # tf_events = find_tf_events(model_dir)
    # print(tf_events)
    # get_tf_event_values(tf_events)
