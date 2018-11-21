import os

import tokenization
from run_classifier import DataProcessor, InputExample
from typing import Iterable
import csv


def get_intents(data_dir) -> Iterable[str]:
    with open(str(data_dir + '/test.tsv'), 'r', encoding='utf8', newline='') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t', lineterminator='\n')

        for row in tsv_reader:
            yield row[1]


class IntentProcessor(DataProcessor):
    """Processor for the intent classification data set."""
    data_dir: str

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        if not self.data_dir:
            raise AssertionError('Set IntentProcessor.data_dir')

        return list(set(get_intents(self.data_dir)))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
