import tensorflow as tf
from functools import lru_cache
import csv
from pathlib import Path
from typing import Tuple, Iterable, List
from rasa_nlu.training_data import Message
import re


def read_tsv(filename: Path, quotechar=None) -> Iterable[List[str]]:
    """Reads a tab separated value file."""
    with tf.gfile.Open(str(filename), "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        next(reader, None)  # skip the header
        for line in reader:
            yield line  # line:  ['What IRC clients are available?', 'Software Recommendation'] type:  <class 'list'>


def convert_annotated_text(annotated: str) -> str:
    return re.sub('[\(].*?[\)]', '', annotated).replace('[', '').replace(']', '')


def convert_line_message(line: List[str]) -> Message:
    """Return message without entities. Good enough for now."""
    message = Message.build(text=convert_annotated_text(line[0]), intent=line[1], entities=[])
    if len(line) == 3:
        message.data['training'] = line[2] == 'True'
    return message


def get_messages(filename: Path) -> Iterable[Message]:
    return map(convert_line_message, read_tsv(filename))


@lru_cache(maxsize=2)
def get_filtered_messages(filename: Path, training: bool) -> Tuple[Message]:
    return tuple(filter(lambda m: m.data['training'] == training, get_messages(filename)))
