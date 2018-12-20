from pathlib import Path
from improv.my_types import NERData
from typing import Iterable, List, Tuple
import ast
from sklearn.metrics import f1_score
from improv.utils import get_project_root
from itertools import chain


def read_file(filename: Path) -> List[str]:
    with open(str(filename), 'r') as f:
        return f.readlines()


def convert_triple(triple: List[str]) -> NERData:
    """Convert a triple of text, true, pred to NERData."""
    def convert_item_list(item: str) -> List[str]:
        return ast.literal_eval(item[6:])

    items = [convert_item_list(item) for item in triple]
    return NERData(items[0], items[1], items[2])


def parse_file(filename: Path) -> Iterable[NERData]:
    lines = read_file(filename)

    lines = [line for line in lines if line != '\n']
    if len(lines) % 3 != 0:
        raise AssertionError('Results file expected to be a multiple of 3.')

    my_range = [3 * i for i in range(len(lines) // 3)]
    ner_datas = map(lambda i: convert_triple(lines[i:i + 3]), my_range)

    return ner_datas


def get_intents(ner_datas: Tuple[NERData]) -> Tuple[List[str], List[str]]:
    y_true = [ner_data.true[0] for ner_data in ner_datas]
    y_pred = [ner_data.pred[0] for ner_data in ner_datas]
    return y_true, y_pred


def get_entities(ner_datas: Tuple[NERData]) -> Tuple[List[str], List[str]]:
    y_true = [ner_data.true[1:] for ner_data in ner_datas]
    y_pred = [ner_data.pred[1:] for ner_data in ner_datas]
    y_true = [item for sublist in y_true for item in sublist]  # flatten list
    y_pred = [item for sublist in y_pred for item in sublist]  # flatten list
    return y_true, y_pred


def rounded_f1(y_true: Iterable, y_pred: Iterable) -> float:
    return round(f1_score(y_true, y_pred, average='weighted'), 3)


def is_ner_not_empty(ner_datas: Tuple[NERData]) -> bool:
    y_true = map(lambda ner_data: ner_data.true[1:], ner_datas)
    y_true = chain(*y_true)

    def is_not_empty(token: str) -> bool:
        return token != 'O'
    return any(map(is_not_empty, y_true))


def print_scores(filename: Path):
    """Determine intent and NER accuracy (weighted f1 score)."""
    ner_datas = tuple(parse_file(filename))

    if ner_datas[0].text[0] == 'INTENT':
        y_true, y_pred = get_intents(ner_datas)
        print('intents weighted f1: {}'.format(rounded_f1(y_true, y_pred)))

    if is_ner_not_empty(ner_datas):
        y_true, y_pred = get_entities(ner_datas)
        print('entities weighted f1: {}'.format(rounded_f1(y_true, y_pred)))


if __name__ == '__main__':
    fn = get_project_root() / 'runs' / '2018-12-20 chatbot' / 'results.txt'
    print_scores(fn)
