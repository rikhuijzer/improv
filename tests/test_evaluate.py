from improv.utils import get_project_root
from improv.evaluate import parse_file, print_scores


# this filename should be updated when output format changes (which possibly will happen)
filename = get_project_root() / 'runs' / '2018-12-20 chatbot' / 'results.txt'


def test_parse_file():
    ner_datas = parse_file(filename)
    for ner_data in ner_datas:
        print(ner_data)


def test_evaluate():
    print_scores(filename)
