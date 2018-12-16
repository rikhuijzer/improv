# from src.kyzhouhzau_ner import NerProcessor, DataProcessor
from src.utils import get_project_root


def test_read_data():
    # processor = DataProcessor
    input_file = str(get_project_root() / 'data' / 'chatbot' / 'test.txt')
    # data = processor._read_data(input_file)
    # print(data)


def test_read_ner_file():
    raise NotImplementedError


def test_get_unique_labels():
    raise NotImplementedError
