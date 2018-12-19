from enum import Enum, auto
from typing import NamedTuple, List


class Corpus(Enum):
    ASKUBUNTU = auto()
    CHATBOT = auto()
    WEBAPPLICATIONS = auto()
    SNIPS2017 = auto()


class NERData(NamedTuple):
    text: List[str]
    true: List[str]
    pred: List[str]
