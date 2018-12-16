from typing import List
from src.config import HParams
from pathlib import Path


def get_ner_contents(filename: Path) -> str:
    raise NotImplementedError


def get_unique_labels(h_params: HParams) -> List[str]:
    """Returns unique labels for some NER file."""
    return []
