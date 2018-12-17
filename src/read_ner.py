from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Set
from itertools import chain


@lru_cache(maxsize=1)
def get_ner_lines(filename: Path) -> List[List[str]]:
    """Returns list of two strings. For example:
        ['O O O O B-StationDest O O']
        ['i want to go to marienplatz when is']
        Tokens are separated by spaces. New sentences are ignored.
        (Which is very odd considering BERT should be able to handle that information.)
        AH! The code assumes every line ends with a dot.
    """
    with open(str(filename)) as f:
        lines = []
        words = []
        labels = []
        for line in f:
            contents = line.strip()
            word = contents.split(' ')[0]
            label = contents.split(' ')[-1]

            if len(contents) == 0:  # append previous sentence to lines
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])

                # append and reset words and labels
                lines.append([l, w])
                words = []
                labels = []
                continue
            words.append(word)
            labels.append(label)
        return lines


@lru_cache(maxsize=1)
def get_unique_labels(folder: Path) -> Tuple[str]:
    """Returns unique labels for some NER file. Using tuple since this function should always return same order."""

    def helper(filename: Path) -> Set[str]:
        lines = get_ner_lines(filename)
        entities = map(lambda line: set(line[0].split(' ')), lines)
        return set(chain(*entities))

    filenames = [folder / 'test.txt', folder / 'train.txt']
    entities_per_file = map(helper, filenames)
    all_entities = set(chain(*entities_per_file))
    return tuple(all_entities)