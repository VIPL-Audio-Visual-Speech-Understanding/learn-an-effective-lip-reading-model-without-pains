from typing import List


def load_labels() -> List[str]:
    with open('label_sorted.txt') as f:
        labels = f.read().splitlines()
    return labels
