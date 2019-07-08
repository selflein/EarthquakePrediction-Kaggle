import functools
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import pandas as pd

TRAIN_CSV = Path('./data/train.csv')
TEST_FOLDER = Path('./data/test')


def cache(path):
    """Wrapper to cache function results on disk.

    Params:
        path: Path to disk location where to cache function result.

    Returns:
         Wrapped function.
    """
    def wrapper(function):
        @functools.wraps(function)
        def f(*args, **kwargs):
            if Path(path).exists():
                print('Loading cache from {}'.format(path))
                return torch.load(path)
            else:
                print('Caching function result at {}'.format(path))
                result = function(*args, **kwargs)
                torch.save(result, path)
        return f
    return wrapper


@cache(path='./.cache/train')
def read_train_from_disk():
    df = pd.read_csv(
        TRAIN_CSV,
        dtype={
            'acoustic_data': np.int16,
            'time_to_failure': np.float32
        }
    )
    return df


def read_test():
    return [(f.stem, pd.read_csv(f, dtype={'acoustic_data': np.int16}))
            for f in TEST_FOLDER.iterdir()]


class StatsTracker:
    """Utility to keep track of training stats and print them to CSV.
    """
    def __init__(self):
        self.stats = defaultdict(list)

    def add_entry(self, stats):
        for stat, value in stats.items():
            self.stats[stat].append(value)

    def compare(self, metric):
        try:
            return True if self.stats[metric][-1] <= min(self.stats[metric]) else False
        except IndexError:
            return True

    def save(self, path):
        pd.DataFrame(self.stats).to_csv(path)


class EpochLogger:
    def __init__(self):
        self.metrics = defaultdict(float)

    def update(self, stats):
        for stat, value in stats.items():
            self.metrics[stat] = (self.metrics[stat] + value) / 2.

