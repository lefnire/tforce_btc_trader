import numpy as np
from enum import Enum


class ScoreMode(Enum):
    """Different ways we might consider scoring our runs. This is for BO's sake, not for our RL agent -
    ie helps us decide which hyper combos to pursue."""
    MEAN = 1  # mean of all episodes
    LAST = 2  # final episode (the one w/o killing)
    POS = 3  # max # positive tests
    CONSECUTIVE_POS = 4  # max # *consecutive* positives
    TOTAL = 5
    MIX = 6


MODE = ScoreMode.MIX


def calculate_score(advantages):
    for i, a in enumerate(advantages):
        if a == 0.: advantages[i] = -1.
    if MODE == ScoreMode.MEAN:
        return np.mean(advantages)
    elif MODE == ScoreMode.LAST:
        return advantages[-1]
    elif MODE == ScoreMode.MIX:
        return np.mean(advantages[:-1]) + advantages[-1]
    elif MODE == ScoreMode.POS:
        return sum(1 for x in advantages if x > 0)
    elif MODE == ScoreMode.TOTAL:
        return sum(x for x in advantages)
    elif MODE == ScoreMode.CONSECUTIVE_POS:
        score, curr_consec = 0, 0
        for i, adv in enumerate(advantages):
            if adv > 0:
                curr_consec += 1
                continue
            if curr_consec > score:
                score = curr_consec
            curr_consec = 0
        return score


def add_common_args(parser):
    # parser.add_argument('-g', '--gpu-split', type=float, default=1, help="Num ways we'll split the GPU (how many tabs you running?)")
    parser.add_argument('-n', '--net-type', type=str, default='lstm')
    parser.add_argument('-t', '--n-tests', type=int, default=30, help="Number of times to split to training and run a test. This slows things down, so balance graph resolution w/ performance.")
    parser.add_argument('-s', '--n-steps', type=int, default=80, help="Number of 1k timesteps total to train. (using 50 means 500,000)")
    parser.add_argument('--autoencode', action="store_true", default=False, help="If you're running out of GPU memory, try --autoencode which scales things down")
    parser.add_argument('--clear-scalers', action="store_true", default=False, help="Should we delete the saved reward/state scaler.pkl objects, start over?")


# One array per running instance (ie, if you have 2 separate tabs running hypersearch.py, then you'll want an array of
# two arrays. `--guess 0` will go through all the overrides in the first array, `--guess 1` all the overrides in the
# second array
guess_overrides = [
    [
        {},  # usually want 1 empty dict, which means "try the hard-coded defaults"
    ]
]


class DotDict(object):
    """
    Utility class that lets you get/set attributes with a dot-seperated string key, like `d = a['b.c.d']` or `a['b.c.d'] = 1`
    """
    def __init__(self, obj):
        self._data = obj
        self.update = self._data.update

    def __getitem__(self, path):
        v = self._data
        for k in path.split('.'):
            if k not in v:
                return None
            v = v[k]
        return v

    def __setitem__(self, path, val):
        v = self._data
        path = path.split('.')
        for i, k in enumerate(path):
            if i == len(path) - 1:
                v[k] = val
                return
            elif k in v:
                v = v[k]
            else:
                v[k] = {}
                v = v[k]

    def to_dict(self):
        return self._data
