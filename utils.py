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


MODE = ScoreMode.MEAN


def calculate_score(scores):
    for i, a in enumerate(scores):
        if a == 0.: scores[i] = -1.
    if MODE == ScoreMode.MEAN:
        return np.mean(scores)
    elif MODE == ScoreMode.LAST:
        return scores[-1]
    elif MODE == ScoreMode.MIX:
        return np.mean(scores[:-1]) + scores[-1]
    elif MODE == ScoreMode.POS:
        return sum(1 for x in scores if x > 0)
    elif MODE == ScoreMode.TOTAL:
        return sum(x for x in scores)
    elif MODE == ScoreMode.CONSECUTIVE_POS:
        score, curr_consec = 0, 0
        for i, adv in enumerate(scores):
            if adv > 0:
                curr_consec += 1
                continue
            if curr_consec > score:
                score = curr_consec
            curr_consec = 0
        return score


def add_common_args(parser):
    parser.add_argument('-g', '--gpu-split', type=float, default=1, help="Num ways we'll split the GPU (how many tabs you running?)")
    parser.add_argument('--autoencode', action="store_true", help="If you're running out of GPU memory, try --autoencode which scales things down")


last_good_commit = '6a6e49c'


def raise_refactor():
    raise NotImplemented(f'Restore from {last_good_commit}')