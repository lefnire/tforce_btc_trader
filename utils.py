import numpy as np
from enum import Enum


class ScoreMode(Enum):
    MEAN = 1  # mean of all episodes
    LAST = 2  # final episode (the one w/o killing)
    CONSECUTIVE = 3  # max # consecutive-positives


MODE = ScoreMode.MEAN


def calculate_score(run):
    advantages = run['advantages']

    if MODE == ScoreMode.MEAN:
        mean = np.mean(advantages)
        if mean == 0: return -100  # no holders allowed
        return mean
    elif MODE == ScoreMode.LAST:
        return advantages[-1]
    elif MODE == ScoreMode.CONSECUTIVE:
        score, curr_consec = 0, 0
        for i, adv in enumerate(advantages):
            if adv > 0:
                curr_consec += 1
                continue
            if curr_consec > score:
                score = curr_consec
            curr_consec = 0
        return score
