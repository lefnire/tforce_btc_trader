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


# One array per running instance (ie, if you have 2 separate tabs running hypersearch.py, then you'll want an array of
# two arrays. `--guess 0` will go through all the overrides in the first array, `--guess 1` all the overrides in the
# second array
guess_overrides = [
    [
        {},  # usually want 1 empty dict, which means "try the hard-coded defaults"
        {'pct_change': True},
        {'pct_change': True, 'scale': False},
        {'single_action': False},
        {'net.width': 4},
        {'step_optimizer.learning_rate': 6.3, 'optimization_steps': 19},
        {'net.depth_post': 2},
        {'punish_repeats': 5000},
        {'batch_size': 10},
        {'step_window': 400}
    ],
    [
        # Winner roughly according to PPO paper / TensorForce defaults (doesn't work for me)
        {'arbitrage': False,
         'baseline_mode': True,
         'batch_size': 10,
         'discount': 0.99,
         'entropy_regularization': 2.,
         'gae_lambda': 0.95,
         'indicators': True,
         'keep_last_timestep': True,
         'likelihood_ratio_clipping': .2,
         'net.activation': 'tanh',
         'net.depth_mid': 3,
         'net.depth_post': 2,
         'net.dropout': 0.5,
         'net.funnel': True,
         'net.l1': 7.,  # this exeeds threshold, so it's "off"
         'net.l2': 2.,
         'net.stride': 3,
         'net.type': 'conv2d',
         'net.width': 8,
         'net.window': 1,
         'optimization_steps': 15,
         'pct_change': False,
         'punish_repeats': 20000,
         'scale': True,
         'step_optimizer.learning_rate': 6.5,
         'step_optimizer.type': 'adam',
         'step_window': 250,
         'single_action': False},
    ]
]