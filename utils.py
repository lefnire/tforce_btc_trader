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
        {},  # guess 0.0 should always be an empty dict, which means "try the hard-coded defaults"
        {'net.l1': 2.5, 'net.l2': 5.},
        {'pct_change': True},
    ],
    [
        {'net.width': 4},
        {'step_optimizer.learning_rate': 5.5},
        {'step_optimizer.learning_rate': 7.5},
    ],
    [
        {'net.width': 8},
        {'net.activation': 'relu'},
        {'net.stride': 2},
    ],
    [
        {'unimodal': True},
        {'scale': False},
        {'step_window': 400},
    ],
    [
        {'net.depth_post': 2},
        {'net.window': 1},
        {'net.window': 3},
    ],
    [
        {'baseline_mode': False},
        {'arbitrage': False},
        # {'punish_repeats': True},
    ],
    [
        # Winner from last runs
        {'arbitrage': False,
         'baseline_mode': True,
         'batch_size': 5,
         'discount': 0.9403733613952013,
         'entropy_regularization': 2.45655628599285,
         'gae_lambda': 0.9732554076031417,
         'indicators': True,
         'keep_last_timestep': False,
         'likelihood_ratio_clipping': 0.09788877896855552,
         'net.activation': 'tanh',
         'net.depth_mid': 3,
         'net.depth_post': 1,
         'net.dropout': 0.27533954873162136,
         'net.funnel': True,
         'net.l1': 5.815603903753169,
         'net.l2': 1.6711235997971334,
         'net.stride': 2,
         'net.type': 'conv2d',
         'net.width': 8,
         'net.window': 3,
         'optimization_steps': 29,
         'pct_change': False,
         # 'punish_repeats': False,
         'scale': True,
         'step_optimizer.learning_rate': 7.918741845681779,
         'step_optimizer.type': 'adam',
         'step_window': 229,
         'unimodal': True},

        {'batch_size': 5},
        {'batch_size': 10},
    ],
    [
        # Winner roughly according to PPO paper / TensorForce defaults (doesn't work for me)
        {'arbitrage': True,
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
         'unimodal': False},
    ]
]