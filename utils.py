import numpy as np
from enum import Enum


class ScoreMode(Enum):
    """Different ways we might consider scoring our runs. This is for BO's sake, not for our RL agent -
    ie helps us decide which hyper combos to pursue."""
    MEAN = 1  # mean of all episodes
    LAST = 2  # final episode (the one w/o killing)
    POS = 3  # max # positive tests
    CONSECUTIVE_POS = 4  # max # *consecutive* positives


MODE = ScoreMode.MEAN


def calculate_score(advantages):
    if MODE == ScoreMode.MEAN:
        mean = np.mean(advantages)
        if mean == 0: return -100  # no holders allowed
        return mean
    elif MODE == ScoreMode.LAST:
        return advantages[-1]
    elif MODE == ScoreMode.POS:
        return sum(1 for x in advantages if x > 0)
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
    parser.add_argument('-g', '--gpu-split', type=float, default=1, help="Num ways we'll split the GPU (how many tabs you running?)")
    parser.add_argument('-n', '--net-type', type=str, default='conv2d')
    parser.add_argument('-t', '--n-tests', type=int, default=30, help="Number of times to split to training and run a test. This slows things down, so balance graph resolution w/ performance.")
    parser.add_argument('-s', '--n-steps', type=int, default=300, help="Number of thousands of timesteps total to train.")


# One array per running instance (ie, if you have 2 separate tabs running hypersearch.py, then you'll want an array of
# two arrays. `--guess 0` will go through all the overrides in the first array, `--guess 1` all the overrides in the
# second array
guess_overrides = [
    [
        {},  # usually want 1 empty dict, which means "try the hard-coded defaults"
        {'scale': False},
        {'step_window': 400},
        {'batch_size': 10},
        {'net.depth_post': 2},
        {'pct_change': False}
    ],
    [
        {'repeat_last_state': True},
        {'punish_repeats': 5000},
        {'net.width': 4},
        {'single_action': False},
        {'step_optimizer.learning_rate': 7, 'optimization_steps': 20},
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
         'net.depth_post': 1,
         'net.dropout': .001,
         'net.funnel': True,
         'net.l1': 7.,  # this exeeds threshold, so it's "off"
         'net.l2': 2.,
         'net.stride': 3,
         'net.type': 'conv2d',
         'net.width': 8,
         'net.window': 1,
         'optimization_steps': 20,
         'pct_change': False,
         'punish_repeats': 20000,
         'scale': True,
         'step_optimizer.learning_rate': 3.,
         'step_optimizer.type': 'adam',
         'step_window': 250,
         'single_action': True},
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