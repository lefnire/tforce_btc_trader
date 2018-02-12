"""This searches combinations of hyperparameters for the best fit using Bayesian Optimization (or optionally
Gradient Boosting). See README for more details.

Each hyper is specified as `key: {type, vals, guess, pre/post/hydrate}`.
- The key can be dot-separated, like `memory.type` (which will get expanded as dict-form)
- type: (int|bounded|bool). bool is True|False param, bounded is a float between min & max, int is "choose one"
    eg 'activation' one of (tanh|elu|selu|..)`)
- vals: the vals this hyper can take on. If type(vals) is primitive, hard-coded at this value. If type is list, then
    (a) min/max specified inside (for bounded); (b) all possible options (for 'int'). If type is dict, then the keys
    are used in the searching (eg, look at the network hyper) and the values are used as the configuration.
- guess: initial guess (supplied by human) to explore
- pre/post/hydrate: hooks that transform this hyper before plugging it. Eg, we'd use type='bounded' for batch size since
    we want to range from min to max (instead of listing all possible values); but we'd cast it to an int inside
    hook before using it.
    - pre: transform it immediately, it'll be saved in the runs table this way
    - post: transform it after all the pre-hooks are run, in case this depends on other hypers
    - hydrate: big-time transformation to the whole hypers dict, based on this hyper val. It won't be saved to the
        database looking like this. Eg, baseline_mode, when set to True, does a number on many other hypers.
"""

import argparse, json, math, time, pdb, os
from pprint import pprint
from box import Box
import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy.sql import text
from tensorforce import TensorForceError
from tensorforce.agents import agents as agents_dict
from tensorforce.core.networks import layer as TForceLayers
from tensorforce.core.networks.network import LayeredNetwork
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from btc_env import BitcoinEnv
import utils
from data import data


def build_net_spec(hypers):
    """Builds an array of dicts that conform to TForce's network specification (see their docs) by mix-and-matching
    different network hypers
    """
    net = Box(hypers['net'])

    dense = {
        'type': 'dense',
        'activation': net.activation,
        'l2_regularization': net.l2,
        'l1_regularization': net.l1
    }
    conv2d = {
        'type': 'conv2d',
        # 'bias': net.bias,
        'l2_regularization': net.l2,
        'l1_regularization': net.l1
    }
    lstm = {'type': 'internal_lstm', 'dropout': net.dropout}
    dropout = {'type': 'dropout', 'rate': net.dropout}

    arr = []

    # Pre-layer (TMK only makes sense for LSTM)
    if 'depth_pre' in net:
        for i in range(net.depth_pre):
            size = int(net.width/(net.depth_pre-i+1)) if net.funnel else net.width
            arr.append({'size': size, **dense})
            if net.dropout: arr.append({**dropout})

    # Mid-layer
    if net.type == 'conv2d':
        steps_out = hypers['step_window']

    for i in range(net.depth_mid):
        if net.type == 'lstm':
            # arr.append({'size': net.width, 'return_final_state': (i == net.depth-1), **lstm})
            arr.append({'size': net.width, **lstm})
            continue

        # For each Conv2d layer, the window/stride is a function of the step-window size. So `net.window=1` means
        # divide the step-window 10 ways; `net.window=2` means divide it 20 ways. This gives us flexibility to
        # define window/stride relative to step_window without having to know either. Note, each layer is reduced
        # from the prior, so window/stride gets recalculated
        step_window = math.ceil(steps_out / (net.window * 10))
        step_stride = math.ceil(step_window / net.stride)

        # next = (length - window)/stride + 1
        steps_out = (steps_out - step_window)/step_stride + 1

        # Ensure there's some minimal amount of reduction at the lower levels (else, we get layers that map 1-1
        # to next layer). TODO this is ugly, better way?
        min_window, min_stride = 3, 2
        step_window = max([step_window, min_window])
        step_stride = max([step_stride, min_stride])

        # This is just my hunch from CNNs I've seen; the filter sizes are much smaller than the downstream denses
        # (like 32-64-64 -> 512-256). If anyone has better intuition...
        size = max([32, int(net.width // 4)])
        # if i == 0: size = int(size / 2)  # Most convs have their first layer smaller... right? just the first, or what?
        arr.append({
            'size': size,
            'window': (step_window, 1),
            'stride': (step_stride, 1),
            **conv2d
        })
    if net.type == 'conv2d':
        arr.append({'type': 'flatten'})

    # Post Dense layers
    for i in range(net.depth_post):
        size = int(net.width / (i + 1)) if net.funnel else net.width
        arr.append({'size': size, **dense})
        if net.dropout: arr.append({**dropout})

    if net.extra_stationary:
        arr.append({'size': 9, **dense})  # TODO fiddle with size? Found 9 from a book, seems legit.
        if net.dropout: arr.append({**dropout})

    return arr


def custom_net(hypers, print_net=False):
    """First builds up an array of dicts compatible with TForce's network spec. Then passes off to a custom neural
    network architecture, rather than using TForce's default LayeredNetwork. The only reason for this is so we can pipe
    in the "stationary" inputs after the LSTM/Conv2d layers. Think about it. LTSM/Conv2d are tracking time-series data
    (price actions, volume, etc). We don't necessarily want to track our own USD & BTC balances for every time-step.
    We _could_, and it _might_ help the agent (I'm not convinced it would); but it actually causes lots of problems
    when we go live (eg, right away we take the last 6k time-steps to have a full window, but we don't have any
    BTC/USD history for that window. There are other issues). So instead we pipe the stationary inputs into the neural
    network downstream, after the time-series layers. Makes more sense to me that way: imagine the conv layers saying
    "the price is right, buy!" and then getting handed a note with "you have $0 USD". "Oh.. nevermind..."
    """
    layers_spec = build_net_spec(hypers)
    if print_net: pprint(layers_spec)

    class CustomNet(LayeredNetwork):
        def __init__(self, **kwargs):
            super(CustomNet, self).__init__(layers_spec, **kwargs)

        def tf_apply(self, x, internals, update, return_internals=False):
            """This method is copied from LayeredNetwork and modified slightly to insert stationary after the series
            layers. If anything's confusing, or if anything changes, consult original function.
            """
            series = x['series']
            stationary = x['stationary']
            x = series

            # Apply stationary to the first Dense after the last LSTM. in the case of Baseline, there's no LSTM,
            # so apply it to the start
            apply_stationary_here = 0
            for i, layer in enumerate(self.layers):
                if hypers['net']['extra_stationary'] and isinstance(layer, TForceLayers.Dense):
                    # Last Dense layer
                    apply_stationary_here = i
                elif isinstance(layer, TForceLayers.InternalLstm) or isinstance(layer, TForceLayers.Flatten):
                    # Last LSTM layer, peg to the next layer (a Dense)
                    apply_stationary_here = i + 1

            next_internals = dict()
            for i, layer in enumerate(self.layers):
                if i == apply_stationary_here:
                    x = tf.concat([x, stationary], axis=1)

                layer_internals = {name: internals['{}_{}'.format(layer.scope, name)] for name in
                                   layer.internals_spec()}
                if len(layer_internals) > 0:
                    x, layer_internals = layer.apply(x=x, update=update, **layer_internals)
                    for name, internal in layer_internals.items():
                        next_internals['{}_{}'.format(layer.scope, name)] = internal
                else:
                    x = layer.apply(x=x, update=update)

            if return_internals:
                return x, next_internals
            else:
                return x
    return CustomNet


def bins_of_8(x): return int(x // 8) * 8

def two_to_the(x, _): return 2**x

def ten_to_the_neg(x, _): return 10**-x

def min_threshold(thresh, fallback):
    """Returns x or `fallback` if it doesn't meet the threshold. Note, if you want to turn a hyper "off" below,
    set it to "outside the threshold", rather than 0.
    """
    return lambda x, _: x if (x and x > thresh) else fallback

def min_ten_neg(thresh, fallback):
    """Returns 10**-x, or `fallback` if it doesn't meet the threshold. Note, if you want to turn a hyper "off" below,
    set it to "outside the threshold", rather than 0.
    """
    return lambda x, _: min_threshold(thresh, fallback)(ten_to_the_neg(x, _), _)

def hydrate_baseline(x, flat):
    return {
        False: {'baseline_mode': None},
        True: {
            'baseline': {'type': 'custom'},
            'baseline_mode': 'states',
            'baseline_optimizer': {
                'type': 'multi_step',
                # Consider having baseline_optimizer learning hypers independent of the main learning hypers.
                # At least with PPO, it seems the step_optimizer learning hypers function quite differently than
                # expected; where baseline_optimizer's function more as-expected. TODO Investigate.
                'num_steps': 5,  # flat['baseline_optimizer.num_steps'],
                'optimizer': {
                    'type': flat['step_optimizer.type'],
                    'learning_rate': 10 ** -flat['step_optimizer.learning_rate']
                }
            },
        }
    }[x]


# Most hypers come directly from tensorforce/tensorforce/agents/ppo_agent.py, see that for documentation
hypers = {}
hypers['agent'] = {
    # 'states_preprocessing': None,
    # 'actions_exploration': None,
    # 'reward_preprocessing': None,
    'discount': {
        'type': 'bounded',
        'vals': [.9, .99],
        'guess': .97
    },
}
hypers['memory_model'] = {
    'update_mode.unit': 'episodes',
    'update_mode.batch_size': 3,  # {
        # 'type': 'bounded',
        # 'vals': [1, 10],
        # 'guess': 10,
        # 'pre': round,
    # },
    'update_mode.frequency': {
        'type': 'bounded',
        'vals': [1, 3],
        'guess': 2,
        'pre': round
    },

    'memory.type': 'latest',
    'memory.include_next_states': False,
    'memory.capacity': BitcoinEnv.EPISODE_LEN * 4,  # {
    #     'type': 'bounded',
    #     'vals': [2000, 20000],
    #     'guess': 5000
    # }
}
hypers['distribution_model'] = {
    # 'distributions': None,
    'entropy_regularization': .01, #{
        # 'type': 'bounded',
        # 'vals': [0, 5],
        # 'guess': 2.,
        # 'hydrate': min_ten_neg(1e-4, 0.)
    # },
    # 'variable_noise': TODO
}
hypers['pg_model'] = {
    'baseline_mode': {
        'type': 'bool',
        'guess': True,
        'hydrate': hydrate_baseline
    },
    'gae_lambda': {
        'type': 'bounded',
        'vals': [.8, 1.],
        'guess': .8,  # .97,
        # Pretty ugly: says "use gae_lambda if baseline_mode=True, and if gae_lambda > .9" (which is why `vals`
        # allows a number below .9, so we can experiment with it off when baseline_mode=True)
        'post': lambda x, others: x if (x and x > .9 and others['baseline_mode']) else None
    },
    # 'baseline_optimizer.num_steps': 5
}
hypers['pg_prob_ration_model'] = {
    'likelihood_ratio_clipping': .2, #{
    #     'type': 'bounded',
    #     'vals': [0., 1.],
    #     'guess': .2,
    #     'hydrate': min_threshold(.05, None)
    # }
}
hypers['ppo_model'] = {
    # Doesn't seem to matter; consider removing
    'step_optimizer.type': {
        'type': 'int',
        'vals': ['nadam', 'adam'],
        'guess': 'adam'
    },
    'step_optimizer.learning_rate': {
        'type': 'bounded',
        'vals': [0., 9.],
        'guess': 6.,
        'hydrate': ten_to_the_neg
    },
    'optimization_steps': 25, #{
    #     'type': 'bounded',
    #     'vals': [1, 50],  # want to try higher, but too slow to test
    #     'guess': 25,
    #     'pre': round
    # },
    'subsampling_fraction': .2, #{
    #     'type': 'bounded',
    #     'vals': [0., 1.],
    #     'guess': .2
    # },
}

hypers['ppo_agent'] = {  # vpg_agent, trpo_agent
    **hypers['agent'],
    **hypers['memory_model'],
    **hypers['distribution_model'],
    **hypers['pg_model'],
    **hypers['pg_prob_ration_model'],
    **hypers['ppo_model']
}

hypers['custom'] = {
    # Use a handful of TA-Lib technical indicators (SMA, EMA, RSI, etc). Which indicators used and for what time-frame
    # not optimally chosen at all; just figured "if some randos are better than nothing, there's something there and
    # I'll revisit". Help wanted.
    'indicators_count': {
        'type': 'bounded',
        'vals': [0, 5],
        'guess': 3,
        'pre': round
    },
    'indicators_window': {
        'type': 'bounded',
        'vals': [0, 600],
        'guess': 300,
        'pre': int,
    },

    # This is special. "Risk arbitrage" is the idea of watching two exchanges for the same
    # instrument's price. Let's say BTC is $10k in GDAX and $9k in Kraken. Well, Kraken is a smaller / less popular
    # exchange, so it tends to play "follow the leader". Ie, Kraken will likely try to get to $10k
    # to match GDAX (oversimplifying obviously). This is called "risk arbitrage" ("arbitrage"
    # by itself is slightly different, not useful for us). Presumably that's golden info for the neural net:
    # "Kraken < GDAX? Buy in Kraken!". It's not a gaurantee, so this is a hyper in hypersearch.py.
    # Incidentally I have found it detrimental, I think due to imperfect time-phase alignment (arbitrage code in
    # data.py) which makes it hard for the net to follow.
    # Note: not valuable if GDAX is main (ie, not valuable if the bigger exchange is the main, only
    # if the smaller exchange (eg Kraken) is main)
    'arbitrage': {
        'type': 'bool',
        'guess': False
    },

    # Conv / LSTM layers
    'net.depth_mid': {
        'type': 'bounded',
        'vals': [1, 3],
        'guess': 2,
        'pre': round
    },
    # Dense layers
    'net.depth_post': {
        'type': 'bounded',
        'vals': [1, 3],
        'guess': 2,
        'pre': round
    },
    # Network depth, in broad-strokes of 2**x (2, 4, 8, 16, 32, 64, 128, 256, 512, ..) just so you get a feel for
    # small-vs-large. Later you'll want to fine-tune.
    'net.width': {
        'type': 'bounded',
        'vals': [3, 9],
        'guess': 6,
        'pre': round,
        'hydrate': two_to_the
    },
    # Whether to expand-in and shrink-out the nueral network. You know the look, narrower near the inputs, gets wider
    # in the hidden layers, narrower again on hte outputs.
    'net.funnel': {
        'type': 'bool',
        'guess': True
    },
    # tanh vs "the relu family" (relu, selu, crelu, elu, *lu). Broad-strokes here by just pitting tanh v relu; then,
    # if relu wins you can fine-tune "which type of relu" later.
    'net.activation': {
        'type': 'int',
        'vals': ['tanh', 'relu'],
        'guess': 'relu'
    },

    # Whether to append one extra tiny layer at the network's end for merging in the stationary data. This would give
    # stationary data extra oomph. Currently, stationary (which is 2-3 features) gets merged in after flatten (in conv)
    # which takes 256+ neurons, so stationary can easily get ignored without this hyper.
    'net.extra_stationary': {
        'type': 'bool',
        'guess': True
    },

    # Regularization: Dropout, L1, L2. You'd be surprised (or not) how important is the proper combo of these. The RL
    # papers just role L2 (.001) and ignore the other two; but that hasn't jived for me. Below is the best combo I've
    # gotten so far, and I'll update as I go.
    'net.dropout': None,  # FIXME not yet supported in tensorforce#memory (https://github.com/reinforceio/tensorforce/issues/317)
    # {
    #    'type': 'bounded',
    #    'vals': [0., .2],
    #    'guess': .001,
    #    'hydrate': min_threshold(.1, None)
    #},
    'net.l2': {
        'type': 'bounded',
        'vals': [0, 7],  # to disable, set to 7 (not 0)
        'guess': 3.,
        'hydrate': min_ten_neg(1e-6, 0.)
    },
    'net.l1': {
        'type': 'bounded',
        'vals': [0, 7],
        'guess': 7.,
        'hydrate': min_ten_neg(1e-6, 0.)
    },

    # Instead of using absolute price diffs, use percent-change.
    'pct_change': {
        'type': 'bool',
        'guess': False
    },
    # single = one action (-$x to +$x). multi = two actions: (buy|sell|hold) and (how much?). all_or_none = buy/sell
    # w/ all the cash or value owned
    'action_type': {
        'type': 'int',
        'vals': ['single', 'multi', 'all_or_none'],
        'guess': 'all_or_none'
    },
    # Should rewards be as-is (PNL), or "how much better than holding" (advantage)? if `sharpe` then we discount 1.0
    # and calculate sharpe score at episode-terminal
    'reward_type': {
        'type': 'int',
        'vals': ['raw', 'advantage', 'sharpe'],
        'guess': 'raw'
    },
    # Scale the inputs and rewards
    'scale': {
        'type': 'bool',
        'guess': True
    },
}

hypers['lstm'] = {
    # Number of dense layers before the LSTM layers
    'net.depth_pre': {
        'type': 'bounded',
        'vals': [0, 3],
        'guess': 2,
        'pre': int
    },
}
hypers['conv2d'] = {
    # 'net.bias': True,  # TODO valuable?

    # T-shirt size window-sizes, smaller # = more destructive. See comments in build_net_spec()
    'net.window': {
        'type': 'bounded',
        'vals': [1, 3],
        'guess': 2,
        'pre': round,
    },
    # How many ways to divide a window? 1=no-overlap, 2=half-overlap (smaller # = more destructive). See comments
    # in build_net_spec()
    'net.stride': {
        'type': 'bounded',
        'vals': [1, 3],
        'guess': 2,
        'pre': round
    },
    # Size of the window to look at w/ the CNN (ie, width of the image). Would like to have more than 400 "pixels" here,
    # but it causes memory issues the way PPO's MemoryModel batches things. This is made up for via indicators
    'step_window': {
        'type': 'bounded',
        'vals': [100, 400],
        'guess': 400,
        'pre': round,
    },

    # Because ConvNets boil pictures down (basically downsampling), the precise current timestep numbers can get
    # averaged away. This will repeat them in state['stationary'] downstream ("sir, you dropped this")
    # TODO removing for now while figuring out dimensionality/autoencoder stuff
    'repeat_last_state': False,  # {
    #     'type': 'bool',
    #     'guess': False
    # }
}


# Fill in implicit 'vals' (eg, 'bool' with [True, False])
for _, section in hypers.items():
    for k, v in section.items():
        if type(v) != dict: continue  # hard-coded vals
        if v['type'] == 'bool': v['vals'] = [0, 1]


class HSearchEnv(object):
    """This was once a TensorForce environment of its own, when I was using RL to find the best hyper-combo for RL.
    I turned from that to Bayesian Optimization, but that's why this is an awkward class of its own - it should be
    merged with the `main()` code below.

    TODO only tested with ppo_agent. Test with other agents
    """
    def __init__(self, cli_args, agent='ppo_agent'):
        net_type = cli_args.net_type
        hypers_ = hypers[agent].copy()
        hypers_.update(hypers['custom'])
        hypers_['net.type'] = net_type  # set as hard-coded val
        hypers_.update(hypers[net_type])

        hardcoded = {}
        for k, v in hypers_.copy().items():
            if type(v) != dict: hardcoded[k] = v

        self.hypers = hypers_
        self.agent = agent
        self.hardcoded = hardcoded
        self.cli_args = cli_args
        self.conn = data.engine.connect()
        self.conn_runs = data.engine_runs.connect()

    def close(self):
        self.conn.close()
        self.conn_runs.close()

    def get_hypers(self, actions):
        """
        Bit of confusing logic here where I construct a `flat` dict of hypers from the actions - looks like how hypers
        are specified above ('dot.key.str': val). Then from that we hydrate it as a proper config dict (hydrated).
        Keeping `flat` around because I save the run to the database so can be analyzed w/ a decision tree
        (for feature_importances and the like) and that's a good format, rather than a nested dict.
        :param actions: the hyperparamters
        """
        self.flat = flat = {}


        # Preprocess hypers
        for k, v in actions.items():
            try: v = v.item()  # sometimes primitive, sometimes numpy
            except Exception: pass
            hyper = self.hypers[k]
            if 'pre' in hyper:
                v = hyper['pre'](v)
            flat[k] = v
        # TODO remove this special-handling
        if flat['reward_type'] == 'sharpe':
            flat['gae_lambda'] = None
            flat['discount'] = 1.
        flat.update(self.hardcoded)

        # Post-process hypers (allow for dependency handling, etc)
        for k, v in flat.items():
            hyper = self.hypers[k]
            if type(hyper) == dict and 'post' in hyper:
                flat[k] = hyper['post'](v, flat)

        # change all a.b=c to {a:{b:c}} (note DotDict class above, I hate and would rather use an off-the-shelf)
        main, custom = utils.DotDict({}), utils.DotDict({})
        for k, v in flat.items():
            obj = main if k in hypers[self.agent] else custom
            try:
                v = self.hypers[k]['hydrate'](v, self.flat)
                if type(v) == dict: obj.update(v)
                else: obj[k] = v
            except: obj[k] = v
        main, custom = main.to_dict(), custom.to_dict()

        network = custom_net(custom, print_net=True)
        if flat['baseline_mode']:
            if type(self.hypers['baseline_mode']) == bool:
                main.update(hydrate_baseline(self.hypers['baseline_mode'], flat))

            main['baseline']['network_spec'] = network

        ## GPU split
        ## FIXME add back to tensorforce#memory
        # session_config = None
        # gpu_split = self.cli_args.gpu_split
        # if gpu_split != 1:
        #     fraction = .9 / gpu_split if gpu_split > 1 else gpu_split
        #     session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=fraction))
        # main['session_config'] = session_config

        print('--- Flat ---')
        pprint(flat)
        print('--- Hydrated ---')
        pprint(main)

        return flat, main, network

    def execute(self, actions):
        flat, hydrated, network = self.get_hypers(actions)

        env = BitcoinEnv(flat, self.cli_args)
        agent = agents_dict[self.agent](
            states=env.states,
            actions=env.actions,
            network=network,
            **hydrated
        )

        env.train_and_test(agent, self.cli_args.n_steps, self.cli_args.n_tests, -1)

        step_acc, ep_acc = env.acc.step, env.acc.episode
        adv_avg = utils.calculate_score(ep_acc.sharpes)
        print(flat, f"\nAdvantage={adv_avg}\n\n")

        sql = """
          insert into runs (hypers, sharpes, returns, uniques, prices, signals, agent, flag)
          values (:hypers, :sharpes, :returns, :uniques, :prices, :signals, :agent, :flag)
          returning id;
        """
        row = self.conn_runs.execute(
            text(sql),
            hypers=json.dumps(flat),
            sharpes=list(ep_acc.sharpes),
            returns=list(ep_acc.returns),
            uniques=list(ep_acc.uniques),
            prices=list(env.prices),
            signals=list(step_acc.signals),
            agent=self.agent,
            flag=self.cli_args.net_type
        ).fetchone()

        if ep_acc.sharpes[-1] > 0:
            _id = str(row[0])
            directory = os.path.join(os.getcwd(), "saves", _id)
            filestar = os.path.join(directory, _id)
            os.makedirs(directory, exist_ok=True)
            agent.save_model(filestar)

        agent.close()
        env.close()
        return adv_avg

    def get_winner(self, id=None):
        if id:
            sql = "select id, hypers from runs where id=:id"
            winner = self.conn_runs.execute(text(sql), id=id).fetchone()
            winner = winner.hypers
            print(winner)
        else:
            winner = {}
            for k,v in self.hypers.items():
                if k not in self.hardcoded:
                    winner[k] = v['guess']
            winner.update(self.hardcoded)
        self.hardcoded = winner
        return self.get_hypers({})


def print_feature_importances(X, Y, feat_names):
    if len(X) < 5: return
    model = GradientBoostingRegressor()
    model_hypers = {
        'max_features': [None, 'sqrt', 'log2'],
        'max_depth': [None, 10, 20],
        'n_estimators': [100, 200, 300],
    }
    model = GridSearchCV(model, param_grid=model_hypers, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    model.fit(X, np.squeeze(Y))
    feature_imp = sorted(zip(model.best_estimator_.feature_importances_, feat_names), key=lambda x: x[0],
                         reverse=True)
    print('\n\n--- Feature Importances ---\n')
    print('\n'.join([f'{x[1]}: {round(x[0],4)}' for x in feature_imp]))
    return model


def boost_optimization(model, loss_fn, bounds, x_list=[], y_list=[], n_pre_samples=5):
    # Handle any specifically-asked for "guesses" first
    for i, v in enumerate(y_list):
        if v[0] is None:
            print("Running guess values")
            y_list[i] = loss_fn(x_list[i])

    n_pre_samples -= len(x_list)
    if n_pre_samples > 0:
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
            x_list.append(params)
            y_list.append(loss_fn(params))

    # generate a big number of random hyper combos, pick the best one. The number of combos
    # we generate is smaller early on, larger later on. That way the early runs are more random,
    # allowing for better exploration, and the later runs give boost a bigger selection to choose
    # from, allowing it to be more spot-on
    n_experiments = int(1e4 * len(y_list))
    best_params, best_score = None, -1000
    for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_experiments, bounds.shape[0])):
        prediction = model.predict([params])[0]
        if prediction > best_score:
            best_params = params
            best_score = prediction
    loss_fn(best_params)


def main():
    import gp
    from sklearn.feature_extraction import DictVectorizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--guess', type=int, default=-1, help="Run the hard-coded 'guess' values first before exploring")
    parser.add_argument('--boost', action="store_true", default=False, help="Use custom gradient-boosting optimization, or bayesian optimization?")
    parser.add_argument('--autoencode', action="store_true", default=False, help="If you're running out of GPU memory, try --autoencode which scales things down")
    utils.add_common_args(parser)
    args = parser.parse_args()

    # Encode features
    hsearch = HSearchEnv(cli_args=args)
    hypers_, hardcoded = hsearch.hypers, hsearch.hardcoded
    hypers_ = {k: v for k, v in hypers_.items() if k not in hardcoded}
    hsearch.close()

    # Build a matrix of features,  length = max feature size
    max_num_vals = 0
    for v in hypers_.values():
        l = len(v['vals'])
        if l > max_num_vals: max_num_vals = l
    empty_obj = {k: None for k in hypers_}
    mat = pd.DataFrame([empty_obj.copy() for _ in range(max_num_vals)])
    for k, hyper in hypers_.items():
        for i, v in enumerate(hyper['vals']):
            mat.loc[i,k] = v
    mat.ffill(inplace=True)

    # Above is Pandas-friendly stuff, now convert to sklearn-friendly & pipe through OneHotEncoder
    vectorizer = DictVectorizer()
    vectorizer.fit(mat.T.to_dict().values())
    feat_names = vectorizer.get_feature_names()

    # Map TensorForce actions to GP-compatible `domain`
    # instantiate just to get actions (get them from hypers above?)
    bounds = []
    for k in feat_names:
        hyper = hypers_.get(k, False)
        bounded = False
        if hyper:
            bounded, min_, max_ = hyper['type'] == 'bounded', min(hyper['vals']), max(hyper['vals'])
        b = [min_, max_] if bounded else [0, 1]
        bounds.append(b)

    def hypers2vec(obj):
        h = dict()
        for k, v in obj.items():
            if k in hardcoded: continue
            if type(v) == bool: h[k] = float(v)
            else: h[k] = v or 0.
        return vectorizer.transform(h).toarray()[0]

    def vec2hypers(vec):
        # Reverse the encoding
        # https://stackoverflow.com/questions/22548731/how-to-reverse-sklearn-onehotencoder-transform-to-recover-original-data
        # https://github.com/scikit-learn/scikit-learn/issues/4414
        reversed = vectorizer.inverse_transform([vec])[0]
        obj = {}
        for k, v in reversed.items():
            if '=' not in k:
                obj[k] = v
                continue
            if k in obj: continue  # we already handled this x=y logic (below)
            # Find the winner (max) option for this key
            score, attr, val = v, k.split('=')[0], k.split('=')[1]
            for k2, score2 in reversed.items():
                if k2.startswith(attr + '=') and score2 > score:
                    score, val = score2, k2.split('=')[1]
            obj[attr] = val

        # Bools come in as floats. Also, if the result is False they don't come in at all! So we start iterate
        # hypers now instead of nesting this logic in reversed-iteration above
        for k, v in hypers_.items():
            if v['type'] == 'bool':
                obj[k] = bool(round(obj.get(k, 0.)))
        return obj

    # Specify the "loss" function (which we'll maximize) as a single rl_hsearch instantiate-and-run
    def loss_fn(params):
        hsearch = HSearchEnv(cli_args=args)
        reward = hsearch.execute(vec2hypers(params))
        hsearch.close()
        return [reward]

    guess_i = 0
    while True:
        # Every iteration, re-fetch from the database & pre-train new model. Acts same as saving/loading a model to disk,
        # but this allows to distribute across servers easily
        conn_runs = data.engine_runs.connect()
        sql = "select hypers, sharpes from runs where flag=:f"
        runs = conn_runs.execute(text(sql), f=args.net_type).fetchall()
        conn_runs.close()
        X, Y = [], []
        for run in runs:
            X.append(hypers2vec(run.hypers))
            Y.append([utils.calculate_score(run.sharpes)])
        boost_model = print_feature_importances(X, Y, feat_names)

        if args.guess != -1:
            guess = {k: v['guess'] for k, v in hypers_.items()}
            guess.update(utils.guess_overrides[args.guess][guess_i])
            loss_fn(hypers2vec(guess))

            guess_i += 1
            if guess_i > len(utils.guess_overrides[args.guess])-1:
                args.guess = -1  # start on GP

            continue

        if args.boost:
            print('Using gradient-boosting')
            boost_optimization(
                model=boost_model,
                loss_fn=loss_fn,
                bounds=np.array(bounds),
                x_list=X,
                y_list=Y
            )
        else:
            # Evidently duplicate values break GP. Many of these are ints, so they're definite duplicates. Either way,
            # tack on some small epsilon to make them different (1e-6 < gp.py's min threshold, make sure that #'s not a
            # problem). I'm concerned about this since many hypers can go below that epislon (eg learning-rate).
            for x in X:
                for i, v in enumerate(x):
                    x[i] += np.random.random() * 1e-6
            gp.bayesian_optimisation2(
                loss_fn=loss_fn,
                bounds=np.array(bounds),
                x_list=X,
                y_list=Y
            )

if __name__ == '__main__':
    main()
