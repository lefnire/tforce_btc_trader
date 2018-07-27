import argparse, json, math, time, pdb, os, copy, uuid
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

from sqlalchemy.dialects import postgresql as psql
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope

from btc_env import BitcoinEnv
import utils
from data import data


def build_net_spec(hypers, baseline):
    """Builds an array of dicts that conform to TForce's network specification (see their docs) by mix-and-matching
    different network hypers
    """
    net = hypers.net
    batch_norm = {"type": "tf_layer", "layer": "batch_normalization"}
    arr = []

    def add_dense(s):
        dense = {
            'size': s,
            'l2_regularization': net.l2,
            'l1_regularization': net.l1
        }
        if not net.batch_norm:
            arr.append({'type': 'dense', 'activation': net.activation, **dense})
            return
        arr.append({'type': 'linear', **dense})
        arr.append(batch_norm)
        arr.append({'type': 'nonlinearity','name': net.activation})
        # FIXME dense dropout bug https://github.com/reinforceio/tensorforce/issues/317
        # if net.dropout: arr.append({'type': 'dropout', 'rate': net.dropout})

    # Mid-layer
    for i in range(net.depth_mid):
        arr.append({
            'size': net.width,
            'window': (net.kernel_size, 1),
            'stride': (net.stride, 1),
            'type': 'conv2d',
            # 'bias': net.bias,
            'l2_regularization': net.l2,
            'l1_regularization': net.l1
        })
    arr.append({'type': 'flatten'})

    # Post Dense layers
    for i in range(int(net.get('depth_post', 0))):
        size = net.width * 4   # 32-filter conv => 128 node FC  TODO smart?
        size = int(size / (i + 1)) if net.funnel else size
        add_dense(size)

    if net.extra_stationary:
        add_dense(9)  # TODO fiddle with size? Found 9 from a book, seems legit.

    return arr


def custom_net(hypers, print_net=False, baseline=False):
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
    hypers = Box(hypers)
    layers_spec = build_net_spec(hypers, baseline=baseline)
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

            if hypers.net.repeat_last_state:
                # stationary.shape=(?, 2), series.shape=(?, 400, 1, 6)
                # full batch, last window-step, 1 (height), all features. tf.squeeze removes the 1(height) dim (note
                # a dim was already removed via -1, hence axis=1)
                last_states = tf.squeeze(series[:, -1, :, :], axis=1)
                stationary = tf.concat([stationary, last_states], axis=1)

            # Apply stationary to the first Dense after the last LSTM. in the case of Baseline, there's no LSTM,
            # so apply it to the start
            apply_stationary_here = 0
            for i, layer in enumerate(self.layers):
                if hypers.net.extra_stationary and isinstance(layer, TForceLayers.Dense):
                    # Last Dense layer
                    apply_stationary_here = i
                elif isinstance(layer, TForceLayers.Flatten) or isinstance(layer, TForceLayers.InternalLstm):
                    # Last LSTM layer, peg to the next layer (a Dense)
                    apply_stationary_here = i + 1

            next_internals = dict()
            for i, layer in enumerate(self.layers):
                if i == apply_stationary_here:
                    x = tf.concat([x, stationary], axis=1)

                layer_internals = {name: internals['{}_{}'.format(layer.scope, name)] for name in layer.internals_spec()}

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

@scope.define
def two_to_the(x):
    return 2**int(x)

@scope.define
def ten_to_the_neg(x):
    return 10**-int(x)

@scope.define
def min_threshold(x, thresh, fallback):
    """Returns x or `fallback` if it doesn't meet the threshold. Note, if you want to turn a hyper "off" below,
    set it to "outside the threshold", rather than 0.
    """
    return x if (x and x > thresh) else fallback

@scope.define
def min_ten_neg(x, thresh, fallback):
    """Returns 10**-x, or `fallback` if it doesn't meet the threshold. Note, if you want to turn a hyper "off" below,
    set it to "outside the threshold", rather than 0.
    """
    x = 10**-x
    return x if (x and x > thresh) else fallback

def post_process(hypers):
    hypers = copy.deepcopy(hypers)  # don't modify original
    agent, custom = hypers['ppo_agent'], hypers['custom']

    o = agent['update_mode']
    o['frequency'] = math.ceil(o['batch_size'] / o['frequency'])
    agent['memory']['capacity'] = BitcoinEnv.EPISODE_LEN * o['batch_size']

    agent.update(agent['baseline_stuff'])
    del agent['baseline_stuff']
    if agent['baseline_mode']:
        o = agent['baseline_optimizer']
        # o['num_steps'] = agent['optimization_steps']
        o['optimizer']['learning_rate'] = agent['step_optimizer']['learning_rate']
        o['optimizer']['type'] = agent['step_optimizer']['type']

        agent['baseline']['network'] = custom_net(custom, print_net=True, baseline=True)
        # if main['gae_lambda']: main['gae_lambda'] = main['discount']
    return hypers


# Most hypers come directly from tensorforce/tensorforce/agents/ppo_agent.py, see that for documentation
# Note: Name this something other than "hypers" (eg "space"), easy conflicts with other methods
space = {}
space['agent'] = {
    # 'states_preprocessing': None,
    # 'actions_exploration': None,
    # 'reward_preprocessing': None,

    # I'm pretty sure we don't want to experiment any less than .99 for non-terminal reward-types (which are 1.0).
    # .99^500 ~= .6%, so looses value sooner than makes sense for our trading horizon. A trade now could effect
    # something 2-5k steps later. So .999 is more like it (5k steps ~= .6%)
    'discount': 1.,  # hp.uniform('discount', .9, .99),
}

MAX_BATCH_SIZE = 20
space['memory_model'] = {
    'update_mode': {
        'unit': 'episodes',
        'batch_size': scope.int(hp.quniform('batch_size', 1, MAX_BATCH_SIZE, 5)),
        'frequency': scope.int(hp.quniform('frequency', 1, 3, 1)),  # t-shirt sizes, reverse order
    },

    'memory': {
        'type': 'latest',
        'include_next_states': False,
        'capacity': None,  # 5000  # BitcoinEnv.EPISODE_LEN * MAX_BATCH_SIZE,  # hp.uniform('capacity', 2000, 20000, 500)
    }
}

space['distribution_model'] = {
    # 'distributions': None,
    'entropy_regularization': .01,  # hp.choice('entropy_regularization', [None, .01]), # scope.min_ten_neg(hp.uniform('entropy_regularization', 0., 5.), 1e-4, .01),
    # 'variable_noise': TODO
}

space['pg_model'] = {
    'baseline_stuff': hp.choice('baseline_stuff', [
        {'baseline_mode': None},
        {
            'baseline': {'type': 'custom'},
            'baseline_mode': 'states',
            'baseline_optimizer': {
                'type': 'multi_step',
                # Consider having baseline_optimizer learning hypers independent of the main learning hypers.
                # At least with PPO, it seems the step_optimizer learning hypers function quite diff0erently than
                # expected; where baseline_optimizer's function more as-expected. TODO Investigate.
                'num_steps': scope.int(hp.quniform('num_steps', 1, 20, 5)),
                'optimizer': {}  # see post_process()
            },
            'gae_lambda': hp.choice('gae_lambda', [1., None]),
            # scope.min_threshold(hp.uniform('gae_lambda', .8, 1.), .9, None)
        }
    ])
}
space['pg_prob_ration_model'] = {
    'likelihood_ratio_clipping': .2,  # scope.min_threshold(hp.uniform('likelihood_ratio_clipping', 0., 1.), .05, None),
}

space['ppo_model'] = {
    # Doesn't seem to matter; consider removing
    'step_optimizer': {
        'type': 'adam',  # hp.choice('type', ['nadam', 'adam']),
        'learning_rate': scope.ten_to_the_neg(hp.uniform('learning_rate', 2, 4.5)),
    },

    'optimization_steps': scope.int(hp.quniform('optimization_steps', 1, 50, 5)),

    'subsampling_fraction': .1,  # hp.uniform('subsampling_fraction', 0.,  1.),
}

ppo_agent = {
    **space['agent'],
    **space['memory_model'],
    **space['distribution_model'],
    **space['pg_model'],
    **space['pg_prob_ration_model'],
    **space['ppo_model']
}

space = {
    'ppo_agent': ppo_agent, # 'vpg_agent': ppo_agent, 'trpo_agent': ppo_agent,
    # TODO dqn, ddpg (hierarchical hyperopt)
}

space['custom'] = {
    'agent': 'ppo_agent',
    # TODO indicators overhaul

    # Use a handful of TA-Lib technical indicators (SMA, EMA, RSI, etc). Which indicators used and for what time-frame
    # not optimally chosen at all; just figured "if some randos are better than nothing, there's something there and
    # I'll revisit". Help wanted.

    # Currently disabling indicators in general. A good CNN should "see" those automatically in the window, right?
    # If I'm wrong, experiment with these (see commit 6fc4ed2)
    'indicators_count': 0,
    'indicators_window': 0,

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
    'arbitrage': False,  # see 6fc4ed2

    # single = one action (-$x to +$x). multi = two actions: (buy|sell|hold) and (how much?). all_or_none = buy/sell
    # w/ all the cash or value owned
    'action_type': 'single_discrete',  # hp.choice('action_type', ['single_discrete', 'single_continuous', 'multi']),

    # Should rewards be as-is (PNL), or "how much better than holding" (advantage)? if `sharpe` then we discount 1.0
    # and calculate sharpe score at episode-terminal.
    # See 6fc4ed2 for handling Sharpe rewards
    'reward_type': 'sharpe',  # hp.choice('reward_type', ['raw', 'advantage', 'sharpe']),

}
space['custom']['net'] = {
    # Conv / LSTM layers
    'depth_mid': scope.int(hp.quniform('depth_mid', 1, 4, 1)),

    # Dense layers
    'depth_post': scope.int(hp.quniform('depth_post', 1, 2, 1)),

    # Network depth, in broad-strokes of 2**x (2, 4, 8, 16, 32, 64, 128, 256, 512, ..) just so you get a feel for
    # small-vs-large. Later you'll want to fine-tune.
    'width': scope.two_to_the(hp.quniform('width', 4, 6, 1)),

    'batch_norm': hp.choice('batch_norm', [True, False]),

    # Whether to expand-in and shrink-out the nueral network. You know the look, narrower near the inputs, gets wider
    # in the hidden layers, narrower again on hte outputs.
    'funnel': hp.choice('funnel', [True, False]),

    # tanh vs "the relu family" (relu, selu, crelu, elu, *lu). Broad-strokes here by just pitting tanh v relu; then,
    # if relu wins you can fine-tune "which type of relu" later.
    'activation': hp.choice('activation', ['tanh', 'relu']),

    # Regularization: Dropout, L1, L2. You'd be surprised (or not) how important is the proper combo of these. The RL
    # papers just role L2 (.001) and ignore the other two; but that hasn't jived for me. Below is the best combo I've
    # gotten so far, and I'll update as I go.
    # 'dropout': scope.min_threshold(hp.uniform('dropout', 0., .5), .1, None),
    # 'l2': scope.min_ten_neg(hp.uniform('l2', 0., 7.), 1e-6, None),
    # 'l1': scope.min_ten_neg(hp.uniform('l1', 0., 7.), 1e-6, None),
    'dropout': None, 'l2': 0., 'l1': 0.,

    # LSTM at {last_good_commit}

    # T-shirt size window-sizes, smaller # = more destructive. See comments in build_net_spec()
    'kernel_size': hp.choice('window', [3, 5]),

    # How many ways to divide a window? 1=no-overlap, 2=half-overlap (smaller # = more destructive). See comments
    # in build_net_spec()
    'stride': 2,

    # Size of the window to look at w/ the CNN (ie, width of the image). Would like to have more than 400 "pixels" here,
    # but it causes memory issues the way PPO's MemoryModel batches things. This is made up for via indicators
    'step_window': 300,  # scope.int(hp.quniform('step_window', 200, 500, 50)),

    # Because ConvNets boil pictures down (basically downsampling), the precise current timestep numbers can get
    # averaged away. This will repeat them in state['stationary'] downstream ("sir, you dropped this")
    'repeat_last_state': hp.choice('repeat_last_state', [True, False]),

    # Whether to append one extra tiny layer at the network's end for merging in the stationary data. This would give
    # stationary data extra oomph. Currently, stationary (which is 2-3 features) gets merged in after flatten (in conv)
    # which takes 256+ neurons, so stationary can easily get ignored without this hyper.
    'extra_stationary': hp.choice('extra_stationary', [True, False])
}

# TODO restore get_winner() from git & fix-up

def main():
    parser = argparse.ArgumentParser()
    utils.add_common_args(parser)
    args = parser.parse_args()

    # Specify the "loss" function (which we'll maximize) as a single rl_hsearch instantiate-and-run
    def loss_fn(hypers):
        processed = post_process(hypers)
        network = custom_net(processed['custom'], print_net=True)

        agent = processed['ppo_agent']
        ## GPU split
        gpu_split = args.gpu_split
        if gpu_split != 1:
            fraction = .9 / gpu_split if gpu_split > 1 else gpu_split
            session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=fraction))
            agent['execution'] = {'type': 'single', 'session_config': session_config}

        print(processed)
        print(network)

        env = BitcoinEnv(processed, args)
        agent = agents_dict['ppo_agent'](
            states=env.states,
            actions=env.actions,
            network=network,
            **agent
        )

        env.train_and_test(agent)

        acc = env.acc.test
        adv_avg = utils.calculate_score(acc.ep.returns)
        print(hypers, f"\nScore={adv_avg}\n\n")

        df = pd.DataFrame([dict(
            id=uuid.uuid4(),
            hypers=json.dumps(hypers),
            returns=list(acc.ep.returns),
            uniques=list(acc.ep.uniques),
            prices=list(env.data.get_prices()),
            signals=list(acc.step.signals),
            flag=None, # args.net_type
        )])
        dtype = {
            'hypers': psql.JSONB
        }
        with data.engine_runs.connect() as conn:
            df.to_sql('runs', conn, if_exists='append', index=False, dtype=dtype)

        # TODO restore save_model() from git

        agent.close()
        env.close()
        return -adv_avg  # maximize

        # TODO restore fetching between runs so can pick up where left off, or
        # get updates from other servers

    trials = Trials()
    max_evals = 100
    best = fmin(loss_fn, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)


if __name__ == '__main__':
    main()
