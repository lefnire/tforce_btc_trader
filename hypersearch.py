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
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll.base import scope

from btc_env import BitcoinEnv
import utils
from data import data


def network_spec(hypers):
    arch = hypers['net']['arch']

    if arch == 1:
        return [
            *[{
                "type": "conv2d",
                "size": 64,
                "window": 3,
                "stride": 2
            } for _ in range(4)],
            {
                "type": "flatten"
            },
            {
                "type": "dense",
                "size": 512
            },
            {
                "type": "dense",
                "size": 128
            }
        ]
    if arch == 2:
        return [
            *[{
                "type": "conv2d",
                "size": 32,
                "window": 3,
                "stride": 2
            } for _ in range(4)],
            {
                "type": "flatten"
            },
            {
                "type": "dense",
                "size": 128
            }
        ]
    # if arch == 3:
    #     return [
    #         {
    #             "type": "dense",
    #             "size": size,
    #         } for size in [300, 200, 100, 32]
    #     ]

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
    # agent['memory']['capacity'] = BitcoinEnv.EPISODE_LEN * o['batch_size']
    #agent['memory']['capacity'] = BitcoinEnv.EPISODE_LEN * MAX_BATCH_SIZE + 1

    agent.update(agent['baseline_stuff'])
    del agent['baseline_stuff']
    if agent['baseline_mode']:
        o = agent['baseline_optimizer']
        # o['num_steps'] = agent['optimization_steps']
        o['optimizer']['learning_rate'] = agent['step_optimizer']['learning_rate']
        o['optimizer']['type'] = agent['step_optimizer']['type']

        agent['baseline']['network'] = network_spec(custom)
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

BATCH_SIZE = 10
space['memory_model'] = {
    'update_mode': {
        'unit': 'episodes',
        'batch_size': BATCH_SIZE,  # scope.int(hp.quniform('batch_size', 1, MAX_BATCH_SIZE, 1)),  # 5 FIXME
        'frequency': scope.int(hp.quniform('frequency', 1, 3, 1)),  # t-shirt sizes, reverse order
    },

    'memory': {
        'type': 'latest',
        'include_next_states': False,
        'capacity': BitcoinEnv.EPISODE_LEN * BATCH_SIZE,  # hp.uniform('capacity', 2000, 20000, 500)
    }
}

space['distribution_model'] = {
    # 'distributions': None,
    'entropy_regularization': .01, # hp.choice('entropy_regularization', [None, .01]), # scope.min_ten_neg(hp.uniform('entropy_regularization', 0., 5.), 1e-4, .01),
    # 'variable_noise': TODO
}

space['pg_model'] = {
    # 'baseline_stuff': hp.choice('baseline_stuff', [
        # {'baseline_mode': None},
    'baseline_stuff': {
            'baseline': {
                'type': 'custom',
                'network': None
            },
            'baseline_mode': 'states',
            'baseline_optimizer': {
                'type': 'multi_step',
                # Consider having baseline_optimizer learning hypers independent of the main learning hypers.
                # At least with PPO, it seems the step_optimizer learning hypers function quite diff0erently than
                # expected; where baseline_optimizer's function more as-expected. TODO Investigate.
                'optimizer': {},  # see post_process()
                'num_steps': 5,  # scope.int(hp.quniform('num_steps', 1, 20, 1)),  # 5 FIXME
            },
            # scope.min_threshold(hp.uniform('gae_lambda', .8, 1.), .9, None)
        },
    # ])
    'gae_lambda': hp.choice('gae_lambda', [1., None]),
}
space['pg_prob_ration_model'] = {
    'likelihood_ratio_clipping': .2,  # scope.min_threshold(hp.uniform('likelihood_ratio_clipping', 0., 1.), .05, None),
}

space['ppo_model'] = {
    # Doesn't seem to matter; consider removing
    'step_optimizer': {
        'type': 'adam',  # hp.choice('type', ['nadam', 'adam']),
        'learning_rate': 1e-3,  # scope.ten_to_the_neg(hp.uniform('learning_rate', 2., 5.)),
    },

    'optimization_steps': hp.choice('optimization_steps', [25, 50]),

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

    # Use a handful of TA-Lib technical indicators (SMA, EMA, RSI, etc). Which indicators used and for what time-frame
    # not optimally chosen at all; just figured "if some randos are better than nothing, there's something there and
    # I'll revisit". Help wanted.

    # Currently disabling indicators in general. A good CNN should "see" those automatically in the window, right?
    # If I'm wrong, experiment with these (see commit 6fc4ed2)
    # TODO indicators overhaul
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
    'arch': hp.choice('arch', [1, 2]),


    # Size of the window to look at w/ the CNN (ie, width of the image). Would like to have more than 400 "pixels" here,
    # but it causes memory issues the way PPO's MemoryModel batches things. This is made up for via indicators
    'step_window': 300,  # scope.int(hp.quniform('step_window', 200, 500, 50)),
}

# TODO restore get_winner() from git & fix-up

def main():
    parser = argparse.ArgumentParser()
    utils.add_common_args(parser)
    args = parser.parse_args()

    # Specify the "loss" function (which we'll maximize) as a single rl_hsearch instantiate-and-run
    def loss_fn(hypers):
        processed = post_process(hypers)
        network = network_spec(processed['custom'])

        agent = processed['ppo_agent']
        ## GPU split
        gpu_split = args.gpu_split
        if gpu_split != 1:
            fraction = .9 / gpu_split if gpu_split > 1 else gpu_split
            session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=fraction))
            agent['execution'] = {'type': 'single', 'session_config': session_config, 'distributed_spec': None}

        pprint(processed)
        pprint(network)

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
            prices=list(env.data.get_prices(acc.ep.i, 0)),
            signals=list(acc.step.signals),
        )]).set_index('id')
        dtype = {
            'hypers': psql.JSONB,
            **{k: psql.ARRAY(psql.DOUBLE_PRECISION) for k in ['returns', 'signals', 'prices', 'uniques']},
        }
        with data.engine_runs.connect() as conn:
            df.to_sql('runs', conn, if_exists='append', index_label='id', dtype=dtype)

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
