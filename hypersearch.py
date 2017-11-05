"""
TODO
* add DQN
* make into class
* __init__ takes custom{}
* hyper-vals can be dict{requires:{str or dict}}. str='baseline_mode', dict={'baseline_mode':['states','network']}
  * properly order requires to ensure "off" vals (None comes last?)
"""
import tensorflow as tf
import pdb, json, random
from pprint import pprint
import numpy as np
import pandas as pd
from sqlalchemy.sql import text
from tensorforce.core.networks.layer import Dense
from tensorforce.core.networks.network import LayeredNetwork
from btc_env.btc_env import BitcoinEnvTforce
from data import conn


# TODO make these part of the hyper search
AGENT_K = 'ppo_agent'


def layers2net(layers, a, d, l2, l1=0., b=True):
    arr = []
    if d: arr.append(dict(type='dropout', rate=d))
    for l in layers:
        conv_args = {}
        type = l[0]
        if len(l)>1: size = l[1]
        if len(l)>2: conv_args['window'] = l[2]
        if len(l)>3: conv_args['stride'] = l[3]
        if type == 'd':
            arr.append(dict(
                type='dense',
                size=size,
                activation=a,
                l1_regularization=l1,
                l2_regularization=0. if d else l2)
            )
            if d: arr.append(dict(type='dropout', rate=d))
        elif type == 'L':
            arr.append(dict(type='lstm', size=size, dropout=d))
        elif type == 'C':
            arr.append(dict(type='conv2d', size=size, bias=b, **conv_args))
        elif type == 'F':
            arr.append(dict(type='flatten'))
    return arr


def custom_net(layers, net_type, **kwargs):
    layers_spec = layers2net(layers, **kwargs)
    for l in layers_spec:
        l.pop('l1_regularization', None)  # not yet in TF2
    if net_type != 'conv2d': return layers_spec

    class ConvNetwork(LayeredNetwork):
        def __init__(self, **kwargs):
            super(ConvNetwork, self).__init__(layers_spec, **kwargs)

        def tf_apply(self, x, internals=(), return_internals=False, training=None):
            image = x['state0']  # 150x7x2-dim, float
            money = x['state1']  # 1x2-dim, float
            x = image
            money_applied = False

            internal_outputs = list()
            index = 0
            for layer in self.layers:
                layer_internals = [internals[index + n] for n in range(layer.num_internals)]
                index += layer.num_internals
                if isinstance(self.layers, Dense) and not money_applied:
                    x = tf.concat([x, money], axis=1)
                    money_applied = True
                x = layer.apply(x, *layer_internals, training=training)

                if not isinstance(x, tf.Tensor):
                    internal_outputs.extend(x[1])
                    x = x[0]

            if return_internals:
                return x, internal_outputs
            else:
                return x
    return ConvNetwork


agent_types = ['ppo_agent']  # dqn_agent
net_specs = {
    # 'lstm': {  # TODO revisit network types (how to flatten into agent_hypers?)
    #     5: [('d', 256), ('L', 512), ('L', 512), ('L', 512), ('d', 256), ('d', 128)],
    #     4: [('d', 128), ('L', 256), ('L', 256), ('d', 192), ('d', 128)]
    # },
    'conv2d': {
        # 5: [
        #     ('C', 64, (8, 3), (4, 1)),
        #     ('C', 96, (4, 2), (2, 1)),
        #     ('C', 96, (3, 2), 1),
        #     ('C', 64, (2, 2), 1),
        #     ('F'),
        #     ('d', 512),
        #     ('d', 256),
        #     ('d', 196),
        #     ('d', 128),
        #     ('d', 64)
        # ],
        4: [
            ('C', 64, (8, 3), (4, 1)),
            ('C', 96, (4, 2), (2, 1)),
            ('C', 96, (3, 2), 1),
            ('F'),
            ('d', 512),
            ('d', 256),
            ('d', 128),
            ('d', 64)
        ],
        3: [
            ('C',32,(8,3),(4,2)),
            ('C',64,(4,2),(2,1)),
            ('C',64,(3,2),1),
            ('F'),
            ('d',512),
            ('d',256),
            ('d',128)
        ],
        2: [
            ('C', 32, (8, 3), (4, 2)),
            ('C', 64, (4, 2), (2, 1)),
            ('C', 64, (3, 2), 1),
            ('F'),
            ('d',256),
            ('d',128),
            ('d',64)
        ],
        1: [  # loser
            ('C',32,(8,3),(4,2)),
            ('C',64,(4,2),(2,1)),
            ('F'),
            ('d',256)
        ],
    }
}

batch_hypers = {
    'batch_size': [8, 32, 64, 128, 256],
    'discount': [.97, .95, .99],
    'keep_last_timestep': [False, True],
    'optimizer.learning_rate': [1e-5, 5e-5, 1e-6],
    'optimizer.type': ['nadam', 'adam'],
    'optimization_steps': [10, 20],

    # Special handling
    'net_type': ['conv2d'],
    'indicators': [False],
    'cryptowatch': [False],  # for now must be set manually in data.py
    'scale': [False],
    'penalize_inaction': [True, False],
    'network': [4, 3],
    'activation': ['tanh', 'elu', 'relu'],
    'dropout': [.4, None],
    'l2': [1e-3, 1e-4, 1e-5],
    'diff': ['percent', 'absolute'],
    'steps': [2048*3+3],
}
memory_hypers = {
    'memory_capacity': [50000, 1000000],  # STEPS*2
    'target_update_frequency': [1, 20],  # STEPS
    'first_update': [500, 10000], # STEPS//2
    'target_update_weight': [None, .001, 1],
    'update_frequency': [None, 1],
    'repeat_update': [1, 4]
}

hypers = {}
hypers['ppo_agent'] = {  # vpg_agent, trpo_agent
    **batch_hypers,
    'gae_lambda': [None, .99, .95],
    'baseline_optimizer.optimizer.learning_rate': [5e-5, 1e-6, 1e-5],
    'baseline_optimizer.num_steps': [10, 20],
    'normalize_rewards': [True],  # True is definite winner

    # I don't know what values to use besides the defaults, just guessing. Look into
    'entropy_regularization': [1e-1, 1e-3, 1e-2],
    'likelihood_ratio_clipping': [.01, .2, .9],

    # Special handling
    'baseline_mode': ['states'],

    # TODO
    # variable_noise
    # distributions? (gaussian, beta, etc). Pretty sure meant to handle under-the-hood, investigate
    # preprocessing / reward_preprocessing
}
hypers['ppo_agent']['step_optimizer.learning_rate'] = hypers['ppo_agent'].pop('optimizer.learning_rate')
hypers['ppo_agent']['step_optimizer.type'] = hypers['ppo_agent'].pop('optimizer.type')

hypers['dqn_agent'] = {
    **batch_hypers,
    **memory_hypers,
    'double_q_model': [True, False],
    'huber_loss': [None, .5, 1.]
    # TODO reward_preprocessing=[dict(type='clip', min=-1, max=1)]
    # TODO memory(type=replay, random_sampling)
    # TODO exploration (epsilon_decay(1,.1,1e10), epsilon_anneal)
}

hypers['naf_agent'] = {
    **batch_hypers,
    **memory_hypers,
    'clip_loss': [0., .5, 1.]  # TODO this still relevant?
    # TODO exploration (ornstein_uhlenbeck_no_params, epsilon_decay)
}


class DotDict(object):
    def __init__(self, data):
        self._data = data

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


def get_hypers(use_winner=False):
    # We'll return two versions of hypers. Flat (w/ dot-separated keys, for saving in db & computing w/ randomForest)
    # and hydrated (the real hypers to pass to the agent). Generate hydated from flat
    if use_winner:
        # first item in each list is the winner (remember to update these)
        winner = conn.execute(text("select hypers from runs where flag=:flag"), flag=use_winner).fetchone()
        print(winner)
        if winner:
            print('Using winner from database')
            flat = winner.hypers
        else:
            print('Using winner from dict')
            flat = {k: v[0] for k, v in hypers[AGENT_K].items()}
    else:
        if conn.execute('select count(*) as ct from runs').fetchone().ct > 20:  # give random hypers a shot for a while
            # flat = conn.execute('select * from runs where flag is null order by reward_avg desc limit 1').fetchone()

            # TODO handle this elsewhere
            runs = conn.execute('select * from runs where array_length(rewards,1)>200').fetchall()
            flat, gt0 = None, -1e6
            for run in runs:
                gt0_curr = (np.array(run.rewards[-50:])>0).sum()
                if  gt0_curr > gt0:
                    flat, gt0 = run, gt0_curr

            print(f'Using conf.id={flat.id}, reward={flat.reward_avg}')
            flat = flat.hypers
        else:
            # Priority-pick winning attrs. Remember to order attrs in hypers dict best-to-worst.
            flat = {k: v[0] for k, v in hypers[AGENT_K].items()}
            print('Using in-code flat')
        # After electing current winning attrs, now random-permute them with some prob - like evolution.
        # TODO use np.choice(p=[..]) to weight best-to-worst (left-to-right)
        for k in list(flat):
            if random.random() > .1:
                flat[k] = random.choice(hypers[AGENT_K][k])

    hydrated = DotDict({
        # 'tf_session_config': tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.2)),
        'tf_session_config': None,
        'baseline_mode': 'states',
        'baseline': {'type': 'custom'},
        'baseline_optimizer': {'type': 'multi_step', 'optimizer': {'type': 'nadam'}},
    })

    # change all a.b=c to {a:{b:c}} (note DotDict class above, I hate and would rather use an off-the-shelf)
    for k, v in flat.items():
        hydrated[k] = v
    hydrated = hydrated.to_dict()

    if flat['baseline_mode'] is None:
        flat['gae_lambda'] = hydrated['gae_lambda'] = None
        # TODO need to do anything w/ flat['baseline*'], or will DecisionTree know to ignore when baseline_mode=None?
        hydrated['baseline'] = None
        hydrated['baseline_optimizer'] = None

    # Will be manually handling certain attributes (not passed to Config()). Pop those off so they don't get in the way
    extra_keys = ['network', 'activation', 'l2', 'dropout', 'diff', 'steps', 'net_type', 'indicators',
                  'penalize_inaction', 'scale', 'cryptowatch']
    extra = {k: hydrated.pop(k) for k in extra_keys}

    net_spec = net_specs[extra['net_type']][extra['network']]
    network = custom_net(net_spec, extra['net_type'], a=extra['activation'], d=extra['dropout'], l2=extra['l2'])
    if hydrated['baseline_mode']:
        hydrated['baseline']['network_spec'] = custom_net(net_spec, extra['net_type'], a=extra['activation'], d=extra['dropout'], l2=extra['l2'])

    print('--- Flat ---')
    pprint(flat)
    print('--- Hydrated ---')
    pprint(hydrated)

    return flat, hydrated, network


def create_env(flat):
    # name = f"{agent_k}_{hyper_k}_{hyper_v}"
    return BitcoinEnvTforce(name=AGENT_K, hypers=flat)


def generate_and_save_hypers(rand=True):
    # Insert row. TODO deprecated. Change async to unify config by generating/fetching then passing .id as arg
    # (maybe this fn will still be needed for that)
    flat, _, _ = get_hypers(from_db=False, rand=rand)
    flag = 'random' if rand else 'winner'
    sql = """
    delete from runs where flag=:flag;
    insert into runs (reward, hypers, flag) values (0, :hypers, :flag)
    """
    conn.execute(text(sql), hypers=json.dumps(flat), flag=flag)