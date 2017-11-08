"""
TODO
* vals can be dict {5: [{type=dense}]..}
* separate out & release open-source
* agent (ppo v dqn) part of randomization, not specified up front
"""
import tensorflow as tf
import pdb, json, random
from pprint import pprint
import numpy as np
import pandas as pd
from sqlalchemy.sql import text
from tensorforce.core.networks.layer import Dense
from tensorforce.core.networks.network import LayeredNetwork
from data import conn


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
        elif type == 'U':
            arr.append(dict(type='dueling', size=size, activation='none'))
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


explorations = {
    'epsilon_decay': {
        "type": "epsilon_decay",
        "epsilon": 1.0,
        "epsilon_final": 0.1,
        "epsilon_timesteps": 1e9
    },
}

hypers = {}
hypers['agent'] = {
    'exploration': ['epsilon_decay'],  # TODO epsilon_anneal
    # TODO preprocessing, batch_observe, reward_preprocessing[dict(type='clip', min=-1, max=1)]
}
hypers['memory_agent'] = {
    'batch_size': [8, 32, 64],
    'memory.type': ['replay'],  # prioritized_replay - bring back, currently causing "Trying to replace unseen observations: Memory is at capacity and contains only unseen observations"
    'memory.random_sampling': {
        '$requires': {'memory.type': 'replay'},
        '$vals': [True, False]
    },
    'memory.capacity': [100000],
    'first_update': [10000],
    'update_frequency': [4, 20],
    'repeat_update': [1, 4]
}
hypers['batch_agent'] = {
    'batch_size': [8, 32, 64, 128, 256],
    'keep_last_timestep': [False, True],
}
hypers['model'] = {
    'optimizer.type': ['nadam', 'adam'],  # TODO rmsprop
    'optimizer.learning_rate': [5e-5, 1e-5, 1e-6],
    'optimization_steps': [10, 20],
    'discount': [.97, .95, .99],
    'normalize_rewards': [True],  # True is definite winner
    # TODO variable_noise
}
hypers['distribution_model'] = {
    'entropy_regularization': [1e-1, 1e-3, 1e-2],
    # distributions_spec (gaussian, beta, etc). Pretty sure meant to handle under-the-hood, investigate
}
hypers['pg_model'] = {
    'baseline_mode': ['states'],
    'gae_lambda': {
        '$requires': 'baseline_mode',
        '$vals': [None, .99, .95]
    },
    'baseline_optimizer.optimizer.learning_rate': {
        '$requires': 'baseline_mode',
        '$vals': [5e-5, 1e-6, 1e-5]
    },
    'baseline_optimizer.num_steps': {
        '$requires': 'baseline_mode',
        '$vals': [10, 20]
    },
}
hypers['pg_prob_ration_model'] = {
    # I don't know what values to use besides the defaults, just guessing. Look into
    'likelihood_ratio_clipping': [.01, .2, .9],
}
hypers['q_model'] = {
    'target_sync_frequency': [10000],  # This effects speed the most - make it a high value
    'target_update_weight': [.001, 1.],
    'double_q_model': [True, False],
    'huber_loss': [None, .5, 1.]
}

hypers['dqn_agent'] = {
    **hypers['agent'],
    **hypers['memory_agent'],
    **hypers['model'],
    **hypers['distribution_model'],
    **hypers['q_model'],
}
hypers['ppo_agent'] = {  # vpg_agent, trpo_agent
    **hypers['agent'],
    **hypers['batch_agent'],
    **hypers['model'],
    **hypers['distribution_model'],
    **hypers['pg_model'],
    **hypers['pg_prob_ration_model']

}
hypers['ppo_agent']['step_optimizer.learning_rate'] = hypers['ppo_agent'].pop('optimizer.learning_rate')
hypers['ppo_agent']['step_optimizer.type'] = hypers['ppo_agent'].pop('optimizer.type')
del hypers['ppo_agent']['exploration']

# TODO pass this as __init__ arg
hypers['custom'] = {
    'net_type': ['conv2d'],
    'indicators': [False],
    'scale': [False],
    'cryptowatch': [False],
    'penalize_inaction': [True, False],
    'network': [4, 3, 2],
    'activation': ['tanh', 'elu', 'relu'],
    'dropout': [None, .4],
    'l2': [1e-3, 1e-4, 1e-5],
    'diff': ['percent', 'absolute'],
    'steps': [2048*3+3],
    'dueling': [True, False]
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


class HyperSearch(object):
    def __init__(self, agent='ppo_agent', overrides={}, custom={}, debug=True):
        self.agent = agent
        self.debug = debug

    def _flat_defaults(self, h):
        flat = {}
        for k, v in h.items():
            vals = v if type(v) is list else v['$vals']
            flat[k] = vals[0]
        return flat

    def get_hypers(self, use_winner=False):
        hypers_ = hypers[self.agent].copy()
        hypers_.update(hypers['custom'])
        if self.agent == 'ppo_agent': del hypers_['dueling']

        # We'll return two versions of hypers. Flat (w/ dot-separated keys, for saving in db & computing w/ randomForest)
        # and hydrated (the real hypers to pass to the agent). Generate hydated from flat
        if use_winner:
            # first item in each list is the winner (remember to update these)
            sql = "select id, hypers from runs where flag=:flag and agent=:agent"
            winner = conn.execute(text(sql), flag=use_winner, agent=self.agent).fetchone()
            if winner:
                if self.debug: print('Using winner from database')
                self.run_id = winner.id
                flat = winner.hypers
            else:
                flat = self._flat_defaults(hypers_)
                if self.debug: print('Using winner from dict')
        else:
            sql = 'select count(*) as ct from runs where agent=:agent'
            if conn.execute(text(sql), agent=self.agent).fetchone().ct > 20:  # give random hypers a shot for a while
                # flat = conn.execute('select * from runs where flag is null order by reward_avg desc limit 1').fetchone()

                # TODO handle this elsewhere
                sql = 'select * from runs where array_length(rewards,1)>200 and agent=:agent'
                runs = conn.execute(text(sql), agent=self.agent).fetchall()
                flat, gt0 = None, -1e6
                for run in runs:
                    gt0_curr = (np.array(run.rewards[-50:])>0).sum()
                    if gt0_curr > gt0:
                        flat, gt0 = run, gt0_curr

                print(f'Using conf.id={flat.id}, reward={flat.reward_avg}')
                flat = flat.hypers
            else:
                # Priority-pick winning attrs. Remember to order attrs in hypers dict best-to-worst.
                flat = self._flat_defaults(hypers_)
                print('Using in-code flat')
            # After selecting current winning attrs, now random-permute them with some prob - like evolution.
            for k in list(flat):
                hyper = hypers_[k]
                if random.random() > .1:
                    vals = hyper if type(hyper) is list else hyper['$vals']
                    flat[k] = random.choice(vals)

        # Ensure dependencies (do after above to make sure the randos have "settled")
        for k in list(flat):
            hyper = hypers_[k]
            if not (type(hyper) is dict and '$requires' in hyper): continue
            req = hyper['$requires']
            # Requirement is a string (require the value's not None). TODO handle nested deps.
            if type(req) is str:
                if not flat[req]: del flat[k]
                continue
            # Requirement is a dict of type {key: value_it_must_equal}. TODO handle multiple deps
            dep_k, dep_v = list(req.items())[0]
            if flat[dep_k] != dep_v:
                del flat[k]

        if self.agent == 'ppo_agent':
            hydrated = DotDict({
                # 'tf_session_config': tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.2)),
                'tf_session_config': None,
                'baseline_mode': 'states',
                'baseline': {'type': 'custom'},
                'baseline_optimizer': {'type': 'multi_step', 'optimizer': {'type': 'nadam'}},
            })
        else:
            hydrated = DotDict({'tf_session_config': None})

        # change all a.b=c to {a:{b:c}} (note DotDict class above, I hate and would rather use an off-the-shelf)
        for k, v in flat.items():
            if k not in hypers['custom']:
                hydrated[k] = v
        hydrated = hydrated.to_dict()

        extra = {k: v for k, v in flat.items() if k in hypers['custom']}
        net_spec = net_specs[extra['net_type']][extra['network']]
        if extra.get('dueling', False):
            net_spec = net_spec + [('U',16)]
        network = custom_net(net_spec, extra['net_type'], a=extra['activation'], d=extra['dropout'], l2=extra['l2'])

        if flat.get('baseline_mode', None):
            hydrated['baseline']['network_spec'] = custom_net(net_spec, extra['net_type'], a=extra['activation'], d=extra['dropout'], l2=extra['l2'])
        if self.agent != 'ppo_agent':
            hydrated['exploration'] = explorations[flat['exploration']]

        self.flat = flat
        if self.debug:
            print('--- Flat ---')
            pprint(flat)
            print('--- Hydrated ---')
            pprint(hydrated)

        return flat, hydrated, network

    def run_finished(self, rewards, min_len=150):
        # when done, save reward to database
        if len(rewards) < min_len: return
        reward_avg = np.mean(rewards[-50:])
        sql = "insert into runs (hypers, reward_avg, rewards, agent) values (:hypers, :reward_avg, :rewards, :agent)"
        conn.execute(text(sql), hypers=json.dumps(self.flat), reward_avg=reward_avg, rewards=rewards, agent=self.agent)

    def generate_and_save_hypers(self, rand=True):
        # Insert row. TODO deprecated. Change async to unify config by generating/fetching then passing .id as arg
        # (maybe this fn will still be needed for that)
        flat, _, _ = self.get_hypers(from_db=False, rand=rand)
        flag = 'random' if rand else 'winner'
        sql = """
        delete from runs where flag=:flag;
        insert into runs (reward, hypers, flag) values (0, :hypers, :flag)
        """
        conn.execute(text(sql), hypers=json.dumps(flat), flag=flag)