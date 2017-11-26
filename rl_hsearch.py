import argparse, json, math, time, pdb
from pprint import pprint
import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy.sql import text
from tensorforce.agents import agents as agents_dict
from tensorforce.core.networks.layer import Dense
from tensorforce.core.networks.network import LayeredNetwork
from tensorforce.environments import Environment
from tensorforce.execution import Runner

from btc_env import BitcoinEnv
from data import engine


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

        def tf_apply(self, x, internals, update, return_internals=False):
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
                x = layer.apply(x, update, *layer_internals)

                if not isinstance(x, tf.Tensor):
                    internal_outputs.extend(x[1])
                    x = x[0]

            if return_internals:
                return x, internal_outputs
            else:
                return x
    return ConvNetwork


lookups = {}
lookups['epsilon_decay'] = {
    "type": "epsilon_decay",
    "epsilon": 1.0,
    "epsilon_final": 0.1,
    "epsilon_timesteps": 1e9
}
lookups['nets'] = {
    'lstm': {
        3: [('d', 256), ('L', 512), ('L', 512), ('L', 512), ('d', 256), ('d', 128)],
        2: [('d', 128), ('L', 256), ('L', 256), ('d', 192), ('d', 128)],
        1: [('d', 64), ('L', 128), ('L', 128), ('d', 64), ('d', 32)],
        0: [('d', 32), ('L', 64), ('L', 64), ('d', 32)]
    },
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
        3: [
            ('C', 64, (8, 3), (4, 1)),
            ('C', 96, (4, 2), (2, 1)),
            ('C', 96, (3, 2), 1),
            ('F'),
            ('d', 512),
            ('d', 256),
            ('d', 128),
            ('d', 64)
        ],
        2: [
            ('C',32,(8,3),(4,2)),
            ('C',64,(4,2),(2,1)),
            ('C',64,(3,2),1),
            ('F'),
            ('d',512),
            ('d',256),
            ('d',128)
        ],
        1: [
            ('C', 32, (8, 3), (4, 2)),
            ('C', 64, (4, 2), (2, 1)),
            ('C', 64, (3, 2), 1),
            ('F'),
            ('d',256),
            ('d',128),
            ('d',64)
        ],
        0: [  # loser
            ('C',32,(8,3),(4,2)),
            ('C',64,(4,2),(2,1)),
            ('F'),
            ('d',256)
        ],
    }
}

hypers = {}
hypers['agent'] = {
    'exploration': {
        'type': 'int',
        'vals': lookups['epsilon_decay']
    },  # TODO epsilon_anneal
    # TODO preprocessing, batch_observe, reward_preprocessing[dict(type='clip', min=-1, max=1)]
}
hypers['memory_agent'] = {
    'batch_size': {
        'type': 'bounded',
        'vals': [8, 256],
        'hook': lambda x: int(x // 8) * 8
    },
    'memory.type': {
        'type': 'int',
        'vals': ['replay', 'naive-prioritized-replay']
    },
    'memory.random_sampling': {
        'type': 'bool',
        'requires': {'memory.type': 'replay'},
    },
    'memory.capacity': {
        'type': 'bounded',
        'vals': [10000, 100000],  # ensure > batch_size
        'hook': int
    },
    'first_update': {
        'type': 'bounded',
        'vals': [1000, 10000],
        'hook': int
    },
    'update_frequency': {
        'type': 'bounded',
        'vals': [4, 20],
        'hook': int
    },
    'repeat_update': {
        'type': 'bounded',
        'vals': [1, 4],
        'hook': int
    }
}
hypers['batch_agent'] = {
    'batch_size': {
        'type': 'bounded',
        'vals': [8, 256],
        'hook': lambda x: int(x // 8) * 8
    },
    'keep_last_timestep': {
        'type': 'bool'
    }
}
hypers['model'] = {
    'optimizer.type': {
        'type': 'int',
        'vals': ['nadam', 'adam'],  # TODO rmsprop
    },
    'optimizer.learning_rate': {
        'type': 'bounded',
        'vals': [1e-7, 1e-2],
    },
    'optimization_steps': {
        'type': 'bounded',
        'vals': [5, 20],
        'hook': int
    },
    'discount': {
        'type': 'bounded',
        'vals': [.95, .99],
    },
    'normalize_rewards': True,  # True is definite winner
    # TODO variable_noise
}
hypers['distribution_model'] = {
    'entropy_regularization': {
        'type': 'bounded',
        'vals': [0., 1,],
    }
    # distributions_spec (gaussian, beta, etc). Pretty sure meant to handle under-the-hood, investigate
}
hypers['pg_model'] = {
    'baseline_mode': 'states',
    'gae_lambda': {
        'requires': 'baseline_mode',
        'type': 'bounded',
        'vals': [.94, .99],
        'hook': lambda x: None if x < .95 else x
    },
    'baseline_optimizer.optimizer.learning_rate': {
        'requires': 'baseline_mode',
        'type': 'bounded',
        'vals': [1e-7, 1e-2]
    },
    'baseline_optimizer.num_steps': {
        'requires': 'baseline_mode',
        'type': 'bounded',
        'vals': [5, 20],
        'hook': int
    },
}
hypers['pg_prob_ration_model'] = {
    # I don't know what values to use besides the defaults, just guessing. Look into
    'likelihood_ratio_clipping': {
        'type': 'bounded',
        'vals': [0., 1.],
    }
}
hypers['q_model'] = {
    'target_sync_frequency': 10000,  # This effects speed the most - make it a high value
    'target_update_weight': {
        'type': 'bounded',
        'vals': [0., 1.],
    },
    'double_q_model': True,
    'huber_loss': {
        'type': 'bounded',
        'vals': [0., 1.],
        'hook': lambda x: None if x < .001 else x
    }
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

hypers['custom'] = {
    'net_type': {
        'type': 'int',
        'vals': ['conv2d', 'lstm']
    },
    'indicators': False,
    'scale': False,
    # 'cryptowatch': False,
    'punish_inaction': {
        'type': 'int',
        'vals': ['off', 'hold_double', 'hold_spank', 'unique_double', 'unique_spank']
    },
    'network': {
        'type': 'bounded',
        'vals': lookups['nets']['conv2d'],
        'hook': lambda x: math.floor(x)
    },
    'activation': {
        'type': 'int',
        'vals': ['tanh', 'elu', 'relu', 'selu'],
    },
    'dropout': {
        'type': 'bounded',
        'vals': [0., .5],
        'hook': lambda x: None if x < .1 else x
    },
    'l2': {
        'type': 'bounded',
        'vals': [1e-5, 1e-1]
    },
    'diff': {
        'type': 'int',
        'vals': ['percent', 'absolute']
    },
    'steps': 2048*3+3,
    'dueling': {
        'type': 'bool'
    }
}

# Fill in implicit 'vals' (eg, 'bool' with [True, False])
for _, section in hypers.items():
    for k, v in section.items():
        if type(v) != dict: continue  # hard-coded vals
        if v['type'] == 'bool': v['vals'] = [0, 1]

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


class HSearchEnv(Environment):
    def __init__(self, agent='ppo_agent', gpu_split=1):
        hypers_ = hypers[agent].copy()
        hypers_.update(hypers['custom'])
        if agent == 'ppo_agent': del hypers_['dueling']

        self.agent = agent
        self.hypers = hypers_
        self.hardcoded = {}
        self.actions_ = {}
        self.gpu_split = gpu_split  # careful, GPU-splitting may be handled in a calling class already

        self.conn = engine.connect()

        for k, v in hypers_.items():
            if type(v) != dict:
                self.hardcoded[k] = v
            elif v['type'] == 'int':
                self.actions_[k] = dict(type='int', shape=(), num_actions=len(v['vals']))
            elif v['type'] == 'bounded':
                # cast to list in case the keys are the min/max (as in network)
                min, max = np.min(list(v['vals'])), np.max(list(v['vals']))
                self.actions_[k] = dict(type='float', shape=(), min_value=min, max_value=max)
            elif v['type'] == 'bool':
                self.actions_[k] = dict(type='bool', shape=())

    def __str__(self):
        return 'HSearchEnv'

    def close(self):
        self.conn.close()

    @property
    def actions(self):
        return self.actions_

    @property
    def states(self):
        return {'shape': 1, 'type': 'float'}

    def _action2val(self, k, v):
        # from TensorForce, v is a numpy object - unpack. From Bayes, it's a primitive. TODO handle better
        try: v = v.item()
        except Exception: pass

        hyper = self.hypers[k]
        if 'hook' in hyper:
            v = hyper['hook'](v)
        if hyper['type'] == 'bool': return bool(round(v))
        if hyper['type'] == 'int':
            if type(hyper['vals']) == list and type(v) == int:
                return hyper['vals'][v]
            # Else it's a dict. Don't map the values till later (keep them as keys in flat)
        return v

    def _key2val(self, k, v):
        hyper = self.hypers[k]
        if type(hyper) == dict and type(hyper.get('vals', None)) == dict:
            # FIXME special case, refactor
            if k == 'network':
                return lookups['nets'][self.flat['net_type']][v]

            return hyper['vals'][v]
        return v

    def reset(self):
        return [1.]

    def execute(self, actions):
        flat = {k: self._action2val(k, v) for k, v in actions.items()}
        flat.update(self.hardcoded)
        self.flat = flat

        # Ensure dependencies (do after above to make sure the randos have "settled")
        for k in list(flat):
            if k in self.hardcoded: continue
            hyper = self.hypers[k]
            if not (type(hyper) is dict and 'requires' in hyper): continue
            req = hyper['requires']
            # Requirement is a string (require the value's not None). TODO handle nested deps.
            if type(req) is str:
                if not flat[req]: del flat[k]
                continue
            # Requirement is a dict of type {key: value_it_must_equal}. TODO handle multiple deps
            dep_k, dep_v = list(req.items())[0]
            if flat[dep_k] != dep_v:
                del flat[k]

        session_config = None if self.gpu_split == 1 else \
            tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.82 / self.gpu_split))
        if self.agent == 'ppo_agent':
            hydrated = DotDict({
                'session_config': session_config,
                'baseline_mode': 'states',
                'baseline': {'type': 'custom'},
                'baseline_optimizer': {'type': 'multi_step', 'optimizer': {'type': flat['step_optimizer.type']}},
            })
        else:
            hydrated = DotDict({'session_config': session_config})

        # change all a.b=c to {a:{b:c}} (note DotDict class above, I hate and would rather use an off-the-shelf)
        for k, v in flat.items():
            if k not in hypers['custom']:
                hydrated[k] = self._key2val(k, v)
        hydrated = hydrated.to_dict()

        extra = {k: self._key2val(k, v) for k, v in flat.items() if k in hypers['custom']}
        net_spec = extra['network']
        if extra.get('dueling', False):
            net_spec = net_spec + [('U', 16)]
        network = custom_net(net_spec, extra['net_type'], a=extra['activation'], d=extra['dropout'], l2=extra['l2'])

        if flat.get('baseline_mode', None):
            baseline_net = [l for l in net_spec if l[0] != 'L']
            hydrated['baseline']['network_spec'] = custom_net(baseline_net, extra['net_type'], a=extra['activation'], d=extra['dropout'], l2=extra['l2'])

        print('--- Flat ---')
        pprint(flat)
        print('--- Hydrated ---')
        pprint(hydrated)

        hydrated['scope'] = 'hypersearch'

        env = BitcoinEnv(flat, name=self.agent)
        agent = agents_dict[self.agent](
            states_spec=env.states,
            actions_spec=env.actions,
            network_spec=network,
            **hydrated
        )

        # n_train, n_test = 2, 1
        n_train, n_test = 230, 20
        runner = Runner(agent=agent, environment=env)
        runner.run(episodes=n_train)  # train
        runner.run(episodes=n_test, deterministic=True)  # test

        ep_results = runner.environment.episode_results
        reward = np.mean(ep_results['rewards'][-n_test:])
        sql = """
          insert into runs (hypers, reward_avg, rewards, agent, prices, actions) 
          values (:hypers, :reward_avg, :rewards, :agent, :prices, :actions)
        """
        self.conn.execute(
            text(sql),
            hypers=json.dumps(flat),
            reward_avg=reward,
            rewards=ep_results['rewards'],
            agent='ppo_agent',
            prices=env.prices.tolist(),
            actions=env.signals
        )
        print(flat, f"\nReward={reward}\n\n")

        runner.agent.close()
        runner.environment.close()

        next_state, terminal = [1.], False
        return next_state, terminal, reward




def main_gp():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workers', type=int, default=1, help="Number of workers")
    parser.add_argument('-g', '--gpu_split', type=int, default=1, help="Num ways we'll split the GPU (how many tabs you running?)")
    parser.add_argument('--load', action="store_true", default=False, help="Load model from save")
    parser.add_argument('--pretrain', action="store_true", default=False, help="Pre-train model from database (current issues)")
    args = parser.parse_args()

    import gp, threading
    from sklearn.feature_extraction import DictVectorizer

    # Encode features
    hsearch = HSearchEnv(gpu_split=args.gpu_split)
    actions, hypers, hardcoded = hsearch.actions, hsearch.hypers, hsearch.hardcoded
    hypers = {k: v for k, v in hypers.items() if k not in hardcoded}
    hsearch.close()

    # Build a matrix of features,  length = max feature size
    max_num_vals = max(actions.values(), key=lambda v: v.get('num_actions', 0))['num_actions']
    empty_obj = {k: None for k in hypers}
    mat = pd.DataFrame([empty_obj.copy() for _ in range(max_num_vals)])
    for k, hyper in hypers.items():
        for i, v in enumerate(hyper['vals']):
            # mat.loc[i,k] = v if hyper['type'] == 'bounded' else i
            mat.loc[i,k] = v
    mat.ffill(inplace=True)

    # Above is Pandas-friendly stuff, now convert to sklearn-friendly & pipe through OneHotEncoder
    vectorizer = DictVectorizer()
    vectorizer.fit(mat.T.to_dict().values())
    feat_names = vectorizer.get_feature_names()

    # Map TensorForce actions to GPyOpt-compatible `domain`
    # instantiate just to get actions (get them from hypers above?)
    bounds = []
    for k in feat_names:
        hyper = hypers.get(k, False)
        if hyper and hyper['type'] == 'bounded':
            b = [min(hyper['vals']), max(hyper['vals'])]
        else: b = [0, 1]
        bounds.append(b)

    # Specify the "loss" function (which we'll maximize) as a single rl_hsearch instantiate-and-run
    def loss_fn(params):
        hsearch = HSearchEnv(gpu_split=args.gpu_split)

        # Reverse the encoding
        # https://stackoverflow.com/questions/22548731/how-to-reverse-sklearn-onehotencoder-transform-to-recover-original-data
        # https://github.com/scikit-learn/scikit-learn/issues/4414
        reversed = vectorizer.inverse_transform([params])[0]
        actions = {}
        for k, v in reversed.items():
            if '=' not in k:
                actions[k] = v
                continue
            if k in actions: continue  # we already handled this x=y logic (below)
            # Find the winner (max) option for this key
            score, attr, val = v, k.split('=')[0], k.split('=')[1]
            for k2, score2 in reversed.items():
                if k2.startswith(attr + '=') and score2 > score:
                    score, val = score2, k2.split('=')[1]
            actions[attr] = val

        _, _, reward = hsearch.execute(actions)
        hsearch.close()
        return reward

    filename = 'saves/gp.pkl' if args.load else None
    model = gp.make_model(filename=filename)

    # Fetch existing runs from database to pre-train GP
    if args.pretrain:
        conn = engine.connect()
        runs = conn.execute("select hypers, reward_avg from runs where flag is null").fetchall()
        conn.close()
        X, Y = [], []
        for run in runs:
            h_ = dict()
            for k, v in run.hypers.items():
                if k in hardcoded: continue
                if type(v) == bool: h_[k] = float(v)
                else: h_[k] = v or 0.
            vec = vectorizer.transform(h_).toarray()[0]
            X.append(vec)
            Y.append(run.reward_avg)
        model.fit(X, Y)

    def thread_fn():
        gp.bayesian_optimisation(
            n_iters=1000,
            sample_loss=loss_fn,
            bounds=np.array(bounds),
            model=model,
            filename=filename
        )

    if args.workers > 1:
        threads = [threading.Thread(target=thread_fn) for _ in range(args.workers)]
        for t in threads:
            time.sleep(3)
            t.start()
        # [t.join(10) for t in threads]
    else: thread_fn()


if __name__ == '__main__':
    main_gp()
