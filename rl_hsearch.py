import argparse, json, math, time, pdb
from pprint import pprint
from box import Box
import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy.sql import text
from tensorforce.agents import agents as agents_dict
from tensorforce.core.networks.layer import Dense
from tensorforce.core.networks.network import LayeredNetwork
from tensorforce.execution import Runner
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


from btc_env import BitcoinEnv
from data import engine

"""
Each hyper is specified as `key: {type, vals, requires, hook}`. 
- type: (int|bounded|bool). bool is True|False param, bounded is a float between min & max, int is "choose one" 
    eg 'activation' one of (tanh|elu|selu|..)`)
- vals: the vals this hyper can take on. If type(vals) is primitive, hard-coded at this value. If type is list, then
    (a) min/max specified inside (for bounded); (b) all possible options (for 'int'). If type is dict, then the keys
    are used in the searching (eg, look at the network hyper) and the values are used as the configuration.
- guess: initial guess (supplied by human) to explore      
- pre/post (hooks): transform this hyper before plugging it into Configuration(). Eg, we'd use type='bounded' for batch size since
    we want to range from min to max (insteaad of listing all possible values); but we'd cast it to an int inside
    hook before using it. (Actually we clamp it to blocks of 8, as you'll see)

The special sauce is specifying hypers as dot-separated keys, like `memory.type`. This allows us to easily 
mix-and-match even within a config-dict. Eg, you can try different combos of hypers within `memory{}` w/o having to 
specify the whole block combo (`memory=({this1,that1}, {this2,that2})`). To use this properly, make sure to specify
a `requires` field where necessary. 
"""


def build_net_spec(net, indicators, baseline=False):
    """Builds a net_spec from some specifications like width, depth, etc"""
    dense = {'type': 'dense', 'activation': net.activation, 'l2_regularization': net.l2, 'l1_regularization': net.l1}
    dropout = {'type': 'dropout', 'rate': net.dropout}
    conv2d = {'type': 'conv2d', 'bias': True, 'l2_regularization': net.l2, 'l1_regularization': net.l1}  # TODO bias as hyper?
    lstm = {'type': 'internal_lstm', 'dropout': net.dropout}
    only_net_end = net.type == 'lstm' and baseline

    arr = []
    if net.dropout: arr.append({**dropout})

    # Pre-layer
    if 'pre_depth' in net and not only_net_end:
        for i in range(net.pre_depth):
            size = int(net.width/(net.pre_depth-i+1)) if net.funnel else net.width
            arr.append({'size': size, **dense})
            if net.dropout: arr.append({**dropout})

    # Mid-layer
    if not only_net_end:
        n_cols = BitcoinEnv.n_cols(conv2d=net.type == 'conv2d', indicators=indicators)
        for i in range(net.depth):
            if net.type == 'conv2d':
                size = max([32, int(net.width/4)])
                if i == 0: size = int(size/2) # FIXME most convs have their first layer smaller... right? just the first, or what?
                arr.append({'size': size, 'window': (net.window, n_cols), 'stride': (net.stride, 1), **conv2d})
            else:
                # arr.append({'size': net.width, 'return_final_state': (i == net.depth-1), **lstm})
                arr.append({'size': net.width, **lstm})
        if net.type == 'conv2d':
            arr.append({'type': 'flatten'})

    # Dense
    for i in range(net.depth):
        size = int(net.width / (i + 2)) if net.funnel else net.width
        arr.append({'size': size, **dense})
        if net.dropout: arr.append({**dropout})

    return arr


def custom_net(net, indicators, print_net=False, baseline=False):
    net = Box(net)
    layers_spec = build_net_spec(net, indicators, baseline)
    if print_net: pprint(layers_spec)
    if net.type != 'conv2d': return layers_spec

    class ConvNetwork(LayeredNetwork):
        def __init__(self, **kwargs):
            super(ConvNetwork, self).__init__(layers_spec, **kwargs)

        def tf_apply(self, x, internals, update, return_internals=False):
            image = x['state0']  # 150x7x2-dim, float
            money = x['state1']  # 1x2-dim, float
            x = image

            internal_outputs = list()
            index = 0

            apply_money_here = None
            for i, layer in enumerate(self.layers):
                if isinstance(layer, Dense):
                    apply_money_here = i
                    if not net.money_last: break  # this pegs money at the first Dense.

            for i, layer in enumerate(self.layers):
                layer_internals = [internals[index + n] for n in range(layer.num_internals)]
                index += layer.num_internals
                if i == apply_money_here:
                    x = tf.concat([x, money], axis=1)
                x = layer.apply(x, update, *layer_internals)

                if not isinstance(x, tf.Tensor):
                    internal_outputs.extend(x[1])
                    x = x[0]

            if return_internals:
                return x, internal_outputs
            else:
                return x
    return ConvNetwork


def bins_of_8(x): return int(x // 8) * 8

hypers = {}
hypers['agent'] = {}
hypers['batch_agent'] = {
    'batch_size': {
        'type': 'bounded',
        'vals': [8, 2048],
        'guess': 176,
        'pre': bins_of_8
    },
    'keep_last_timestep': True
}
hypers['model'] = {
    'optimizer.type': 'nadam',
    'optimizer.learning_rate': {
        'type': 'bounded',
        'vals': [1e-7, 1e-1],
        'guess': .005,
    },
    'optimization_steps': {
        'type': 'bounded',
        'vals': [2, 20],
        'guess': 10,
        'pre': int
    },
    'discount': {
        'type': 'bounded',
        'vals': [.95, .99],
        'guess': .97
    },
    'reward_preprocessing_spec': {
        'type': 'int',
        'vals': ['None', 'normalize', 'clip'],  # TODO others?
        'guess': 'normalize',
        'hydrate': lambda x, flat: {
            'None': {'reward_preprocessing_spec': None},
            'normalize': {'reward_preprocessing_spec': {'type': 'normalize'}},
            'clip': {'reward_preprocessing_spec': {'type': 'clip', 'min_value': -5, 'max_value': 5}},
        }[x]
    },
    # TODO variable_noise
}
hypers['distribution_model'] = {
    'entropy_regularization': {
        'type': 'bounded',
        'vals': [0., 1.],
        'guess': .4
    }
    # distributions_spec (gaussian, beta, etc). Pretty sure meant to handle under-the-hood, investigate
}
hypers['pg_model'] = {
    'baseline_mode': {
        'type': 'bool',
        'guess': False,
        'hydrate': lambda x, flat: {
            False: {'baseline_mode': None},
            True: {
                'baseline': {'type': 'custom'},
                'baseline_mode': 'states',
                'baseline_optimizer': {
                    'type': 'multi_step',
                    'num_steps': flat['optimization_steps'],
                    'optimizer': {
                        'type': flat['step_optimizer.type'],
                        'learning_rate': flat['step_optimizer.learning_rate']
                    }
                },
            }
        }[x]
    },
    'gae_lambda': {
        'type': 'bounded',
        'vals': [.85, .99],
        'guess': .96,
        'post': lambda x, others: x if others['baseline_mode'] and x > .9 else None
    },
}
hypers['pg_prob_ration_model'] = {
    # I don't know what values to use besides the defaults, just guessing. Look into
    'likelihood_ratio_clipping': {
        'type': 'bounded',
        'vals': [0., 1.],
        'guess': .5
    }
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

hypers['custom'] = {
    'indicators': {
        'type': 'bool',
        'guess': False
    },
    'net.depth': {
        'type': 'bounded',
        'vals': [1, 5],
        'guess': 2,
        'pre': int
    },
    'net.width': {
        'type': 'bounded',
        'vals': [32, 768],
        'guess': 384,
        'pre': bins_of_8
    },
    'net.funnel': {
        'type': 'bool',
        'guess': True
    },
    # 'net.type': {'type': 'int', 'vals': ['lstm', 'conv2d']},  # gets set from args.net_type
    'net.activation': {
        'type': 'int',
        'vals': ['tanh', 'elu'],  # , 'relu', 'selu'],
        'guess': 'elu'
    },
    'net.dropout': {
        'type': 'bounded',
        'vals': [0., .5],
        'guess': .2,
        'pre': lambda x: None if x < .1 else x
    },
    'net.l2': {
        'type': 'bounded',
        'vals': [1e-5, .1],
        'guess': .05
    },
    'net.l1': {
        'type': 'bounded',
        'vals': [1e-5, .1],
        'guess': .05
    },
    'diff': {
        'type': 'int',
        'vals': ['raw', 'percent', 'standardize'],
        'guess': 'percent',
    },
    'steps': {
        'type': 'bounded',
        'vals': [2048*3+3, 3*(2048*3+3)],
        'guess': 2048*3+3,
        'pre': int
    },
    'unimodal': {
        'type': 'bool',
        'guess': False
    },
    'deterministic': {
        # Whether to use deterministic=True during testing phase https://goo.gl/6GgLJo
        'type': 'bool',
        'guess': True
    }
}

hypers['lstm'] = {
    'net.pre_depth': {
        'type': 'bounded',
        'vals': [0, 3],
        'guess': 1,
        'pre': int,
    },
}
hypers['conv2d'] = {
    # 'net.bias': {'type': 'bool'},
    'net.window': {
        'type': 'bounded',
        'vals': [3, 8],
        'guess': 4,
        'pre': int,
    },
    'net.stride': {
        'type': 'bounded',
        'vals': [1, 4],
        'guess': 2,
        'pre': int,
    },
    'net.money_last': {
        'type': 'bool',
        'guess': True
    },
}


# Fill in implicit 'vals' (eg, 'bool' with [True, False])
for _, section in hypers.items():
    for k, v in section.items():
        if type(v) != dict: continue  # hard-coded vals
        if v['type'] == 'bool': v['vals'] = [0, 1]

class DotDict(object):
    """
    Utility class that lets you get/set attributes with a dot-seperated string key, like `d = a['b.c.d']` or `a['b.c.d'] = 1`
    """
    def __init__(self, data):
        self._data = data
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


class HSearchEnv(object):
    """
    This is the "wrapper" environment (the "inner" environment is the one you're testing against, like Cartpole-v0).
    This env's actions are all the hyperparameters (above). The state is nothing (`[1.]`), and a single episode is
    running the inner-env however many episodes (300). The inner's last-few reward avg is outer's one-episode reward.
    That's one run: make inner-env, run 300, avg reward, return that. The next episode will be a new set of
    hyperparameters (actions); run inner-env from scratch using new hypers.
    """
    def __init__(self, agent='ppo_agent', gpu_split=1, net_type='conv2d'):
        """
        TODO only tested with ppo_agent. There's some code for dqn_agent, but I haven't tested. Nothing else
        is even attempted implemtned
        """
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
        self.gpu_split = gpu_split
        self.net_type = net_type
        self.conn = engine.connect()

    def close(self):
        self.conn.close()

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
        flat.update(self.hardcoded)

        # Post-process hypers (allow for dependency handling, etc)
        for k, v in flat.items():
            hyper = self.hypers[k]
            if type(hyper) == dict and 'post' in hyper:
                flat[k] = hyper['post'](v, flat)

        # change all a.b=c to {a:{b:c}} (note DotDict class above, I hate and would rather use an off-the-shelf)
        main, custom = DotDict({}), DotDict({})
        for k, v in flat.items():
            obj = main if k in hypers[self.agent] else custom
            try: obj.update(self.hypers[k]['hydrate'](v, self.flat))
            except: obj[k] = v
        main, custom = main.to_dict(), custom.to_dict()

        # FIXME handle in diff.hydrate()
        if custom['diff'] == 'standardize':
            main['states_preprocessing_spec'] = {'type': 'running_standardize', 'reset_after_batch': False}

        network = custom_net(custom['net'], custom['indicators'], print_net=True)
        if flat['baseline_mode']:
            main['baseline']['network_spec'] = custom_net(custom['net'], custom['indicators'], baseline=True)

        # GPU split
        session_config = None
        if self.gpu_split != 1:
            fraction = .90 / self.gpu_split if self.gpu_split > 1 else self.gpu_split
            session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=fraction))
        main['session_config'] = session_config

        print('--- Flat ---')
        pprint(flat)
        print('--- Hydrated ---')
        pprint(main)

        return flat, main, network

    def execute(self, actions):
        flat, hydrated, network = self.get_hypers(actions)

        env = BitcoinEnv(flat, name=self.agent)
        agent = agents_dict[self.agent](
            states_spec=env.states,
            actions_spec=env.actions,
            network_spec=network,
            **hydrated
        )

        # n_train, n_test = 2, 1
        n_train, n_test = 250, 50
        runner = Runner(agent=agent, environment=env)
        runner.run(episodes=n_train)  # train
        env.testing = True
        for i in range(n_test):
            next_state, terminal = env.reset(), False
            while not terminal:
                actions = agent.act(next_state, deterministic=flat['deterministic'])  # test
                next_state, terminal, reward = env.execute(actions)
        # You may need to remove runner.py's close() calls so you have access to runner.episode_rewards, see
        # https://github.com/lefnire/tensorforce/commit/976405729abd7510d375d6aa49659f91e2d30a07

        # I personally save away the results so I can play with them manually w/ scikitlearn & SQL
        rewards = env.episode_rewards
        reward = np.mean(rewards[-n_test:])
        sql = """
          insert into runs (hypers, reward_avg, rewards, agent, prices, actions, flag) 
          values (:hypers, :reward_avg, :rewards, :agent, :prices, :actions, :flag)
        """
        self.conn.execute(
            text(sql),
            hypers=json.dumps(flat),
            reward_avg=reward,
            rewards=rewards,
            agent='ppo_agent',
            prices=env.prices.tolist(),
            actions=env.signals,
            flag=self.net_type
        )
        print(flat, f"\nReward={reward}\n\n")

        runner.close()
        return reward

    def get_winner(self):
        sql = "select id, hypers from runs where flag is null and agent=:agent order by reward_avg desc limit 1"
        winner = self.conn.execute(text(sql), agent=self.agent).fetchone()
        self.hardcoded = winner.hypers
        print(f'Using winner {winner.id}')
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


def main_gp():
    import gp, GPyOpt
    from sklearn.feature_extraction import DictVectorizer

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu_split', type=float, default=.3, help="Num ways we'll split the GPU (how many tabs you running?)")
    parser.add_argument('-n', '--net_type', type=str, default='lstm', help="(lstm|conv2d) Which network arch to use")
    parser.add_argument('--guess', action="store_true", default=False, help="Run the hard-coded 'guess' values first before exploring")
    parser.add_argument('--gpyopt', action="store_true", default=False, help="Use GPyOpt library? Or use basic sklearn GP implementation?")
    args = parser.parse_args()

    # Encode features
    hsearch = HSearchEnv(gpu_split=args.gpu_split, net_type=args.net_type)
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

    # Map TensorForce actions to GPyOpt-compatible `domain`
    # instantiate just to get actions (get them from hypers above?)
    bounds = []
    for k in feat_names:
        hyper = hypers_.get(k, False)
        if hyper:
            bounded, min_, max_ = hyper['type'] == 'bounded', min(hyper['vals']), max(hyper['vals'])
        if args.gpyopt:
            b = {'name': k, 'type': 'discrete', 'domain': (0, 1)}
            if bounded: b.update(type='continuous', domain=(min_, max_))
        else:
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
        if not args.gpyopt: vec = [vec]  # gp.py passes as flat, GPyOpt as wrapped
        reversed = vectorizer.inverse_transform(vec)[0]
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
        hsearch = HSearchEnv(gpu_split=args.gpu_split, net_type=args.net_type)
        reward = hsearch.execute(vec2hypers(params))
        hsearch.close()
        return [reward]

    while True:
        """
        Every iteration, re-fetch from the database & pre-train new model. Acts same as saving/loading a model to disk, 
        but this allows to distribute across servers easily
        """
        conn = engine.connect()
        sql = "select hypers, reward_avg from runs where flag=:f"
        runs = conn.execute(text(sql), f=args.net_type).fetchall()
        conn.close()
        X, Y = [], []
        for run in runs:
            X.append(hypers2vec(run.hypers))
            Y.append([run.reward_avg])
        print_feature_importances(X, Y, feat_names)

        if args.guess:
            guesses = {k: v['guess'] for k, v in hypers_.items()}
            X.append(hypers2vec(guesses))
            Y.append([None])
            args.guess = False

        if args.gpyopt:
            pretrain = {'X': np.array(X), 'Y': np.array(Y)} if X else {}
            opt = GPyOpt.methods.BayesianOptimization(
                f=loss_fn,
                domain=bounds,
                maximize=True,
                **pretrain
            )
            opt.run_optimization(max_iter=5)
        else:
            gp.bayesian_optimisation2(
                n_iters=5,
                loss_fn=loss_fn,
                bounds=np.array(bounds),
                x_list=X,
                y_list=Y
            )

if __name__ == '__main__':
    main_gp()
