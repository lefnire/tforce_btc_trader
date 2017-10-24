import tensorflow as tf
from tensorforce import Configuration, agents
from tensorforce.core.networks.layer import Dense
from tensorforce.core.networks.network import LayeredNetwork

from btc_env.btc_env import BitcoinEnvTforce

CONV2D = True
AGENT_TYPE = 'ppo_agent'
STEPS = 2048 #* 3 + 3


def optimizer(type='nadam', lr=5e-6):
    k = 'step_optimizer' if AGENT_TYPE == 'ppo_agent' else 'optimizer'
    return {
        k: dict(
            type=type,
            learning_rate=lr
        )
    }


def network(layers, a='elu', d=.4, l2=.0001, l1=.0005, b=True):
    arr = []
    if d: arr.append(dict(type='dropout', rate=d))
    for l in layers:
        conv_args = {}
        type = l[0]
        if len(l)>1: size = l[1]
        if len(l)>2: conv_args['window'] = l[2]
        if len(l)>3: conv_args['stride'] = l[3]
        elif type == 'd':
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


def custom_net(layers, **kwargs):
    layers_spec = network(layers, **kwargs)
    for l in layers_spec:
        l.pop('l1_regularization', None)  # not yet in TF2
    if not CONV2D: return layers_spec

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

if CONV2D:
    nets = {
        '5x': [
            ('C', 64, (8, 3), (4, 1)),
            ('C', 96, (4, 2), (2, 1)),
            ('C', 96, (3, 2), 1),
            ('C', 64, (2, 2), 1),
            ('F'),
            ('d', 512),
            ('d', 256),
            ('d', 196),
            ('d', 128),
            ('d', 64)
        ],
        '4x': [
            ('C', 64, (8, 3), (4, 1)),
            ('C', 96, (4, 2), (2, 1)),
            ('C', 96, (3, 2), 1),
            ('F'),
            ('d', 512),
            ('d', 256),
            ('d', 128),
            ('d', 64)
        ],
        '3x': [
            ('C',32,(8,3),(4,2)),
            ('C',64,(4,2),(2,1)),
            ('C',64,(3,2),1),
            ('F'),
            ('d',512),
            ('d',256),
            ('d',128)
        ],
        '2x': [
            ('C', 32, (8, 3), (4, 2)),
            ('C', 64, (4, 2), (2, 1)),
            ('C', 64, (3, 2), 1),
            ('F'),
            ('d',256),
            ('d',128),
            ('d',64)
        ],
        # '1x': [  # loser
        #     ('C',32,(8,3),(4,2)),
        #     ('C',64,(4,2),(2,1)),
        #     ('F'),
        #     ('d',256)
        # ],
    }
    net_default = nets['4x']
    # for k in ('3x', '4x', '5x'): del nets[k]
else:
    nets = {
        '5x': [('d', 256), ('L', 512), ('L', 512), ('L', 512), ('d', 256), ('d', 128)],
        '4x': [('d', 128), ('L', 256), ('L', 256), ('d', 192), ('d', 128)]
    }
    net_default = nets['4x']
    # del nets['4x']


def baseline(mode='states', type='custom', lr=5e-6, net_args={'network_spec': custom_net(net_default)}, num_steps=20):
    return {
        "baseline_mode": mode,
        "baseline": {
            "type": type,
            **net_args
        },
        "baseline_optimizer": {
            "type": "multi_step",
            "optimizer": {
                "type": "nadam",
                "learning_rate": lr
            },
            "num_steps": num_steps
        }
    }


network_experiments = dict(k='network', v=[
    dict(k=k, v=dict(network=custom_net(v)))
    for k, v in nets.items()
])
dropout_experiments = dict(k='dropout', v=[
    # dict(k='.2', v=dict(network=custom_net(net_default, d=.2))),
    dict(k='None', v=dict(network=custom_net(net_default, d=None))),
])
activation_experiments = dict(k='activation', v=[
    dict(k='tanh', v=dict(network=custom_net(net_default, a='tanh'))),
    dict(k='selu', v=dict(network=custom_net(net_default, a='selu'))),
    dict(k='relu', v=dict(network=custom_net(net_default, a='relu')))
])
main_experiment = dict(k='main', v=[dict(k='-', v=dict())])


if AGENT_TYPE in ['ppo_agent', 'vpg_agent', 'trpo_agent']:
    confs = [
        # TODO entropy_regularization: None (default=1e-2)
        # likelihood_ratio_clipping: ? (default=0.2)
        # distributions? (gaussian, beta, etc). Pretty sure meant to handle under-the-hood, investigate
        # preprocessing / reward_preprocessing

        main_experiment,
        dict(k='learning_rate', v=[
            dict(k='1e-5', v={
                **baseline(lr=1e-5),
                **optimizer(lr=1e-5)
            }),
            dict(k='5e-5', v=optimizer(lr=5e-5)),
            dict(k='1e-7', v=optimizer(lr=1e-7)),
        ]),
        dict(k='gae_lambda', v=[
            dict(k='.95', v=dict(gae_lambda=.95)),  # Prior winner gae_lambda=None
        ]),
        dict(k='abs_diff', v=[dict(k='-', v=dict(
            env_args={'diff': 'absolute'}
        ))]),
        dropout_experiments,
        dict(k='baseline', v=[  # lr -6 too low, -5 slightly unstable, 5e-6 just right
            dict(k='(64,64)', v=baseline(mode='network', type='mlp', net_args={'sizes': [64,64]})),
            dict(k='lr-6', v=baseline(lr=1e-6)),
            dict(k='lr-5', v=baseline(lr=1e-5)),
            dict(k='steps10', v=baseline(num_steps=10)),  # 5=loser
            dict(k='None', v=dict(baseline_mode=None, baseline=None, baseline_optimizer=None)),
        ]),
        network_experiments,
        dict(k='batch_size', v=[
            # dict(k='512', v=dict(batch_size=512)),
            dict(k='2048', v=dict(batch_size=2048)),
            dict(k='1024', v=dict(batch_size=1024)),
            dict(k='256', v=dict(batch_size=256)),
            dict(k='128', v=dict(batch_size=128)),
            # dict(k='64', v=dict(batch_size=64)),
            dict(k='8', v=dict(batch_size=8)),
        ]),
        dict(k='keep_last_timestep', v=[
            dict(k='True', v=dict(keep_last_timestep=True)),
        ]),
        dict(k='regularization', v=[
           dict(k='l21e-5l15e-5', v=dict(network=custom_net(net_default, l2=1e-5, l1=5e-5))),
           dict(k='l2.001l1.005', v=dict(network=custom_net(net_default, l2=.001, l1=.005))),
        ]),

        dict(k='optimization_steps', v=[
            # dict(k='1', v=dict(optimization_steps=1)),  # ??
            # dict(k='5', v=dict(optimization_steps=5)),  # loser
            dict(k='10', v=dict(optimization_steps=10)),
            # dict(k='20', v=dict(optimization_steps=20)),  # b/w 10 & 20
        ]),

        activation_experiments,
        dict(k='discount', v=[
            dict(k='.97', v=dict(discount=.97)),
            dict(k='.95', v=dict(discount=.95)),
        ]),
        # dict(k='optimizer', v=[
        #     dict(k='adam', v=dict(optimizer='adam')),  # winner=Nadam
        # ]),
        # dict(k='bias', v=[  # winner=True
        #     dict(k='False', v=dict(network=custom_net(net_default, b=False)))
        # ]),
    ]

elif AGENT_TYPE in ['naf_agent']:
    confs = [
        main_experiment,
        dict(k='repeat_update', v=[
            dict(k='4', v=dict(repeat_update=4))
        ]),
        # dropout_experiments,
        network_experiments,
        activation_experiments,
        dict(k='clip_loss', v=[
            dict(k='.5', v=dict(clip_loss=.5)),
            dict(k='0.', v=dict(clip_loss=0.)),
        ]),
        dict(k='exploration', v=[
            dict(k='ornstein_uhlenbeck_no_params', v=dict(exploration="ornstein_uhlenbeck")),
            dict(k='epsilon_decay', v=dict(exploration=dict(
                type="epsilon_decay",
                epsilon=1.0,
                epsilon_final=0.1,
                epsilon_timesteps=1e6
            ))),
        ]),
        dict(k='memory_capacity', v=[
            dict(k='5e4', v=dict(memory_capacity=50000, first_update=500)),
            dict(k='1e6', v=dict(memory_capacity=1000000, first_update=10000)),
        ]),
        dict(k='target_update_frequency', v=[
            dict(k='1', v=dict(target_update_frequency=1)),
            dict(k='20', v=dict(target_update_frequency=20)),
        ]),
        dict(k='update_target_weight', v=[
            dict(k='1', v=dict(update_target_weight=1.))
        ]),
        dict(k='update_frequency', v=[  # maybe does nothing (since target_update_frequency used)
            dict(k='1', v=dict(update_frequency=1))
        ]),
        dict(k='batch_size', v=[
            dict(k='100', v=dict(batch_size=100)),
            dict(k='1', v=dict(batch_size=1)),
        ]),
        dict(k='learning_rate', v=[
            dict(k='1e-6', v=dict(learning_rate=1e-6)),
            dict(k='1e-7', v=dict(learning_rate=1e-7)),
        ]),
        # dict(k='memory', v=[  # loser
        #     dict(k='prioritized_replay', v=dict(memory='prioritized_replay')),
        # ]),
    ]

elif AGENT_TYPE in ['dqn_agent', 'dqn_nstep_agent']:
    confs = [
        # main_experiment,
        network_experiments,
        dict(k='reward_preprocessing', v=[
            dict(k='clip', v=dict(reward_preprocessing=[dict(type='clip', min=-1, max=1)]))
        ]),
        dict(k="repeat_update", v=[
           dict(k='4', v=dict(repeat_update=4)),
        ]),
        dict(k='target_update_weight', v=[
            dict(k='.001', v=dict(target_update_weight=.001)),
        ]),
        dict(k='huber_loss', v=[
            dict(k='1.', v=dict(huber_loss=1.)),
            dict(k='.5', v=dict(huber_loss=.5))
        ]),
        dict(k='batch_size', v=[
            dict(k='16', v=dict(batch_size=16)),
            dict(k='128', v=dict(batch_size=128)),
            dict(k='512', v=dict(batch_size=512)),
            dict(k='2048', v=dict(batch_size=1024)),
        ]),

        dict(k='learning_rate', v=[
            dict(k='.001', v=optimizer(lr=.001)),
            dict(k='.004', v=optimizer(lr=.004)),
            dict(k='.0001', v=optimizer(lr=.0001)),
            dict(k='1e-6', v=optimizer(lr=1e-6)),
            # dict(k='1e-7', v=optimizer(lr=1e-7)),
        ]),
        dropout_experiments,
        activation_experiments,
        dict(k='discount', v=[
            dict(k='.95', v=dict(discount=.95)),
            dict(k='.97', v=dict(discount=.97)),
        ]),
        dict(k='memory', v=[
            dict(k='replay:False', v=dict(memory=dict(type='replay', random_sampling=False))),
        ]),
        dict(k="keep_last_timestep", v=[
            dict(k='True', v=dict(keep_last_timestep=True))
        ]),
        dict(k="exploration", v=[
            dict(k='epsilon_anneal', v=dict(exploration=dict(
                type="epsilon_anneal",
                epsilon=1.0,
                epsilon_final=0.,
                epsilon_timesteps=1.1e6
            )))
        ]),
    ]


confs = [
    dict(
        name=c['k'] + ':' + permu['k'],
        conf=permu['v']
    )
    for c in confs for permu in c['v']
]


def conf(overrides, name='main', env_args={}, with_agent=True):
    name = f"{AGENT_TYPE}{'_conv2' if CONV2D else ''}|{name}"
    agent_class = agents.agents[AGENT_TYPE]

    if 'env_args' in overrides:
        env_args.update(overrides['env_args'])
        del overrides['env_args']

    env = BitcoinEnvTforce(steps=STEPS, name=name, conv2d=CONV2D, **env_args)

    conf = dict(
        tf_session_config=None,
        # tf_session_config=tf.ConfigProto(device_count={'GPU': 0}),
        # tf_session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.2)),
        # summary_logdir=f"saves/boards/tforce/{name}",
        # summary_labels=['loss'],
        # summary_frequency=5,

        network=custom_net(net_default),
        keep_last_timestep=True,
        discount=.99,
        exploration=dict(
            type="epsilon_decay",
            epsilon=1.,
            epsilon_final=.1,
            epsilon_timesteps=1e10
        ),
        batch_size=512
    )

    if agent_class in (agents.TRPOAgent, agents.PPOAgent, agents.VPGAgent):
        # learning_rate: long=1e-6, short=-5
        conf.update(
            # batch_size=16,
            normalize_rewards=True,  # definite winner=True
            optimization_steps=20,
            keep_last_timestep=True,  # False seems winner for conv2d - try again
            gae_lambda=.95,
            **optimizer(lr=5e-6),
            **baseline(mode='states', type='custom', lr=5e-6, net_args={'network_spec': custom_net(net_default)})
        )
        if agent_class == agents.PPOAgent:
            del conf['exploration']  # https://gitter.im/reinforceio/TensorForce?at=59f1ed7632e080696e2a8884

    elif agent_class == agents.NAFAgent:
        conf.update(
            network=custom_net(net_default),
            batch_size=8,
            memory_capacity=800,
            first_update=80,
            exploration=dict(
                type="ornstein_uhlenbeck",
                sigma=0.2,
                mu=0,
                theta=0.15
            ),
            update_target_weight=.001,
            clip_loss=1.
        )

    elif agent_class in (agents.DQNAgent, agents.DQNNstepAgent):
        conf.update(
            double_q_model=True,
            # batch_size=16,
            network=custom_net(nets['3x']),
            memory_capacity=STEPS*2,
            target_sync_frequency=STEPS,
            first_update=STEPS//2,
            **optimizer(lr=1e-6)
        )
        # conf['optimizer']['learning_rate'] = 1e-4


    conf.update(overrides)
    network = conf.pop('network')
    print(conf)
    conf = Configuration(**conf)

    # If we don't want the agent from the calling script (want to modify additional configs, or don't want it
    # in general) return the function to make it later
    def make_agent():
        return agent_class(
            states_spec=env.states,
            actions_spec=env.actions,
            network_spec=network,
            config=conf
        )

    return dict(
        agent=make_agent() if with_agent else make_agent,
        conf=conf,
        env=env,
        name=name
    )