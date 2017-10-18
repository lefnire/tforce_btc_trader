import tensorflow as tf
from tensorforce import Configuration, agents
from tensorforce.core.networks.layer import Dense
from tensorforce.core.networks.network import LayeredNetwork

from btc_env.btc_env import BitcoinEnvTforce

CONV2D = True
AGENT_TYPE = 'ppo_agent'
STEPS = 2048 * 3 + 3


def baseline(sizes=(32,32), mode='network', lr=.001, num_steps=10, type='mlp'):
    return {
        "baseline_mode": mode,
        "baseline": {
            "type": type,
            "sizes": sizes
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


def network(layers, a='elu', d=.4, l2=.001, l1=.005, b=True):
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
            ('d',512),
            ('d',256)
        ],
        # '1x': [  # loser
        #     ('C',32,(8,3),(4,2)),
        #     ('C',64,(4,2),(2,1)),
        #     ('F'),
        #     ('d',256)
        # ],
    }
    net_default = nets['3x']
    # for k in ('3x', '4x', '5x'): del nets[k]
else:
    nets = {
        # '6x': [('d', 256), ('L', 512), ('L', 512), ('L', 512), ('d', 256), ('d', 128)],
        '5x': [('d', 256), ('L', 512), ('L', 512), ('d', 256), ('d', 128)],
        '4x': [('d', 128), ('L', 256), ('L', 256), ('d', 192), ('d', 128)]
    }
    net_default = nets['4x']
    # del nets['4x']

network_experiments = dict(k='network', v=[
    dict(k=k, v=dict(network=custom_net(v)))
    for k, v in nets.items()
])
dropout_experiments = dict(k='dropout', v=[
    dict(k='.2', v=dict(network=custom_net(net_default, d=.2))),
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
        # main_experiment,
        dropout_experiments,
        activation_experiments,
        dict(k='discount', v=[
            dict(k='.99', v=dict(discount=.99)),
            dict(k='.95', v=dict(discount=.95)),
        ]),
        dict(k='gae_lambda', v=[
            dict(k='.95', v=dict(gae_rewards=.95)),  # Prior winner gae_rewards=False
        ]),
        dict(k='baseline', v=[
            dict(k='main', v=baseline()),
            # TODO states, type  # dict(k='mode_network', v=baseline(mode='states')),
            dict(k='(64,64)', v=baseline(sizes=[64,64])),
            dict(k='lr-4', v=baseline(lr=1e-4)),
            dict(k='steps5', v=baseline(num_steps=5)),
        ]),
        network_experiments,
        dict(k='batch_size', v=[
            # dict(k='8', v=dict(batch_size=8)),  # (8, 128) get stuck, always 0
            dict(k='128', v=dict(batch_size=128)),
            # dict(k='256', v=dict(batch_size=256)),  # try 1 by itself, currently crashing when w/ others
            dict(k='512', v=dict(batch_size=512)),
            dict(k='1024', v=dict(batch_size=1024)),
            dict(k='2048', v=dict(batch_size=2048)),
        ]),
        dict(k='regularization', v=[
           dict(k='l21e-5l15e-5', v=dict(network=custom_net(net_default, l2=1e-5, l1=5e-5))),
           # dict(k='l2.001l1.005', v=dict(network=custom_net(net_default, l2=.001, l1=.005))),
        ]),
        dict(k='optimization_steps', v=[
            # dict(k='10', v=dict(optimization_steps=10)),
            dict(k='5', v=dict(optimization_steps=5)),
            dict(k='1', v=dict(optimization_steps=1)),
        ]),
        # dict(k='batch', v=[
        #     dict(k='b2048.o1024', v=dict(batch_size=2048, optimizer_batch_size=1024)),
        #     dict(k='b1024.o128', v=dict(batch_size=1024, optimizer_batch_size=128)),
        # ]),
        # dict(k='keep_last', v=[
        #     dict(k='True', v=dict(keep_last=True)),
        # ]),
        # dict(k='learning_rate', v=[
        #     dict(k='1e-6', v=dict(learning_rate=1e-6)), # 6 on long-run, too slow for experiments
        # ]),
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
    prio_replay_batch = 16
    confs = [
        main_experiment,
        dict(k='learning_rate', v=[
            dict(k='1e-7', v=dict(learning_rate=1e-7)),
            dict(k='1e-8', v=dict(learning_rate=1e-8)),
            dict(k='.004', v=dict(learning_rate=.004)),
            dict(k='.0001', v=dict(learning_rate=.0001)),
        ]),
        dict(k='batch_size', v=[
            dict(k='50', v=dict(batch_size=50)),
            dict(k='100', v=dict(batch_size=100)),
        ]),
        activation_experiments,
        network_experiments,
        dict(k='discount', v=[
            dict(k='.95', v=dict(discount=.95)),
            dict(k='.99', v=dict(discount=.99)),
        ]),
        dict(k='reward_preprocessing', v=[
            dict(k='clip', v=dict(reward_preprocessing=[dict(type='clip', min=-1, max=1)]))
        ]),
        dict(k="repeat_update", v=[
           dict(k='4', v=dict(repeat_update=4)),
        ]),
        dict(k='clip_loss', v=[
            dict(k='1.', v=dict(clip_loss=1.)),
            dict(k='.5', v=dict(clip_loss=.5))
        ]),
        # TODO test target_update_frequency, first_update, memory_capacity
        dict(k='target_update_weight', v=[
            dict(k='.001', v=dict(target_update_weight=.001)),
        ]),
        # dropout_experiments,
        dict(k='memory', v=[
            dict(k='replay:False', v=dict(memory=dict(type='replay', random_sampling=False))),
        ]),
        dict(k="keep_last", v=[
           dict(k='True', v=dict(keep_last=True))
        ]),
        dict(k="exploration", v=[
            dict(k='epsilon_anneal', v=dict(
                type="epsilon_anneal",
                epsilon=1.0,
                epsilon_final=0.,
                epsilon_timesteps=1.1e6
            ))
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
        # log_level="info",
        # tf_saver=False,
        # tf_saver="saves",

        # tf_summary=None,
        # tf_summary_level=0,
        # summary_logdir=f"saves/boards/tforce/{name}",
        # summary_level=3,
        # summary_frequency=5,

        network=custom_net(net_default),
        keep_last=True,
        discount=.97,
        exploration=dict(
            type="epsilon_decay",
            epsilon=1.,
            epsilon_final=.1,
            epsilon_timesteps=1e10
        ),
        optimizer=dict(
            type='nadam',
            learning_rate=5e-6
        ),
    )

    if agent_class in (agents.TRPOAgent, agents.PPOAgent, agents.VPGAgent):
        # learning_rate: long=1e-6, short=-5
        conf.update(
            batch_size=2048,
            normalize_rewards=True,  # definite winner=True
            optimization_steps=7,
            keep_last=False,  # False seems winner for conv2d - try again
        )
        conf['step_optimizer'] = conf.pop('optimizer')
        # conf['step_optimizer']['learning_rate'] = 1e-6
        # FIXME
        if 'learning_rate' in overrides:
            print(f"overriding lr={overrides['learning_rate']}")
            conf['step_optimizer']['learning_rate'] = overrides.pop('learning_rate')
        if 'optimizer' in overrides:
            conf['step_optimizer']['type'] = overrides.pop('optimizer')

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
            # batch_size=8,
            batch_size=1024,
            memory_capacity=STEPS*2,
            target_sync_frequency=STEPS,
            first_update=STEPS//2,
        )
        # conf['optimizer']['learning_rate'] = 1e-4


    conf.update(overrides)
    network = conf.pop('network')
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