AGENT_TYPE = 'PPOAgent'


def baseline(**kwargs):
    b = dict(baseline = dict(
        type="mlp",
        sizes=[128, 128],
        epochs=10,
        update_batch_size=512,  # 1024
        learning_rate=.001
    ))
    b['baseline'].update(**kwargs)
    return b


def network(layers, a='elu', d=None, l2=.001, l1=.005):
    arr = []
    if d: arr.append(dict(type='dropout', dropout=d))
    for type, size in layers:
        if type == 'D':
            arr.append(dict(type='dense2', size=size, dropout=d, activation=a))
        elif type == 'd':
            arr.append(dict(
                type='dense',
                size=size,
                activation=a,
                l1_regularization=l1,
                l2_regularization=0. if d else l2)
            )
            if d: arr.append(dict(type='dropout', dropout=d))
        elif type == 'L':
            arr.append(dict(type='lstm', size=size, dropout=d))
    return arr


# TODO remove this
def network_old(arch='DLLDD', n=512, d=.4, a='elu'):
    return dict(network=network(
        list(map(lambda l: (l, n), arch))
        , d=d, a=a, l2=0., l1=0.
    ))


net1x = [('L', 64), ('d', 64)]
net2x = [('L', 128), ('L', 128), ('d', 64)]
net3x = [('L',256), ('L',256), ('d',192), ('d',128)]
net4x = [('L',512), ('L',512), ('d',256), ('d',128)]
net5x = [('L',192), ('L',512), ('L',512), ('d',256), ('d',128)]


if AGENT_TYPE in ['PPOAgent', 'VPGAgent', 'TRPOAgent']:
    confs = [
        dict(k='main', v=[dict(k='-', v=dict())]),
        dict(k='learning_rate', v=[
            dict(k='1e-3', v=dict(learning_rate=1e-3)),
            dict(k='1e-5', v=dict(learning_rate=1e-5)),
        ]),
        dict(k='batch', v=[
            dict(k='b4096.o2048', v=dict(batch_size=4096, optimizer_batch_size=2048)),
            dict(k='b2048.o512', v=dict(batch_size=2048, optimizer_batch_size=512)),
            dict(k='b128.o64', v=dict(batch_size=128, optimizer_batch_size=64)),
            dict(k='b2048.o64(ppo1)', v=dict(batch_size=2048, optimizer_batch_size=64)),
            dict(k='b256.o128', v=dict(batch_size=256, optimizer_batch_size=128)),
            dict(k='b1024.o256', v=dict(batch_size=1024, optimizer_batch_size=256)),
        ]),
        dict(k='dropout', v=[
            dict(k='dropout', v=[
                # dict(k='None(3x)', v=dict(network=network(net3x, d=None))),
                dict(k='.2(3x)', v=dict(network=network(net3x, d=.2))),
                dict(k='.5(4x)', v=dict(network=network(net4x, d=.5))),
            ]),
        ]),
        dict(k='activation', v=[
            dict(k='tanh', v=dict(network=network(net3x, a='tanh'))),
            dict(k='selu', v=dict(network=network(net3x, a='selu'))),
            dict(k='relu', v=dict(network=network(net3x, a='relu')))
        ]),
        dict(k='epochs', v=[
            dict(k='1', v=dict(epochs=1)),
            dict(k='10', v=dict(epochs=10)),
            dict(k='20', v=dict(epochs=20)),
        ]),
        dict(k='network', v=[
            dict(k='5x', v=dict(network=network(net5x))),
            dict(k='4x', v=dict(network=network(net4x))),
            # dict(k='3x', v=dict(network=network(net3x))),
            dict(k='2x', v=dict(network=network(net2x))),
            dict(k='1x', v=dict(network=network(net1x))),
        ]),
        dict(k='baseline', v=[
            dict(k='2x64', v=baseline(sizes=[64, 64])),
            dict(k='2x256', v=baseline(sizes=[256, 256])),
            dict(k='epochs5', v=baseline(epochs=5)),
            dict(k='update_batch_size128', v=baseline(update_batch_size=128)),
            dict(k='update_batch_size1024', v=baseline(update_batch_size=1024)),
            dict(k='learning_rate.1e-6', v=baseline(learning_rate=1e-6)),
        ]),
        dict(k='gae_rewards', v=[
            dict(k='True', v=dict(gae_rewards=True)),  # winner=False
        ]),
        dict(k='keep_last', v=[
            dict(k='False', v=dict(keep_last=False)),
        ]),
        dict(k='random_sampling', v=[
            dict(k='False', v=dict(random_sampling=False)),
        ]),
        dict(k='optimizer', v=[
            dict(k='adam', v=dict(optimizer='adam')),
        ]),
        dict(k='deterministic', v=[
            dict(k='true', v=dict(deterministic=True))
        ]),
        dict(k='discount', v=[
            dict(k='.95', v=dict(discount=.95)),
            dict(k='.97', v=dict(discount=.97)),
        ]),
    ]

elif AGENT_TYPE in ['NAFAgent']:
    confs = [
        dict(k='main', v=[dict(k='-', v=dict())]),
        dict(k='repeat_update', v=[
            dict(k='4', v=dict(repeat_update=4))
        ]),
        dict(k='dropout', v=[
            # These two are a tie, but I *think* .5(big) is slightly better. Roll back to 3x + .2 otherwise
            dict(k='None(3x)', v=dict(network=network(net3x, d=None))),
            dict(k='.2(3x)', v=dict(network=network(net3x, d=.2))),
            dict(k='.5(4x)', v=dict(network=network(net4x, d=.5))),
        ]),
        dict(k='network', v=[
            # dict(k='1x', v=dict(network=network([('L',64), ('d',64)]))),  # loser
            # dict(k='2x', v=dict(network=network([('L',128), ('L',128), ('d',64)]))),  # loser
            dict(k='3x', v=dict(network=network(net3x))),
            dict(k='4x', v=dict(network=network(net4x)))
        ]),
        dict(k='activation', v=[
            dict(k='tanh', v=dict(network=network(net4x, a='tanh'))),
            dict(k='selu', v=dict(network=network(net4x, a='selu'))),
            dict(k='relu', v=dict(network=network(net4x, a='relu')))
        ]),
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

elif AGENT_TYPE in ['DQNAgent', 'DQNNstepAgent']:
    prio_replay_batch = 16
    confs = [
        dict(k='main', v=[
            dict(k='-', v=dict()),
            dict(k='json', v=dict(
                exploration=dict(
                    type="epsilon_anneal",
                    epsilon=1.,
                    epsilon_final=0.,
                    epsilon_timesteps=1.5e6
                ),
                reward_preprocessing=[dict(type="clip", min=-1, max=1)],
                batch_size=6,
                repeat_update=1,
                keep_last=True,
                target_update_frequency=10000,
                update_target_weight=1.0,
                clip_loss=1.0
            )),
        ]),
        dict(k="target_update_frequency", v=[
            dict(k='5000', v=dict(target_update_frequency=5000))
        ]),
        dict(k='update_target_weight', v=[
            dict(k='.001', v=dict(update_target_weight=.001)),
        ]),
        dict(k='activation', v=[
            dict(k='tanh', v=dict(network=network(net3x, a='tanh'))),
            dict(k='selu', v=dict(network=network(net3x, a='selu'))),
            dict(k='relu', v=dict(network=network(net3x, a='relu')))
        ]),
        dict(k='dropout', v=[
            # dict(k='None(3x)', v=dict(network=network(net3x, d=None))),
            dict(k='.2(3x)', v=dict(network=network(net3x, d=.2))),
            dict(k='.5(4x)', v=dict(network=network(net4x, d=.5))),
        ]),
        dict(k='network', v=[
            dict(k='5x', v=dict(network=network(net5x))),
            dict(k='4x', v=dict(network=network(net4x))),
            # dict(k='3x', v=dict(network=network(net3x))),
            dict(k='2x', v=dict(network=network([('L', 128), ('L', 128), ('d', 64)]))),
            dict(k='1x', v=dict(network=network([('L', 64), ('d', 64)]))),
        ]),
        dict(k='batch_size', v=[
            dict(k='100', v=dict(batch_size=100))
        ]),
        dict(k='batch_combo', v=[
            dict(k='8', v=dict(
                batch_size=8,
                memory_capacity=800,
                first_update=80,
                target_update_frequency=20,
            )),
            dict(k='16', v=dict(
                batch_size=16,
                memory_capacity=800,
                first_update=80,
                target_update_frequency=20,
            )),
        ]),
        dict(k='reward_preprocessing', v=[
            dict(k='clip', v=dict(reward_preprocessing=[dict(type='clip', min=-1, max=1)]))
        ]),
        dict(k='learning_rate', v=[
            dict(k='.004', v=dict(learning_rate=.004)),
            dict(k='.0001', v=dict(learning_rate=.0001)),
        ]),
        dict(k="repeat_update", v=[
           dict(k='4', v=dict(repeat_update=4)),
        ]),
        dict(k='clip_loss', v=[
            dict(k='1.', v=dict(clip_loss=1.)),
            dict(k='.5', v=dict(clip_loss=.5))
        ]),
        dict(k='memory', v=[
            dict(k='replay:False', v=dict(memory=dict(type='replay', random_sampling=False))),
            dict(k='prio-tforce', v=dict(
                memory='prioritized_replay',
                batch_size=8,
                memory_capacity=50,
                first_update=20,
                target_update_frequency=10,
            )),
            # dict(k='prio-custom', v=dict(
            #     memory='prioritized_replay',
            #     batch_size=prio_replay_batch,
            #     memory_capacity=int(prio_replay_batch * 6.25),
            #     first_update=int(prio_replay_batch * 2.5),
            #     target_update_frequency=int(prio_replay_batch * 1.25)
            # )),
            # # https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
            # dict(k='prio-blog', v=dict(
            #     memory='prioritized_replay',
            #     batch_size=32,
            #     memory_capacity=200000,
            #     first_update=int(32 * 2.5),
            #     target_update_frequency=10000
            # ))
        ]),
        dict(k="keep_last", v=[
           dict(k='True', v=dict(keep_last=True))
        ]),
        dict(k="exploration", v=[
            dict(k='epsilon_anneal', v=dict(
                type="epsilon_anneal",
                epsilon=1.0,
                epsilon_final=0.,
                epsilon_timesteps=1.5e6
            ))
        ]),
        # "discount": 0.99,
        # update_frequency=500,
    ]

    if AGENT_TYPE == 'DQNNstepAgent':
        confs += []
    elif AGENT_TYPE == 'DQNAgent':
        confs += [
            # dict(k='double_dqn', v=[dict(k='False', v=dict(double_dqn=False))])
        ]

confs = [
    dict(
        name=c['k'] + ':' + permu['k'],
        conf=permu['v']
    )
    for c in confs for permu in c['v']
]