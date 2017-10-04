from collections import namedtuple
AGENT_TYPE = 'NAFAgent'

def baseline(**kwargs):
    b = dict(baseline=dict(
        type="mlp",
        sizes=[64, 64],
        epochs=10,
        update_batch_size=128,  # 1024
        learning_rate=.01
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
                l2_regularization=l2)
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


if AGENT_TYPE in ['PPOAgent', 'VPGAgent', 'TRPOAgent']:
    confs = [
        dict(k='main', v=[dict(k='-', v=dict())]),
        dict(k='learning_rate', v=[
            dict(k='1e-5', v=dict(learning_rate=1e-5)),
            dict(k='.01', v=dict(learning_rate=.01)),
        ]),
        dict(k='batch', v=[
            dict(k='b128.o64', v=dict(batch_size=128, optimizer_batch_size=64)),
            dict(k='b2048.o64(ppo1)', v=dict(batch_size=2048, optimizer_batch_size=64)),
            dict(k='b256.o128', v=dict(batch_size=256, optimizer_batch_size=128)),
            dict(k='b1024.o256', v=dict(batch_size=1024, optimizer_batch_size=256)),
            dict(k='b2048.o512', v=dict(batch_size=2048, optimizer_batch_size=512)),
            dict(k='b4096.o2048', v=dict(batch_size=4096, optimizer_batch_size=2048)),
        ]),
        dict(k='dropout', v=[
            dict(k='.2', v=network_old(d=.2)),
            dict(k='l2_reg', v=network_old(d=None)),
            dict(k='.1', v=network_old(d=.1)),
        ]),
        dict(k='activation', v=[
            dict(k='tanh', v=network_old(a='tanh')),  # TODO experiment
            dict(k='selu', v=network_old(a='selu')),
            dict(k='dense1', v=network_old('LLdd')),
            dict(k='dense1.l2', v=network_old('LLdd', d=None))
        ]),
        dict(k='epochs', v=[
            dict(k='1', v=dict(epochs=1)),
            dict(k='10', v=dict(epochs=10)),
            dict(k='20', v=dict(epochs=20)),
        ]),
        dict(k='baseline', v=[
            dict(k='None', v=dict(baseline=None)),
            dict(k='2x64', v=baseline(sizes=[64, 64])),
            dict(k='2x256', v=baseline(sizes=[256, 256])),
            dict(k='epochs10', v=baseline(epochs=10)),
            dict(k='update_batch_size64', v=baseline(update_batch_size=64)),
            dict(k='update_batch_size1024', v=baseline(update_batch_size=1024)),
            dict(k='learning_rate.001', v=baseline(learning_rate=.001)),
        ]),
        dict(k='network', v=[
            dict(k=f'{arch}.{neur}', v=network_old(arch, neur))
            for neur in [512, 256]
            for arch in [
                # 'L',  # loser
                # 'DL', 'LD', 'LL',  # losers
                # 'DLD', 'DLL', 'LDD', 'LLD', 'LLL',
                # 'DLLD', 'LLDD', 'LLLD',
                # 'DLLLD'
                'LLdd', 'LLDD', 'DLLLD', 'DLLD', 'LLD'
            ]
            # Good: LLDD64, DLLLD64; ~DLL128, LLD128, LLDD128
            # D{0|1}L{2+}D{1+}, LLDD seems best
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


net3x = [('L',256), ('L',256), ('d',192), ('d',128)]
net4x = [('L',512), ('L',512), ('d',256), ('d',128)]


if AGENT_TYPE in ['NAFAgent']:
    confs = [
        dict(k='main', v=[dict(k='-', v=dict())]),
        dict(k='dropout', v=[
            # These two are a tie, but I *think* .5(big) is slightly better. Roll back to 3x + .2 otherwise
            dict(k='None(3x)', v=dict(network=network(net3x, d=None))),
            dict(k='.2(3x)', v=dict(network=network(net3x, d=.2))),
            dict(k='.5(4x)', v=dict(network=network(net4x, d=.5))),
        ]),
        dict(k='clip_loss', v=[
            dict(k='.5', v=dict(clip_loss=.5)),
            dict(k='0.', v=dict(clip_loss=0.)),
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
        dict(k='exploration', v=[
            dict(k='ornstein_uhlenbeck_params', v=dict(exploration=dict(
                type="ornstein_uhlenbeck",
                sigma=0.2,
                mu=0,
                theta=0.15
            ))),
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
        dict(k='repeat_update', v=[
            dict(k='10', v=dict(repeat_update=10))
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
            dict(k='1e-8', v=dict(learning_rate=1e-8)),
        ]),
        # dict(k='memory', v=[  # loser
        #     dict(k='prioritized_replay', v=dict(memory='prioritized_replay')),
        # ]),
    ]

confs = [
    dict(
        name=c['k'] + ':' + permu['k'],
        conf=permu['v']
    )
    for c in confs for permu in c['v']
]