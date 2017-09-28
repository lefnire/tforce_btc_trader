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


def network(arch='LLDD', n=512, d=.2, a='elu'):
    net = [dict(type='dropout', size=n, dropout=d)] if d else []
    for layer in arch:
        if layer == 'L':
            net += [dict(type='lstm', size=n, dropout=d)]
        elif layer == 'D':
            net += [dict(type='dense2', size=n, dropout=d, activation=a)]
        elif layer == 'd':
            if d:
                net += [dict(type='dense', size=n), dict(type='dropout', size=n, dropout=d)]
            else:
                net += [dict(type='dense', size=n, l2_regularization=.001)]
    return dict(network=net)

confs = [
    dict(k='main', v=[dict(k='-', v=dict())]),
    dict(k='activation', v=[
        dict(k='tanh', v=network(a='tanh')),
        dict(k='selu', v=network(a='selu')),
        dict(k='dense1', v=network('LLdd')),
        dict(k='dense1.l2', v=network('LLdd', d=None))
    ]),
    # dict(k='network', v=[
        # dict(k='LLDD.64', v=network(n=64)),
        # dict(k='LLDD.128', v=network(n=128)),
        # dict(k='LLDD.256', v=network(n=256)),  # clear winner
        # dict(k='LLDD.512', v=network(n=512)),
    # ]),
    dict(k='baseline', v=[
        dict(k='None', v=dict(baseline=None)),  # Baseline is good
        dict(k='2x128', v=baseline(sizes=[128, 128])),  # TODO test 64 vs 128
        # dict(k='2x256', v=baseline(sizes=[256, 256])),  # loser
        dict(k='epochs10', v=baseline(epochs=10)),
        dict(k='update_batch_size64', v=baseline(update_batch_size=64)),
        dict(k='update_batch_size512', v=baseline(update_batch_size=512)),
        dict(k='learning_rate.001', v=baseline(learning_rate=.001)),
    ]),
    dict(k='network', v=[
        dict(k=f'{arch}.{neur}', v=network(arch, neur))
        for neur in [64, 128, 256, 512]
        for arch in [
            # 'L',  # loser
            # 'DL', 'LD', 'LL',  # losers
            # 'DLD', 'DLL', 'LDD', 'LLD', 'LLL',
            # 'DLLD', 'LLDD', 'LLLD',
            # 'DLLLD'
            'LLD', 'LLDD', 'DLLD', 'DLLDD', #'DLLLD
        ]

        # Good: LLDD64, DLLLD64; ~DLL128, LLD128, LLDD128
        # D{0|1}L{2+}D{1+}, LLDD seems best
    ]),
    dict(k='batch', v=[
        dict(k='b2048.o64(ppo1)', v=dict(batch_size=2048, optimizer_batch_size=64)),
        dict(k='b128.o64', v=dict(batch_size=128, optimizer_batch_size=64)),
        dict(k='b256.o128', v=dict(batch_size=256, optimizer_batch_size=128)),
        dict(k='b1024.o256', v=dict(batch_size=1024, optimizer_batch_size=256)),
        # dict(k='b1024.o512', v=dict(batch_size=1024, optimizer_batch_size=512)),
        # dict(k='b4096.o128', v=dict(batch_size=4096, optimizer_batch_size=128)),
        dict(k='b4096.o1024', v=dict(batch_size=4096, optimizer_batch_size=1024)),
        # dict(k='b4096.o2048', v=dict(batch_size=4096, optimizer_batch_size=2048)),
    ]),
    dict(k='epochs', v=[
        dict(k='1', v=dict(epochs=1)),
        dict(k='10', v=dict(epochs=10)),
        # dict(k='40', v=dict(epochs=40)),  # loser
    ]),
    dict(k='gae_rewards', v=[
        dict(k='True', v=dict(gae_rewards=True)),  # winner=False
    ]),
    dict(k='keep_last', v=[
        dict(k='True', v=dict(keep_last=True)),  # unclear (winner~=False)
    ]),
    dict(k='random_sampling', v=[
        dict(k='False', v=dict(random_sampling=False)),
    ]),
    dict(k='discount', v=[
        dict(k='.95', v=dict(discount=.95)),
        dict(k='.97', v=dict(discount=.97)),
    ]),
    dict(k='learning_rate', v=[
        dict(k='.01', v=dict(learning_rate=.01)),
        dict(k='.0001', v=dict(learning_rate=.0001)),
    ]),
    dict(k='optimizer', v=[
        dict(k='adam', v=dict(optimizer='adam')),
    ]),
    dict(k='dropout', v=[
        dict(k='l2_reg', v=network(d=None)),
        dict(k='.1', v=network(d=.1)),
        dict(k='.5', v=network(d=.5))
    ]),
    dict(k='deterministic', v=[
        dict(k='true', v=dict(deterministic=True))
    ])
]

confs = [
    dict(
        name=c['k'] + ':' + permu['k'],
        conf=permu['v']
    )
    for c in confs for permu in c['v']
]