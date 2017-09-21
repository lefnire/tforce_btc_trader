import tensorflow as tf
from tensorforce import Configuration, TensorForceError
from tensorforce.agents import VPGAgent, PPOAgent, DQNAgent
from tensorforce.core.networks import layered_network_builder
from pprint import pprint

from data import conn
from btc_env import BitcoinEnv

EPISODES = 500000
STEPS = 5000
AGENT_NAME = 'PPOAgent|misc'


def conf(**kwargs):
    agent_type = AGENT_NAME.split('|')[0]
    env = BitcoinEnv(
        limit=STEPS, agent_type=agent_type, agent_name=AGENT_NAME,
        scale_features=False,
        abs_reward=False
    )

    neurons = 256
    dropout = .2
    # Global conf
    # try-next: diff dropout, dqn, discount
    # possible winners: no-scale, relative-reward, normalize_rewards=False, 512>256, elu/he_init LSTM
    # definite winners: dropout, dense2, 256>150 4L, baseline=None (try MLP for DQN), random_sampling=True, nadam
    # losers: 3x-baseline, dense(original)
    # unclear: vpg
    conf = dict(
        # tf_session_config=None,
        # tf_session_config=tf.ConfigProto(device_count={'GPU': 0}),
        tf_session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.2)),

        network=[
            dict(type='dropout', size=neurons, dropout=dropout),
            dict(type='dense2', size=neurons, dropout=dropout),  # combine attrs into attr-combos (eg VWAP)
            dict(type='lstm', size=neurons, dropout=dropout),  # merge those w/ history
            # dict(type='dense2', size=neurons, dropout=dropout),  # combine those into indicators (eg SMA)
        ],

        # Main
        discount=.99,  # TODO experiment
        exploration=dict(
            type="epsilon_decay",
            epsilon=1.0,
            epsilon_final=0.1,
            epsilon_timesteps=2e6
        ),
        optimizer="nadam", # winner=nadam
        states=env.states,
        actions=env.actions,
    )

    if not env.actions['continuous']:
        conf.update(
            baseline=dict(
                type="mlp",
                sizes=[128, 128],  # losers: 2x256, winners: 2x128
                epochs=5,
                update_batch_size=128,
                learning_rate=.01
            ),
        )

    # PolicyGradientModel
    if agent_type in ['PPOAgent', 'VPGAgent', 'TRPOAgent']:
        conf.update(
            batch_size=1024,  # TODO experiment
            gae_rewards=True,  # winner
            keep_last=True,
            max_timesteps=-1,
        )
        # VPGAgent
        if agent_type == 'VPGAgent':
            agent_class = VPGAgent
            conf.update(dict(
                # normalize_rewards=True,  # winner
                normalize_rewards=False,
                random_sampling=True,
                learning_rate=.001
            ))
        # PPOAgent
        elif agent_type == 'PPOAgent':
            agent_class = PPOAgent
            conf.update(dict(
                epochs=50,
                optimizer_batch_size=1024,
                random_sampling=True,  # seems winner
                normalize_rewards=False,  # winner (even when scale_features=True)
                learning_rate=.001  # .001 best, currently speed-running
            ))

    # Q-model
    else:
        agent_class = DQNAgent
        conf.update(
            # memory_capacity=STEPS
            # first_update=int(STEPS/10),
            # update_frequency=500,
            baseline=None,
            memory='replay',
            clip_loss=.1,
            double_dqn=True,
        )
        if conf['memory'] == 'prioritized_replay':
            approach, batch = 'tforce', 16
            conf.update(**dict(
                tforce=dict(
                    batch_size=8,
                    memory_capacity=50,
                    first_update=20,
                    target_update_frequency=10,
                ),
                custom=dict(
                    batch_size=batch,
                    memory_capacity=int(batch * 6.25),
                    first_update=int(batch * 2.5),
                    target_update_frequency=int(batch * 1.25)
                ),
                # https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
                blog=dict(
                    batch_size=32,
                    memory_capacity=200000,
                    first_update=int(32 * 2.5),
                    target_update_frequency=10000
                )
            )[approach])

    # From caller (A3C v single-run)
    conf.update(**kwargs)
    pprint(conf)
    # Allow overrides to network above, then run it through configurator
    conf['network'] = layered_network_builder(conf['network'])
    conf = Configuration(**conf)

    return dict(
        agent=agent_class(config=conf),
        conf=conf,
        env=env
    )