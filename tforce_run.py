import copy
import numpy as np
from pprint import pprint
from sqlalchemy.sql import text

import tensorflow as tf
from tensorforce import Configuration, TensorForceError, util
from tensorforce.agents import PPOAgent, DQNAgent, NAFAgent, TRPOAgent
from tensorforce.core.networks import layered_network_builder
from tensorforce.execution import Runner


from tforce_env import BitcoinEnv
from helpers import conn

EPISODES = 50000
STEPS = 10000

AGENT_NAME = 'DQNAgent;2xlstm'
overrides = dict(
    # tf_session_config=None
    # tf_session_config=tf.ConfigProto(device_count={'GPU': 0}),
    tf_session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.2)),  # .284 .44

    memory='prioritized_replay',
    network=layered_network_builder([
        dict(type='lstm', size=150),
        dict(type='lstm', size=150)
    ]),
)

BATCH = 16
overrides.update(**dict(
    tforce=dict(
        batch_size=8,
        memory_capacity=50,
        first_update=20,
        target_update_frequency=10,
    ),
    custom=dict(
        batch_size=BATCH,
        memory_capacity=int(BATCH * 6.25),
        first_update=int(BATCH * 2.5),
        target_update_frequency=int(BATCH * 1.25)
    ),
    # https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
    blog=dict(
        batch_size=32,
        memory_capacity=200000,
        first_update=int(32 * 2.5),
        target_update_frequency=10000
    ),
    none=dict()
)['tforce'])

""" Hyper-parameter tuning
Current 
- raw/standardize: according to goo.gl/8Z4or9 StandardScaler doesn't help much, and clip_loss mitigates. But try again
- remove prioritized_replay since it samples randomly, bad for LSTM

Next
- batch_size: >8 important! (16 seems only one working; want 32+ try lets_make_dqn example)
- TRPO
- discount
- lstm w/ peephole (& other flags)
- NAF, VPG
- PPO: need to tweak parameters, poor perfomance with defaults

Winners 
- delta-score
- dbl-dqn
- no-fee (FIXME)
- rmsprop

Losers 
- dense64-64/150-150: dense always performs worse
- lstm256-128-64, 64-64, stacked;256-256
- absolute-score

Unclear (try again later)
- A3C (distributed=True, cluster_spec=?). https://www.tensorflow.org/deploy/distributed, openai_gym_async.py. Or lets_make_dqn CartPole-A3C.py
- Architectures. lstm150-150 seems winning, try again after normalization
- prioritized_replay (goo.gl/8Z4or9): True seems winning, but doesn't progress past first 10 episodes & doesn't avg>start
- use_indicators: True seems winning
- dropout
- clip
- learning_rate
"""

agent_type = AGENT_NAME.split(';')[0]  # (DQNAgent|PPOAgent|NAFAgent)
env = BitcoinEnv(limit=STEPS, agent_type=agent_type, agent_name=AGENT_NAME)

mem_agent_conf = dict(
    # memory_capacity=STEPS
    # first_update=int(STEPS/10),
    # update_frequency=500,
    clip_loss=.1,
    double_dqn=True,
    discount=.99
)

common_conf = dict(
    network=layered_network_builder([
        dict(type='lstm', size=128, dropout=.2),
        dict(type='lstm', size=128, dropout=.2),
    ]),
    batch_size=150,
    states=env.states,
    actions=env.actions,
    exploration=dict(
        type="epsilon_decay",
        epsilon=1.0,
        epsilon_final=0.1,
        epsilon_timesteps=STEPS * 50  # int(STEPS * 400)  # 1e6
    ),
    optimizer={
        "type": "rmsprop",
        "momentum": 0.95,
        "epsilon": 0.01
    },
    learning_rate=0.00025
)

agents = dict(
    DQNAgent=dict(
        agent=DQNAgent,
        config=mem_agent_conf,
    ),
    NAFAgent=dict(
        agent=NAFAgent,
        config=mem_agent_conf,
    ),
    PPOAgent=dict(
        agent=PPOAgent,
        config=dict(
            max_timesteps=STEPS
        )
    )
)

def episode_finished(r):
    """ Callback function printing episode statistics"""
    # Logging is done in BitcoinEnv
    return True

config = {}
config.update(common_conf)
config.update(agents[agent_type]['config'])
config.update(overrides)
print(AGENT_NAME)
pprint(config)
conn.execute("delete from episodes where agent_name='{}'".format(AGENT_NAME))
agent = agents[agent_type]['agent'](config=Configuration(**config))
runner = Runner(agent=agent, environment=env)
runner.run(episodes=EPISODES, episode_finished=episode_finished)

# Print statistics
print("Learning finished. Total episodes: {ep}. AVG(rewards[-100:])={ar}.".format(
    ep=runner.episode, ar=round(np.median(runner.episode_rewards[-100:]), 1)))