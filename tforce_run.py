import copy
import numpy as np
from pprint import pprint
from sqlalchemy.sql import text

import tensorflow as tf
from tensorforce import Configuration, TensorForceError, util
from tensorforce.agents import PPOAgent, DQNAgent, NAFAgent
from tensorforce.core.networks import layered_network_builder
from tensorforce.execution import Runner


from tforce_env import BitcoinEnv
from helpers import conn

EPISODES = 10000  # 100000
STEPS = 10000  # 10000

AGENT_NAME = 'DQNAgent;tmp'
overrides = dict(
    tf_session_config=tf.ConfigProto(device_count={'GPU': 0})
    # tf_session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.4))  # .284 .44
)
# Current: layers/neurons, PPO, (this weekend: prioritized_replay)
# After: raw/standardize, discount, batch_size, agent(VPG, VRPO, NAF)

# Winners: delta, dbl-dqn
# Losers: 2dense64n, absolute
# Unclear (try later): use_indicators, dropout, 2lstm150n, clip

agent_type = AGENT_NAME.split(';')[0]  # (DQNAgent|PPOAgent|NAFAgent)
env = BitcoinEnv(use_indicators=False, limit=STEPS, agent_type=agent_type, agent_name=AGENT_NAME)

mem_agent_conf = dict(
    # memory='prioritized_replay',
    # memory_capacity=STEPS,
    # first_update=int(STEPS/10),
    # update_frequency=500,
    clip_loss=.1,
    double_dqn=True,
    discount=.99
)

common_conf = dict(
    network=layered_network_builder([
        dict(type='lstm', size=256, dropout=.2),
        dict(type='lstm', size=128, dropout=.2),
        dict(type='lstm', size=64, dropout=.2),
    ]),
    batch_size=150,
    states=env.states,
    actions=env.actions,
    exploration=dict(
        type="epsilon_decay",
        epsilon=1.0,
        epsilon_final=0.1,
        epsilon_timesteps=int(STEPS * 100)  # 1e6
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
            max_timesteps=STEPS,
            learning_rate=.001
        )
    )
)

def episode_finished(r):
    """ Callback function printing episode statistics"""
    # if r.episode % int(EPISODES/100) != 0: return True
    # if r.episode % 5 != 0: return True
    agent_name = r.environment.name
    # r.environment.plotTrades(r.episode, r.episode_rewards[-1], agent_name)
    avg_len = int(np.median(r.episode_lengths[-20:]))
    avg_reward = int(np.median(r.episode_rewards[-20:]))
    avg_cash = round(np.median(r.environment.episode_cashs[-20:]), 1)
    avg_value = round(np.median(r.environment.episode_values[-20:]), 1)
    print("Ep.{} time:{}, reward:{} cash_val:{}".format(
        r.episode, r.environment.time, avg_reward, round(avg_cash + avg_value, 2)
    ))

    # save a snapshot of the actual graph & the buy/sell signals so we can visualize elsewhere
    y = list(r.environment.y_train[:500])
    signals = list(r.environment.signals[:500])

    q = text("""
        insert into episodes (episode, reward, cash, value, agent_name, steps, y, signals) 
        values (:episode, :reward, :cash, :value, :agent_name, :steps, :y, :signals)
    """)
    conn.execute(q,
                 episode=r.episode,
                 reward=r.episode_rewards[-1],
                 cash=r.environment.cash,
                 value=r.environment.value,
                 agent_name=agent_name,
                 steps=r.episode_lengths[-1],
                 y=y,
                 signals=signals
    )
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