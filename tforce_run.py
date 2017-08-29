import copy, random, time
import numpy as np
from pprint import pprint

from tensorforce import Configuration, TensorForceError, util
from tensorforce.agents import PPOAgent, DQNAgent, NAFAgent
from tensorforce.core.networks import layered_network_builder
from tensorforce.execution import Runner


from tforce_env import BitcoinEnv

EPISODES = 50000  # 100000
STEPS = 10000  # 10000
AGENT = 'DQNAgent'
# AGENT = 'PPOAgent'
# AGENT = 'NAFAgent'


env = BitcoinEnv(use_indicators=False, limit=STEPS, agent_type=AGENT)

t = time.time()
# Callback function printing episode statistics
def episode_finished(r):
    global t
    # if r.episode % int(EPISODES/100) != 0: return True
    # if r.episode % 5 != 0: return True
    r.environment.plotTrades(r.episode, r.episode_rewards[-1], AGENT)
    print("Ep.{} steps:{} reward:{} time:{}s".format(
        r.episode,
        int(np.mean(r.episode_lengths[-20:])),
        int(np.mean(r.episode_rewards[-20:])),
        round(time.time() - t)
    ))
    t = time.time()
    return True

# TODO implement custom LSTM multi-layer w/ dropout
# https://medium.com/@erikhallstrm/using-the-dropout-api-in-tensorflow-2b2e6561dfeb

mem_agent_conf = dict(
    memory=dict(type='replay', random_sampling=False),
    memory_capacity=STEPS,
    first_update=int(STEPS/10),
    update_frequency=100,
    discount=.97,
)
blank_grid = dict(batch_size=[150])

agents = dict(
    common=dict(
        network=layered_network_builder([
            dict(type='lstm', size=64),
            dict(type='lstm', size=64)
        ]),
        batch_size=150,
        states=env.states,
        actions=env.actions,
        exploration=dict(
            type="epsilon_decay",
            epsilon=1.0,
            epsilon_final=0.1,
            epsilon_timesteps=int(EPISODES*(1/3))  # 1e6
        ),
        optimizer="adam"  # rmsprop, nadam
    ),
    PPOAgent=dict(
        agent=PPOAgent,
        grid=blank_grid,
        config=dict(
            random_sampling=False
        )
    ),
    DQNAgent=dict(
        agent=DQNAgent,
        grid=dict(
            network=[layered_network_builder([
                dict(type='lstm', size=150),
                dict(type='lstm', size=150)
            ])],
        ),
        config=mem_agent_conf
    ),
    NAFAgent=dict(
        agent=NAFAgent,
        grid=blank_grid,
        config=mem_agent_conf
    )
)

grid = agents[AGENT]['grid']
scores = copy.deepcopy(grid)
for attr, vals in grid.items():
    for idx, val in enumerate(vals):
        print("{}: {}".format(attr, val))
        config = {}
        config.update(agents['common'])
        config.update({attr: val})
        agent = agents[AGENT]['agent'](config=Configuration(**config))
        runner = Runner(agent=agent, environment=env)
        runner.run(episodes=EPISODES, episode_finished=episode_finished)

        last_100_avg = round(np.mean(runner.episode_rewards[-100:]), 1)
        scores[attr][idx] = last_100_avg

        # Print statistics
        print("Learning finished. Total episodes: {ep}. AVG(rewards[-100:])={ar}.".format(
            ep=runner.episode, ar=last_100_avg))
        print("Attr scores:")
        pprint(scores)

print('Final attrs:')
pprint({k: grid[k][np.argmax(v)]} for k,v in scores.items())