import copy
import numpy as np
from pprint import pprint

from tensorforce import Configuration
from tensorforce.agents import PPOAgent, DQNAgent
from tensorforce.core.networks import layered_network_builder
from tensorforce.execution import Runner

# import gym_trading
# import gym
# csv = "gym_trading/data/EURUSD60.csv"
# env = gym.make('trading-v0')
# env.initialise_simulator(csv, trade_period=50, train_split=0.7)

from tforce_env import BitcoinEnv

EPISODES = 50000  # 100000
LIMIT = 10000  # 10000

env = BitcoinEnv(limit=LIMIT)

grid = dict(
    discount=[.99, .95],
    optimizer=["adam", "rmsprop"],
    lstm_size=[150, 64]
)
scores = copy.deepcopy(grid)

for attr, vals in grid.items():
    for idx, val in enumerate(vals):

        lstm_size = val if attr == 'lstm_size' else grid['lstm_size'][0]
        agent = DQNAgent(config=Configuration(
            exploration={
                "type": "epsilon_decay",
                "epsilon": 1.0,
                "epsilon_final": 0.1,
                "epsilon_timesteps": int(EPISODES*(2/3)) # 1e6
            },
            discount=val if attr == 'discount' else grid['discount'][0],
            batch_size=150,
            # memory_capacity=LIMIT,  # 800
            optimizer=val if attr == 'optimizer' else grid['optimizer'][0],
            memory=dict(
                type='replay',
                random_sampling=False
            ),
            first_update=500,  # TODO ???

            states=env.states,
            actions=env.actions,
            network=layered_network_builder([
                dict(type='lstm', size=lstm_size),
                dict(type='lstm', size=lstm_size)
            ])
        ))

        # Create the runner
        runner = Runner(agent=agent, environment=env)


        # Callback function printing episode statistics
        def episode_finished(r):
            if r.episode % int(EPISODES/100) != 0: return True
            # r.environment.plotTrades(r.episode, r.episode_rewards[-1])
            print("Ep {}) avg[-20:] steps:{} reward:{}".format(
                r.episode,
                int(np.mean(r.episode_lengths[-20:])),
                int(np.mean(r.episode_rewards[-20:]))
                #int(r.environment.total_reward),
            ))
            return True


        # Start learning
        runner.run(episodes=EPISODES, episode_finished=episode_finished)

        last_100_avg = round(np.mean(runner.episode_rewards[-100:]), 1)
        scores[attr][idx] = last_100_avg

        # Print statistics
        print("Learning finished. Total episodes: {ep}. AVG(rewards[-100:])={ar}.".format(
            ep=runner.episode, ar=last_100_avg))
        print("Attr scores:")
        pprint(scores)


for k,v in scores.items():
    scores[k] = grid[k][np.argmax(v)]
print('Final attrs:')
pprint(scores)
# {'discount': [19612.5, 19827.799999999999],
#  'lstm_size': [19492.0, 21316.700000000001],
#  'optimizer': [22245.400000000001, 17938.200000000001]}
# Final attrs:
# {'discount': 0.95, 'lstm_size': 64, 'optimizer': 'adam'}