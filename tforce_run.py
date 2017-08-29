import copy
import numpy as np
from pprint import pprint
from sqlalchemy.sql import text

from tensorforce import Configuration, TensorForceError, util
from tensorforce.agents import PPOAgent, DQNAgent, NAFAgent
from tensorforce.core.networks import layered_network_builder
from tensorforce.execution import Runner


from tforce_env import BitcoinEnv
from helpers import conn

EPISODES = 10000  # 100000
STEPS = 10000  # 10000


def run_agent(agent_type='DQNAgent'):
    """
    :param agent_type: (DQNAgent|PPOAgent|NAFAgent)
    """
    env = BitcoinEnv(use_indicators=False, limit=STEPS, agent_type=agent_type)

    mem_agent_conf = dict(
        # memory='prioritized_replay',
        memory_capacity=STEPS,
        # first_update=int(STEPS/10),
        # update_frequency=500,
        clip_loss=.1,
        double_dqn=True,
        discount=.99,
        learning_rate=0.00025,

    )
    common_conf = dict(
        network=layered_network_builder([
            dict(type='lstm', size=150, dropout=.2),
            dict(type='lstm', size=150, dropout=.2)
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
        }
    )
    blank_grid = dict(batch_size=[150])

    agents = dict(
        # Current: layers/neurons, PPO (this weekend: prioritized_replay)
        # After: layers/neurons, raw/standardize, discount, batch_size, agent(VPG, VRPO, NAF)

        # Winners: delta, dbl-dqn
        # Losers: 2dense64n, absolute
        # Unclear (try later): use_indicators, dropout, 2lstm150n, clip
        DQNAgent=dict(
            agent=DQNAgent,
            grid=dict(
                network=[
                    dict(
                        label='delta;prioritized;2lstm.2d',
                        # Try 256 -> 128 -> 64 -> 32
                        val=layered_network_builder([
                            dict(type='lstm', size=150, dropout=.2),
                            dict(type='lstm', size=150, dropout=.2),
                        ])
                    )
                ]
            ),
            config=mem_agent_conf
        ),
        NAFAgent=dict(
            agent=NAFAgent,
            grid=blank_grid,
            config=mem_agent_conf
        ),
        PPOAgent=dict(
            agent=PPOAgent,
            config=dict(
                max_timesteps=STEPS,
                learning_rate=.001
            ),
            grid=dict(
                network=[
                    dict(
                        label='delta;2lstm150n',
                        val=layered_network_builder([
                            dict(type='lstm', size=150, dropout=.2),
                            dict(type='lstm', size=150, dropout=.2)
                        ]),
                    )
                ]
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

    grid = agents[agent_type]['grid']
    scores = copy.deepcopy(grid)
    for attr, vals in grid.items():
        for idx, val in enumerate(vals):

            # Setup for attr key:val
            val_str = val
            if type(val) == dict and val.get('label', None):
                val_str = val['label']
                val = val['val']
            agent_name = '{}_{}_{}'.format(agent_type, attr, val_str)
            env.name = agent_name
            conn.execute("delete from episodes where agent_name='{}'".format(agent_name))
            print("{}: {}".format(attr, val_str))

            config = {}
            config.update(common_conf)
            config.update(agents[agent_type]['config'])
            config.update({attr: val})
            pprint(config)
            agent = agents[agent_type]['agent'](config=Configuration(**config))
            runner = Runner(agent=agent, environment=env)
            runner.run(episodes=EPISODES, episode_finished=episode_finished)

            last_100_avg = round(np.median(runner.episode_rewards[-100:]), 1)
            scores[attr][idx] = last_100_avg

            # Print statistics
            print("Learning finished. Total episodes: {ep}. AVG(rewards[-100:])={ar}.".format(
                ep=runner.episode, ar=last_100_avg))
            print("Attr scores:")
            pprint(scores)

    print('Final attrs:')
    pprint({k: grid[k][np.argmax(v)]} for k,v in scores.items())


# run_agent('DQNAgent')
run_agent('PPOAgent')
# run_agent('NAFAgent')