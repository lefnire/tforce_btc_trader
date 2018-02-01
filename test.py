import pdb
from data import data
from data.data import F, Z
from btc_env import BitcoinEnv, Mode
from hypersearch import HSearchEnv
from box import Box
import pandas as pd

COUNT = 101


def count_rows(*args, **kwargs): return COUNT


def db_to_dataframe_wrapper(direction=1):
    def db_to_dataframe(*args, **kwargs):
        features = []
        for i in range(COUNT):
            num = i if direction == 1 else COUNT - i
            features.append(dict(
               a_o=num,
               a_h=num,
               a_l=num,
               a_c=num,
               a_v=num
           ))
        return pd.DataFrame(features)
    return db_to_dataframe


def reset(env):
    env.start_cash = env.start_value = 1000
    env.use_dataset(Mode.TRAIN)
    env.reset()

def main():
    hs = HSearchEnv(Box(net_type='conv2d', gpu_split=1))
    flat, hydrated, network = hs.get_winner()
    flat['unimodal'] = True
    flat['arbitrage'] = False
    flat['indicators'] = False
    flat['step_window'] = 10

    data.tables = [
        dict(
            name='a',
            ts='ts',
            cols=dict(o=F, h=F, l=F, c=F, v=Z)
        )
    ]
    data.target = 'a_c'
    data.count_rows = count_rows
    data.db_to_dataframe = db_to_dataframe_wrapper(1)

    env = BitcoinEnv(flat, name='ppo_agent')
    env.n_steps = 300  # fixme

    # Hold
    reset(env)
    for i in range(90):  # step_window - start_timestep
        next_state, terminal, reward = env.execute(0)
    env.episode_finished(None)
    assert env.acc.episode.advantages[-1] == 0

    # > 1
    reset(env)
    for i in range(90):
        next_state, terminal, reward = env.execute(1)
    env.episode_finished(None)
    assert env.acc.episode.advantages[-1] > 0

    # < 1
    reset(env)
    for i in range(90):
        next_state, terminal, reward = env.execute(-1)
    env.episode_finished(None)
    assert env.acc.episode.advantages[-1] < 0

    # Try just one
    reset(env)
    env.execute(0)
    env.episode_finished(None)
    assert env.acc.episode.advantages[-1] == 0

    reset(env)
    env.execute(1)
    env.episode_finished(None)
    assert env.acc.episode.advantages[-1] > 0

    reset(env)
    env.execute(-1)
    env.episode_finished(None)
    assert env.acc.episode.advantages[-1] < 0


    # Now for a bear market
    data.db_to_dataframe = db_to_dataframe_wrapper(-1)

    # Hold
    reset(env)
    for i in range(90):  env.execute(0)
    env.episode_finished(None)
    assert env.acc.episode.advantages[-1] == 0

    # > 1
    reset(env)
    for i in range(90): env.execute(1)
    env.episode_finished(None)
    assert env.acc.episode.advantages[-1] < 0

    # < 1
    reset(env)
    for i in range(90): env.execute(-1)
    env.episode_finished(None)
    assert env.acc.episode.advantages[-1] > 0

    # Try just one
    reset(env)
    env.execute(0)
    env.episode_finished(None)
    assert env.acc.episode.advantages[-1] == 0

    reset(env)
    env.execute(1)
    env.episode_finished(None)
    assert env.acc.episode.advantages[-1] < 0

    reset(env)
    env.execute(-1)
    env.episode_finished(None)
    assert env.acc.episode.advantages[-1] > 0


if __name__ == '__main__':
    main()
