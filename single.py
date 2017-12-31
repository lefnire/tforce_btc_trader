import argparse
import numpy as np
from tensorforce.agents import agents as agents_dict
from tensorforce.execution import Runner

import data
from btc_env import BitcoinEnv
from rl_hsearch import HSearchEnv

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu_split', type=float, default=4, help="Num ways we'll split the GPU (how many tabs you running?)")
parser.add_argument('--id', type=int, help="Load winner from DB or hard-coded guess?")
args = parser.parse_args()


def main():
    hs = HSearchEnv(gpu_split=args.gpu_split)
    flat, hydrated, network = hs.get_winner(id=args.id)
    flat['steps'] = -1
    env = BitcoinEnv(flat, name='ppo_agent')
    agent = agents_dict['ppo_agent'](
        states_spec=env.states,
        actions_spec=env.actions,
        network_spec=network,
        **hydrated
    )

    env.train_and_test(agent)
    agent.close()
    env.close()

if __name__ == '__main__':
    while True: main()
