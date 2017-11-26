import argparse

import tensorflow as tf
from tensorforce.agents import agents as agents_dict
from tensorforce.execution import Runner

from btc_env import BitcoinEnv
from rl_hsearch import HSearchEnv

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent', type=str, default='ppo_agent', help="(ppo_agent|dqn_agent) agent to use")
parser.add_argument('-g', '--gpu-split', type=int, default=1, help="Num ways we'll split the GPU (how many tabs you running?)")
parser.add_argument('--use-winner', action="store_true", default=True)
args = parser.parse_args()


def main():
    hs = HSearchEnv(agent=args.agent, gpu_split=args.gpu_split)
    flat, hydrated, network = hs.get_winner()
    env = BitcoinEnv(flat, name=args.agent)
    agent = agents_dict[args.agent](
        states_spec=env.states,
        actions_spec=env.actions,
        network_spec=network,
        **hydrated
    )

    runner = Runner(agent=agent, environment=env)
    runner.run(episodes=300)
    if args.use_winner:
        runner.run(deterministic=bool(args.use_winner))
    hs.run_finished(runner.environment.episode_results['rewards'])
    runner.agent.close()
    runner.environment.close()


if __name__ == '__main__':
    while True: main()