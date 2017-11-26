import argparse

import tensorflow as tf
from tensorforce.agents import agents as agents_dict
from tensorforce.execution import Runner

from btc_env import BitcoinEnvTforce
from hypersearch import HyperSearch

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent', type=str, default='ppo_agent', help="(ppo_agent|dqn_agent) agent to use")
parser.add_argument('-g', '--gpu-split', type=int, default=0, help="Num ways we'll split the GPU (how many tabs you running?)")
parser.add_argument('--use-winner', type=str, help="(mode|predicted) Don't random-search anymore, use the winner (w/ either statistical-mode or ML-prediced hypers)")
args = parser.parse_args()


def main():
    hs = HyperSearch(agent=args.agent)
    flat, hydrated, network = hs.get_hypers(use_winner=args.use_winner)
    env = BitcoinEnvTforce(name=args.agent, hypers=flat)
    if args.gpu_split:
        hydrated['session_config'] = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.82/args.gpu_split))
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
    hs.run_finished(runner.environment.gym.env.episode_results['rewards'])
    runner.agent.close()
    runner.environment.close()


if __name__ == '__main__':
    while True: main()