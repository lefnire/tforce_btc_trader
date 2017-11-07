import argparse, pdb
import tensorflow as tf
from hypersearch import HyperSearch
from tensorforce import Configuration
from tensorforce.agents import agents as agents_dict
from btc_env.btc_env import BitcoinEnvTforce
from tensorforce.execution import Runner

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
        hydrated['tf_session_config'] = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.82/args.gpu_split))
    agent = agents_dict[args.agent](
        states_spec=env.states,
        actions_spec=env.actions,
        network_spec=network,
        config=Configuration(**hydrated)
    )

    episodes = None if args.use_winner else 300
    # Note: unlike threaded which requires we close the env manually (good), single-runner closes automatically -
    # so need to do run_finished in last episode_finished()
    def ep_fin(r):
        nonlocal episodes
        if episodes and r.agent.episode >= episodes - 1:
            hs.run_finished(r.environment.gym.env.episode_results['rewards'])
            return False
        return True
    runner = Runner(agent=agent, environment=env)
    runner.run(episodes=episodes, episode_finished=ep_fin)


if __name__ == '__main__':
    while True: main()