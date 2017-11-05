import argparse
import tensorflow as tf
from hypersearch import get_hypers, generate_and_save_hypers, create_env
from tensorforce import Configuration
from tensorforce.agents import agents as agents_dict

from tensorforce.execution import Runner

AGENT_K = 'ppo_agent'  # FIXME

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu-split', type=int, default=0, help="Num ways we'll split the GPU (how many tabs you running?)")
parser.add_argument('--deterministic', action="store_true", help="Now test for real (winning hypers).")
args = parser.parse_args()


def main():
    flat, hydrated, network = get_hypers(deterministic=args.deterministic)
    env = create_env(flat)
    if args.gpu_split:
        hydrated['tf_session_config'] = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.82/args.gpu_split))
    agent = agents_dict[AGENT_K](
        states_spec=env.states,
        actions_spec=env.actions,
        network_spec=network,
        config=Configuration(**hydrated)
    )

    episodes = None if args.deterministic else 350
    # Note: unlike threaded which requires we close the env manually (good), single-runner closes automatically -
    # so need to do run_finished in last episode_finished()
    def ep_fin(r):
        global episodes
        if episodes and r.agent.episode >= episodes - 1:
            r.environment.gym.env.run_finished()
            return False
        return True
    runner = Runner(agent=agent, environment=env)
    runner.run(episodes=episodes)


if __name__ == '__main__':
    while True: main()