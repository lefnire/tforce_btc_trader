import argparse
import numpy as np
from tensorforce.agents import agents as agents_dict
from tensorforce.execution import Runner

from btc_env import BitcoinEnv
from rl_hsearch import HSearchEnv

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu_split', type=float, default=.3, help="Num ways we'll split the GPU (how many tabs you running?)")
parser.add_argument('--id', type=int, help="Load winner from DB or hard-coded guess?")
parser.add_argument('--early-stop', action="store_true", default=False, help="Should stop early after some success")
args = parser.parse_args()


def main():
    hs = HSearchEnv(gpu_split=args.gpu_split)
    flat, hydrated, network = hs.get_winner(id=args.id)
    env = BitcoinEnv(flat, name='ppo_agent')
    agent = agents_dict['ppo_agent'](
        states_spec=env.states,
        actions_spec=env.actions,
        network_spec=network,
        **hydrated
    )

    runner = Runner(agent=agent, environment=env)

    def train():
        env.testing = False
        runner.run(episodes=5)

    def test():
        env.testing = True
        next_state, terminal = env.reset(), False
        while not terminal:
            next_state, terminal, reward = env.execute(agent.act(next_state, deterministic=True))
        if not args.early_stop: return False

        batch = 4
        last_few = np.array([
            r if env.episode_uniques[i] > 1 else -.01
            for i, r in enumerate(env.episode_rewards['human'][-batch:])
        ])
        # return len(last_few) > batch and np.all(last_few > 0)
        return len(last_few) > batch and np.mean(last_few) > 0

    should_stop = False
    while not should_stop:
        train()
        should_stop = test()
    print('Stopped training')
    while True: test()


if __name__ == '__main__':
    while True: main()
