"""
This file is for when you've found a solid hyper combo from hypersearch.py and you want to run it in the wild. Either
live, or "dry-run live" (--test-live), etc. Note, you need to run this file once first without live/test-live to
train and save the model (hypersearch doesn't save models).
"""

import argparse, os
from tensorforce.agents import agents as agents_dict
import shutil

import utils
from btc_env import BitcoinEnv
from hypersearch import HSearchEnv

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int, help="Load winner from DB or hard-coded guess?")
parser.add_argument('--live', action="store_true", default=False, help="Run in live mode")
parser.add_argument('--test-live', action="store_true", default=False, help="Dry-run live mode")
parser.add_argument('--early-stop', type=int, default=-1, help="Stop model after x successful runs")
parser.add_argument('--name', type=str, required=True, help="Name of the folder to save this run.")
utils.add_common_args(parser)
args = parser.parse_args()


def main():
    directory = os.path.join(os.getcwd(), "saves", args.name)
    filestar = os.path.join(directory, args.name)

    live_ish = args.live or args.test_live
    if not live_ish:
        try: shutil.rmtree(directory)
        except: pass
        os.mkdir(directory)

    hs = HSearchEnv(cli_args=args)
    flat, hydrated, network = hs.get_winner(id=args.id)
    env = BitcoinEnv(flat, name='ppo_agent')

    agent = agents_dict['ppo_agent'](
        states=env.states,
        actions=env.actions,
        network=network,
        **hydrated
    )

    if live_ish:
        agent.restore_model(directory)
        env.run_live(agent, test=args.test_live)
    else:
        env.train_and_test(agent, args.n_steps, args.n_tests, args.early_stop)
        agent.save_model(filestar)
        agent.close()
        env.close()


if __name__ == '__main__':
    main()
