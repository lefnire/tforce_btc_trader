import argparse

import tensorflow as tf
from tensorforce.agents import agents as agents_dict
from tensorforce.execution.threaded_runner import ThreadedRunner, WorkerAgentGenerator

from btc_env import BitcoinEnvTforce
from hypersearch import HyperSearch

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent', type=str, default='ppo_agent', help="(ppo_agent|dqn_agent) agent to use")
parser.add_argument('-w', '--workers', type=int, default=5, help="Number of workers")
parser.add_argument('-g', '--gpu-split', type=int, default=0, help="Num ways we'll split the GPU (how many tabs you running?)")
parser.add_argument('--load', action="store_true", default=False, help="Load model from save")
parser.add_argument('--use-winner', type=str, help="(mode|predicted) Don't random-search anymore, use the winner (w/ either statistical-mode or ML-prediced hypers)")
args = parser.parse_args()


def main():
    main_agent = None
    agents, envs = [], []
    hs = HyperSearch(args.agent)
    flat, hydrated, network = hs.get_hypers(use_winner=args.use_winner)
    hydrated['saver_spec'] = dict(directory='saves/model', load=args.load)
    if args.gpu_split:
        hydrated['session_config'] = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.82/args.gpu_split))

    for i in range(args.workers):
        write_graph = False  # i == 0 and args.use_winner
        name = f'{args.agent}_{hs.run_id}' if write_graph else args.agent
        env = BitcoinEnvTforce(name=name, hypers=flat, write_graph=write_graph)
        envs.append(env)

        config = hydrated.copy()

        if i == 0:
            # let the first agent create the model, then create agents with a shared model
            agent = main_agent = agents_dict[args.agent](
                states_spec=envs[0].states,
                actions_spec=envs[0].actions,
                network_spec=network,
                **config
            )
        else:
            agent = WorkerAgentGenerator(agents_dict[args.agent])(
                states_spec=envs[0].states,
                actions_spec=envs[0].actions,
                network_spec=network,
                model=main_agent.model,
                **config
            )
        agents.append(agent)

    # When ready, look at original threaded_ale for save/load & summaries
    def summary_report(x): pass
    threaded_runner = ThreadedRunner(agents, envs)
    threaded_runner.run(
        episodes=-1 if args.use_winner else 300 * (args.workers-1),
        summary_interval=2000,
        summary_report=summary_report
    )
    for e in envs:
        hs.run_finished(e.gym.env.episode_results['rewards'])
        e.close()
    main_agent.model.close()

if __name__ == '__main__':
    while True:
        # while loop down here so vars in function above get cleaned up b/w calls
        main()