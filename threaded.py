import argparse, time, pdb
import numpy as np
import tensorflow as tf
from tensorforce import Configuration
from tensorforce.agents import agents as agents_dict
from tensorforce.execution import ThreadedRunner
from tensorforce.execution.threaded_runner import WorkerAgent
from btc_env.btc_env import BitcoinEnvTforce
from hypersearch import HyperSearch

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent', type=str, default='ppo_agent', help="(ppo_agent|dqn_agent) agent to use")
parser.add_argument('-w', '--workers', type=int, default=5, help="Number of workers")
parser.add_argument('-g', '--gpu-split', type=int, default=0, help="Num ways we'll split the GPU (how many tabs you running?)")
parser.add_argument('--use-winner', type=str, help="(mode|predicted) Don't random-search anymore, use the winner (w/ either statistical-mode or ML-prediced hypers)")
args = parser.parse_args()


def main():
    main_agent = None
    agents, envs = [], []
    hs = HyperSearch(args.agent)
    flat, hydrated, network = hs.get_hypers(use_winner=args.use_winner)
    if args.gpu_split:
        hydrated['tf_session_config'] = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.82/args.gpu_split))

    for i in range(args.workers):
        env = BitcoinEnvTforce(name=args.agent, hypers=flat)
        envs.append(env)

        conf = hydrated.copy()
        # optionally overwrite epsilon final values
        if "exploration" in conf and "epsilon" in conf['exploration']['type']:
            # epsilon annealing is based on the global step so divide by the total workers
            # conf.exploration.epsilon_timesteps = conf.exploration.epsilon_timesteps // WORKERS
            conf['exploration']['epsilon_timesteps'] = conf['exploration']['epsilon_timesteps'] // 2
            if i != 0:  # for the worker which logs, let it expire
                # epsilon final values are [0.5, 0.1, 0.01] with probabilities [0.3, 0.4, 0.3]
                # epsilon_final = np.random.choice([0.5, 0.1, 0.01], p=[0.3, 0.4, 0.3])
                epsilon_final = [.4, .1][i % 2]
                conf['exploration']['epsilon_final'] = epsilon_final
        conf = Configuration(**conf)

        if i == 0:
            # let the first agent create the model, then create agents with a shared model
            main_agent = agent = agents_dict[args.agent](
                states_spec=envs[0].states,
                actions_spec=envs[0].actions,
                network_spec=network,
                config=conf
            )
        else:
            conf.default(main_agent.default_config)
            agent = WorkerAgent(
                states_spec=envs[0].states,
                actions_spec=envs[0].actions,
                network_spec=network,
                config=conf,
                model=main_agent.model
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