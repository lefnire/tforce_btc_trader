import argparse
import numpy as np
import tensorflow as tf
from tensorforce.agents import agents as AgentsDictionary
from tensorforce.execution import ThreadedRunner

import experiments, agent_conf

WORKERS = 7

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=int, default=0, help="Show debug outputs")
    args = parser.parse_args()

    # exp = experiments.confs[args.experiment]
    for exp in experiments.confs[0:]:
        exp['conf'].update(tf_session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.41)))

        main_agent = None
        agents, envs = [], []
        for i in range(WORKERS):
            setup = agent_conf.conf(exp['conf'], experiments.AGENT_TYPE, exp['name'], env_args=dict(is_main=i == 0), no_agent=True)
            conf = setup['conf']
            env = setup['env']
            # optionally overwrite epsilon final values
            if "exploration" in conf and "epsilon" in conf.exploration.type:
                # epsilon annealing is based on the global step so divide by the total workers
                # conf.exploration.epsilon_timesteps = conf.exploration.epsilon_timesteps // WORKERS
                if i != 0:  # for the worker which logs, let it expire
                    # epsilon final values are [0.5, 0.1, 0.01] with probabilities [0.3, 0.4, 0.3]
                    # epsilon_final = np.random.choice([0.5, 0.1, 0.01], p=[0.3, 0.4, 0.3])
                    epsilon_final = [.5, .1][i % 2]
                    conf.exploration.epsilon_final = epsilon_final

            if i == 0:
                # let the first agent create the model, then create agents with a shared model
                main_agent = agent = AgentsDictionary[experiments.AGENT_TYPE](config=conf)
            else:
                conf.default(dict(states=envs[0].states, actions=envs[0].actions))
                agent = AgentsDictionary[experiments.AGENT_TYPE](config=conf, model=main_agent.model)
            agents.append(agent)
            envs.append(env)

        # When ready, look at original threaded_ale for save/load & summaries
        def summary_report(r): pass
        threaded_runner = ThreadedRunner(agents, envs)
        threaded_runner.run(episodes=300*WORKERS, summary_interval=1000, summary_report=summary_report)
        [e.close() for e in envs]


if __name__ == '__main__':
    main()
