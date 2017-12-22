import argparse
import numpy as np
import tensorflow as tf
from tensorforce.execution import Runner
from tensorforce.agents import agents as agents_dict
from rl_hsearch import HSearchEnv
from btc_env import BitcoinEnv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workers', type=int, default=4, help="Number of worker agents")
    parser.add_argument('-P', '--parameter-server', action='store_true', help="Parameter server")
    parser.add_argument('-i', '--task-index', type=int, default=0, help="Task index")
    parser.add_argument('--from-db', action="store_true", default=False, help="Load winner from DB or hard-coded guess?")
    args = parser.parse_args()

    is_main = args.task_index == 0

    ps_hosts = ['127.0.0.1:12222']
    worker_hosts = []
    port = 12223
    for _ in range(args.workers):
        worker_hosts.append(f'127.0.0.1:{port}')
        port += 1
    cluster = {'ps': ps_hosts, 'worker': worker_hosts}
    cluster_spec = tf.train.ClusterSpec(cluster)
    device = '/job:{}/task:{}'.format('ps' if args.parameter_server else 'worker', args.task_index)  # '/cpu:0'

    hs = HSearchEnv(gpu_split=.9/args.workers)
    flat, hydrated, network = hs.get_winner(from_db=args.from_db)
    env = BitcoinEnv(flat, name='ppo_agent')
    agent = agents_dict['ppo_agent'](
        states_spec=env.states,
        actions_spec=env.actions,
        network_spec=network,
        distributed_spec=dict(
            cluster_spec=cluster_spec,
            parameter_server=args.parameter_server,
            task_index=args.task_index,
            device=device,
            local_model=(not args.parameter_server),
        ),
        **hydrated
    )

    n_train, n_test = 30, 3
    runner = Runner(agent=agent, environment=env)
    if not is_main:
        runner.run()
    else:
        while True:
            print("Train")
            env.testing = False
            runner.run(episodes=n_train)  # train
            print("Test")
            for i in range(n_test):  # test
                env.testing = True
                next_state, terminal = env.reset(), False
                while not terminal:
                    next_state, terminal, reward = env.execute(agent.act(next_state, deterministic=True))


    rewards = env.episode_rewards
    reward = np.mean(rewards[-n_test:])
    print(flat, f"\nReward={reward}\n\n")
    runner.close()
