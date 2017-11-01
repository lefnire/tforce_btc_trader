import argparse
import tensorflow as tf
from tensorforce.execution import Runner
from tensorforce import Configuration, agents as agents_dict

from hypersearch import get_hypers, run_finished


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='random', help="'random' or 'winner', determines whether searching or running")
    parser.add_argument('-w', '--num-workers', type=int, default=4, help="Number of worker agents")
    parser.add_argument('-P', '--parameter-server', action='store_true', help="Parameter server")
    parser.add_argument('-i', '--task-index', type=int, default=0, help="Task index")
    args = parser.parse_args()

    is_main = args.task_index == 0

    ps_hosts = ['127.0.0.1:12222']
    worker_hosts = []
    port = 12223
    for _ in range(args.num_workers):
        worker_hosts.append(f'127.0.0.1:{port}')
        port += 1
    cluster = {'ps': ps_hosts, 'worker': worker_hosts}
    cluster_spec = tf.train.ClusterSpec(cluster)
    device = '/job:{}/task:{}'.format('ps' if args.parameter_server else 'worker', args.task_index)  # '/cpu:0'

    flat, hydrated, network, env = get_hypers(rand=args.mode == 'random', from_db=True)
    hydrated.update(
        tf_session_config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.75/args.num_workers)
        ),
        cluster_spec=cluster_spec,
        parameter_server=args.parameter_server,
        task_index=args.task_index,
        device=device,
        local_model=(not args.parameter_server),
    )
    # env_args = dict(is_main=args.task_index == 0)
    print('--- Network ---')
    print(network)
    agent = agents_dict.agents['ppo_agent'](
        states_spec=env.states,
        actions_spec=env.actions,
        network_spec=network,
        config=Configuration(**hydrated)
    )

    runner = Runner(
        agent=agent,
        environment=env,
        repeat_actions=1
    )
    runner.run(
        episodes=300 if args.mode == 'random' else None,
    )
    run_finished(env, flat)

    # First to it kills the others
    from async import restart
    restart.delay()