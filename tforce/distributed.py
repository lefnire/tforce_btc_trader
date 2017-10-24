import argparse, inspect, os, sys
import tensorflow as tf
from six.moves import xrange, shlex_quote
from tensorforce.execution import Runner

import agent_conf
import data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--experiment', type=int, default=0, help="Index of experiment to run")
    parser.add_argument('-w', '--num-workers', type=int, default=4, help="Number of worker agents")
    parser.add_argument('-M', '--mode', choices=['tmux', 'child'], default='tmux', help="Starter mode")
    parser.add_argument('-L', '--logdir', default='saves/async', help="Log directory")
    parser.add_argument('-K', '--kill', action='store_true', default=False, help="Kill runners")
    parser.add_argument('-C', '--is-child', action='store_true')
    parser.add_argument('-P', '--parameter-server', action='store_true', help="Parameter server")
    parser.add_argument('-i', '--task-index', type=int, default=0, help="Task index")

    args = parser.parse_args()

    session_name = 'btc'
    shell = '/bin/bash'

    kill_cmds = [
        "kill $( lsof -i:12222-{} -t ) > /dev/null 2>&1".format(12222 + args.num_workers),
        f"tmux kill-session -t {session_name}",
    ]
    if args.kill:
        os.system("\n".join(kill_cmds))
        return 0

    if not args.is_child:
        # start up child processes
        target_script = os.path.abspath(inspect.stack()[0][1])

        def wrap_cmd(session, name, cmd):
            if isinstance(cmd, list):
                cmd = ' '.join(shlex_quote(str(arg)) for arg in cmd)
            if args.mode == 'tmux':
                return 'tmux send-keys -t {}:{} {} Enter'.format(session, name, shlex_quote(cmd))
            elif args.mode == 'child':
                return '{} > {}/{}.{}.out 2>&1 & echo kill $! >> {}/kill.sh'.format(
                    cmd, args.logdir, session, name, args.logdir
                )

        def build_cmd(ps, index):
            cmd_args = [
                # 'CUDA_VISIBLE_DEVICES=',
                sys.executable, target_script,
                '--is-child',
                '--num-workers', args.num_workers,
                '--task-index', index,
                '--experiment', args.experiment
            ]
            if ps:
                cmd_args = ['CUDA_VISIBLE_DEVICES='] + cmd_args + ['--parameter-server']
            return cmd_args

        if args.mode == 'tmux':
            cmds = kill_cmds + ['tmux new-session -d -s {} -n ps'.format(session_name)]
        elif args.mode == 'child':
            cmds = ['mkdir -p {}'.format(args.logdir),
                    'rm -f {}/kill.sh'.format(args.logdir),
                    'echo "#/bin/bash" > {}/kill.sh'.format(args.logdir),
                    'chmod +x {}/kill.sh'.format(args.logdir)]

        cmds.append(wrap_cmd(session_name, 'ps', build_cmd(ps=True, index=0)))

        for i in xrange(args.num_workers):
            name = 'worker{}'.format(i)
            if args.mode == 'tmux':
                cmds.append('tmux new-window -t {} -n {} -d {}'.format(session_name, name, shell))
            cmds.append(wrap_cmd(session_name, name, build_cmd(ps=False, index=i)))

        # add one PS call
        # cmds.append('tmux new-window -t {} -n ps -d {}'.format(session_name, shell))

        print("\n".join(cmds))
        os.system("\n".join(cmds))
        return 0

    ps_hosts = ['127.0.0.1:{}'.format(12222)]
    worker_hosts = []
    port = 12223
    for _ in range(args.num_workers):
        worker_hosts.append('127.0.0.1:{}'.format(port))
        port += 1
    cluster = {'ps': ps_hosts, 'worker': worker_hosts}
    cluster_spec = tf.train.ClusterSpec(cluster)
    device = '/job:{}/task:{}'.format('ps' if args.parameter_server else 'worker', args.task_index)  # '/cpu:0'

    # environment = OpenAIGym(args.gym_id)
    c = agent_conf.confs[args.experiment]
    c['conf'].update(
        tf_session_config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.2)
        ),
        cluster_spec=cluster_spec,
        parameter_server=args.parameter_server,
        task_index=args.task_index,
        device=device,
        local_model=(not args.parameter_server),
    )
    conf = agent_conf.conf(c['conf'], c['name'], env_args=dict(is_main=args.task_index == 0))
    environment = conf['env']
    agent = conf['agent']

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )
    runner.run()


if __name__ == '__main__':
    main()
