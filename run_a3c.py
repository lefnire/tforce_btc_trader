from __future__ import absolute_import, division, print_function
import argparse, inspect, logging, os, sys, time, math
import tensorflow as tf
from six.moves import xrange, shlex_quote
import numpy as np
from tensorforce.execution import Runner
from tensorforce.util import log_levels

import agent_conf
import data
import experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='btc', help="Name of your tmux session")
    parser.add_argument('-e', '--experiment', type=int, default=0, help="Index of experiment to run")
    parser.add_argument('-w', '--num-workers', type=int, default=7, help="Number of worker agents")
    parser.add_argument('-m', '--monitor', help="Save results to this file")
    parser.add_argument('-M', '--mode', choices=['tmux', 'child'], default='tmux', help="Starter mode")
    parser.add_argument('-L', '--logdir', default='logs_async', help="Log directory")
    parser.add_argument('-C', '--is-child', action='store_true')
    parser.add_argument('-i', '--task-index', type=int, default=0, help="Task index")
    parser.add_argument('-K', '--kill', action='store_true', default=False, help="Kill runners")

    args = parser.parse_args()

    session_name = args.name
    shell = '/bin/bash'

    kill_cmds = [
        "kill $( lsof -i:12222-{} -t ) > /dev/null 2>&1".format(12222 + args.num_workers),
        "tmux kill-session -t {}".format(session_name),
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

        def build_cmd(index):
            cmd_args = [
                sys.executable, target_script,
                '--is-child',
                '--num-workers', args.num_workers,
                '--task-index', index,
                '--experiment', args.experiment
            ]
            if index == -1:
                cmd_args = ['CUDA_VISIBLE_DEVICES='] + cmd_args
            return cmd_args

        if args.mode == 'tmux':
            cmds = kill_cmds + ['tmux new-session -d -s {} -n ps'.format(session_name)]
        elif args.mode == 'child':
            cmds = ['mkdir -p {}'.format(args.logdir),
                    'rm -f {}/kill.sh'.format(args.logdir),
                    'echo "#/bin/bash" > {}/kill.sh'.format(args.logdir),
                    'chmod +x {}/kill.sh'.format(args.logdir)]
        cmds.append(wrap_cmd(session_name, 'ps', build_cmd(-1)))

        for i in xrange(args.num_workers):
            name = 'w_{}'.format(i)
            if args.mode == 'tmux':
                cmds.append('tmux new-window -t {} -n {} -d {}'.format(session_name, name, shell))
            cmds.append(wrap_cmd(session_name, name, build_cmd(i)))

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
    device = ('/job:ps' if args.task_index == -1 else '/job:worker/task:{}'.format(args.task_index))

    c = experiments.confs[args.experiment]
    c['conf'].update(
        distributed=True,
        cluster_spec=cluster_spec,
        global_model=(args.task_index == -1),
        device=device,
        log_level="info",
        tf_saver=False,
        tf_summary=None,
        tf_summary_level=0,
        preprocessing=None,
    )
    conf = agent_conf.conf(c['conf'], 'PPOAgent', c['name'])

    logger = logging.getLogger(__name__)
    logger.setLevel(log_levels[conf['conf'].log_level])
    logger.info("Starting distributed agent for OpenAI Gym '{gym_id}'".format(gym_id='btc'))
    logger.info("Config:")
    logger.info(conf['conf'])

    if not args.is_child:
        data.wipe_rows(conf['agent_name'])
    runner = Runner(
        agent=conf['agent'],
        environment=conf['env'],
        repeat_actions=1,
        cluster_spec=cluster_spec,
        task_index=args.task_index
    )

    summary_writer = tf.summary.FileWriter(f"./a3c/saves/train/{conf['agent_name']}")
    def episode_finished(r):
        if args.task_index == 0:
            results = r.environment.episode_results
            total = float(results['cash'][-1] + results['values'][-1])
            reward = float(results['rewards'][-1])
            summary = tf.Summary()
            summary.value.add(tag='Perf/Total', simple_value=total)
            summary.value.add(tag='Perf/Reward', simple_value=reward)
            summary_writer.add_summary(summary, r.episode)
            summary_writer.flush()
        return True

    runner.run(1000, agent_conf.STEPS, episode_finished=episode_finished, num_workers=args.num_workers)


if __name__ == '__main__':
    main()
