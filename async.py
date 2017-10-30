"""
Distributed tmux children will communicate with calling parent via RabbitMQ (Celery). Would be simpler to have a handle
to child threads w/ a threading solution (see threaded_ale.py and ThreadedRunner.py), but it seems threading isn't a
real async-update RL approach (https://gitter.im/reinforceio/TensorForce?at=59f840fbb20c6424296c2318), instead need to
use async (tmux / child processes)

1. run file:
celery -A async worker --loglevel=info

2. From other tab: (http://docs.celeryproject.org/en/latest/getting-started/first-steps-with-celery.html)
python -c 'from async import restart; restart.delay()'
tmux attach
"""

import inspect, os, sys, pdb

from celery import Celery
from six.moves import xrange, shlex_quote

celery_args = {
    'url': 'pyamqp://',  # amqp://localhost pyamqp://guest@localhost//
    'backend': 'rpc://'
}
app = Celery('async_mq', **celery_args)


@app.task
def restart():
    workers = 4
    shell = '/bin/bash'
    port = 12222
    sess_name = 'btc'

    # Kill first (so nobody's straggling w/ database writes)
    kill_cmds = [
        f"kill $( lsof -i:{port}-{port + workers} -t ) > /dev/null 2>&1",
        f"tmux kill-session -t {sess_name}",
    ]
    print("\n".join(kill_cmds))
    os.system("\n".join(kill_cmds))

    # Setup new experiment (re-import here, else pg connection is killed & get error)
    from hypersearch import generate_and_save_hypers
    generate_and_save_hypers(rand=True)

    # start up child processes
    target_script = os.path.abspath(inspect.stack()[0][1]).replace('.py', '_child.py')

    def wrap_cmd(session, name, cmd):
        if isinstance(cmd, list):
            cmd = ' '.join(shlex_quote(str(arg)) for arg in cmd)
        return f'tmux send-keys -t {session}:{name} {shlex_quote(cmd)} Enter'

    def build_cmd(ps, index):
        cmd_args = [
            # 'CUDA_VISIBLE_DEVICES=',
            sys.executable, target_script,
            '--num-workers', workers,
            '--task-index', index,
        ]
        if ps:
            cmd_args = ['CUDA_VISIBLE_DEVICES='] + cmd_args + ['--parameter-server']
        return cmd_args

    cmds = [f'tmux new-session -d -s {sess_name} -n ps']
    cmds.append(wrap_cmd(sess_name, 'ps', build_cmd(ps=True, index=0)))

    for i in xrange(workers):
        name = f'worker{i}'
        cmds.append(f'tmux new-window -t {sess_name} -n {name} -d {shell}')
        cmds.append(wrap_cmd(sess_name, name, build_cmd(ps=False, index=i)))

    print("\n".join(cmds))
    os.system("\n".join(cmds))

#
