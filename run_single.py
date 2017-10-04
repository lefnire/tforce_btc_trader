import time
import tensorflow as tf
import numpy as np
from tensorforce.execution import Runner
import agent_conf, data
from experiments import confs

START = 0  # 3
for conf in confs[START::4]:
    conf['conf'].update(
        tf_session_config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.21)
        ),
    )
    conf = agent_conf.conf(
        conf['conf'],
        agent_type='NAFAgent',
        mods=conf['name'],
        # env_args=dict(log_states=True, is_main=True, log_results=False),
        env_args=dict(is_main=True, log_results=True, scale_features=False)
    )
    print(conf['agent_name'])
    runner = Runner(agent=conf['agent'], environment=conf['env'])
    is_main = conf['agent_name'] == 'NAFAgent|main:-'
    if is_main: print('is_main=True')
    runner.run(episodes=600 if is_main else 300)
    print(conf['agent_name'], 'done')