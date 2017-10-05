import tensorflow as tf
from tensorforce.execution import Runner
import agent_conf, data
from experiments import confs, AGENT_TYPE

START = 1
for conf in confs[START::4]:
    conf['conf'].update(
        tf_session_config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.21)
        ),
    )
    conf = agent_conf.conf(
        conf['conf'],
        agent_type=AGENT_TYPE,
        mods=conf['name'],
        # env_args=dict(log_states=True, is_main=True, log_results=False),
        env_args=dict(is_main=True, log_results=True, scale_features=False, indicators=False)
    )
    print(conf['agent_name'])
    runner = Runner(agent=conf['agent'], environment=conf['env'])
    is_main = 'main:-' in conf['agent_name']
    runner.run(episodes=600 if is_main else 300)
    print(conf['agent_name'], 'done')