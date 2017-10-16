import tensorflow as tf
from tensorforce.execution import Runner
import agent_conf, data
from agent_conf import confs, AGENT_TYPE

for conf in confs[1:]:
    conf['conf'].update(
        tf_session_config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.41)
        ),
        # log_level="info",
        # tf_saver="saves",
        # tf_summary=f"saves/boards/{AGENT_TYPE}|{conf['name']}",
        # tf_summary_level=3,
        # tf_summary_interval=50,
    )
    conf = agent_conf.conf(
        conf['conf'],
        name=conf['name'],
        env_args=dict(is_main=True)
    )
    print(conf['name'])
    runner = Runner(agent=conf['agent'], environment=conf['env'])
    runner.run(episodes=350)
    print(conf['name'], 'done')