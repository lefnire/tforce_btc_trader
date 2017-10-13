import tensorflow as tf
from tensorforce.execution import Runner
import agent_conf, data
from experiments import confs, AGENT_TYPE

for conf in confs[3:]:
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
        agent_type=AGENT_TYPE,
        mods=conf['name'],
        # env_args=dict(log_states=True, is_main=True),
        env_args=dict(is_main=True, scale_features=False, indicators=False)
    )
    print(conf['agent_name'])
    runner = Runner(agent=conf['agent'], environment=conf['env'])
    runner.run(episodes=350)
    print(conf['agent_name'], 'done')