import tensorflow as tf
from tensorforce.execution import Runner
import agent_conf, data
from agent_conf import AGENT_NAME

summary_writer = tf.summary.FileWriter(f"./a3c/saves/train/{AGENT_NAME}")
data.wipe_rows(AGENT_NAME)
conf = agent_conf.conf(
    tf_saver=False,
    tf_summary="logs_async",
    tf_summary_level=3,
)


def episode_finished(r):
    global summary_writer
    results = r.environment.episode_results
    total = float(results['cash'][-1] + results['values'][-1])
    reward = float(results['rewards'][-1])
    summary = tf.Summary()
    summary.value.add(tag='Perf/Total', simple_value=total)
    summary.value.add(tag='Perf/Reward', simple_value=reward)
    summary_writer.add_summary(summary, r.episode)
    summary_writer.flush()
    return True


runner = Runner(agent=conf['agent'], environment=conf['env'])
runner.run(episodes=agent_conf.EPISODES, episode_finished=episode_finished)