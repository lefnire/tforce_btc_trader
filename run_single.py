import tensorflow as tf
import numpy as np
from tensorforce.execution import Runner
import agent_conf, data
from experiments import confs

from queue import Queue
from threading import Thread


def run_experiment(q):
    while True:
        conf = q.get()
        conf = agent_conf.conf(
            conf['conf'],
            agent_type='PPOAgent',
            mods=conf['name'],
        )
        agent_name = conf['agent_name']
        summary_writer = tf.summary.FileWriter(f"./a3c/saves/train/{agent_name}")
        data.wipe_rows(agent_name)

        def episode_finished(r):
            results = r.environment.episode_results
            total = float(results['cash'][-1] + results['values'][-1])
            reward = float(results['rewards'][-1])
            reward_avg = np.mean(results['rewards'][-100:])
            summary = tf.Summary()
            if r.episode > 15:
                # Tensorboard smoothing is affected by all data points, but early points are random
                summary.value.add(tag='Perf/Total', simple_value=total)
                summary.value.add(tag='Perf/Reward', simple_value=reward)
            summary.value.add(tag='Perf/Reward_AVG', simple_value=reward_avg)
            summary_writer.add_summary(summary, r.episode)
            summary_writer.flush()
            return True

        print(conf['agent_name'])
        runner = Runner(agent=conf['agent'], environment=conf['env'])
        runner.run(episodes=400, episode_finished=episode_finished)
        print(conf['agent_name'], 'done')
        q.task_done()


q = Queue(0)
num_threads = 15
for i in range(num_threads):
  worker = Thread(target=run_experiment, args=(q,))
  worker.setDaemon(True)
  worker.start()

for c in confs: q.put(c)
q.join()