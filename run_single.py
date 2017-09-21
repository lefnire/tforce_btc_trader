import tensorflow as tf
from tensorforce.execution import Runner
import agent_conf, data

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
            summary = tf.Summary()
            summary.value.add(tag='Perf/Total', simple_value=total)
            summary.value.add(tag='Perf/Reward', simple_value=reward)
            summary_writer.add_summary(summary, r.episode)
            summary_writer.flush()
            return True

        print(conf['agent_name'])
        runner = Runner(agent=conf['agent'], environment=conf['env'])
        runner.run(episodes=900, episode_finished=episode_finished)
        print(conf['agent_name'], 'done')
        q.task_done()


def baseline(**kwargs):
    b = dict(baseline=dict(
        type="mlp",
        sizes=[128, 128],
        epochs=5,
        update_batch_size=128,
        learning_rate=.01
    ))
    b['baseline'].update(**kwargs)
    return b


def network(l=2, n=256, d=.2):
    if l == 1:
        return dict(network=[
            dict(type='dropout', size=n, dropout=d),
            dict(type='lstm', size=n, dropout=d),
        ])
    if l == 2:
        return dict(network=[
            dict(type='dropout', size=n, dropout=d),
            dict(type='dense2', size=n, dropout=d),
            dict(type='lstm', size=n, dropout=d),
        ])
    if l == 3:
        return dict(network=[
            dict(type='dropout', size=n, dropout=d),
            dict(type='dense2', size=n, dropout=d),
            dict(type='lstm', size=n, dropout=d),
            dict(type='dense2', size=n, dropout=d),
        ])
    if l == 4:
        return dict(network=[
            dict(type='dropout', size=n, dropout=d),
            dict(type='dense2', size=n, dropout=d),
            dict(type='lstm', size=n, dropout=d),
            dict(type='lstm', size=n, dropout=d),
            dict(type='dense2', size=n, dropout=d),
        ])

confs = [
    dict(k='main', v=[dict(k='-', v=dict())]),
    dict(k='network', v=[
        dict(k='1L', v=network(1)),
        dict(k='3L', v=network(3)),
        dict(k='4L', v=network(4)),
        dict(k='2L128N', v=network(2, n=128)),
        dict(k='3L128N', v=network(3, n=128)),
        dict(k='2L512N', v=network(2, n=512)),
        dict(k='3L512N', v=network(3, n=512)),
        dict(k='.5D', v=network(2, d=.5)),
    ]),
    dict(k='baseline', v=[
        dict(k='main', v=baseline()),
        dict(k='2x64', v=baseline(sizes=[64, 64])),
        dict(k='2x256', v=baseline(sizes=[256, 256])),
        dict(k='epochs10', v=baseline(epochs=10)),
        dict(k='epochs100', v=baseline(epochs=100)),
        dict(k='update_batch_size64', v=baseline(update_batch_size=64)),
        dict(k='update_batch_size1024', v=baseline(update_batch_size=1024)),
        dict(k='learning_rate.001', v=baseline(learning_rate=.001)),
    ]),
    dict(k='epochs', v=[
        dict(k='4', v=dict(discount=4)),
        dict(k='40', v=dict(discount=40)),
        dict(k='400', v=dict(discount=400)),
        dict(k='4000', v=dict(discount=4000))
    ]),
    dict(k='batch_size', v=[
        dict(k='1024', v=dict(batch_size=1028)),
        dict(k='2048', v=dict(batch_size=2048)),
        dict(k='4096', v=dict(batch_size=4096)),
    ]),
    dict(k='optimizer_batch_size', v=[
        dict(k='64', v=dict(optimizer_batch_size=64)),
        dict(k='128', v=dict(optimizer_batch_size=128)),
        dict(k='512', v=dict(optimizer_batch_size=512)),
        dict(k='1024', v=dict(optimizer_batch_size=1024)),
    ]),


    dict(k='learning_rate', v=[
        dict(k='.01', v=dict(learning_rate=.01)),
        dict(k='.0001', v=dict(learning_rate=.0001)),
    ]),
    dict(k='discount', v=[
        dict(k='.95', v=dict(discount=.95)),
        dict(k='.97', v=dict(discount=.97)),
    ]),
    dict(k='gae_rewards', v=[
        # dict(k='True', v=dict(gae_rewards=True)),
        dict(k='False', v=dict(gae_rewards=False)),
    ]),
    dict(k='random_sampling', v=[
        # dict(k='True', v=dict(random_sampling=True)),
        dict(k='False', v=dict(random_sampling=False)),
    ]),
    dict(k='normalize_rewards', v=[
        dict(k='True', v=dict(normalize_rewards=True)),
        # dict(k='False', v=dict(normalize_rewards=False)),
    ]),
    dict(k='keep_last', v=[
        # dict(k='True', v=dict(keep_last=True)),
        dict(k='False', v=dict(keep_last=False)),
    ]),
]

q = Queue(0)
num_threads = 10
for i in range(num_threads):
  worker = Thread(target=run_experiment, args=(q,))
  worker.setDaemon(True)
  worker.start()

for c in confs:
    for permu in c['v']:
        q.put(dict(
            name=c['k'] + ':' + permu['k'],
            conf=permu['v']
        ))

q.join()













