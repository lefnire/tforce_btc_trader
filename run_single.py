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
        runner.run(episodes=500, episode_finished=episode_finished)
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


def network(arch, n=256, d=.2):
    net = [
        dict(type='dropout', size=n, dropout=d)
    ]
    for layer in arch:
        if layer == 'L':
            net.append(dict(type='lstm', size=n, dropout=d))
        elif layer == 'D':
            net.append(dict(type='dense2', size=n, dropout=d))
    return dict(network=net)

confs = [
    dict(k='main', v=[dict(k='-', v=dict())]),
    dict(k='network', v=[
        dict(k=f'{arch}.{neur}', v=network(arch, neur))
        for neur in [64, 128, 256, 512]
        for arch in [
            'L',
            'DL', 'LD', 'LL'
            'DLD', 'DLL', 'LDD', 'LLD', 'LLL',
            'DLLD', 'LLDD', 'LLLD',
            'DLLLD'
        ] # Good were DLD, DLLD, LLDD(winner)

    ]),
    dict(k='gae_rewards', v=[
        dict(k='False', v=dict(gae_rewards=False)),  # unclear
    ]),
    dict(k='random_sampling', v=[
        dict(k='False', v=dict(random_sampling=False)),  # unclear
    ]),
    dict(k='normalize_rewards', v=[
        dict(k='True', v=dict(normalize_rewards=True)),  # unclear
    ]),
    dict(k='keep_last', v=[
        dict(k='False', v=dict(keep_last=False)),  # unclear
    ]),
    dict(k='learning_rate', v=[
        dict(k='.001', v=dict(learning_rate=.001)),
        dict(k='.0001', v=dict(learning_rate=.0001)),  # unclear
    ]),
    dict(k='discount', v=[
        dict(k='.95', v=dict(discount=.95)),  # winner? :o
        dict(k='.99', v=dict(discount=.99)),  # unclear
    ]),
    dict(k='baseline', v=[  # loser
        dict(k='main', v=baseline()), # winner (of baseline)
        dict(k='2x64', v=baseline(sizes=[64, 64])),
        dict(k='2x256', v=baseline(sizes=[256, 256])),
        dict(k='epochs10', v=baseline(epochs=10)),
        dict(k='epochs100', v=baseline(epochs=100)),
        dict(k='update_batch_size64', v=baseline(update_batch_size=64)),
        dict(k='update_batch_size1024', v=baseline(update_batch_size=1024)),
        dict(k='learning_rate.001', v=baseline(learning_rate=.001)),
    ]),
    dict(k='epochs', v=[
        dict(k='1', v=dict(epochs=1)),  # unclear
        dict(k='40', v=dict(epochs=40)),  # unclear (slow, but maybe catches up strong?)
        dict(k='400', v=dict(epochs=400)),  # loser & prohibitive
    ]),
    dict(k='optimizer_batch_size', v=[
        dict(k='128', v=dict(optimizer_batch_size=128)),  # unclear
        dict(k='1024', v=dict(optimizer_batch_size=1024)), # winner?
        dict(k='4096', v=dict(optimizer_batch_size=4096)),  # 2048=winner (seems higher better)
    ]),
    dict(k='batch_size', v=[
        dict(k='128', v=dict(batch_size=1028)),  # unclear
        dict(k='1024', v=dict(batch_size=2048)),  # unclear
        dict(k='4096', v=dict(batch_size=4096)),  # loser
    ]),
]

q = Queue(0)
num_threads = 5
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