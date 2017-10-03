# ================================================
# Modified from the work of Arthur Juliani:
#       Simple Reinforcement Learning with Tensorflow Part 8: Asynchronus Advantage Actor-Critic (A3C)
#       https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb
#
#       Implementation of Asynchronous Methods for Deep Reinforcement Learning
#       Algorithm details can be found here:
#           https://arxiv.org/pdf/1602.01783.pdf
#
# Modified to work with OpenAI Gym environments (currently working with cartpole)
# Author: Liam Pettigrew
# =================================================

import os, threading, multiprocessing, shutil, time
import numpy as np
import tensorflow as tf

import data
from btc_env.btc_env import BitcoinEnvTforce
from a3c.worker import Worker
from a3c.ac_network import AC_Network

# ===========================
#   Gym Utility Parameters
# ===========================
# Gym environment

# Directory for storing gym results
MONITOR_DIR = './saves/monitor/'

# ==========================
#   Training Parameters
# ==========================
RANDOM_SEED = 1234
# Load previously trained model
LOAD_MODEL = False
# Test and visualise a trained model
TEST_MODEL = False
# Directory for storing session model
MODEL_DIR = './saves/model/'
# Discount rate for advantage estimation and reward discounting
GAMMA = 0.99



def main(_):
    global master_network
    global global_episodes

    # Set train/test mode for database for all workers
    data.set_mode('TEST' if TEST_MODEL else 'TRAIN')

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # try: indicators, reward_factor, dense last (2L), peepholes
    # next: layers, funnel, dropout, tanh
    # short-term: -6, high-batch, high-arch; long-tail: high-batch, -7 (try -8?)
    h_lr_batch = []
    defaults = dict(funnel=False, dropout=.5, lr=1e-8, batch=2048, epochs=25, mult=4, l_units=512, d_units=256, l_layers=3, d_layers=2)
    for batch in [256, 512, 1024]:
        for lr in [1e-6, 1e-8, 1e-10]:
            if batch == defaults['batch'] and lr == defaults['lr']: continue
            h = defaults.copy()
            h.update(lr=lr, batch=batch)
            h_lr_batch.append(h)
    h_mult = []
    for mult in [1, 2, 3, 4, 5]:
        if mult == defaults['mult']: continue
        h = defaults.copy()
        h.update(
            mult=mult,
            l_units={1:64, 2:128, 3:256, 4:512, 5:512}[mult],
            d_units={1:64, 2:128, 3:256, 4:256, 5:256}[mult],
            l_layers={1:2, 2:2, 3:3, 4:3, 5:4}[mult],
            d_layers={1:1, 2:1, 3:2, 4:2, 5:3}[mult]
        )
        h_mult.append(h)
    h_epochs = []
    for epoch in [5, 20, 50]:
        if epoch == defaults['epochs']: continue
        h = defaults.copy()
        h.update(epochs=epoch)
        h_epochs.append(h)
    arr = h_lr_batch + h_mult + h_epochs
    # for i, hyper in enumerate(arr):
    h = defaults.copy()
    for i, hyper in enumerate([h]):
        agent_name = f"A3CAgent|lr{hyper['lr']}bs{hyper['batch']}x{hyper['mult']}ep{hyper['epochs']}"
        tf.reset_default_graph()

        btc_env = BitcoinEnvTforce(agent_name=agent_name, is_main=False)
        STATE_DIM = btc_env.states['shape'][0]
        ACTION_DIM = btc_env.actions['shape'][0]

        # with tf.device("/cpu:0"):
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.contrib.opt.NadamOptimizer(learning_rate=hyper['lr'])
        master_network = AC_Network(STATE_DIM, ACTION_DIM, 'global', None, hyper)  # Generate global network
        num_workers = multiprocessing.cpu_count()  # Set workers to number of available CPU threads

        # For testing and visualisation we only need one worker
        if TEST_MODEL:
            num_workers = 1

        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(i, STATE_DIM, ACTION_DIM, trainer, MODEL_DIR, global_episodes,
                                  RANDOM_SEED, TEST_MODEL, hyper, agent_name))
        saver = tf.train.Saver(max_to_keep=5)

        # Gym monitor
        if not TEST_MODEL:
            env = workers[0].get_env()
            #env.monitor.start(MONITOR_DIR, video_callable=False, force=True)
        # END with tf.device("/cpu:0"):

        tf_session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.43))
        with tf.Session(config=tf_session_config) as sess:
        # with tf.Session() as sess:
            coord = tf.train.Coordinator()
            if LOAD_MODEL or TEST_MODEL:
                print('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            if TEST_MODEL:
                env = workers[0].get_env()
                #env.monitor.start(MONITOR_DIR, force=True)
                workers[0].work(GAMMA, sess, coord, saver)
            else:
                # This is where the asynchronous magic happens.
                # Start the "work" process for each worker in a separate thread.
                worker_threads = []
                for worker in workers:
                    worker_work = lambda: worker.work(GAMMA, sess, coord, saver)
                    t = threading.Thread(target=(worker_work))
                    t.start()
                    time.sleep(.5)
                    worker_threads.append(t)
                coord.join(worker_threads)

if __name__ == '__main__':
    tf.app.run()