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

from itertools import product  # TODO https://stackoverflow.com/questions/44802939/hyperparameter-tuning-of-tensorflow-model
import os, threading, multiprocessing, shutil, time
import numpy as np
import tensorflow as tf
from box import Box

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
MODEL_DIR = 'saves/model/'
# Discount rate for advantage estimation and reward discounting
GAMMA = 0.99



def main(_):
    global master_network
    global global_episodes

    # Set train/test mode for database for all workers
    data.set_mode('TEST' if TEST_MODEL else 'TRAIN')

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # TODO: indicators, reward_factor, peepholes, dropout, activations
    defaults = dict(dropout=.4, lr=1e-8, batch=2048, epochs=25, net=0, scale=True)

    h_mult = []
    nets = [
        # [0, [64] * 2, [64]],
        # [64, [64] * 2, [64]],

        # [0, [128] * 2, [128]],
        # [128, [128] * 2, [128]],
        # [128, [128] * 2, [128] * 2],

        [0, [256] * 2, [192]],  # 3,1=4  (reward,value-loss)  TODO experiment w/ more 0-starts + funnel, exp w/ above
        [192, [256] * 2, [192]],  # 5,2=7
        [192, [256] * 2, [192, 128]],  # 9,6=15
        [192, [256] * 3, [192, 128]],  # 4,3=7
        [256, [256] * 3, [256] * 2],  # 7(cutoff),4=11

        [0, [512] * 2, [256, 192]],  # 1,5=6
        [256, [512] * 2, [256, 192]],  # 2,10=12
        [256, [512] * 3, [256, 192]],  # 8,7=15
        [512, [512] * 2, [512] * 2],  # 10,9=19
        [512, [512] * 3, [512] * 2],  # 6,8=14
    ]
    for i, net in enumerate(nets):
        arch = str(net).replace(' ', '')
        if i == defaults['net']:
            defaults.update(arch=arch, net=net)
        else:
            h = defaults.copy()
            h.update(arch=arch, net=net)
            h_mult.append(h)

    # TODO these 3 should be random or grid -searched
    h_batch = []
    for batch in [256, 512, 1024, 4096]:
        if batch == defaults['batch']: continue
        h = defaults.copy()
        h.update(batch=batch)
        h_batch.append(h)
    h_lr = []
    for lr in [1e-6, 1e-8, 1e-10]:
        if lr == defaults['lr']: continue
        h = defaults.copy()
        h.update(lr=lr)
        h_lr.append(h)
    h_epochs = []
    for epoch in [5, 20, 50]:
        if epoch == defaults['epochs']: continue
        h = defaults.copy()
        h.update(epochs=epoch)
        h_epochs.append(h)

    # arr = [defaults] + h_batch + h_lr + h_mult + h_epochs
    arr = [defaults] + h_mult
    for i, hyper in enumerate(arr):
        hyper = Box(hyper)
        agent_name = f"A3C_Le{hyper.lr}_Ba{hyper.batch}_Ar{hyper.arch}_Ep{hyper.epochs}_Sc{hyper.scale}"
        tf.reset_default_graph()

        btc_env = BitcoinEnvTforce(agent_name=agent_name, is_main=False)
        STATE_DIM = btc_env.states['shape'][0]
        ACTION_DIM = btc_env.actions['shape'][0]

        # with tf.device("/cpu:0"):
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.contrib.opt.NadamOptimizer(learning_rate=hyper.lr)
        master_network = AC_Network(STATE_DIM, ACTION_DIM, 'global', None, hyper, summary_level=0)  # Generate global network
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

        tf_session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.41))
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
                    time.sleep(1)
                    worker_threads.append(t)
                coord.join(worker_threads)

if __name__ == '__main__':
    tf.app.run()