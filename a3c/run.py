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
from btc_env import BitcoinEnv
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
# Learning rate
LEARNING_RATE = 0.001
# Discount rate for advantage estimation and reward discounting
GAMMA = 0.99



def main(_):
    global master_network
    global global_episodes

    # Set train/test mode for database for all workers
    data.set_mode('TEST' if TEST_MODEL else 'TRAIN')

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # definite winners: N256, L1-2, batch150, elu, 2L (batch normalization is crap-shoot)
    # try: indicators, reward_factor, dense last (2L), peepholes
    # for hyper in ['neurons:256', 'neurons:512', 'layers:2', 'layers:3', 'layers:4', 'activation:tanh', 'activation:elu', 'dropout:off', 'dropout:on']:
    for hyper in ['continuous:200-5k']:
        agent_name = 'A3CAgent|' + hyper
        data.wipe_rows(agent_name)
        tf.reset_default_graph()

        btc_env = BitcoinEnv(agent_name=agent_name, agent_type=agent_name.split('|')[0], indicators=hyper == 'indicators:on')
        STATE_DIM = btc_env.num_features()
        # ACTION_DIM = btc_env.actions['num_actions']
        ACTION_DIM = 1

        # with tf.device("/cpu:0"):
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.contrib.opt.NadamOptimizer(learning_rate=LEARNING_RATE)
        master_network = AC_Network(STATE_DIM, ACTION_DIM, 'global', None, hyper)  # Generate global network
        num_workers = multiprocessing.cpu_count()  # Set workers to number of available CPU threads

        # For testing and visualisation we only need one worker
        if TEST_MODEL:
            num_workers = 1

        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(i, STATE_DIM, ACTION_DIM, trainer, MODEL_DIR, global_episodes,
                                  RANDOM_SEED, TEST_MODEL, hyper))
        saver = tf.train.Saver(max_to_keep=5)

        # Gym monitor
        if not TEST_MODEL:
            env = workers[0].get_env()
            #env.monitor.start(MONITOR_DIR, video_callable=False, force=True)
        # END with tf.device("/cpu:0"):

        # tf_session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.4))
        # with tf.Session(config=tf_session_config) as sess:
        with tf.Session() as sess:
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