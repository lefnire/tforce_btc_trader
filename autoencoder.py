"""
https://github.com/ageron/handson-ml/blob/master/15_autoencoders.ipynb
"""

import tensorflow as tf
import numpy as np


class AutoEncoder:

    def __init__(self): pass

    def fit_transform_tied(self, matrix, n_dims=6):
        n_inputs = matrix.shape[1]
        n_hidden1 = int(np.mean([n_inputs, n_dims]))
        n_hidden2 = n_dims
        n_hidden3 = n_hidden1
        n_outputs = n_inputs

        learning_rate = 0.005
        l2_reg = 0.0005

        activation = tf.nn.elu
        regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        initializer = tf.contrib.layers.variance_scaling_initializer()

        X = tf.placeholder(tf.float32, shape=[None, n_inputs])

        weights1_init = initializer([n_inputs, n_hidden1])
        weights2_init = initializer([n_hidden1, n_hidden2])

        weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
        weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
        weights3 = tf.transpose(weights2, name="weights3")  # tied weights
        weights4 = tf.transpose(weights1, name="weights4")  # tied weights

        biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
        biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
        biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
        biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")

        hidden1 = activation(tf.matmul(X, weights1) + biases1)
        hidden2_raw = tf.matmul(hidden1, weights2) + biases2
        hidden2 = activation(hidden2_raw)
        hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
        outputs = tf.matmul(hidden3, weights4) + biases4

        reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
        reg_loss = regularizer(weights1) + regularizer(weights2)
        loss = reconstruction_loss + reg_loss

        optimizer = tf.contrib.opt.NadamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        n_epochs = 50
        batch_size = 150

        session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.9 / 2))
        with tf.Session(config=session_config) as sess:
            init.run()
            print("Auto-encoding time-series data for dimensionality reduction (see comment in btc_env.py)")
            for epoch in range(n_epochs):
                n_batches = matrix.shape[0] // batch_size
                for i in range(n_batches):
                    print("\r{}%".format(100 * i // n_batches), end="")
                    # sys.stdout.flush()
                    X_batch = matrix[i:i+batch_size]
                    sess.run(training_op, feed_dict={X: X_batch})
                loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
                print("\r{}".format(epoch), "Train MSE:", loss_train)
                if loss_train < .02: break
            ret = hidden2_raw.eval(feed_dict={X: matrix})
            sess.close()
            return ret
