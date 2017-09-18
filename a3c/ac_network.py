import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

# Clipping ratio for gradients
CLIP_NORM = 40.0
# Cell units
# CELL_UNITS = 256
DROPOUT = .2

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer, hyper):
        hyper_k, hyper_v = hyper.split(':')
        use_dropout = hyper != 'dropout:off'
        use_tanh = hyper == 'activation:tanh'
        CELL_UNITS = int(hyper_v) if hyper_k == 'neurons' else 256

        with tf.variable_scope(scope):
            he_init = tf.contrib.layers.variance_scaling_initializer()

            # Input
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.training = training = tf.placeholder_with_default(False, shape=())

            # net = tf.layers.batch_normalization(self.inputs, training=training, momentum=.9)
            net = tf.layers.dropout(self.inputs, rate=DROPOUT, training=training)
            # net = self.inputs

            # Layer 1 (Dense)
            net = tf.layers.dense(net, CELL_UNITS, kernel_initializer=he_init, use_bias=False)
            net = tf.layers.batch_normalization(net, training=training, momentum=.9)
            net = tf.nn.elu(net)
            net = tf.layers.dropout(net, rate=DROPOUT, training=training)

            # Recurrent network for temporal dependencies
            # Original: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
            # TODO Multi-layer: https://medium.com/@erikhallstrm/using-the-tensorflow-multilayered-lstm-api-f6e7da7bbe40
            lstm_cell = tf.nn.rnn_cell.LSTMCell(CELL_UNITS, state_is_tuple=True)
            keep_prob = tf.cond(training, lambda: 1-DROPOUT, lambda: 1.)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(net, [0])
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in,
                initial_state=state_in,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, CELL_UNITS])

            net = rnn_out
            net = tf.layers.batch_normalization(net, training=training, momentum=.9)

            # Output layers for policy and value estimations
            self.policy = slim.fully_connected(net, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(net, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                with tf.control_dependencies(update_ops):
                    self.actions = tf.placeholder(shape=[None, a_size], dtype=tf.float32)
                    self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                    self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                    self.responsible_outputs = tf.reduce_sum(self.policy * self.actions, [1])

                    # Value loss function
                    self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))

                    # Softmax policy loss function
                    self.policy_loss = -tf.reduce_sum(tf.log(tf.maximum(self.responsible_outputs, 1e-12)) * self.advantages)

                    # Softmax entropy function
                    self.entropy = - tf.reduce_sum(self.policy * tf.log(tf.maximum(self.policy, 1e-12)))

                    self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                    # Get gradients from local network using local losses
                    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                    self.gradients = tf.gradients(self.loss, local_vars)
                    self.var_norms = tf.global_norm(local_vars)
                    grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                    # Apply local gradients to global network
                    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                    self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))