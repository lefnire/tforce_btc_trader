import tensorflow as tf
import numpy as np

# Clipping ratio for gradients
CLIP_NORM = 40.0
CELL_UNITS = 512
D_LAYERS, L_LAYERS = 2, 2
FUNNEL = True
DROPOUT = .4


class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer, hyper):
        hyper_k, hyper_v = hyper.split(':')
        # use_dropout = hyper != 'dropout:off'
        # use_tanh = hyper == 'activation:tanh'
        # CELL_UNITS = int(hyper_v) if hyper_k == 'neurons' else 512

        with tf.variable_scope(scope):
            # Input
            he_init = tf.contrib.layers.variance_scaling_initializer()
            self.training = training = tf.placeholder_with_default(False, shape=(), name='training')
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)

            net = self.inputs if self.inputs.get_shape().ndims == 3 else tf.expand_dims(self.inputs, [1])
            net = tf.layers.dropout(net, rate=DROPOUT, training=training)

            # Recurrent network for temporal dependencies
            # Original: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
            # Multi-layer: https://medium.com/@erikhallstrm/using-the-tensorflow-multilayered-lstm-api-f6e7da7bbe40
            cell = [tf.nn.rnn_cell.LSTMCell(CELL_UNITS) for _ in range(L_LAYERS)]
            output_keep = tf.cond(self.training, lambda: 1 - DROPOUT, lambda: 1.)
            if DROPOUT:
                cell = [tf.nn.rnn_cell.DropoutWrapper(c, output_keep_prob=output_keep) for c in cell]
            multi = tf.nn.rnn_cell.MultiRNNCell(cell)

            self.rnn_prev = tf.placeholder(dtype=tf.float32, shape=[L_LAYERS, 2, None, CELL_UNITS], name="rnn_state")
            l = tf.unstack(self.rnn_prev, axis=0)
            rnn_tuple_state = tuple([
                tf.nn.rnn_cell.LSTMStateTuple(l[i][0], l[i][1])
                for i in range(L_LAYERS)
            ])

            output, self.rnn_next = tf.nn.dynamic_rnn(
                multi, net,
                initial_state=rnn_tuple_state,
                time_major=False)
            # self.rnn_next = (lstm_c[:1, :], lstm_h[:1, :])
            net = tf.reshape(output, [-1, CELL_UNITS])

            for i in range(2):
                size = CELL_UNITS//(i+1) if FUNNEL else CELL_UNITS
                net = tf.layers.dense(net, size, kernel_initializer=he_init, activation=tf.nn.elu)
                net = tf.layers.dropout(net, rate=DROPOUT, training=training)

            # Output layers for policy and value estimations
            self.policy = tf.squeeze(tf.layers.dense(net, 1, kernel_initializer=he_init))
            self.value = tf.layers.dense(net, 1, kernel_initializer=he_init)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                with tf.control_dependencies(update_ops):
                    """ https://goo.gl/ZU2Z9a
                    Value Loss: L = Σ(R - V(s))²
                    Policy Loss: L = -log(π(s)) * A(s) - β*H(π)
                    """
                    self.actions = tf.placeholder(shape=[None], dtype=tf.float32)
                    self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                    self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                    # Value loss function
                    self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))

                    # Softmax policy loss function
                    # self.responsible_outputs = tf.reduce_sum(self.policy * self.action, [1])
                    # self.policy_loss = -tf.reduce_sum(tf.log(tf.maximum(self.responsible_outputs, 1e-12)) * self.advantages)
                    self.policy_loss = -tf.reduce_sum(
                        tf.square(self.actions - self.policy)
                        * self.advantages
                    )

                    # Softmax entropy function
                    # self.entropy = - tf.reduce_sum(self.policy * tf.log(tf.maximum(self.policy, 1e-12)))
                    self.entropy = tf.constant(0.)

                    self.loss = self.value_loss + self.policy_loss - self.entropy * 0.01

                    # Get gradients from local network using local losses
                    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                    self.gradients = tf.gradients(self.loss, local_vars)
                    self.var_norms = tf.global_norm(local_vars)
                    grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                    # Apply local gradients to global network
                    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                    self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))