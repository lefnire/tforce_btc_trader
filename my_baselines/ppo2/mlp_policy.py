from baselines.common.mpi_running_mean_std import RunningMeanStd
import numpy as np
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

NEURONS, L_LAYERS, D_LAYERS, DROPOUT = 512, 2, 2, .4
NAME = f'N{NEURONS}L{L_LAYERS}D{D_LAYERS}drop{DROPOUT}' + 'tanh.l1'
# tanh, selu, dropout, scale  - try next: 256, 2L, w/o ob_rms, tforce w/ min/max=5, dense1 (relu + init)
# elu, elu+l1 -> nan


class MlpPolicy(object):
    recurrent = True
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def lstm(self, x, size, name):
        inputs = x if len(x.get_shape()) > 2 else tf.expand_dims(x, [1])

        # Construct network/cells
        cell = [tf.nn.rnn_cell.LSTMCell(size) for _ in range(L_LAYERS)]
        output_keep = tf.cond(self.training, lambda: 1-DROPOUT, lambda: 1.)
        if DROPOUT:
            cell = [tf.nn.rnn_cell.DropoutWrapper(c, output_keep_prob=output_keep) for c in cell]
        multi = tf.nn.rnn_cell.MultiRNNCell(cell)

        # Setup internal state-management
        # https: // stackoverflow.com / questions / 39112622 / how - do - i - set - tensorflow - rnn - state - when - state - is -tuple - true
        # https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
        rnn_prev = U.get_placeholder(name="rnn_state", dtype=tf.float32, shape=[L_LAYERS, 2, None, size])
        l = tf.unstack(rnn_prev, axis=0)
        rnn_tuple_state = tuple([
            tf.nn.rnn_cell.LSTMStateTuple(l[i][0], l[i][1])
            for i in range(L_LAYERS)
        ])
        self.rnn_init = np.zeros([L_LAYERS, 2, size], np.float32)

        output, rnn_next = tf.nn.dynamic_rnn(
            multi, inputs,
            initial_state=rnn_tuple_state,
            time_major=False)
        # self.rnn_next = (lstm_c[:1, :], lstm_h[:1, :])
        return tf.reshape(output, [-1, size]), rnn_prev, rnn_next

    def _init(self, ob_space, ac_space, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)
        self.training = tf.placeholder_with_default(False, shape=(), name='training')

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        if DROPOUT:
            obz = tf.layers.dropout(obz, rate=DROPOUT, training=self.training)

        obz = tf.nn.tanh(U.dense(obz, 256, "first_dense", weight_init=U.normc_initializer(1.0)))

        # Value Function
        last_out = obz
        last_out, rnn_prev, rnn_next = self.lstm(last_out, NEURONS, "vffc_lstm")
        rnn_out = last_out
        for i in range(D_LAYERS):
            last_out = tf.nn.tanh(U.dense(last_out, NEURONS//(i+2), "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            if DROPOUT:
                last_out = tf.layers.dropout(last_out, rate=DROPOUT, training=self.training)
        self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]

        # Policy Function
        last_out = rnn_out
        for i in range(D_LAYERS):
            last_out = tf.nn.tanh(U.dense(last_out, NEURONS//(i+2), "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            if DROPOUT:
                last_out = tf.layers.dropout(last_out, rate=DROPOUT, training=self.training)
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = U.dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function(
            [stochastic, ob, rnn_prev],
            [ac, self.vpred, rnn_next],
        )

    def act(self, stochastic, ob, rnn_state):
        ac1, vpred1, rnn_next = self._act(stochastic, ob[None], np.reshape(rnn_state, [L_LAYERS,2,1,NEURONS]))
        return ac1[0], vpred1[0], rnn_next
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

