from baselines.common.mpi_running_mean_std import RunningMeanStd
import numpy as np
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

NEURONS, LAYERS, DROPOUT = 512, 2, .1
MODE = 'funnel'  # tanh, selu, dropout, scale  - try next: 256, 2L, w/o ob_rms, tforce w/ min/max=5, dense1 (relu + init)


class MlpPolicy(object):
    recurrent = True
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def lstm(self, x, size, name):
        inputs = x if len(x.get_shape()) > 2 else tf.expand_dims(x, [1])

        # Construct network/cells
        cell = [tf.nn.rnn_cell.LSTMCell(size) for _ in range(LAYERS)]
        cell = [tf.nn.rnn_cell.DropoutWrapper(c, output_keep_prob=1-DROPOUT) for c in cell]
        multi = tf.nn.rnn_cell.MultiRNNCell(cell)

        # Setup internal state-management
        # https: // stackoverflow.com / questions / 39112622 / how - do - i - set - tensorflow - rnn - state - when - state - is -tuple - true
        # https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
        self.rnn_prev = U.get_placeholder(name="rnn_state", dtype=tf.float32, shape=[LAYERS, 2, None, size])
        l = tf.unstack(self.rnn_prev, axis=0)
        rnn_tuple_state = tuple([
            tf.nn.rnn_cell.LSTMStateTuple(l[i][0], l[i][1])
            for i in range(LAYERS)
        ])
        self.rnn_init = np.zeros([LAYERS, 2, size], np.float32)

        output, self.rnn_next = tf.nn.dynamic_rnn(
            multi, inputs,
            # sequence_length=(tf.shape(x)[1], 1),
            initial_state=rnn_tuple_state,
            time_major=False)
        # self.rnn_next = (lstm_c[:1, :], lstm_h[:1, :])
        return tf.reshape(output, [-1, size])

    def _init(self, ob_space, ac_space, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        obz = tf.layers.dropout(obz, rate=DROPOUT, training=True)

        # Value Function
        last_out = obz
        last_out = self.lstm(last_out, NEURONS, "vffc_lstm")
        rnn_out = last_out
        for i in range(2):
            last_out = tf.nn.tanh(U.dense(last_out, NEURONS//(i+2), "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            last_out = tf.layers.dropout(last_out, rate=DROPOUT, training=True)
        self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]

        # Policy Function
        last_out = rnn_out
        for i in range(2):
            last_out = tf.nn.tanh(U.dense(last_out, NEURONS//(i+2), "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            last_out = tf.layers.dropout(last_out, rate=DROPOUT, training=True)
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
            [stochastic, ob, self.rnn_prev],
            [ac, self.vpred, self.rnn_next],
        )

    def act(self, stochastic, ob, rnn_state):
        ac1, vpred1, rnn_next = self._act(stochastic, ob[None], np.reshape(rnn_state, [LAYERS,2,1,NEURONS]))
        return ac1[0], vpred1[0], rnn_next
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

