import tensorflow as tf
import scipy.signal
import numpy as np
import random
from a3c.ac_network import AC_Network
from btc_env import BitcoinEnv

MINI_BATCH = 2e3
REWARD_FACTOR = 0.001
STEPS = 6e3; STEPS += STEPS // MINI_BATCH  # tack on some leg-room
EPSILON_STEPS = 3e6
HYPER_SWITCH = 1e10


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
def discounting(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Worker():
    def __init__(self, name, s_size, a_size, trainer, model_path, global_episodes, seed, test, hyper):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_totals = []
        self.episode_totals_true = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("./saves/{}/{}".format('test' if test else 'train', hyper))
        self.summary_writer_true = tf.summary.FileWriter("./saves/{}:true/{}".format('test' if test else 'train', hyper))
        self.is_test = test
        self.a_size = a_size
        self.epsilon = 1
        self.hyper = hyper

        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer, hyper)
        self.update_local_ops = update_target_graph('global', self.name)

        indicators = hyper == 'indicators:on'
        self.env = BitcoinEnv(limit=STEPS, agent_type='A3CAgent', agent_name='A3CAgent|'+hyper, indicators=indicators)
        self.env.seed(seed)

        self.train_itr = 0

    def get_env(self):
        return self.env

    def train(self, rollout, sess, gamma, r):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        rewards_list = np.asarray(rewards.tolist()+[r])*REWARD_FACTOR
        discounted_rewards = discounting(rewards_list, gamma)[:-1]

        # Advantage estimation
        # JS, P Moritz, S Levine, M Jordan, P Abbeel,
        # "High-dimensional continuous control using generalized advantage estimation."
        # arXiv preprint arXiv:1506.02438 (2015).
        values_list = np.asarray(values.tolist()+[r])*REWARD_FACTOR
        advantages = rewards + gamma * values_list[1:] - values_list[:-1]
        discounted_advantages = discounting(advantages, gamma)


        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        # sess.run(self.local_AC.reset_state_op)
        net = self.local_AC
        feed_dict = {
            net.training: not self.is_test,
            net.target_v: discounted_rewards,
            net.inputs: np.vstack(states),
            net.actions: actions,
            net.advantages: discounted_advantages,
            net.state_in[0]: self.batch_rnn_state[0],
            net.state_in[1]: self.batch_rnn_state[1],
        }
        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run([
            net.value_loss,
            net.policy_loss,
            net.entropy,
            net.grad_norms,
            net.var_norms,
            net.state_out,
            net.apply_grads
        ], feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                net = self.local_AC
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_states = []
                episode_reward = 0
                episode_step_count = 0

                # Restart environment
                terminal = False
                s = self.env.reset()

                rnn_state = net.state_init
                self.batch_rnn_state = rnn_state

                # Run an episode
                while not terminal:
                    episode_states.append(s)

                    # Get preferred action distribution
                    a_dist, v, rnn_state = sess.run([net.policy, net.value, net.state_out],
                                         feed_dict={net.inputs: [s],
                                                    net.state_in[0]: rnn_state[0],
                                                    net.state_in[1]: rnn_state[1]})

                    if self.is_test or random.random() > self.epsilon:
                        a = a_dist  # Use maximum when testing
                    else:
                        a = random.randint(-100, 100)  # Use stochastic distribution sampling

                    # s2, r, terminal, info = self.env.step(np.argmax(a))
                    s2, r, terminal = self.env.step(a, action_true=a_dist)
                    episode_reward += r
                    episode_buffer.append([s, a, r, s2, terminal, v[0, 0]])
                    episode_values.append(v[0, 0])

                    # Train on mini batches from episode
                    mini_batch = int(self.hyper.split(':')[1]) if self.hyper.startswith('mini_batch') else MINI_BATCH
                    if len(episode_buffer) == mini_batch and not self.is_test:
                        self.train_itr += 1
                        v1 = sess.run([net.value],
                                      feed_dict={net.inputs: [s],
                                                 net.state_in[0]: rnn_state[0],
                                                 net.state_in[1]: rnn_state[1]})
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1[0][0])
                        episode_buffer = []

                    # Set previous state for next step
                    s = s2
                    total_steps += 1
                    episode_step_count += 1

                    if self.epsilon > 0.1:  # decrement epsilon over time
                        self.epsilon -= (1.0 / EPSILON_STEPS)

                self.episode_rewards.append(episode_reward)
                self.episode_totals.append(self.env.cash + self.env.value)
                self.episode_totals_true.append(self.env.cash_true + self.env.value_true)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if self.train_itr < STEPS // MINI_BATCH:
                    raise Exception(f"Didn't train over all batches ({self.train_itr} of {STEPS // MINI_BATCH})")

                if self.name == 'worker_0':
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Epsilon', simple_value=float(self.epsilon))
                    summary.value.add(tag='Perf/Reward', simple_value=float(self.episode_rewards[-1]))
                    summary.value.add(tag='Perf/Total', simple_value=float(self.episode_totals[-1]))
                    # summary.value.add(tag='Perf/Length', simple_value=float(self.episode_lengths[-1]))
                    summary.value.add(tag='Perf/Value', simple_value=float(self.episode_mean_values[-1]))
                    summary.value.add(tag='Perf/Time', simple_value=float(self.env.time))
                    if not self.is_test:
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        # summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()

                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Total', simple_value=float(self.episode_totals_true[-1]))
                    self.summary_writer_true.add_summary(summary, episode_count)
                    self.summary_writer_true.flush()

                    if not self.is_test:
                        if episode_count % 50 == 0:
                            # pass
                            saver.save(sess, self.model_path + '/model', global_step=self.global_episodes)

                        # stop if we're showing net gains to prevent overfitting
                        n_pos = 50
                        last_n_positive = len(self.episode_totals) > n_pos \
                            and np.all(np.array(self.episode_totals_true[-n_pos:]) > 2000)

                        if episode_count >= HYPER_SWITCH or last_n_positive:
                            coord.request_stop()

                    sess.run(self.increment) # Next global episode

                episode_count += 1
