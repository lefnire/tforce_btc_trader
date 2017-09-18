import tensorflow as tf
import scipy.signal
import numpy as np
import random
import gym
from a3c.ac_network import AC_Network
from btc_env import BitcoinEnv

# Size of mini batches to run training on
MINI_BATCH = 150  # winner=150
REWARD_FACTOR = 0.001

STEPS = 10000
EPSILON_EPISODES = 150
HYPER_SWITCH = 500

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Weighted random selection returns n_picks random indexes.
# the chance to pick the index i is give by the weight weights[i].
def weighted_pick(weights, n_picks, epsilon=0):
    # Epsilon Greedy
    if random.random() < epsilon:  # choose random action
        return np.random.randint(0, len(weights), n_picks)

    # Else, weighted pick
    t = np.cumsum(weights)
    s = np.sum(weights)
    return np.searchsorted(t,np.random.rand(n_picks)*s)


# Discounting function used to calculate discounted returns.
def discounting(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# Normalization of inputs and outputs
def norm(x, upper, lower=0.):
    return (x-lower)/max((upper-lower), 1e-12)

class Worker():
    def __init__(self, name, s_size, a_size, trainer, model_path, global_episodes, env_name, seed, test, hyper):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_totals = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + hyper)
        self.is_test = test
        self.a_size = a_size
        self.epsilon = 1
        self.hyper = hyper

        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer, hyper)
        self.update_local_ops = update_target_graph('global', self.name)

        # self.env = gym.make(env_name)
        indicators = hyper == 'indicators:on'
        self.env = BitcoinEnv(limit=STEPS, agent_type='A3CAgent', agent_name='A3CAgent|'+hyper, indicators=indicators)
        self.env.seed(seed)

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
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(states),
                     self.local_AC.actions: np.vstack(actions),
                     self.local_AC.advantages: discounted_advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_mini_buffer = []
                episode_values = []
                episode_states = []
                episode_reward = 0
                episode_step_count = 0

                # Restart environment
                terminal = False
                s = self.env.reset()

                rnn_state = self.local_AC.state_init

                # Run an episode
                while not terminal:
                    episode_states.append(s)
                    if self.is_test:
                        self.env.render()

                    # Get preferred action distribution
                    a_dist, v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                         feed_dict={self.local_AC.inputs: [s],
                                                    self.local_AC.state_in[0]: rnn_state[0],
                                                    self.local_AC.state_in[1]: rnn_state[1]})

                    a0 = weighted_pick(a_dist[0], 1, self.epsilon) # Use stochastic distribution sampling
                    if self.is_test:
                        a0 = np.argmax(a_dist[0]) # Use maximum when testing
                    a = np.zeros(self.a_size)
                    a[a0] = 1

                    # s2, r, terminal, info = self.env.step(np.argmax(a))
                    s2, r, terminal = self.env.step(np.argmax(a))

                    episode_reward += r

                    episode_buffer.append([s, a, r, s2, terminal, v[0, 0]])
                    episode_mini_buffer.append([s, a, r, s2, terminal, v[0, 0]])

                    episode_values.append(v[0, 0])

                    # Train on mini batches from episode
                    MINI_BATCH = int(self.hyper.split(':')[1]) if self.hyper.startswith('mini_batch') else 150
                    if len(episode_mini_buffer) == MINI_BATCH and not self.is_test:
                        v1 = sess.run([self.local_AC.value],
                                      feed_dict={self.local_AC.inputs: [s],
                                                    self.local_AC.state_in[0]: rnn_state[0],
                                                    self.local_AC.state_in[1]: rnn_state[1]})
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_mini_buffer, sess, gamma, v1[0][0])
                        episode_mini_buffer = []

                    # Set previous state for next step
                    s = s2
                    total_steps += 1
                    episode_step_count += 1

                if self.epsilon > 0.1:  # decrement epsilon over time
                    self.epsilon -= (1.0 / EPSILON_EPISODES)

                self.episode_rewards.append(episode_reward)
                self.episode_totals.append(self.env.cash + self.env.value)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if self.name == 'worker_0':
                    if not self.is_test:
                        summary = tf.Summary()
                        summary.value.add(tag='Perf/Epsilon', simple_value=float(self.epsilon))
                        summary.value.add(tag='Perf/Reward', simple_value=float(self.episode_rewards[-1]))
                        summary.value.add(tag='Perf/Total', simple_value=float(self.episode_totals[-1]))
                        # summary.value.add(tag='Perf/Length', simple_value=float(self.episode_lengths[-1]))
                        summary.value.add(tag='Perf/Value', simple_value=float(self.episode_mean_values[-1]))
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                        self.summary_writer.add_summary(summary, episode_count)
                        self.summary_writer.flush()

                    if episode_count % 100 == 0 and not self.is_test:
                        pass
                        #saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')

                    if episode_count >= HYPER_SWITCH:
                        coord.request_stop()

                    sess.run(self.increment) # Next global episode

                episode_count += 1
