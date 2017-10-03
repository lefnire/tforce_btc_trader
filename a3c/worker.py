import tensorflow as tf
import scipy.signal
import numpy as np
import random
from a3c.ac_network import AC_Network
from btc_env.btc_env import BitcoinEnvTforce

REWARD_FACTOR = 0.001
EPSILON_STEPS = .5e6
HYPER_SWITCH = 2e6


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
    def __init__(self, name, s_size, a_size, trainer, model_path, global_episodes, seed, test, hyper, agent_name):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_mean_values = []
        self.is_test = test
        self.a_size = a_size
        self.epsilon = 1
        self.hyper = hyper
        self.steps = 2048*3 + (2048*3 // self.hyper['batch'])  # tack on some leg-room

        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer, hyper)
        self.update_local_ops = update_target_graph('global', self.name)

        self.env = BitcoinEnvTforce(
            steps=self.steps,
            agent_name=agent_name,
            indicators=hyper.get('indicators', False),
            is_main=name == 0
        )
        self.env.gym.seed(seed)

        self.train_itr = 0

    def get_env(self):
        return self.env.gym.env

    def train(self, rollout, sess, gamma, r):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 5]
        rnn_states = rollout[:, 6]

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
            net.training: True,
            net.target_v: discounted_rewards,
            net.inputs: np.vstack(states),
            net.actions: actions,
            net.advantages: discounted_advantages,
            net.rnn_prev: np.transpose([np.asarray(state) for state in rnn_states], [1,2,0,3])
        }
        for _ in range(self.hyper['epochs']):
            v_l, p_l, e_l, g_n, v_n, _ = sess.run([
                net.value_loss,
                net.policy_loss,
                net.entropy,
                net.grad_norms,
                net.var_norms,
                net.apply_grads
            ], feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                net = self.local_AC
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0

                # Restart environment
                terminal = False
                s = self.env.reset()
                rnn_prev = np.zeros([self.hyper['l_layers'], 2, 1, self.hyper['l_units']])

                # Run an episode
                while not terminal:
                    # Get preferred action distribution
                    a_dist, v, rnn_next = sess.run([net.policy, net.value, net.rnn_next],
                                         feed_dict={net.inputs: [s],
                                                    net.rnn_prev: rnn_prev})

                    if self.is_test or random.random() > self.epsilon:
                        a = a_dist  # Use maximum when testing
                    else:
                        a = random.randint(-100, 100)  # Use stochastic distribution sampling

                    # s2, r, terminal, info = self.env.step(np.argmax(a))
                    s2, r, terminal = self.env.execute([a, a_dist])
                    episode_reward += r
                    episode_buffer.append([s, a, r, s2, terminal, v[0, 0], np.squeeze(rnn_prev)])
                    episode_values.append(v[0, 0])

                    # Train on mini batches from episode
                    if len(episode_buffer) == self.hyper['batch'] and not self.is_test:
                        self.train_itr += 1
                        if not self.should_stop_training():
                            v1 = sess.run([net.value],
                                          feed_dict={net.inputs: [s],
                                                     net.rnn_prev: rnn_prev})
                            v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1[0][0])
                        episode_buffer = []

                    # Set previous state for next step
                    s = s2
                    rnn_prev = rnn_next
                    episode_step_count += 1

                    if self.epsilon > 0.1:  # decrement epsilon over time
                        self.epsilon -= (1.0 / EPSILON_STEPS)

                self.episode_mean_values.append(np.mean(episode_values))

                if self.train_itr < self.steps // self.hyper['batch']:
                    raise Exception(f"Didn't train over all batches ({self.train_itr} of {self.steps // self.hyper['batch']})")

                if self.name == 'worker_0':
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Epsilon', simple_value=float(self.epsilon))
                    summary.value.add(tag='Perf/Value', simple_value=float(self.episode_mean_values[-1]))
                    if not self.is_test:
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        # summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.get_env().summary_writer.add_summary(summary, episode_count)
                    self.get_env().summary_writer.flush()

                    if not self.is_test:
                        if episode_count % 50 == 0:
                            pass
                            # saver.save(sess, self.model_path + '/model', global_step=self.global_episodes)

                        # stop if we're showing net gains to prevent overfitting
                        if episode_count >= HYPER_SWITCH//self.steps or self.should_stop_training():
                            coord.request_stop()

                    sess.run(self.increment) # Next global episode

                episode_count += 1

    def should_stop_training(self):
        return False
        rewards = self.get_env().episode_results['rewards']
        return len(rewards) > 20 and np.mean(rewards[-20:]) > 0 # np.all(np.array(rewards[-20:]) > 0)