https://github.com/liampetti/A3C-LSTM

# Implementation of Asynchronous Advantage Actor-Critic algorithm using Long Short Term Memory Networks (A3C-LSTM)

Modified from the work of Arthur Juliani: [Simple Reinforcement Learning with Tensorflow Part 8: Asynchronous Actor-Critic Agents (A3C)](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)

Paper can be found here: ["Asynchronous Methods for Deep Reinforcement Learning" - Mnih et al., 2016](https://arxiv.org/abs/1602.01783)

Tested on [CartPole](https://gym.openai.com/envs/CartPole-v0)

### Requirements
[Gym](https://github.com/openai/gym#installation) and [TensorFlow](https://www.tensorflow.org/install/).

### Usage

Training only happens on minibatches of greater than 30, effectively preventing poor performing episodes from influencing training. A reward factor is used to allow for effective training at faster learning rates.

Models are saved every 100 episodes. They can be reloaded for further training or visualised for testing by setting either of the global parameters to True.

This is just example code to test an A3C-LSTM implementation. This should not be considered the optimal way to learn for this environment!

