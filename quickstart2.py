from tensorforce import Configuration
from tensorforce.agents import DQNNstepAgent
from tensorforce.core.networks import layered_network_builder
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

env = OpenAIGym('CartPole-v0')
agent = DQNNstepAgent(
    Configuration(
    tf_session_config=None,
    states=env.states,
    actions=env.actions,
    # batch_size=8,
    double_dqn=True,
    network=layered_network_builder([dict(type='dense', size=32)])
))

runner = Runner(agent=agent, environment=env)
def episode_finished(r):
    print(r.episode_rewards[-1])
    return True

runner.run(episodes=3000, max_timesteps=200, episode_finished=episode_finished)