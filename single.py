from hypersearch import get_hypers, run_finished, generate_and_save_hypers
from tensorforce import Configuration, agents as agents_dict
from tensorforce.execution import Runner

generate_and_save_hypers()

flat, hydrated, network, env = get_hypers(rand=True, from_db=True)
agent = agents_dict.agents['ppo_agent'](
    states_spec=env.states,
    actions_spec=env.actions,
    network_spec=network,
    config=Configuration(**hydrated)
)

runner = Runner(
    agent=agent,
    environment=env
)
runner.run(episodes=300)
run_finished(env, flat)