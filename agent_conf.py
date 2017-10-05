from tensorforce import Configuration, TensorForceError, agents, models
from tensorforce.core.networks import layered_network_builder

from btc_env.btc_env import BitcoinEnvTforce
from experiments import network, net4x, net3x

STEPS = 2048 * 3 + 3


def conf(overrides, agent_type, mods='main', env_args={}, no_agent=False):
    agent_name = agent_type + '|' + mods
    agent_class = agents.agents[agent_type]
    env = BitcoinEnvTforce(steps=STEPS, agent_name=agent_name, **env_args)

    conf = dict(
        tf_session_config=None,
        # tf_session_config=tf.ConfigProto(device_count={'GPU': 0}),
        # tf_session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.2)),
        log_level="info",
        tf_saver=False,
        tf_summary=None,
        tf_summary_level=0,

        network=network(net3x),
        learning_rate=1e-7,
        discount=.99,
        exploration=dict(
            type="epsilon_decay",
            epsilon=1.0,
            epsilon_final=0.,
            epsilon_timesteps=1.5e6
        ),
        optimizer="nadam", # winner=nadam
        states=env.states,
        actions=env.actions,
    )

    if agent_class == agents.TRPOAgent:
        pass
    elif issubclass(agent_class.model, models.PolicyGradientModel):
        conf.update(
            baseline=dict(
                type="mlp",
                sizes=[128, 128],
                epochs=10,
                update_batch_size=512,  # 1024
                learning_rate=.001
            ),
            batch_size=2048,  # batch_size must be > optimizer_batch_size
            optimizer_batch_size=1024,
            normalize_rewards=True,  # definite winner=True
            keep_last=True,
            epochs=5
            # gae_rewards winner=False
        )
    elif agent_class == agents.NAFAgent:
        conf.update(
            network=network(net4x, d=.4),
            learning_rate=1e-8,
            batch_size=8,
            memory_capacity=800,
            first_update=80,
            exploration=dict(
                type="ornstein_uhlenbeck",
                sigma=0.2,
                mu=0,
                theta=0.15
            ),
            update_target_weight=.001,
            clip_loss=1.
        )
    elif agent_class == agents.DQNAgent:
        conf.update(double_dqn=True)
    elif agent_class == agents.DQNNstepAgent:
        conf.update(batch_size=8)
        # Investigate graphs: batch-8 setup, random_replay=False, 4x

    # From caller (A3C v single-run)
    conf.update(overrides)
    # Allow overrides to network above, then run it through configurator
    conf['network'] = layered_network_builder(conf['network'])
    conf = Configuration(**conf)

    return dict(
        agent=None if no_agent else agent_class(config=conf),
        conf=conf,
        env=env,
        agent_name=agent_name
    )