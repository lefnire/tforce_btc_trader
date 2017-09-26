from gym.envs.registration import register

register(
    id='BTC-v0',
    entry_point='btc_env.btc_env:BitcoinEnv',
    max_episode_steps=100000,
    # reward_threshold=10
)