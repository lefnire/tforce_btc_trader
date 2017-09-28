#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import btc_env

def train(env_id, num_timesteps, seed):
    # from baselines.ppo1 import mlp_policy, pposgd_simple
    from my_baselines.ppo2 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    env.env.set_opts(agent_name='baselines_ppo')
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=128, num_hid_layers=2)
    # env = bench.Monitor(env, logger.get_dir() and
    #     osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=1, optim_stepsize=3e-4, optim_batchsize=1,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()

def main():
    train("BTC-v0", num_timesteps=1e6, seed=1234)


if __name__ == '__main__':
    main()
