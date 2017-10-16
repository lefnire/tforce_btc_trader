#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import btc_env

def main():
    seed = 1234
    # from baselines.ppo1 import mlp_policy, pposgd_simple
    from blines.ppo2 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make("BTC-v0")
    env.env.init(env,
        agent_name=f'baselines.{mlp_policy.NAME}',
        scale_features='scale' in mlp_policy.NAME
    )
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
    # env = bench.Monitor(env, logger.get_dir() and
    #     osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=1e12,   # infinite
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=128,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()

if __name__ == '__main__':
    main()
