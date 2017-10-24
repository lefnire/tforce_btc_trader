"""
WIP! This file is incomplete, will take some time to hash out what I want here.
1. Create a list of configs to recurse over
2. After each 300, save all the attrs used
  a. Then grab all from db, linear regression, cross-join attrs, predict, take attrs w/ max prediction
  b. set as new default params, print
"""
import pdb
import tensorflow as tf
from tensorforce.execution import Runner
from tensorforce import Configuration, agents as agents_dict
from tensorforce.core.networks.layer import Dense
from tensorforce.core.networks.network import LayeredNetwork
from tensorforce.execution import Runner
import data

from btc_env.btc_env import BitcoinEnvTforce


tree = {
    'agent': ['ppo_agent'],  # dqn_agent
    'net_type': ['lstm']  # conv2d
}

tree['network'] = {
    'lstm': {
        5: [('d', 256), ('L', 512), ('L', 512), ('L', 512), ('d', 256), ('d', 128)],
        4: [('d', 128), ('L', 256), ('L', 256), ('d', 192), ('d', 128)]
    }
}

tree['ppo_agent'] = {
    'step_optimizer.learning_rate': [1e-5, 1e-6, 1e-7],
    'batch_size': [512, 1024, 128, 8],
    'discount': [.95, .97, .99],
}

agent_trees = []
for agent_k in tree['agent']:
    for net_k in tree['net_type']:
        agent_trees.append({
            'network': tree['network'][net_k],
            'agent': agent_k,
            **tree[agent_k]
        })


def get_defaults(tree):
    """
    TODO this should select current winners from database
    """
    conf = {
        'tf_session_config': tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.49)
        ),

    }
    for attr, vals in tree.items():
        conf[attr] = vals[0] if type(vals) is list else next(iter(vals.values()))
    return conf

while True:
    for tree in agent_trees:
        for attr, vals in tree.items():
            # if simple value, v==k. If dict, use store k, use v
            if type(vals) is list:
                vals = {v: v for v in vals}
            for k, v in vals:
                # TODO fetch & apply defaults
                conf = get_defaults(tree)
                conf[k] = v
                conv2d = tree.pop('net_type') == 'conv2d'
                network = tree.pop('network')
                conf = Configuration(**conf)

                name = f"{tree['agent']}.{tree['net_type']}.{k}"
                env = BitcoinEnvTforce(steps=2048, name=name, conv2d=conv2d, is_main=True)

                agent = agents_dict[tree.pop('agent')](
                    states_spec=env.states,
                    actions_spec=env.actions,
                    network_spec=network,
                    config=conf
                )

                print(conf['name'])
                runner = Runner(episodes=2, agent=conf['agent'], environment=conf['env'])
                pdb.set_trace()
                runner.run()