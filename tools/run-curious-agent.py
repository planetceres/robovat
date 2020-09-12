#!/usr/bin/env python

"""
Curious Agent implementing a pushing task
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import os
import random
import socket
import uuid
from builtins import input

import numpy as np
import h5py

import _init_paths  # NOQA
from robovat import envs
from robovat import policies
from robovat.io import hdf5_utils
from robovat.io.episode_generation_curious import generate_episodes
from robovat.simulation.simulator import Simulator
from robovat.utils import time_utils
from robovat.utils.logging import logger
from robovat.utils.yaml_config import YamlConfig
#from tools.pose_log import log_pose

#TODO: check the config and env and policy config files
#TODO: check the success_thresh and MAX_STEPS

args = {'env': 'PushEnv', 'policy': 'HeuristicPushPolicy', 'env_config':'configs/envs/push_env.yaml', 
        'policy_config':'configs/policies/heuristic_push_policy.yaml', 'config_bindings':"{'TASK_NAME': 'data_collection', 'SUCCESS_THRESH':2 ,'MAX_STEPS:':200}",
        'use_simulator': 1, 'assets_dir':'./assets', 'output_dir':None, 'num_steps':None,
        'num_episodes':'None', 'num_episodes_per_file': 1000, 'debug': 1, 'worker_id': 0, 'seed': None,
        'pause': False, 'timeout':120 }


def parse_config_files_and_bindings(args):
    if args['env_config'] is None:
        env_config = None
    else:
        env_config = YamlConfig(args['env_config']).as_easydict()

    if args['policy_config'] is None:
        policy_config = None
    else:
        policy_config = YamlConfig(args['policy_config']).as_easydict()

    if args['config_bindings'] is not None:
        parsed_bindings = ast.literal_eval(args['config_bindings'])
        logger.info('Config Bindings: %r', parsed_bindings)
        env_config.update(parsed_bindings)
        policy_config.update(parsed_bindings)

    return env_config, policy_config



def main():

    # Configuration.
    env_config, policy_config = parse_config_files_and_bindings(args)

    # Set the random seed.
    if args['seed'] is not None:
        random.seed(args['seed'])
        np.random.seed(args['seed'])

    # Simulator.
    if args['use_simulator']:
        simulator = Simulator(worker_id=args['worker_id'],
                              use_visualizer=bool(args['debug']),
                              assets_dir=args['assets_dir'])
    else:
        simulator = None

    # Environment.
    env_class = getattr(envs, args['env'])
    env = env_class(simulator=simulator,
                    config=env_config,
                    debug=args['debug'])

    # Policy.
    policy_class = getattr(policies, args['policy'])
    policy = policy_class(env=env, config=policy_config)

    # Output directory.
    if args['output_dir'] is not None:
        hostname = socket.gethostname()
        hostname = hostname.split('.')[0]
        output_dir = os.path.abspath(args['output_dir'])
        output_dir = os.path.join(output_dir, hostname, '%02d' % (args['key']))
        if not os.path.isdir(output_dir):
            logger.info('Making output directory %s...', output_dir)
            os.makedirs(output_dir)

    # Generate and write episodes.
    logger.info('Start running...')
    env.reset()
    #log_pose()
    num_episodes_this_file = 0
    for episode_ind, episode in generate_episodes(
            env,
            policy,
            num_steps=args['num_steps'],
            num_episodes=args['num_episodes'],
            timeout=args['timeout'],
            debug=args['debug']):

        if args['output_dir']:
            # Create a file for saving the episode data.
            if num_episodes_this_file == 0:
                timestamp = time_utils.get_timestamp_as_string()
                filename = 'episodes_%s.hdf5' % (timestamp)
                output_path = os.path.join(output_dir, filename)
                logger.info('Created a new file %s...', output_path)
        
            # Append the episode to the file.
            logger.info('Saving episode %d to file %s (%d / %d)...',
                        episode_index,
                        output_path,
                        num_episodes_this_file,
                        args['num_episodes_per_file'])

            with h5py.File(output_path, 'a') as fout:
                name = str(uuid.uuid4())
                group = fout.create_group(name)
                hdf5_utils.write_data_to_hdf5(group, episode)

        num_episodes_this_file += 1
        num_episodes_this_file %= args['num_episodes_per_file']

        if args['pause']:
            input('Press [Enter] to start a new episode.')
        #print(pose_logger.uri)



if __name__ == '__main__':
    main()
