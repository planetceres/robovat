#!/usr/bin/env python

"""
Curious Agent implementing a pushing task
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import multiprocessing
import os
import random
import socket
import shutil
import threading
import uuid
from builtins import input

import numpy as np
import matplotlib.pyplot as plt
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


#PARAMETERS
OUTPUT_GRAPH = True         # safe logs
RENDER=True                 # render one worker
LOG_DIR = './log'           # savelocation for logs
N_WORKERS = multiprocessing.cpu_count() # number of workers
MAX_EP_STEP = 200           # maxumum number of steps per episode
MAX_GLOBAL_EP = 2000        # total number of episodes
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10     # sets how often the global net is updated
GAMMA = 0.90                # discount factor
ENTROPY_BETA = 0.01         # entropy factor
LR_A = 0.0001               # learning rate for actor
LR_C = 0.001                # learning rate for critic

#TODO: check the config and env and policy config files
#TODO: check the success_thresh and MAX_STEPS

args = {'env': 'PushEnv', 'policy': 'HeuristicPushPolicy', 'env_config':'configs/envs/push_env.yaml', 
        'policy_config':'configs/policies/heuristic_push_policy.yaml', 'config_bindings':"{'SUCCESS_THRESH':2 ,'MAX_STEPS:':200}",
        'use_simulator': 1, 'assets_dir':'./assets', 'output_dir':None, 'num_steps':None,
        'num_episodes':'None', 'num_episodes_per_file': 1000, 'debug': 1, 'worker_id': 0, 'seed': None,
        'pause': False, 'timeout':120 }

#TODO: figure the next two lines out! 

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

    global_rewards = []
    global_episodes = 0

    sess = tf.Session()
    
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

    N_S = env.observation_space.shape[0]                    # number of states
    N_A = env.action_space.shape[0]                         # number of actions
    A_BOUND = [env.action_space.low, env.action_space.high] # action bounds

    # Policy.
    #policy_class = getattr(policies, args['policy'])
    #policy = policy_class(env=env, config=policy_config)

    # Output directory.
    #if args['output_dir'] is not None:
    #    hostname = socket.gethostname()
    #    hostname = hostname.split('.')[0]
    #   output_dir = os.path.abspath(args['output_dir'])
    #    output_dir = os.path.join(output_dir, hostname, '%02d' % (args['key']))
    #    if not os.path.isdir(output_dir):
    #        logger.info('Making output directory %s...', output_dir)
    #       os.makedirs(output_dir)

    # Generate and write episodes.
    #logger.info('Start running...')
    env.reset()
    #log_pose()
    num_episodes_this_file = 0

    with tf.device("/cpu:0"):
        global_ac = ACNet(GLOBAL_NET_SCOPE,sess)  # we only need its params
        workers = []
        # Create workers
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, global_ac,sess))

    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH: # write log file
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, sess.graph)

        
     # Append the episode to the file.
     #       logger.info('Saving episode %d to file %s (%d / %d)...',
     #                  episode_index,
     #                   output_path,
     #                   num_episodes_this_file,
     #                  args['num_episodes_per_file'])

     #       with h5py.File(output_path, 'a') as fout:
     #           name = str(uuid.uuid4())
     #          group = fout.create_group(name)
     #           hdf5_utils.write_data_to_hdf5(group, episode)

     #  num_episodes_this_file += 1
     #   num_episodes_this_file %= args['num_episodes_per_file']

     #   if args['pause']:
     #       input('Press [Enter] to start a new episode.')
     #  print(pose_logger.uri)

    worker_threads = []
    for worker in workers: #start workers
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)  # wait for termination of workers
    
    plt.plot(np.arange(len(global_rewards)), global_rewards) # plot rewards
    plt.xlabel('step')
    plt.ylabel('total moving reward')
    plt.show()



if __name__ == '__main__':
    main()








