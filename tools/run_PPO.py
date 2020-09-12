#!/usr/bin/env python

"""
ppo implementing a pushing task

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pprint

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from robovat.simulation.simulator import Simulator
from robovat.utils.logging import logger
from robovat.utils.yaml_config import YamlConfig

import policies
from utils import suite_env

 
args = {'root':'$HOME/tmp/ppo/robovat/push','env': 'PushEnv', 'policy': 'RandomPolicy', 'env_config':'configs/envs/push_env.yaml', 
        'policy_config':'configs/policies/push_policy.yaml', 'config_bindings':"{'TASK_NAME': 'clearing','SUCCESS_THRESH':2 ,'MAX_STEPS:':200}",
        'use_simulator': 1, 'assets_dir':'./assets', 'output_dir':None, 'num_steps':None,
        'num_episodes':'None', 'num_episodes_per_file': 1000, 'debug': True, 'worker_id': 0, 'seed': None,
        'pause': False, 'timeout':120 'policy_dir_name' : '~/curis-project/robovat-curis-mona/tools/ppo.py'}

'''flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('policy_dir_name', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Directory name for saved policy model')
flags.DEFINE_string('env_name', 'ShuffleEnv', 'Name of an environment')
flags.DEFINE_integer('num_episodes', 3,
                     'Number of episodes')
flags.DEFINE_integer('random_seed', None,
                     'Number of episodes')
FLAGS = flags.FLAGS'''

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


def parse_config_files_and_bindings(env_config=None, policy_config=None, config_bindings=None):
    """Parse the config files and the bindings.
    Args:
        args: The arguments.
    Returns:
        env_config: Environment configuration.
        policy_config: Policy configuration.
    """
    if env_config is None:
        env_config = dict()
    else:
        env_config = YamlConfig(env_config).as_easydict()

    if policy_config is None:
        policy_config = dict()
    else:
        policy_config = YamlConfig(policy_config).as_easydict()

    if config_bindings is not None:
        parsed_bindings = ast.literal_eval(config_bindings)
        logger.info('Config Bindings: %r', parsed_bindings)
        env_config.update(parsed_bindings)
        policy_config.update(parsed_bindings)

    logger.info('Environment Config:')
    pprint.pprint(env_config)
    logger.info('Policy Config:')
    pprint.pprint(policy_config)

    return env_config, policy_config

'''def env_load_fn(env_name):
    env = ""
    env_config = 'config/envs/grocery_env.yaml'
    debug = 1 
    assets_dir = 'data'
    worker_id = 0
    seed = None
    use_simulator = True

    # Set the random seed.
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    # Configuration.
    env_config, policy_config = parse_config_files_and_bindings(env_config=env_config)

    # Simulator.
    if use_simulator:
        simulator = Simulator(worker_id=worker_id,
                              use_visualizer=bool(debug),
                              assets_dir=assets_dir)
    else:
        simulator = None
    py_env = suite_env.load(env,
                            simulator=simulator,
                            config=env_config,
                            debug=debug,
                            max_episode_steps=None)
    return py_env'''

def run_episodes_and_create_video(policy, eval_tf_env, num_episodes):
    frames = []
    for _ in range(num_episodes):
        start = time.time()
        time_step = eval_tf_env.reset()
        while not time_step.is_last()[0:1]:
            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)
        print(f"time: {time.time()-start}")

@gin.configurable
def run_eval(
    root_dir,
    policy_dir_name,
    env,
    num_episodes = 100):
  """A simple train and eval for PPO."""
  if root_dir is None:
    raise AttributeError('train_eval requires a root_dir.')

  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')
  saved_model_dir = os.path.join(root_dir, 'policy_saved_model')

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    if random_seed is not None:
      tf.compat.v1.set_random_seed(random_seed)
    eval_py_env = env_load_fn(env_name)
    tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    global_step = tf.compat.v1.train.get_global_step()

    saved_policy = tf.compat.v2.saved_model.load(os.path.join(saved_model_dir, policy_dir_name))
    run_episodes_and_create_video(saved_policy, tf_env, num_episodes)


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.config.set_soft_device_placement(True)
  gpus = tf.config.experimental.list_physical_devices('GPU') 
  if gpus:
      try:
          for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
          logical_gpus = tf.config.experimental.list_logical_devices('GPU')
          print(f"{len(gpus)} physical GPUs - {len(logical_gpus)} logical GPUs")
      except RuntimeError as e:
          # Memory growth must be set before GPUs have been initialized
          logging.error(e)

  tf.compat.v1.enable_v2_behavior()

    # Configuration.
    env_config, policy_config = parse_config_files_and_bindings(args)

    # Set the random seed.
    '''if args['seed'] is not None:
        random.seed(args['seed'])
        np.random.seed(args['seed'])'''

    # Simulator.
    if args['use_simulator']:
        simulator = Simulator(worker_id=args['worker_id'],
                              use_visualizer=bool(args['debug']),
                              assets_dir=args['assets_dir'])
    else:
        simulator = None

    # Environment.
    env_class = getattr(envs, args['env'])
    env = env_class(simulator,
                    config=env_config,
                   debug=args['debug'])
  run_eval(args['root'], args['policy_dir_name'],
    env=env,
    num_episodes=args['num_episodes'])


if __name__ == '__main__':
  #flags.mark_flag_as_required('root_dir')
  app.run(main)

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
import A3C_network as nw
import A3C_worker as w
import A3C_config as config

import matplotlib.pyplot as plt
import tensorflow as tf
import h5py

from robovat import envs
from robovat import policies
from robovat.io import hdf5_utils
#from robovat.io.episode_generation_curious import generate_episodes
from robovat.simulation.simulator import Simulator
from robovat.utils import time_utils
from robovat.utils.logging import logger
from robovat.utils.yaml_config import YamlConfig

#from tools.A3C_network import ACNet
#from tools.A3c_worker import Worker 
#from tools.pose_log import log_pose

#PARAMETERS
OUTPUT_GRAPH = True         # safe logs
RENDER=True                 # render one worker
LOG_DIR = './log'           # savelocation for logs
N_WORKERS = multiprocessing.cpu_count() # number of workers
N_WORKERS = 1
MAX_EP_STEP = 200           # maxumum number of steps per episode
MAX_GLOBAL_EP = 2000        # total number of episodes
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10     # sets how often the global net is updated
GAMMA = 0.90                # discount factor
ENTROPY_BETA = 0.01         # entropy factor
LR_A = 0.0001               # learning rate for actor
LR_C = 0.001                # learning rate for critic

#global global_rewards = []
#global global_episodes = 0

#TODO: check the config and env and policy config files
#TODO: check the success_thresh and MAX_STEPS

 
args = {'env': 'PushEnv', 'policy': 'RandomPolicy', 'env_config':'configs/envs/push_env.yaml', 
        'policy_config':'configs/policies/push_policy.yaml', 'config_bindings':"{'TASK_NAME': 'clearing','SUCCESS_THRESH':2 ,'MAX_STEPS:':200}",
        'use_simulator': 1, 'assets_dir':'./assets', 'output_dir':None, 'num_steps':None,
        'num_episodes':'None', 'num_episodes_per_file': 1000, 'debug': True, 'worker_id': 0, 'seed': None,
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

    #sess = tf.compat.v1.Session()
    tf.tf_agents.agents.ppo.ppo_agent()
    
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
    env = env_class(simulator,
                    config=env_config,
                   debug=args['debug'])

    #N_S = env.observation_space.shape[0]                    # number of states
    #N_A = env.action_space.shape[0]                        # number of actions
    
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

    coord = tf.compat.v1.train.Coordinator()

    with tf.device("/cpu:0"):
        global_ac = nw.ACNet(GLOBAL_NET_SCOPE, sess , env)  # we only need its params
        workers = []
        # Create workers
        for i in range(0, N_WORKERS):
            print('\n\n\n\n worker %d \n\n\n\n', (i))

       # Simulator.
            if args['use_simulator']:
                simulator = Simulator(worker_id=i+1,
                                      use_visualizer=bool(args['debug']),
                                      assets_dir=args['assets_dir'])
            else:
                simulator = None

            env = env_class(simulator,
                            config=env_config,
                            debug=args['debug'])

            i_name = 'W_%i' % i   # worker name
            worker = w.Worker(i_name, global_ac ,sess , env)
            workers.append(worker)
  
    #coord = tf.compat.v1.train.Coordinator()
    

    sess.run(tf.compat.v1.global_variables_initializer())

    if OUTPUT_GRAPH: # write log file
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.compat.v1.summary.FileWriter(LOG_DIR, sess.graph)

        
    '''#Append the episode to the file.
        logger.info('Saving episode %d to file %s (%d / %d)...',
                     episode_index,
                    output_path,
                     num_episodes_this_file,
                     args['num_episodes_per_file'])

        with h5py.File(output_path, 'a') as fout:
            name = str(uuid.uuid4())
            group = fout.create_group(name)
            hdf5_utils.write_data_to_hdf5(group, episode)'''

    num_episodes_this_file += 1
    num_episodes_this_file %= args['num_episodes_per_file']

     #    if args['pause']:
     #       input('Press [Enter] to start a new episode.')
     #  print(pose_logger.uri)

    worker_threads = []
    
    for worker in workers: #start workers

        #job = lambda: worker.work()
        tup = (coord, 0 )
        ''''p = multiprocessing.Process(target=spawn)
        p.start()
        coord.join(p) # this line allows you to wait for processes'''
       
        t = threading.Thread(target= worker.work, args = tup)
        worker_threads.append(t)
    coord.join(worker_threads)  # wait for termination of workers'''
    for t in worker_threads:
        t.start()
    
    plt.plot(np.arange(len(config.global_rewards)), config.global_rewards) # plot rewards
    plt.xlabel('step')
    plt.ylabel('total moving reward')
    plt.show()

if __name__ == '__main__':
    main()








