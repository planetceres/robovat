#!/usr/bin/env python

"""
Worker class for the Actor Critic

Modified code from GitHub stefanbo92/A3C-Continuous
"""


"""An ActorPolicy that also returns policy_info needed for PPO training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.ppo import ppo_utils
from tf_agents.networks import network
from tf_agents.policies import actor_policy
from tf_agents.specs import distribution_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts

tfd = tfp.distributions


class PPOPolicy(actor_policy.ActorPolicy):
  """An ActorPolicy that also returns policy_info needed for PPO training."""

  def __init__(self,
               time_step_spec=None,
               action_spec=None,
               actor_network=None,
               value_network=None,
               observation_normalizer=None,
               clip=True,
               collect=True):
    """Builds a PPO Policy given network Templates or functions.
    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      actor_network: An instance of a tf_agents.networks.network.Network, with
        call(observation, step_type, network_state).  Network should
        return one of the following: 1. a nested tuple of tfp.distributions
          objects matching action_spec, or 2. a nested tuple of tf.Tensors
          representing actions.
      value_network:  An instance of a tf_agents.networks.network.Network, with
        call(observation, step_type, network_state).  Network should return
        value predictions for the input state.
      observation_normalizer: An object to use for obervation normalization.
      clip: Whether to clip actions to spec before returning them.  Default
        True. Most policy-based algorithms (PCL, PPO, REINFORCE) use unclipped
        continuous actions for training.
      collect: If True, creates ops for actions_log_prob, value_preds, and
        action_distribution_params. (default True)
    Raises:
      ValueError: if actor_network or value_network is not of type callable or
        tensorflow.python.ops.template.Template.
    """
    info_spec = ()
    if collect:
      # TODO(oars): Cleanup how we handle non distribution networks.
      if isinstance(actor_network, network.DistributionNetwork):
        network_output_spec = actor_network.output_spec
      else:
        network_output_spec = tf.nest.map_structure(
            distribution_spec.deterministic_distribution_from_spec, action_spec)
      info_spec = tf.nest.map_structure(lambda spec: spec.input_params_spec,
                                        network_output_spec)

    super(PPOPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        info_spec=info_spec,
        actor_network=actor_network,
        observation_normalizer=observation_normalizer,
        clip=clip)

    self._collect = collect
    self._value_network = value_network

  def apply_value_network(self, observations, step_types, policy_state):
    """Apply value network to time_step, potentially a sequence.
    If observation_normalizer is not None, applies observation normalization.
    Args:
      observations: A (possibly nested) observation tensor with outer_dims
        either (batch_size,) or (batch_size, time_index). If observations is a
        time series and network is RNN, will run RNN steps over time series.
      step_types: A (possibly nested) step_types tensor with same outer_dims as
        observations.
      policy_state: Initial policy state for value_network.
    Returns:
      The output of value_net, which is a tuple of:
        - value_preds with same outer_dims as time_step
        - policy_state at the end of the time series
    """
    if self._observation_normalizer:
      observations = self._observation_normalizer.normalize(observations)
    return self._value_network(observations, step_types, policy_state)

  def _apply_actor_network(self, time_step, policy_state):
    if self._observation_normalizer:
      observation = self._observation_normalizer.normalize(
          time_step.observation)
      time_step = ts.TimeStep(time_step.step_type, time_step.reward,
                              time_step.discount, observation)
    # Dictionary, value is stacked array across time. state_obs, actions
    # Stores obs, action, create t x 1 array called mass_mask (mark entries that store stuff)
    # mass_mask - [0, 0, 0, 1]. Wherever 1 ignore that element.
    # append new obs to the time_step, or obs. 
    return self._actor_network(
        time_step.observation, time_step.step_type, network_state=policy_state)
  
  def _variables(self):
    var_list = self._actor_network.variables[:]
    var_list += self._value_network.variables[:]
    if self._observation_normalizer:
      var_list += self._observation_normalizer.variables
    return var_list

  def _distribution(self, time_step, policy_state):
    # Actor network outputs nested structure of distributions or actions.
    # TODO: Use policy state to call the mass estimator
    # Consider ading placeholder to policy_state [mask, obs_h, act_h, mass_est_placeholder]
    actions_or_distributions, policy_state = self._apply_actor_network(
        time_step, policy_state) # Overwrite function to take (ts, ps, mass_estimate)

    def _to_distribution(action_or_distribution):
      if isinstance(action_or_distribution, tf.Tensor):
        # This is an action tensor, so wrap it in a deterministic distribution.
        return tfp.distributions.Deterministic(loc=action_or_distribution)
      return action_or_distribution

    distributions = tf.nest.map_structure(_to_distribution,
                                          actions_or_distributions)

    # Prepare policy_info.
    if self._collect:
      policy_info = ppo_utils.get_distribution_params(distributions)
    else:
      policy_info = ()

    return policy_step.PolicyStep(distributions, policy_state, policy_info)



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mlflow
import socket
import time
import traceback

#import tools.curious_agent

from robovat.simulation.body import Body
from robovat.utils import time_utils
from robovat.utils.logging import logger
from pose_log import logger as Plogger
from pose_estimation import find_forward_error as forward_r

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from robovat import envs
from robovat import policies
#from tools.pose_log import log_pose
import A3C_network
import run_A3C
import A3C_config as config


MAX_GLOBAL_EP = 2000        # total number of episodes

# worker class that inits own environment, trains on it and updloads weights to global net
class Worker(object):
    def __init__(self, name, globalAC, sess, env):            
        self.env = env._reset()  # make environment for each worker
        self.name = name
        self.AC = A3C_network.ACNet(name, sess, env, globalAC) # create ACNet for each worker
        self.sess=sess
        #self.coord = coord


    def work(self, *tup):
        (coord, number) = tup
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not coord.should_stop() and config.global_episodes < MAX_GLOBAL_EP:
            print('\n\n\n %s \n\n\n', self.name)
            s = self.env._reset()

            #self.env._reset_scene()
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):

                bodies_i = env.movable_bodies() 

                if self.name == 'W_0':
                    pose_logger = Plogger()
                    pose_logger.end()
                    pose_logger.start()
                    #self.env.render()
                a = self.AC.choose_action(s)         # estimate stochastic action based on policy 
                s_, r, done, info = self.env.step(a) # make step in environment
                bodies_f = env.movable_bodies() 

                r += forward_r('mlruns', bodies_i, bodies_f, a)
                #TODO:learn how to normalize the clearing reward
                r = r/2

                pose_logger.log(info, env.movable_bodies, a)
                #logger.info(
                #'Episode %d finished in %.2f sec. '
                #'In average each episode takes %.2f sec',
                #episode_index, toc - tic, total_time / (episode_index + 1))
                done = True if ep_t == MAX_EP_STEP - 1 else False

                ep_r += r
                logger.info('Worker %s Episode reward: %f', self.name[-1], ep_r) 
                # save actions, states and rewards in buffer
                buffer_s.append(s)          
                buffer_a.append(a)
                buffer_r.append(r) 
                #buffer_r.append((r+8)/8)    # normalize reward
                #TODO:figure out reward and change this

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        pose_logger.end()
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict) # actual training step, update global ACNet
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global() # get global parameters to local ACNet

                s = s_
                total_step += 1
                if done:
                    if len(config.global_rewards) < 5:  # record running episode reward
                        config.global_rewards.append(ep_r)
                    else:
                        config.global_rewards.append(ep_r)
                        config.global_rewards[-1] =(np.mean(config.global_rewards[-5:])) # smoothing 
                    print(
                        self.name,
                        "Ep:", global_episodes,
                        "| Ep_r: %i" % config.global_rewards[-1],
                          )
                    global_episodes += 1
                    break



#TODO:!
