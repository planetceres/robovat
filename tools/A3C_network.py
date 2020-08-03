#!/usr/bin/env python

"""
Network for the Actor Critic
"""

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import matplotlib.pyplot as plt

from robovat import envs
from robovat import policies
from robovat import policies
import A3C_worker as w
#from robovat.agent import run
#from robovat.agent.constants import constants


#from tools.pose_log import log_pose

class ACNet(object):
    def __init__(self, scope, sess, globalAC=None):       
        self.sess=sess
        self.actor_optimizer = tf.compat.v1.train.RMSPropOptimizer(LR_A, name='RMSPropA')  # optimizer for the actor
        self.critic_optimizer = tf.compat.v1.train.RMSPropOptimizer(LR_C, name='RMSPropC') # optimizer for the critic
        self.env = env
        self.N_A = 4
        #env.max_movable_bodies
        self.N_S = 1
        self.A_BOUND = [env.action_space.low, env.action_space.high] # action bounds
        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.compat.v1.variable_scope(scope):
                tf.compat.v1.disable_eager_execution()
                self.s = tf.compat.v1.placeholder(tf.float32, [None, self.N_S], 'S')       # state
                self.a_params, self.c_params = self._build_net(scope)[-2:]  # parameters of actor and critic net
        else:   # local net, calculate losses
            with tf.compat.v1.variable_scope(scope):
                self.s = tf.compat.v1.placeholder(tf.float32, [None, self.N_S], 'S')            # state
                self.a_his = tf.compat.v1.placeholder(tf.float32, [None, self.N_A], 'A')        # action
                self.v_target = tf.compat.v1.placeholder(tf.float32, [None, 1], 'Vtarget') # v_target value

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope) # get mu and sigma of estimated action from neural net

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * self.A_BOUND[1], sigma + 1e-4

                normal_dist = tf1.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), self.A_BOUND[0], self.A_BOUND[1]) # sample a action from distribution
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params) #calculate gradients for the network weights
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'): # update local and global network weights
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = self.actor_optimizer.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.critic_optimizer.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope): # neural network structure of the actor and critic
        w_init = tf.random_normal_initializer(0., .1)
        with tf.compat.v1.variable_scope('actor'):
            l_a = tf.compat.v1.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.compat.v1.layers.dense(l_a, self.N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu') # estimated action value
            sigma = tf.compat.v1.layers.dense(l_a, self.N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma') # estimated variance
        with tf.compat.v1.variable_scope('critic'):
            l_c = tf.compat.v1.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.compat.v1.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # estimated value for state
        a_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return self.sess.run(self.A, {self.s: s})[0]

