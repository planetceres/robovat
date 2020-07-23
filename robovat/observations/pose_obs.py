"""Observation of object pose.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import pybullet as p
import json
from pathlib import Path

from robovat.observations import observation


INF = 2**32 - 1
augmented_obs = True
full_augmented_obs = True
with open("./color_mapping.json", "r") as f: 
    color_map = json.load(f)
with open("./volume_mapping.json", "r") as f: 
    volume_map = json.load(f)
with open("./semantic_mapping.json", "r") as f: 
    semantic_map = json.load(f)

class PoseObs(observation.Observation):
    """Pose observation."""

    def __init__(self,
                 num_bodies,
                 modality='pose',
                 name=None):
        """Initialize."""
        self.name = name or 'pose_obs'
        self.num_bodies = num_bodies
        self.modality = modality
        self.env = None

    def initialize(self, env):
        self.env = env
        assert self.env.simulator is not None

    def on_episode_start(self):
        """Called at the start of each episode."""
        self.bodies = self.env.movable_bodies
        self.full_augmented_state = self.env.config[self.env.grocery_env_name]['sim']['full_state']
        if augmented_obs:
            self.mass_by_id = self.env.mass_by_id

    def get_gym_space(self):
        """Returns gym space of this observation."""
        _shape = (self.num_bodies,)

        if self.modality == 'pose':
            return gym.spaces.Box(-INF, INF, _shape + (6,), dtype=np.float32)
        elif self.modality == 'position':
            if augmented_obs:
                # if self.env.config[self.grocery_env_name]['sim']['full_state']:
                if full_augmented_obs:
                    #x,y,z,mass,mass_variance,intactness,intactness_variance,semantic_label, semantic_variance
                    return gym.spaces.Box(-INF, INF, _shape + (34,), dtype=np.float32)
                return gym.spaces.Box(-INF, INF, _shape + (4,), dtype=np.float32)
            return gym.spaces.Box(-INF, INF, _shape + (3,), dtype=np.float32)
        elif self.modality == 'yaw_cossin':
            return gym.spaces.Box(-INF, INF, _shape + (2,), dtype=np.float32)
        elif self.modality == 'pose2d':
            return gym.spaces.Box(-INF, INF, _shape + (3,), dtype=np.float32)
        else:
            raise ValueError('Unrecognized modality: %r.' % (self.modality))

    def augment_obj_properties(self, body, pose):
        """ Adds object type, softness (intactness coeff), true mass, 
            mass mean, and mass var to the pose obs state

            Obs: [x, y, z, obj_type, intactness coeff (softness), true mass, true mean, mass var]
        """
        obj_type = self.env.obj_type_by_id[body.uid]
        softness, softness_mu, softness_variance = self.env.softness_by_id[body.uid]
        mass, mass_mu, mass_variance = self.mass_by_id[body.uid]
        bounds = np.reshape(p.getAABB(body.uid), -1)
        texture = p.getVisualShapeData(body.uid)[0][4].decode('utf-8')
        key = str(Path(".").joinpath(*Path(texture).parts[-6:]))
        color = color_map[key]
        volume = volume_map[key]
        meat_fruit = semantic_map[key][0]
        chemical_nonchemical = semantic_map[key][1]
        hot_cold = semantic_map[key][2]

        # Total state (20) 11,12,13,14,16,17,19
        # bounds (6), obj_type (1), color(3), volume(1)
        # meat_fruit (1), chemical_nonchemical(1), hot_cold(1),
        # softness(1), softness_mu(1), softness_variance(1) 
        # mass(1), mass_mu(1), mass_variance(1), 

            # True state (provided to state estimator and actor/value net) (10)
            # bounds (6), color(3), volume(1) 

            # Hidden state (used to supervise state estimator) (10)
            # obj_type (1), meat_fruit (1), chemical_nonchemical(1), hot_cold(1),
            # softness(1), softness_mu(1), softness_variance(1) 
            # mass(1), mass_mu(1), mass_variance(1), 

        # Inferred by state estimator (7)
        # meat_fruit (1), chemical_nonchemical(1), hot_cold(1),
        # softness(1), softness_variance(1), mass(1), 
        # mass_variance(1)

        return np.array([*bounds, color[0], color[1], color[2], volume,
                         obj_type, meat_fruit, chemical_nonchemical, hot_cold,
                         softness, softness_mu, softness_variance,
                         mass, mass_mu, mass_variance, *([0]*7) ])


    def get_observation(self):
        """Returns the observation data of the current step."""
        poses = [self.get_pose(body) for body in self.bodies]
        poses += [self.get_zero_pose()] * (self.num_bodies - len(self.bodies))
        return np.stack(poses)

    def get_pose(self, body):
        """Returns the observation data of the current step."""
        pose = body.pose

        if self.modality == 'pose':
            return body.pose.to_array()
        elif self.modality == 'position':
            if augmented_obs:
                if self.full_augmented_state:
                    return self.augment_obj_properties(body, pose)
                return np.array([pose.position.x, pose.position.y, pose.position.z, self.env.softness_by_id[body.uid]])
            return np.array([pose.position.x, pose.position.y, pose.position.z])
        elif self.modality == 'yaw_cossin':
            return np.array([np.cos(pose.yaw), np.sin(pose.yaw)],
                            dtype=np.float32)
        if self.modality == 'pose2d':
            return np.array([pose.x, pose.y, pose.yaw], dtype=np.float32)
        else:
            raise ValueError('Unrecognized modality: %r.' % (self.modality))

    def get_zero_pose(self):
        """Returns the observation data of the current step."""
        if self.modality == 'pose':
            return np.zeros([6], dtype=np.float32)
        elif self.modality == 'position':
            if augmented_obs:
                return np.zeros([4], dtype=np.float32)
            return np.zeros([3], dtype=np.float32)
        elif self.modality == 'yaw_cossin':
            return np.zeros([2], dtype=np.float32)
        if self.modality == 'pose2d':
            return np.zeros([3], dtype=np.float32)
        else:
            raise ValueError('Unrecognized modality: %r.' % (self.modality))
