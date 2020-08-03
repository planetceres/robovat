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
from robovat.agent import worker as w
from robovat.agent import run
from robovat.agent.constants import constants


#from tools.pose_log import log_pose


