# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import random
import time

import gym
import gym_maze  # This is for q-learning experiment 

import agents
import config

FLAGS = config.flags.FLAGS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    return None


if __name__ == '__main__':
    set_seed(FLAGS.seed)

    # Load environment
    print('Environment: {}'.format(FLAGS.env))
    env = gym.make(FLAGS.env)

    # Load agent
    print('Agent: {}'.format(FLAGS.agent))
    agent = agents.load(FLAGS.agent+"/agent.py").Agent(env)

    # start learning
    agent.learn()
    agent.test()
