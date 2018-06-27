# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import tensorflow as tf
import numpy as np
import random
import logging
import time

import gym
import agents
import config
import logger

FLAGS = config.flags.FLAGS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    return None


if __name__ == '__main__':
    
    logger = logging.getLogger('rl')
    set_seed(FLAGS.seed)

    # Load environment
    logger.info('Environment: {}'.format(FLAGS.env))
    env = gym.make(FLAGS.env)

    # Load agent
    logger.info('Agent: {}'.format(FLAGS.agent))
    agent = agents.load(FLAGS.agent+"/agent.py").Agent(env)

    # start learning
    if FLAGS.train:
        start_time = time.time()
        agent.learn()
        finish_time = time.time()
        print("TRAINING TIME (sec)", finish_time - start_time)

    agent.test()
