# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from agents.agent import AbstractAgent
import logging

logger = logging.getLogger('rl.agent')

class Agent(AbstractAgent):

    def __init__(self, env):
        logger.info("Agent")
        self.env = env
        self.action_dim = env.action_space.n
        logger.debug('action dim: {}'.format(self.action_dim))

    def learn(self):
        logger.debug("Learn")
        print(self.env.step)
        return None
    
    def test(self):
        logger.debug("Test")
        return None