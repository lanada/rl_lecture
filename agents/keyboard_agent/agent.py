# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from agents.agent import AbstractAgent
import logging
import config

logger = logging.getLogger('rl.agent')
FLAGS = config.flags.FLAGS


class Agent(AbstractAgent):

    def __init__(self, env):
        super(AbstractAgent, self).__init__(env)
        logger.info("Keyboard Agent")

        self.action_dim = env.action_space.n
        self.train_step = FLAGS.train_step
        self.test_step = FLAGS.test_step

    def learn(self):
        logger.debug("No Learning")

    def test(self, global_step=0):
        logger.debug("Test step: {}".format(global_step))

        global_step = 0
        total_reward = 0
        episode_num = 0

        while global_step < self.test_step:

            episode_num += 1

            self.env.reset()  # Reset environment
            done = False

            while not done:

                action = self.get_action()

                obs_next, reward, done, info = self.env.step(action)

                if FLAGS.gui:
                    self.env.render()

                total_reward += reward

        print("[train_ep: {}, total reward: {}]".format(episode_num, total_reward))

    def get_action(self):
        # 키보드 인풋 받을 것
        return self.env.action_space.sample()
