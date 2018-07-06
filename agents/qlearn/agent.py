# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
from agents.agent import AbstractAgent
from agents.common.input import observation_dim
import logging
import config
import time

logger = logging.getLogger('rl.agent')
FLAGS = config.flags.FLAGS


class Agent(AbstractAgent):

    def __init__(self, env):
        super(Agent, self).__init__(env)
        logger.info("Q-Learning Agent is created")

        self.action_dim = env.action_space.n
        self.obs_dim = np.power(int(env.observation_space.high[0]+1),2)

        # Make Q-table
        self.q_table = np.zeros([self.obs_dim, self.action_dim])

        # parameter setting 
        self.gamma = .99     # discount factor  # TODO: 이거 위에 따로 글로벌로 뺄지 말지 결정 필요
        self.lr = 0.1        # learning rate
        self.train_step = FLAGS.train_step  # It should be larger than 5000
        self.test_step = FLAGS.test_step

    def learn(self):
        logger.debug("Start train for {} steps".format(self.train_step))
        global_step = 0

        while global_step < self.train_step:
            obs = self.env.reset()  # Reset environment
            obs_flatten = self.flatten_obs(obs)
            
            total_reward = 0
            done = False

            while (not done and global_step < self.train_step):
                global_step += 1

                action = self.get_action(obs_flatten, global_step)

                obs_next, reward, done, _ = self.env.step(action)
                obs_next_flatten = self.flatten_obs(obs_next)

                self.train_agent(obs_flatten, action, reward, obs_next_flatten, done)

                if FLAGS.gui:
                    self.env.render()

                obs_flatten = obs_next_flatten
                total_reward += reward

    def test(self, global_step=0):
        logger.debug("Start test for {} steps".format(self.test_step))

        global_step = 0
        episode_num = 0

        while global_step < self.test_step:
            episode_num += 1
            step_in_ep = 0
            total_reward = 0
            done = False

            obs = self.env.reset()  # Reset environment
            obs_flatten = self.flatten_obs(obs)

            while (not done and global_step < self.test_step):
                global_step += 1
                step_in_ep += 1

                action = self.get_action(obs_flatten, global_step, False)

                obs_next, reward, done, _ = self.env.step(action)
                obs_next_flatten =  self.flatten_obs(obs_next)

                if FLAGS.gui:
                    time.sleep(0.05)
                    self.env.render()

                obs_flatten = obs_next_flatten
                total_reward += reward

            print("[ test_ep: {}, total reward: {} ]".format(episode_num, total_reward))

    def get_action(self, obs, global_step, train=True):

        epsilon = 1. / ((global_step // 10) + 1)
        if train and np.random.rand(1) < epsilon:
            action = self.env.action_space.sample()
        else:
            action =int(np.argmax(self.q_table[obs, :]))
        return action

    def train_agent(self, obs, action, reward, obs_next, done):
        # Update new state and reward from environment
        self.q_table[obs, action] = self.q_table[obs, action] + self.lr*(reward + self.gamma*np.max(self.q_table[obs_next,:]) -self.q_table[obs, action])

        return None

    def flatten_obs(self, obs):
        # Flatten obs: 
        # obs [2,3] --> state_flatten 17 (= 2+5*3) where size of map is 5 by 5
        ret = int(obs[1]*(self.env.observation_space.high[0]+1) + obs[0])  
        return ret
