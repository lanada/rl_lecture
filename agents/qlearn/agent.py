# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
from agents.agent import AbstractAgent
from agents.common.input import observation_dim
import logging
import config

logger = logging.getLogger('rl.agent')
FLAGS = config.flags.FLAGS


class Agent(AbstractAgent):

    def __init__(self, env):
        super(Agent, self).__init__(env)
        logger.info("Q-Learning Agent")

        self.action_dim = env.action_space.n
        self.obs_dim = np.power(int(env.observation_space.high[0]+1),2)
        self.model = self.set_model()
        self.train_step = FLAGS.train_step
        self.test_step = FLAGS.test_step

    def set_model(self):
        # model = None
        # model can be q-table or q-network

        # Initialize Q table with all zeros
        Q = np.zeros([self.obs_dim, self.action_dim])
        # parameter setting 
        self.gamma = .99     # discount factor
        self.lr = 0.1        # learning rate
        
        return Q

    def learn(self):
        logger.debug("Start Learn")

        global_step = 0
        episode_num = 0

        while global_step < self.train_step:

            episode_num += 1
            step_in_ep = 0

            obs = self.env.reset()  # Reset environment
            obs_flatten = int(obs[1]*(self.env.observation_space.high[0]+1) + obs[0])  # state [2,3] --> state_flatten 17 == 2+5*3
            total_reward = 0
            done = False

            while not done:

                global_step += 1
                step_in_ep += 1

                action = self.get_action(obs_flatten, global_step)

                obs_next, reward, done, _ = self.env.step(action)
                obs_next_flatten = int(obs_next[1]*(self.env.observation_space.high[0]+1) + obs_next[0])
                # print(obs_next_flatten, obs_next)
                self.train_agent(obs_flatten, action, reward, obs_next_flatten, done)

                if FLAGS.gui:
                    self.env.render()

                obs_flatten = obs_next_flatten
                total_reward += reward

    def test(self, global_step=0):
        logger.debug("Test step: {}".format(global_step))

        global_step = 0
        episode_num = 0
        total_reward = 0

        while global_step < self.test_step:
            episode_num += 1
            step_in_ep = 0

            obs = self.env.reset()  # Reset environment
            obs_flatten = int(obs[1]*(self.env.observation_space.high[0]+1) + obs[0])  # state [2,3] --> state_flatten 17 == 2+5*3

            done = False

            while not done:

                global_step += 1
                step_in_ep += 1

                action = self.get_action(obs_flatten, global_step, False)

                obs_next, reward, done, _ = self.env.step(action)
                obs_next_flatten = int(obs_next[1]*(self.env.observation_space.high[0]+1) + obs_next[0])

                if FLAGS.gui:
                    self.env.render()

                obs_flatten = obs_next_flatten
                total_reward += reward

        print("[ train_ep: {}, total reward: {} ]".format(episode_num, total_reward))

    def get_action(self, obs, global_step, train=True):
        # 최적의 액션 선택 + Exploration (Epsilon greedy)  

        epsilon = 1. / ((global_step // 10) + 1)
        if np.random.rand(1) < epsilon:
            action = self.env.action_space.sample()
        else:
            action =int(np.argmax(self.model[obs, :]))
        return action

    def train_agent(self, obs, action, reward, obs_next, done):
         # Update new state and reward from environment
        # print(obs, action, reward,obs_next,done )
        self.model[obs, action] = self.model[obs, action] + self.lr*(reward + self.gamma*np.max(self.model[obs_next,:]) -self.model[obs, action])
        # print(self.model)
        return None
