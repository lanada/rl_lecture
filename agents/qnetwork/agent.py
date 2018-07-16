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
import sys

from agents.qnetwork.NN import Q_Network
from agents.common.replay_buffer import ReplayBuffer
import tensorflow as tf

logger = logging.getLogger('rl.agent')
FLAGS = config.flags.FLAGS

minibatch_size = 32
pre_train_step = 10
target_update_period = 1000

class Agent(AbstractAgent):

    def __init__(self, env):
        super(Agent, self).__init__(env)
        logger.info("Q-Learning Agent is created")

        self.action_dim = env.action_space.n
        self.obs_dim = np.power(int(env.observation_space.high[0]+1),2)

        # parameter setting 
        self.train_step = FLAGS.train_step  
        self.test_step = FLAGS.test_step

        self.model = self.set_model()
            
        self.replay_buffer = ReplayBuffer(minibatch_size=minibatch_size)
            
    def set_model(self):

        model = Q_Network(self.obs_dim, self.action_dim, self.train_step)
        return model

    def learn(self):
        logger.debug("Start train for {} steps".format(self.train_step))
        global_step = 0
        episode_num = 0

        while global_step < self.train_step:
            episode_num += 1
            step_in_ep = 0
            
            obs_v = self.env.reset() 
            obs = self.one_hot(obs_v)

            total_reward = 0
            done = False

            while (not done and global_step < self.train_step):

                global_step += 1
                step_in_ep += 1

                action = self.get_action(obs, global_step)

                # For debugging
                if global_step % 10000 == 0:
                    self.draw_current_optimal_actions(global_step)

                obs_v_next, reward, done, _ = self.env.step(action)
                obs_next = self.one_hot(obs_v_next)

                self.train_agent(obs, action, reward, obs_next, done, global_step)

                # if FLAGS.gui:
                #     self.env.render()

                obs = obs_next
                total_reward += reward

    def test(self, global_step=0):
        logger.debug("Start test for {} steps".format(self.test_step))

        global_step = 0
        episode_num = 0

        self.draw_current_optimal_actions(0)
        
        while global_step < self.test_step:
            episode_num += 1
            step_in_ep = 0
            total_reward = 0
            done = False

            obs_v = self.env.reset()  # Reset environment
            obs = self.one_hot(obs_v)

            while (not done and global_step < self.test_step):

                global_step += 1
                step_in_ep += 1

                action = self.get_action(obs, global_step, False)

                obs_v_next, reward, done, _ = self.env.step(action)
                obs_next = self.one_hot(obs_v_next)

                if FLAGS.gui:
                    self.env.render()
                    # time.sleep(0.2)

                obs = obs_next
                total_reward += reward

            print("[ test_ep: {}, total reward: {} ]".format(episode_num, total_reward))

    def get_action(self, obs, global_step, train=True):

        eps_min = 0.1
        eps_max = 1.0
        eps_decay_steps = self.train_step
        epsilon = max(eps_min, eps_max - (eps_max - eps_min)*global_step/eps_decay_steps)

        if train and np.random.rand(1) < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.model.get_action(obs)

        return action


    def train_agent(self, obs, action, reward, obs_next, done, global_step):

        self.replay_buffer.add_to_memory((obs, action, reward, obs_next, done))
        
        if len(self.replay_buffer.replay_memory) < minibatch_size * pre_train_step:
            return None

        minibatch = self.replay_buffer.sample_from_memory()
        s, a, r, s_, done = map(np.array, zip(*minibatch))
        self.model.train_network(s, a, r, s_, done)

        if global_step % target_update_period == 0:
            self.model.copy_to_target()

        return

    def one_hot(self, obs):
        idx = int(obs[1]*(self.env.observation_space.high[0]+1) + obs[0])
        return np.eye(int(pow(self.env.observation_space.high[0]+1, 2)))[idx]

    def draw_current_optimal_actions(self, step):
        idx = int(np.sqrt(self.obs_dim))
        directions = ["U", "D", "R", "L"]
        print("optimal actions at step {}".format(step))
        for i in range(idx):
            print("----"*idx+"-")
            row = ""
            for j in range(idx):
                row = row + "| {} ".format(directions[self.model.get_action(np.eye(self.obs_dim)[int(idx*i+j)])]) # one-hot
                # row = row + "| {} ".format(directions[self.model.act([j,i], 0, False)]) # tuple
            row = row + "|"
            print(row)
        print("----"*idx+"-")
        return
    

