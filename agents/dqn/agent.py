# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from agents.agent import AbstractAgent
from agents.common.input import observation_dim
from agents.common.replay_buffer import ReplayBuffer
import logging
import config

logger = logging.getLogger('rl.agent')
FLAGS = config.flags.FLAGS


class Agent(AbstractAgent):

    def __init__(self, env):
        super(Agent, self).__init__(env)
        logger.info("DQN Agent")

        self.action_dim = env.action_space.n
        self.obs_dim = observation_dim(env.observation_space)
        self.model = self.set_model()
        self.replay_buffer = ReplayBuffer() 

        self.train_step = FLAGS.train_step
        self.test_step = FLAGS.test_step

    def set_model(self):
        model = None
        # model can be q-table or q-network
        return model

    def learn(self):
        logger.debug("Start Learn")

        global_step = 0
        episode_num = 0

        while global_step < self.train_step:

            episode_num += 1
            step_in_ep = 0

            obs = self.env.reset()  # Reset environment
            total_reward = 0
            done = False

            while not done:

                global_step += 1
                step_in_ep += 1

                action = self.get_action(obs, global_step)

                obs_next, reward, done, _ = self.env.step(action)

                self.train_agent(obs, action, reward, obs_next, done)

                if FLAGS.gui:
                    self.env.render()

                obs = obs_next
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
            done = False

            while not done:

                global_step += 1
                step_in_ep += 1

                action = self.get_action(obs, global_step, False)

                obs_next, reward, done, _ = self.env.step(action)

                if FLAGS.gui:
                    self.env.render()

                obs = obs_next
                total_reward += reward

        print("[ train_ep: {}, total reward: {} ]".format(episode_num, total_reward))

    def get_action(self, obs, global_step, train=True):
        # 최적의 액션 선택 + Exploration (Epsilon greedy)   
        return self.env.action_space.sample()

    def train_agent(self, obs, action, reward, obs_next, done):
        """
        How to use replay buffer 

        1. Put sample: make tuple and put it to replay buffer #
         - self.replay_buffer.add_to_memory((s, a, r, s_, done))
        2. Get sample
         - minibatch = self.replay_buffer.sample_from_memory()
        """
        return None
