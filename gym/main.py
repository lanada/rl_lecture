import gym

env = gym.make('Pendulum-v0')
env.reset()

for _ in range(1000):
    
    env.step(env.action_space.sample()) # take a random action
    env.render()