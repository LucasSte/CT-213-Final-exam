import os
import gym
from dqn_agent import ImageAgent, RAMAgent

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# env = gym.make('SpaceInvaders-ram-v0')
env = gym.make('SpaceInvaders-v0')
state_size = env.observation_space.shape[0]
action_size =env.action_space.n

agent = ImageAgent(state_size, action_size)
# agent = RAMAgent(state_size, action_size)

agent.make_model()
