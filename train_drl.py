import os
import gym

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


env = gym.make('SpaceInvaders-ram-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n