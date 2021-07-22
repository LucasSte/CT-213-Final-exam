import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent, RAMAgent
from utils import create_env_agent
import tensorflow as tf


def plot_points(point_list, style):
    x = []
    y = []
    for point in point_list:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, style)


RENDER = True
NUM_EPISODES = 20  # Number of episodes used for evaluation
# fig_format = 'png'  # Format used for saving matplotlib's figures
fig_format = 'eps'

# Comment this line to enable training using your GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.compat.v1.disable_eager_execution()

# Initiating the Mountain Car environment
env, agent = create_env_agent('SpaceInvaders-ram-v0')
# env, agent = create_env_agent('SpaceInvaders-v0')


# Checking if weights from previous learning session exists
if os.path.exists(agent.__class__.__name__ + 'space_invaders.h5'):
    print('Loading weights from previous learning session.')
    agent.load(agent.__class__.__name__, "space_invaders.h5")
else:
    print('No weights found from previous learning session. Unable to proceed.')
    exit(-1)
return_history = []

for episodes in range(1, NUM_EPISODES + 1):
    # Reset the environment
    state = env.reset()
    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = 0.0
    for time in range(1, 5000):
        # Render the environment for visualization
        if RENDER:
            env.render()
        # Select action
        action = agent.act(state)
        # Take action, observe reward and new state
        next_state, reward, done, _ = env.step(action)
        # Reshaping to keep compatibility with Keras
        # Making reward engineering to keep compatibility with how training was done
        # reward = reward_engineering_space_invaders(state[0], action, reward, next_state, done, agent.agentType)
        state = next_state
        # Accumulate reward
        cumulative_reward = agent.gamma * cumulative_reward + reward
        if done:
            print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}"
                  .format(episodes, NUM_EPISODES, time, cumulative_reward, agent.epsilon))
            break
    return_history.append(cumulative_reward)

# Prints mean return
print('Mean return: ', np.mean(return_history))

# Plots return history
plt.plot(return_history, 'b')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.savefig('dqn_evaluation.' + fig_format, format=fig_format)
plt.show()
