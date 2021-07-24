import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from network_img import DeepQnetworkImg
import tensorflow as tf


def plot_points(point_list, style):
    x = []
    y = []
    for point in point_list:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, style)


RENDER = False
NUM_EPISODES = 20  # Number of episodes used for evaluation
# fig_format = 'png'  # Format used for saving matplotlib's figures
fig_format = 'eps'

# Comment this line to enable training using your GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.compat.v1.disable_eager_execution()

# Initiating the Mountain Car environment
env = gym.make('SpaceInvaders-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 512  # batch size used for the experience replay
agent = DeepQnetworkImg(state_size, action_size, 'ddqn.h5', batch_size=batch_size)


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
    rewards = []
    for time in range(1, 5000):
        # Render the environment for visualization
        if RENDER:
            env.render()
        #state = np.reshape(state, (1, state_size))
        action = agent.get_greedy_action(state)
        # Take action, observe reward and new state
        #print(action)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        #reward = np.clip(reward, -1, 1)
        # Reshaping to keep compatibility with Keras
        # Making reward engineering to keep compatibility with how training was done
        # reward = reward_engineering_space_invaders(state[0], action, reward, next_state[0], done)
        state = next_state
        # Accumulate reward
        cumulative_reward = agent.gamma * cumulative_reward + reward
        if done:
            print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}, total reward: {}"
                  .format(episodes, NUM_EPISODES, time, cumulative_reward, agent.epsilon, np.sum(np.array(rewards))))
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

