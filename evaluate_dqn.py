import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from network import AgentDoubleDQN
import tensorflow as tf


RENDER = False
NUM_EPISODES = 300
fig_format = 'eps'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.compat.v1.disable_eager_execution()

env = gym.make('SpaceInvaders-ram-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 512  # batch size used for the experience replay
agent = AgentDoubleDQN(state_size, action_size, batch_size=batch_size)


# Checking if weights from previous learning session exists
if os.path.exists(agent.__class__.__name__ + 'space_invaders.h5'):
    print('Loading weights from previous learning session.')
    agent.load(agent.__class__.__name__, "space_invaders.h5")
else:
    print('No weights found from previous learning session. Unable to proceed.')
    exit(-1)
return_history = []
time_history = []
reward_history = []
for episodes in range(1, NUM_EPISODES + 1):
    state = env.reset()
    cumulative_reward = 0.0
    rewards = []
    for time in range(1, 5000):
        if RENDER:
            env.render()
        action = agent.get_greedy_action(state)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)

        state = next_state

        cumulative_reward = agent.gamma * cumulative_reward + reward
        if done:
            rw = np.sum(np.array(rewards))
            reward_history.append(rw)
            time_history.append(time)
            print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}, total reward: {}"
                  .format(episodes, NUM_EPISODES, time, cumulative_reward, agent.epsilon, rw))
            break
    return_history.append(cumulative_reward)

# Prints mean return
print('Mean return: ', np.mean(return_history))
print('Mean total reward: ', np.mean(reward_history))
print('Mean total time: ', np.mean(time_history))

# Plots return history
plt.plot(return_history, 'b')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.savefig('dqn_evaluation_cumulative_rw.' + fig_format, format=fig_format)
plt.show()

plt.plot(reward_history, 'b')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.savefig('dqn_evaluation_total_rw.' + fig_format, format=fig_format)
plt.show()

