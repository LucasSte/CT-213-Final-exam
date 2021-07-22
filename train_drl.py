import os
import gym
from dqn_agent import ImageAgent, RAMAgent
import numpy as np
from utils import create_env_agent, reward_engineering_space_invaders
import matplotlib.pyplot as plt

NUM_EPISODES = 500
RENDER = False  # please change to false after
#  fig_format = 'png'
fig_format = 'eps'

# Comment this line to enable training using your GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Initiating the Space Invaders environment
env, agent = create_env_agent('SpaceInvaders-ram-v0')
# env, agent = create_env_agent('SpaceInvaders-v0')



# Checking if weights from previous learning session exists
if os.path.exists(agent.__class__.__name__ + 'space_invaders.h5'):
    print('Loading weights from previous learning session.')
    agent.load(agent.__class__.__name__, "space_invaders.h5")
else:
    print('No weights found from previous learning session.')
done = False
batch_size = 16  # batch size used for the experience replay
return_history = []

for episodes in range(1, NUM_EPISODES + 1):
    # Reset the environment
    state = env.reset()
    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = 0.0
    for time in range(1, 5000):
        if RENDER:
            env.render()  # Render the environment for visualization
        # Select action
        action = agent.act(state)
        # Take action, observe reward and new state
        next_state, reward, done, _ = env.step(action)
        # Reshaping to keep compatibility with Keras
        # Making reward engineering to allow faster training
        reward = reward_engineering_space_invaders(state, action, reward, next_state[0], done)
        # Appending this experience to the experience replay buffer
        agent.append_experience(state, action, reward, next_state, done)
        state = next_state
        # Accumulate reward
        cumulative_reward = agent.gamma * cumulative_reward + reward
        if done:
            print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}"
                  .format(episodes, NUM_EPISODES, time, cumulative_reward, agent.epsilon))
            break
        # We only update the policy if we already have enough experience in memory
        if len(agent.replay_buffer) > 2 * batch_size:
            loss = agent.replay(batch_size)
        print('Time: ', time, ' Episode: ', episodes)
    return_history.append(cumulative_reward)
    agent.update_epsilon()
    # Every 10 episodes, update the plot for training monitoring
    if episodes % 5 == 0:
        plt.plot(return_history, 'b')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.show(block=False)
        plt.pause(0.1)
        plt.savefig('dqn_training.' + fig_format, format=fig_format)
        # Saving the model to disk
        agent.save(agent.__class__.__name__, "space_invaders.h5")
