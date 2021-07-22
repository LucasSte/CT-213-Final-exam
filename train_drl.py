import os
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
batch_size = 512  # batch size used for the experience replay
return_history = []

for episodes in range(1, NUM_EPISODES + 1):
    state = env.reset()
    cumulative_reward = 0.0
    for time in range(1, 5000):
        if RENDER:
            env.render()  # Render the environment for visualization
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward_engineering_space_invaders(state, action, reward, next_state, done, agent.agentType)
        agent.append_experience(state, action, reward, next_state, done)
        state = next_state
        cumulative_reward = agent.gamma * cumulative_reward + reward
        if done:
            print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}"
                  .format(episodes, NUM_EPISODES, time, cumulative_reward, agent.epsilon))
            break
        if len(agent.replay_buffer) > 2 * batch_size and time % 100 == 0:
            loss = agent.replay(batch_size)
    return_history.append(cumulative_reward)
    agent.update_epsilon()
    if episodes % 5 == 0:
        # plt.plot(return_history, 'b')
        # plt.xlabel('Episode')
        # plt.ylabel('Return')
        # plt.show(block=False)
        # plt.pause(0.1)
        # plt.savefig('dqn_training.' + fig_format, format=fig_format)
        # Saving the model to disk
        agent.save(agent.__class__.__name__, "space_invaders.h5")
