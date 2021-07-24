import os
import numpy as np
import gym
import matplotlib.pyplot as plt
from network_img import DeepQnetworkImg
import tensorflow as tf
# Comment this line to enable training using your GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
  except RuntimeError as e:
    print(e)

NUM_EPISODES = 50
RENDER = False  # please change to false after
#  fig_format = 'png'
fig_format = 'eps'



# Initiating the Space Invaders environment
# env, agent = create_env_agent('SpaceInvaders-ram-v0')

env = gym.make('SpaceInvaders-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 64  # batch size used for the experience replay
agent = DeepQnetworkImg(state_size, action_size, 'ddqn.h5', batch_size=batch_size)


# Checking if weights from previous learning session exists
# if os.path.exists(agent.__class__.__name__ + 'space_invaders.h5'):
#     print('Loading weights from previous learning session.')
#     agent.load(agent.__class__.__name__, "space_invaders.h5")
# else:
#     print('No weights found from previous learning session.')
done = False
return_history = []

for episodes in range(1, NUM_EPISODES + 1):
    state = env.reset()
    cumulative_reward = 0.0
    for time in range(1, 5000):
        if RENDER:
            env.render()  # Render the environment for visualization
        #state = np.reshape(state, (1, state_size))
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        reward = np.clip(reward, -1, 1)
        #next_state = np.reshape(next_state, (1, state_size))
        # reward = reward_engineering_space_invaders(state, action, reward, next_state[0], done)
        agent.add_memory(state, action, next_state, reward, done)
        state = next_state
        cumulative_reward = agent.gamma * cumulative_reward + reward
        if done:
            print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}"
                  .format(episodes, NUM_EPISODES, time, cumulative_reward, agent.epsilon))
            break
        if len(agent.previous_memory) > 2 * batch_size and time % 300 == 0:
            loss = agent.train()
        if time > 0 and time % 400 == 0:
            agent.update_prediction_network()
    return_history.append(cumulative_reward)
    agent.update_epsilon()
    if episodes % 5 == 0:
        plt.plot(return_history, 'b')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        # plt.show(block=False)
        # plt.pause(0.1)
        plt.savefig('dqn_training.' + fig_format, format=fig_format)
        # Saving the model to disk
        agent.save(agent.__class__.__name__, "space_invaders.h5")

