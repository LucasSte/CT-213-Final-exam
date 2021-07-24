import os
import numpy as np
import gym
import matplotlib.pyplot as plt
from network import DeepQnetwork
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

NUM_EPISODES = 300
RENDER = False  # please change to false after
fig_format = 'eps'


env = gym.make('SpaceInvaders-ram-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 512  # batch size used for the experience replay
agent = DeepQnetwork(state_size, action_size, 'ddqn.h5', batch_size=batch_size)



done = False
return_history = []
reward_history = []

for episodes in range(1, NUM_EPISODES + 1):
    state = env.reset()
    cumulative_reward = 0.0
    ep_rw = []
    for time in range(1, 5000):
        if RENDER:
            env.render()  # Render the environment for visualization
        state = np.reshape(state, (1, state_size))
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        ep_rw.append(reward)
        next_state = np.reshape(next_state, (1, state_size))
        agent.add_memory(state, action, next_state, reward, done)
        state = next_state
        cumulative_reward = agent.gamma * cumulative_reward + reward
        if done:
            total = np.sum(np.array(ep_rw))
            reward_history.append(total)
            print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}, total reward: {}"
                  .format(episodes, NUM_EPISODES, time, cumulative_reward, agent.epsilon, total))
            break
        if len(agent.previous_memory) > 2 * batch_size and time % 50 == 0:
            loss = agent.train()
    #if time > 0 and time % 100 == 0:
    agent.update_prediction_network()
    return_history.append(cumulative_reward)
    agent.update_epsilon()
    if episodes % 5 == 0:
        plt.figure()
        plt.plot(return_history, 'b')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.savefig('dqn_training_new.' + fig_format, format=fig_format)
        plt.figure()
        plt.plot(reward_history, 'b')
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.savefig('dqn_training_rw_new.' + fig_format, format=fig_format)

        agent.save(agent.__class__.__name__, "new_space_invaders.h5")

