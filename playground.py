import gym
env = gym.make('SpaceInvaders-ram-v0')
print(env.action_space)
print(env.observation_space)
for i_episode in range(1):
    observation = env.reset()
    for t in range(1000):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
print(env.observation_space)
