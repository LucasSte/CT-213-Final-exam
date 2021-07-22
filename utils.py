import gym
from dqn_agent import RAMAgent, ImageAgent

def reward_engineering_space_invaders(state, action, reward, next_state, done, agentType):
    """cada inimigo a menos contabiliza 1pt. Cada vida perdida cotabiliza -10pts"""
    if agentType == "RAM":
        print("RAm")
        return reward + (36 - state[17]) + (3 - state[73]) * 10
    print("Img")
    return reward

def create_env_agent(env_type: str):
    env = gym.make(env_type)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    if env_type == 'SpaceInvaders-ram-v0':
        return env, RAMAgent(state_size, action_size)
    else:
        return env, ImageAgent(state_size, action_size)
