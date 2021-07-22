import gym
from dqn_agent import RAMAgent, ImageAgent

def reward_engineering_space_invaders(state, action, reward, next_state, done, agentType):
    """cada inimigo a menos contabiliza 1pt. Cada vida perdida cotabiliza -10pts"""
    if agentType == "RAM":
        if next_state[73] < state[73]:  # caso agente perder uma vida
            return reward - 50
        if state[17] == 0:  # Caso nao ha mais inimigo (passou de fase)
            return reward + 100
        if next_state[17] < state[17]:  # Caso matou um inimigo
            return reward + 5
        return reward - 1  # Se não acontecer nada -> penalidade por tempo
    return reward

def create_env_agent(env_type: str):
    env = gym.make(env_type)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    if env_type == 'SpaceInvaders-ram-v0':
        return env, RAMAgent(state_size, action_size)
    else:
        return env, ImageAgent(state_size, action_size)
