
def reward_engineering_space_invaders(state, reward, next_state):
    """cada inimigo a menos contabiliza 1pt. Cada vida perdida cotabiliza -10pts"""
    if next_state[73] < state[73]:  # caso agente perder uma vida
        return reward - 50
    if next_state[17] == 0:  # Caso nao haja mais inimigos (passou de fase)
        return reward + 100
    if next_state[17] < state[17]:  # Caso matou um inimigo
        return reward + 5
    return reward - 1  # Se não acontecer nada -> penalidade por tempo