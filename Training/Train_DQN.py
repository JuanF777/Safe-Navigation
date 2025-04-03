import torch
from rl_agent.dqn_agent import DQNAgent
from environment.spawn_town import spawn_town
import numpy as np

def train_dqn():
    state_dim = 4  # Example dimensions
    action_dim = 3  # Left, right, straight
    agent = DQNAgent(state_dim, action_dim)

    for episode in range(1000):  # Number of episodes
        state = np.zeros(state_dim)  # Initialize state
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = spawn_town(state, action)  # Get next state, reward from environment
            agent.store_experience(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
        agent.update_target()

if __name__ == "__main__":
    train_dqn()
