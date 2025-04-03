import torch
from rl_agent.ddpg_agent import DDPGAgent
from environment.spawn_town import spawn_town
import numpy as np

def train_ddpg():
    state_dim = 4  # Example dimensions
    action_dim = 2  # Steering, throttle
    max_action = 1.0
    agent = DDPGAgent(state_dim, action_dim, max_action)

    for episode in range(1000):  # Number of episodes
        state = np.zeros(state_dim)  # Initialize state
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = spawn_town(state, action)  # Get next state, reward from environment
            agent.store_experience(state, action, reward, next_state, done)
            agent.learn()
            state = next_state

if __name__ == "__main__":
    train_ddpg()
