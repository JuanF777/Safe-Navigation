import torch
from rl_agent.dqn_agent import DQNAgent
from environment.spawn_town import spawn_town
from evaluate import MetricsLogger

def test_dqn():
    state_dim = 4  # Example dimensions
    action_dim = 3  # Left, right, straight
    model_path = "dqn_model.pth"
    agent = DQNAgent(state_dim, action_dim)
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()

    logger = MetricsLogger()

    for episode in range(100):
        state = np.zeros(state_dim)
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = spawn_town(state, action)
            logger.log_reward(reward)
            if collision_detected():
                logger.log_collision()
            state = next_state

    metrics = logger.get_results()
    print(f"DQN Test Results: {metrics}")

if __name__ == "__main__":
    test_dqn()
