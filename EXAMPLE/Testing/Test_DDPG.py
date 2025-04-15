import torch
from rl_agent.ddpg_agent import DDPGAgent
from environment.spawn_town import spawn_town
from evaluate import MetricsLogger

def test_ddpg():
    state_dim = 4  # Example dimensions
    action_dim = 2  # Steering, throttle
    model_path_actor = "ddpg_actor.pth"
    model_path_critic = "ddpg_critic.pth"
    agent = DDPGAgent(state_dim, action_dim, 1.0)
    agent.actor.load_state_dict(torch.load(model_path_actor))
    agent.critic.load_state_dict(torch.load(model_path_critic))
    agent.actor.eval()
    agent.critic.eval()

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
    print(f"DDPG Test Results: {metrics}")

if __name__ == "__main__":
    test_ddpg()
