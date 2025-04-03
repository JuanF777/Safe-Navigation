class MetricsLogger:
    def __init__(self):
        self.rewards = []
        self.collisions = 0

    def log_reward(self, reward):
        self.rewards.append(reward)

    def log_collision(self):
        self.collisions += 1

    def get_results(self):
        avg_reward = np.mean(self.rewards)
        return {
            "average_reward": avg_reward,
            "collision_rate": self.collisions / len(self.rewards)
        }
