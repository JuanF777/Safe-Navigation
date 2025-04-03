import matplotlib.pyplot as plt
import json

def plot_metrics(metrics_dqn, metrics_ddpg):
    labels = ["Collisions", "Jerkiness", "Total Reward"]
    dqn_values = [metrics_dqn["Total Collisions"], metrics_dqn["Average Jerkiness"], metrics_dqn["Total Episode Reward"]]
    ddpg_values = [metrics_ddpg["Total Collisions"], metrics_ddpg["Average Jerkiness"], metrics_ddpg["Total Episode Reward"]]

    x = range(len(labels))
    plt.bar(x, dqn_values, width=0.4, label="DQN", align='center')
    plt.bar([p + 0.4 for p in x], ddpg_values, width=0.4, label="DDPG", align='center')

    plt.xticks([p + 0.2 for p in x], labels)
    plt.ylabel("Metric Value")
    plt.title("DQN vs. DDPG Performance")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    with open("metrics_dqn.json") as f:
        metrics_dqn = json.load(f)

    with open("metrics_ddpg.json") as f:
        metrics_ddpg = json.load(f)

    plot_metrics(metrics_dqn, metrics_ddpg)
