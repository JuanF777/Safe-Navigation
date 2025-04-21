from stable_baselines3 import DDPG
from stable_baselines3.ddpg.policies import MlpPolicy
from carlatest_ddpg import CarlaEnv

from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

env = CarlaEnv()

# Add action noise for better exploration
n_actions = env.action_space.shape[-1]
noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG(
    MlpPolicy,
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=5000,
    batch_size=128,
    tau=0.005,
    gamma=0.99,
    train_freq=(1, "step"),
    gradient_steps=1,
    action_noise=noise,
    tensorboard_log="./ddpg_logs/",
    policy_kwargs=dict(net_arch=[256, 256])
)

model.learn(total_timesteps=200000, tb_log_name="ddpg_carla")
model.save("ddpg")

