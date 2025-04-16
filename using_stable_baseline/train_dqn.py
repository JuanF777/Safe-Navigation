# train_dqn.py
from stable_baselines3 import DQN
from carlatest import CarlaEnv

env = CarlaEnv()
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("dqn")
