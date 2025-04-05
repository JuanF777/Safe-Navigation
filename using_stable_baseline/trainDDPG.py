import sys
import os
import math
import random
import time
import numpy as np
import cv2


# Get the directory of the current script
script_dir = os.getcwd()

# Add the .egg file to Python's path
sys.path.append(os.path.join(script_dir, "carla-0.9.10-py3.7-linux-x86_64.egg"))

import carla
import gym
from gym import spaces

class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()

        # Connect to CARLA
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        # self.world = self.client.get_world()
        try:
            # self.world = self.client.get_world()
            self.world = self.client.load_world('Town02')
            print("Connected to CARLA successfully!")
        except Exception as e:
            print("Error connecting to CARLA:", e)

        # Define Action & Observation Spaces
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)  # Steering, Throttle
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)  # Velocity, position, etc.

        self.vehicle = None
        self.reset()

    def reset(self):        
        # Destroy old vehicles
        if self.vehicle is not None:
            self.vehicle.destroy()

        # Spawn a new vehicle
        blueprint = self.world.get_blueprint_library().filter("model3")[0]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(blueprint, spawn_point)

        # Return initial state
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def step(self, action):
        # Apply control
        # steer, throttle = action # won't work
        steer = float(action[0])
        throttle = float(action[1])
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))

        # Compute reward
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)
        reward = speed  # Encourage speed
        done = False  # Define termination condition

        # Get next state
        next_state = np.array([speed, 0.0, 0.0, 0.0], dtype=np.float32)
        return next_state, reward, done, {}

    def close(self):
        if self.vehicle is not None:
            self.vehicle.destroy()

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

env = CarlaEnv()

# Define action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Initialize DDPG agent
model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=2)

# Train the agent
print("Starting DDPG training...")
model.learn(total_timesteps=10000, log_interval=10, progress_bar=True, reset_num_timesteps=True)
print("Training completed!")

# Save the model
model.save("ddpg_carla")