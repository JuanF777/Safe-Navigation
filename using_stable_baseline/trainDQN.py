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
sys.path.append(os.path.join(script_dir, "carla-0.9.10-py3.7-win-amd64.egg"))

import carla
import gym
from gym import spaces
# import gymnasium as gym
# from gym import spaces
import cv2

class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)

        try:
            # self.world = self.client.get_world()
            self.world = self.client.load_world('Town02')
            print("Connected to CARLA successfully!")
        except Exception as e:
            print("Error connecting to CARLA:", e)

        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # [0: Left, 1: Straight, 2: Right]
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        
        self.vehicle = None
        self.camera = None

        self.latest_image = np.zeros((84, 84, 3), dtype=np.uint8)  # Default black image

        self.collision_sensor = None
        self.collision_detected = False  # To track collision state

        # Get map information
        self.map = self.world.get_map()

        # Timer for safe driving
        self.safe_drive_time = 0  # Tracks timesteps spent driving without collision
        self.safe_drive_threshold = 5  # Threshold for rewarding safe driving


    def reset(self):
        """Resets the environment and returns initial observation."""
        self.destroy_actors()
        
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("model3")[0]  
        spawn_points = self.world.get_map().get_spawn_points()
        # self.vehicle = self.world.spawn_actor(vehicle_bp, random.choice(spawn_points))
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[7])

        spectator = self.world.get_spectator()
        transform = carla.Transform(self.vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)), self.vehicle.get_transform().rotation)
        print(transform)
        spectator.set_transform(transform)

        
        # Collision sensor setup
        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self._on_collision)


        # Camera setup
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "84")
        camera_bp.set_attribute("image_size_y", "84")
        camera_bp.set_attribute("fov", "110")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda image: self._process_image(image))

        # Wait until at least one image has been received
        while self.latest_image is None or self.latest_image.sum() == 0:
            time.sleep(0.01)

        return self.latest_image
    
        # return np.zeros((84, 84, 3), dtype=np.uint8)  # Dummy observation for now


    def _on_collision(self, event):
        """Callback function for collision sensor."""
        self.collision_detected = True

    def _process_image(self, image):
        """Converts CARLA image to numpy array."""
        img = np.array(image.raw_data).reshape((image.height, image.width, 4))[:, :, :3]
        img = cv2.resize(img, (84, 84))  # Ensure size match
        self.latest_image = img

    def step(self, action):
        """Applies action and returns (observation, reward, done, info)."""
        throttle, steer = 0.5, 0.0  # Default acceleration
        if action == 0:
            steer = -0.3  # Left
        elif action == 2:
            steer = 0.3  # Right
        
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
        
        # Reset collision detection flag for this step
        self.collision_detected = False
        
        # observation = np.zeros((84, 84, 3), dtype=np.uint8)  # Dummy observation

        # Wait briefly to ensure the image has updated
        time.sleep(0.05)
        observation = self.latest_image

        reward = 0.0  # Default reward
        done = False

        # Collision detection: if a collision occurs, terminate the episode
        if self.collision_detected:
            reward = -10.0  # Large penalty for collision
            done = True
            print("Collision detected! Episode terminated.")

        # Example of off-road detection
        is_off_road = self._is_off_road(self.vehicle.get_location())
        if is_off_road:
            reward = -5.0  # Penalty for going off-road
            done = True
            print("Vehicle is off-road! Episode terminated.")

        # Positive reward for driving without collision for a period of time
        if not self.collision_detected and not is_off_road and not done:
            self.safe_drive_time += 1  # Increment safe drive time
            if self.safe_drive_time >= self.safe_drive_threshold:
                reward += 3.0  # Positive reward for safe driving
                self.safe_drive_time = 0  # Reset after giving reward
            elif done == False:
                reward = 0.0


        print(f"Step Action: {action}, Reward: {reward}, Collision: {self.collision_detected}, safedrivetime: {self.safe_drive_time}, safedrivethreshold: {self.safe_drive_threshold}")
        return observation, reward, done, {}
    
    def _is_off_road(self, location):
        try:
            waypoint = self.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
            return False  # It's on-road
        except:
            return True  # No valid waypoint, off-road

    def destroy_actors(self):
        """Destroys existing actors in the environment."""
        if self.vehicle is not None:
            self.vehicle.destroy()
        if self.camera is not None:
            self.camera.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()

    def close(self):
        self.destroy_actors()

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environment
env = make_vec_env(CarlaEnv, n_envs=1)

# Initialize the DQN model
model = DQN("CnnPolicy", env, verbose=2, buffer_size=50000, learning_rate=1e-4, gamma=0.99)
# verbose=2 will give more detailed information, such as the value of the loss function during training.

# Train the model
print("Starting DQN training...")
model.learn(total_timesteps=10000, log_interval=10, progress_bar=True)
print("Training completed!")

# Save the model
model.save("dqn_carla")

# # Test the model
# env = CarlaEnv()
# model = DQN.load("dqn_carla")

# obs = env.reset()
# done = False

# while not done:
#     action, _ = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     print(f"Reward: {reward}, Done: {done}")
