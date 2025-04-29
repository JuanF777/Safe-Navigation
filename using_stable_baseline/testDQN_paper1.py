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
# import gymnasium as gym
# from gym import spaces
import cv2

from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.global_route_planner import GlobalRoutePlanner


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
        self.action_space = spaces.Discrete(5)  # Take 5 actions; go straight, turn left, turn right, turn slightly left, turn slightly right
        # self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -180.0, -5.0, -10.0, 0.0], dtype=np.float32),
            high=np.array([100.0, 180.0, 5.0, 10.0, 500.0], dtype=np.float32),
            dtype=np.float32
        )
        # [speed_kmh, phi_deg, lateral_error, distance_to_goal]


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
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])

        traj = self.trajectory()
        self.path = []
        for el in traj:
            self.path.append(el[0])

        self.episode_start = time.time()

        # # spawn NPC vehicle
        # NPCvehicle_bp = blueprint_library.filter('vehicle.*')

        # spawn_points = self.world.get_map().get_spawn_points()

        # NPCvehicles = []

        # for i in range(50):  # spawn 50 vehicles
        #     blueprint = random.choice(NPCvehicle_bp)
        #     transform = random.choice(spawn_points)
        #     NPCvehicle = self.world.try_spawn_actor(blueprint, transform)
        #     if NPCvehicle:
        #         NPCvehicles.append(NPCvehicle)

        # # Enable traffic manager
        # traffic_manager = self.client.get_trafficmanager()
        # traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        # for NPCvehicle in NPCvehicles:
        #     NPCvehicle.set_autopilot(True, traffic_manager.get_port())

        # print(f"Spawned {len(NPCvehicles)} vehicles.")



        spectator = self.world.get_spectator()
        transform = carla.Transform(self.vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)), self.vehicle.get_transform().rotation)
        print(transform)
        spectator.set_transform(transform)

        
        # Collision sensor setup
        self.collision_history = [] # store any collision detected
        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        # Lane-crossing sensor setup
        self.lanecrossing_history = []
        lane_crossing_sensor = self.world.get_blueprint_library().find("sensor.other.lane_invasion")
        # keeping the location of the sensor to be same as RGM Camera
        self.lanecrossing_sensor = self.world.spawn_actor(lane_crossing_sensor, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to = self.vehicle)
        self.lanecrossing_sensor.listen(lambda event: self._on_lanecrossing(event))

        # Get vehicle state
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        pos = transform.location
        rot = transform.rotation
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h

        # Waypoints
        waypoint_index = 1
        waypoint = self.path[waypoint_index]
        next_waypoint = self.path[min(waypoint_index + 1, len(self.path) - 1)]
        waypoint_loc = waypoint.transform.location
        waypoint_rot = waypoint.transform.rotation
        next_waypoint_loc = next_waypoint.transform.location

        # phi
        orientation_diff = waypoint_rot.yaw - rot.yaw
        phi = orientation_diff % 360 - 360 * (orientation_diff % 360 > 180)

        # abs dist
        abs_dist = math.sqrt((pos.x - next_waypoint_loc.x)**2 + (pos.y - next_waypoint_loc.y)**2)

        # signed lateral distance
        u = [waypoint_loc.x - next_waypoint_loc.x, waypoint_loc.y - next_waypoint_loc.y]
        v = [pos.x - next_waypoint_loc.x, pos.y - next_waypoint_loc.y]
        if np.linalg.norm(u) > 0.1 and np.linalg.norm(v) > 0.1:
            cross = np.cross(u, v)
            angle = np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
            signed_dis = np.linalg.norm(v) * np.sin(np.sign(cross) * angle)
        else:
            signed_dis = 0.0

        # distance to goal
        dist_from_goal = np.sqrt((pos.x - self.goal.x) ** 2 + (pos.y - self.goal.y) ** 2)




        observation = np.array([speed, phi, signed_dis, abs_dist, dist_from_goal], dtype=np.float32)
        return observation

        # # Camera setup
        # camera_bp = blueprint_library.find("sensor.camera.rgb")
        # camera_bp.set_attribute("image_size_x", "84")
        # camera_bp.set_attribute("image_size_y", "84")
        # camera_bp.set_attribute("fov", "110")
        # camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        # self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        # self.camera.listen(lambda image: self._process_image(image))

        # # Wait until at least one image has been received
        # timeout = time.time() + 5  # 5-second timeout
        # while (self.latest_image is None or self.latest_image.sum() == 0) and time.time() < timeout:
        #     time.sleep(0.01)

        # return self.latest_image

    def trajectory(self):
        # Get spawn points
        spawn_points = self.map.get_spawn_points()

        # Define start and goal (you can pick indices or define manually with carla.Location)
        start0 = spawn_points[0]
        self.start = start0.location
        goal0 = spawn_points[9]
        self.goal = goal0.location

        # Snap to road
        start_wp = self.map.get_waypoint(self.start)
        end_wp = self.map.get_waypoint(self.goal)

        # Set up the route planner
        dao = GlobalRoutePlannerDAO(self.map, sampling_resolution=2.0)  # 2 meters between waypoints
        grp = GlobalRoutePlanner(dao)
        grp.setup()

        # Generate the route (route will be a list of (waypoint, road_option) tuples)
        route = grp.trace_route(start_wp.transform.location, end_wp.transform.location)

        # Visualize Waypoints
        for waypoint, _ in route:
            self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                    color=carla.Color(r=0, g=255, b=0), life_time=600.0,
                                    persistent_lines=True)
            
        return route

    def get_closest_waypoint(self, waypoint_list, target_waypoint):
        closest_waypoint = None
        closest_distance = float('inf')
        for i, waypoint in enumerate(waypoint_list):
            distance = math.sqrt((waypoint.transform.location.x - target_waypoint.transform.location.x)**2 +
                                 (waypoint.transform.location.y - target_waypoint.transform.location.y)**2)
            if distance < closest_distance:
                closest_waypoint = i
                closest_distance = distance
        return closest_waypoint

    def _on_collision(self, event):
        """Callback function for collision sensor."""
        self.collision_history.append(event)

    def _on_lanecrossing(self, event):
        self.lanecrossing_history.append(event)

    def _process_image(self, image):
        """Converts CARLA image to numpy array."""
        img = np.array(image.raw_data).reshape((image.height, image.width, 4))[:, :, :3]
        img = cv2.resize(img, (84, 84))  # Ensure size match
        self.latest_image = img

    def step(self, action):
        # Get vehicle state
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
        # to get the position and orientation of the car
        pos = self.vehicle.get_transform().location
        rot = self.vehicle.get_transform().rotation

        target_waypoint = self.client.get_world().get_map().get_waypoint(pos, project_to_road=True)
        # target_waypoint = self.path[0]
        waypoint_index = self.get_closest_waypoint(self.path, target_waypoint) + 1
        waypoint = self.path[waypoint_index]
        if len(self.path) != 1:
            next_waypoint = self.path[waypoint_index+1]
        else:
            next_waypoint = waypoint
        waypoint_loc = waypoint.transform.location
        waypoint_rot = waypoint.transform.rotation
        next_waypoint_loc = next_waypoint.transform.location
        next_waypoint_rot = next_waypoint.transform.rotation

        # abs dist
        abs_dist = math.sqrt((pos.x - next_waypoint_loc.x)**2 + (pos.y - next_waypoint_loc.y)**2)
        print(waypoint_loc, ", " ,next_waypoint_loc, ", ", abs_dist)

        # to get the orientation difference between the car and the road "phi"
        orientation_diff = waypoint_rot.yaw - rot.yaw
        phi = orientation_diff%360 -360*(orientation_diff%360>180)
        
        # find d
        u = [waypoint_loc.x-next_waypoint_loc.x, waypoint_loc.y-next_waypoint_loc.y]
        v = [pos.x-next_waypoint_loc.x, pos.y-next_waypoint_loc.y]
        if np.linalg.norm(u) > 0.1 and np.linalg.norm(v) > 0.1:
            signed_dis = np.linalg.norm(v)*np.sin(np.sign(np.cross(u,v))*np.arccos(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))))
        else:
            signed_dis = 0

        dist_from_goal = np.sqrt((pos.x - self.goal.x)**2 + (pos.y-self.goal.y)**2)


        """Applies action and returns (observation, reward, done, info)."""
        # throttle, steer = 0.5, 0.0  # Default acceleration
        # if action == 0:
        #     steer = -0.3  # Left
        # elif action == 2:
        #     steer = 0.3  # Right
        
        # self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))


        # Take 5 actions; go straight, turn left, turn right, turn slightly left, turn slightly right
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0))

        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.1, steer=-0.6))

        if action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.1, steer=0.6))

        if action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=-0.1))

        if action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.1))

        
        # Reset collision detection flag for this step
        self.collision_detected = False
        

        # # Wait briefly to ensure the image has updated
        # time.sleep(0.05)
        # observation = self.latest_image
        observation = np.array([speed, phi, signed_dis, abs_dist, dist_from_goal], dtype=np.float32)


        reward = 0.0
        done = False



        '''
        Rewards
        '''
        # Maintain proper orientation relative to the road direction
        if abs(phi)<5:  # phi is very small (good alignment)
            if action == 0:  # going straight
                reward += 2
            else:
                reward -= 1
        elif abs(phi)<10:  # phi is moderate
            if phi<0:  # angled to the left
                if action == 3:  # slight left turn
                    reward += 2
                elif action == 1:  # sharp left turn
                    reward += 1
                else:
                    reward -= 1
            else:  # angled to the right
                if action == 4:  # slight right turn
                    reward += 2
                elif action == 2:  # sharp right turn
                    reward += 1
                else:
                    reward -= 1
        else:  # phi is large (poor alignment)
            if phi<0:  # severely angled left
                if action == 1:  # sharp left turn
                    reward += 2
                elif action == 3:  # slight left turn
                    reward += 1
                else:
                    reward -= 1
            else:  # severely angled right
                if action == 2:  # sharp right turn
                    reward += 2
                elif action == 4:  # slight right turn
                    reward += 1
                else:
                    reward -= 1
        # Stay close to the centerline of the road
        if abs(signed_dis)>2:
            reward -= 10
        if abs(signed_dis)<0.1:  # very close to centerline
            if action == 0:  # going straight
                reward += 4
            else:
                reward -= 2
        elif abs(signed_dis)<0.5:  # moderate distance from centerline
            if signed_dis<0:  # to the left of centerline
                if action == 3:  # slight left turn
                    reward += 2
                elif action == 1:  # sharp left turn
                    reward += 1
                else:
                    reward -= 1
            else:  # to the right of centerline
                if action == 4:  # slight right turn
                    reward += 2
                elif action == 2:  # sharp right turn
                    reward += 1
                else:
                    reward -= 1
        else:  # far from centerline
            if signed_dis<0:  # far to the left
                if action == 1:  # sharp left turn
                    reward += 2
                elif action == 3:  # slight left turn
                    reward += 1
                else:
                    reward -= 1
            else:  # far to the right
                if action == 2:  # sharp right turn
                    reward += 2
                elif action == 4:  # slight right turn
                    reward += 1
                else:
                    reward -= 1





        # # Default positive reward for moving forward without errors
        # reward += 0.1  # small encouragement for moving

        # Check for collision
        if len(self.collision_history) != 0:
            reward = -20
            done = True
            print("Collision detected! Episode terminated.")

        # Excessive orientation deviation
        if abs(phi)>100:
            reward = -10
            done = True
            print("Excessive orientation deviation! Episode terminated.")

        # Excessive lateral deviation
        if abs(signed_dis)>3:
            reward = -10
            done = True
            print("Excessive lateral deviation! Episode terminated.")

        # Excessive abs deviation
        if abs_dist>8:
            reward = -10
            done = True
            print("Excessive abs deviation! Episode terminated.")

        # to avoid lanecrossing
        if len(self.lanecrossing_history) != 0:
            done = True
            reward = - 10
            print("Lane crossed! Episode terminated.")

        # Time limit exceeded
        if self.episode_start + 200 < time.time():
            done = True
            print("Time exceeded! Episode terminated.")

        # # Check for off-road
        # is_off_road = self._is_off_road(self.vehicle.get_location())
        # if is_off_road:
        #     reward = -5.0
        #     done = True
        #     print("Vehicle is off-road! Episode terminated.")

        # # Encourage maintaining decent speed on road
        # if not done:
        #     if speed < 5:
        #         reward -= 0.5  # discourage crawling
        #     elif 5 <= speed <= 30:
        #         reward += 0.5  # sweet spot
        #     elif speed > 40:
        #         reward -= 0.5  # discourage speeding

        # # Reward for safe continuous driving
        # if not self.collision_detected and not is_off_road and not done:
        #     self.safe_drive_time += 1
        #     if self.safe_drive_time >= self.safe_drive_threshold:
        #         reward += 3.0
        #         self.safe_drive_time = 0
        # else:
        #     self.safe_drive_time = 0  # reset if unsafe
        
        # Print debug info
        # print(f"Action: {action}, Speed: {speed:.2f} km/h, phi_deg: {phi}, lateral_error: {signed_dis}, abs_dist_error: {abs_dist}, distance_to_goal: {dist_from_goal}")
        return observation, reward, done, {}
    
    # def _is_off_road(self, location):
    #     try:
    #         waypoint = self.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    #         return False  # It's on-road
    #     except:
    #         return True  # No valid waypoint, off-road

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
env = make_vec_env(CarlaEnv, n_envs=1, wrapper_kwargs={"verbose": 2})

# Initialize the DQN model
# model = DQN("CnnPolicy", env, verbose=2, buffer_size=50000, learning_rate=1e-4, gamma=0.99)
# model = DQN("MlpPolicy", env, verbose=2, buffer_size=50000, learning_rate=1e-4, gamma=0.99)
model = DQN(
    "MlpPolicy", 
    env, 
    verbose=2,
    buffer_size=100000,         # Experience replay buffer size
    learning_rate=3e-4,         # Adam optimizer learning rate
    gamma=0.99,                 # Discount factor
    exploration_fraction=0.2,   # Fraction of training to explore
    exploration_initial_eps=1.0, # Initial exploration rate
    exploration_final_eps=0.05,  # Final exploration rate
    train_freq=4,               # Train every 4 steps
    gradient_steps=1,           # How many gradient steps after each rollout
    target_update_interval=1000, # How often to update target network
    learning_starts=10000,      # Collect experiences before training
    batch_size=64,              # Batch size for training
    tensorboard_log="./DQN_tensorboard_logs/" # Tensorboard logging
)
# verbose=2 will give more detailed information, such as the value of the loss function during training.

# # Train the model
# print("Starting DQN training...")
# model.learn(total_timesteps=100000, log_interval=10, progress_bar=True)
# print("Training completed!")

# # Save the model
# model.save("dqn_carla")

# Test the model
env = CarlaEnv()
model = DQN.load("dqn_carla2")

obs = env.reset()
done = False


try:
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}")
except KeyboardInterrupt:
    print("\nInterrupted! Exiting...")
