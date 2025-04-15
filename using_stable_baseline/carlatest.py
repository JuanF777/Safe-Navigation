import sys
import os
import gym
import numpy as np
from gym import spaces

script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, "carla-0.9.10-py3.7-win-amd64.egg"))
import carla
import math

class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        # self.world = self.client.get_world()
        self.world = self.client.load_world("Town02")
        print("Loaded map:", self.world.get_map().name)


        self.vehicle = None
        self.collision_sensor = None
        self.collision_hist = []

        self.map = self.world.get_map()
        self.waypoint_list = []
        self.current_wp_index = 0
        self.prev_distance = None

        self.total_distance = 0.0
        self.last_location = None




    def reset(self):
        spawn_points = self.map.get_spawn_points()

        if self.vehicle is not None:
            self.vehicle.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        self.collision_hist = []

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        spawn_point = spawn_points[0]  # Start at index 0
        print(f"[DEBUG] Spawn Location: {spawn_point.location}")

        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        spectator = self.world.get_spectator()
        transform = self.vehicle.get_transform()
        location = transform.location + carla.Location(z=30)  # Height above car
        rotation = carla.Rotation(pitch=-90)  # Look straight down
        spectator.set_transform(carla.Transform(location, rotation))


        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.collision_hist.append(event))

        self.waypoint_list = []
        current_wp = self.map.get_waypoint(spawn_point.location)

        for i in range(40):
            next_wps = current_wp.next(3.0)
            if not next_wps:
                break

            if i == 0:
                chosen_wp = next_wps[0]
                for wp in next_wps:
                    angle_diff = abs(wp.transform.rotation.yaw - current_wp.transform.rotation.yaw)
                    if 20 < angle_diff < 160:  # angle change means it's turning
                        chosen_wp = wp
                        print(f"[DEBUG] Turn selected with angle diff: {angle_diff:.2f}")
                        break
                current_wp = chosen_wp
            else:
                current_wp = next_wps[0]

            self.waypoint_list.append(current_wp)

        for wp in self.waypoint_list:
            self.world.debug.draw_string(wp.transform.location, 'O', draw_shadow=False,color=carla.Color(r=0, g=255, b=0), life_time=10.0, persistent_lines=True)

        self.current_wp_index = 0
        self.prev_distance = self._distance_to_waypoint()

        self.total_distance = 0.0
        self.last_location = self.vehicle.get_location()

        return self._get_obs()

    def step(self, action):
        control = carla.VehicleControl()

        # Distance to next waypoint
        current_distance = self._distance_to_waypoint()

        # Only brake if we're approaching the final few waypoints
        is_near_goal = (len(self.waypoint_list) - self.current_wp_index) <= 3

        if is_near_goal and current_distance < 3.0:
            control.throttle = 0.0  # Soft brake
            control.brake = 0.3
        else:
            control.throttle = 0.5
            control.brake = 0.0

        # Steering action
        control.steer = [-0.3, 0.0, 0.3][action]
        self.vehicle.apply_control(control)

        # Track distance traveled
        current_location = self.vehicle.get_location()
        dx = current_location.x - self.last_location.x
        dy = current_location.y - self.last_location.y
        step_distance = math.sqrt(dx**2 + dy**2)
        self.total_distance += step_distance
        self.last_location = current_location


        self.world.tick()

        done = False
        reward = 1.0  # base reward

        if len(self.collision_hist) > 0:
            reward = -100.0
            done = True
        else:
            # Reward if agent gets closer to waypoint
            if current_distance < self.prev_distance:
                reward += 0.5
            self.prev_distance = current_distance

            # Reached a waypoint?
            if current_distance < 2.0:
                self.current_wp_index += 1
                if self.current_wp_index >= len(self.waypoint_list):
                    print("Reached final waypoint!")
                    reward += 100.0  # Goal reached
                    done = True

        print(f"[DEBUG] Current Distance: {self.total_distance:.2f}m")

        # Penalty if far from center of waypoint (encourage staying close)
        drift_distance = current_distance
        if drift_distance > 2.5:
            reward -= 0.5  # big penalty for drifting
        elif drift_distance > 1.5:
            reward -= 0.2  # mild penalty


        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        vel = self.vehicle.get_velocity()
        loc = self.vehicle.get_location()
        return np.array([vel.x, vel.y, loc.x, loc.y], dtype=np.float32)

    def _distance_to_waypoint(self):
        if self.current_wp_index >= len(self.waypoint_list):
            return 0.0
        vehicle_loc = self.vehicle.get_location()
        wp_loc = self.waypoint_list[self.current_wp_index].transform.location
        dx = vehicle_loc.x - wp_loc.x
        dy = vehicle_loc.y - wp_loc.y
        return math.sqrt(dx ** 2 + dy ** 2)
    
    def get_total_distance(self):
        return self.total_distance

    def close(self):
        if self.vehicle:
            self.vehicle.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
