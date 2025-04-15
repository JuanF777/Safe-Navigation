import sys
import os
import gym
import numpy as np
from gym import spaces
import math

script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, "carla-0.9.10-py3.7-win-amd64.egg"))
import carla

class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)


        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
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
        if self.vehicle:
            self.vehicle.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        self.collision_hist = []

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        spawn_point = carla.Transform(carla.Location(x=-7.42, y=142.24, z=0.5), carla.Rotation(yaw=90))
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.collision_hist.append(event))

        spectator = self.world.get_spectator()
        location = self.vehicle.get_transform().location + carla.Location(z=30)
        rotation = carla.Rotation(pitch=-90)
        spectator.set_transform(carla.Transform(location, rotation))

        coords = [
    (-7.42, 142.24),  # Forward straight path
    (-7.42, 145.24),
    (-7.42, 148.24),
    (-7.42, 151.24),
    (-7.42, 154.24),
    (-7.42, 157.24),
    (-7.42, 160.24),
    (-7.42, 163.24),
    (-7.42, 166.24),
    (-7.42, 169.24),
    (-7.42, 172.24),
    (-7.46, 175.24),
    (-7.46, 178.24),
    (-7.46, 181.24),
    (-7.45, 184.24),
    (-7.45, 187.24),
    (-3.86, 189.54),
    (0.51, 191.65),
    (3.40, 191.55),
    (6.40, 191.56),
    (9.40, 191.56),
    (12.4, 191.56),
    (15.4, 191.56),
    (18.4, 191.56),
    (21.4, 191.56),
    (24.4, 191.56),
    (27.4, 191.56),
    (30.4, 191.57),
    (33.4, 191.57),
    (37.70, 191.57),
    (40.70, 191.57),
    (43.70, 191.57),
    (46.70, 191.57),
    (49.55, 191.65),
    (52.16, 191.58),
    (55.7, 191.58), 
    (58.7, 191.58),
    (61.7, 191.58),
    (64.7, 191.58),
    (67.7, 191.58)
        ]

        self.waypoint_list = []
        for x, y in coords:
            wp = self.map.get_waypoint(carla.Location(x=x, y=y, z=0.5))
            self.waypoint_list.append(wp)
            self.world.debug.draw_string(wp.transform.location, 'O', draw_shadow=False,
                                         color=carla.Color(r=0, g=255, b=0), life_time=20.0)

        self.current_wp_index = 0
        self.prev_distance = self._distance_to_waypoint()
        self.total_distance = 0.0
        self.last_location = self.vehicle.get_location()

        return self._get_obs()

    def step(self, action):
        control = carla.VehicleControl()
        current_distance = self._distance_to_waypoint()

        is_near_goal = (len(self.waypoint_list) - self.current_wp_index) <= 3
        control.throttle = 0.5 if not (is_near_goal and current_distance < 3.0) else 0.0
        control.brake = 0.3 if is_near_goal and current_distance < 3.0 else 0.0
        control.steer = [-0.3, 0.0, 0.3][action]

        self.vehicle.apply_control(control)
        self.world.tick()

        current_location = self.vehicle.get_location()
        dx = current_location.x - self.last_location.x
        dy = current_location.y - self.last_location.y
        step_distance = math.sqrt(dx**2 + dy**2)
        self.total_distance += step_distance
        self.last_location = current_location

        reward = 1.0
        done = False

        if len(self.collision_hist) > 0:
            reward = -100.0
            done = True
        else:
            if current_distance < self.prev_distance:
                reward += 0.5
            self.prev_distance = current_distance

            if current_distance < 2.0:
                self.current_wp_index += 1
                if self.current_wp_index >= len(self.waypoint_list):
                    print("Reached final waypoint!")
                    reward += 100.0
                    done = True

        if current_distance > 2.5:
            reward -= 0.5
        elif current_distance > 1.5:
            reward -= 0.2

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        vel = self.vehicle.get_velocity()
        loc = self.vehicle.get_location()
        next_wp = self.waypoint_list[self.current_wp_index].transform

        # Heading difference (radians)
        vehicle_yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
        wp_yaw = math.radians(next_wp.rotation.yaw)
        heading_diff = math.sin(wp_yaw - vehicle_yaw)

        # Relative vector to waypoint
        dx = next_wp.location.x - loc.x
        dy = next_wp.location.y - loc.y

        return np.array([vel.x, vel.y, dx, dy, heading_diff], dtype=np.float32)


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
