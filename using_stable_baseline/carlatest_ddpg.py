import os
import sys
import gym
import numpy as np
from gym import spaces
import math

# Add CARLA Python API path
script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, "carla-0.9.10-py3.7-win-amd64.egg"))
import carla

class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()

        # Action: steering in range [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)

        # Observation: [forward_dist, lateral_dist, velocity_x, velocity_y, angle_to_wp]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.load_world("Town02")
        print("Loaded map:", self.world.get_map().name) # Confirm correct map is loaded

        self.vehicle = None
        self.collision_sensor = None
        self.collision_hist = []

        self.map = self.world.get_map()
        self.waypoint_list = []
        self.current_wp_index = 0
        self.prev_distance = None
        self.total_distance = 0.0
        self.last_location = None

    def reset(self): # Clean up and reset environement
        self.episode_steps = 0

        # Clean up actors from previous episode
        if self.vehicle:
            self.vehicle.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        self.collision_hist = []

        blueprint_library = self.world.get_blueprint_library() # vehicle defintion
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        # spawn random vehicle 
        spawn_point = carla.Transform(carla.Location(x=-7.42, y=142.24, z=0.5), carla.Rotation(yaw=90))
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Attach collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle) # spawn sensor
        self.collision_sensor.listen(lambda event: self.collision_hist.append(event))

        # Set top-down spectator view
        spectator = self.world.get_spectator()
        location = self.vehicle.get_transform().location + carla.Location(z=30)
        rotation = carla.Rotation(pitch=-90)
        spectator.set_transform(carla.Transform(location, rotation))

        # Define route with waypoints
        # Should be a full foward, then left turn onto new road
        coords = [
            (-7.42, 142.24), (-7.42, 145.24), (-7.42, 148.24), (-7.42, 151.24),
            (-7.42, 154.24), (-7.42, 157.24), (-7.42, 160.24), (-7.42, 163.24),
            (-7.42, 166.24), (-7.42, 169.24), (-7.42, 172.24), (-7.46, 175.24),
            (-7.46, 178.24), (-7.46, 181.24), (-7.45, 184.24), (-7.45, 186.24),
            (-6.5, 187.50), (-5.3, 188.5), (-4.1, 189.3), (-3.0, 190.1),
            (-1.6, 190.9), (0.2, 191.5), (2.0, 191.55), (4.0, 191.56),
            (6.0, 191.56), (9.0, 191.56), (12.0, 191.56), (15.0, 191.56),
            (18.0, 191.56), (21.0, 191.56), (24.0, 191.56), (27.0, 191.56),
            (30.0, 191.57), (33.0, 191.57), (37.0, 191.57), (40.0, 191.57),
            (43.0, 191.57), (46.0, 191.57), (49.0, 191.57), (52.0, 191.58),
            (55.0, 191.58), (58.0, 191.58), (61.0, 191.58), (64.0, 191.58),
            (67.0, 191.58)
        ]
        self.waypoint_list = []
        # MARK WAYPOINTS
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
        reward = 1.0
        done = False
        self.episode_steps += 1

        # Apply steering control
        steer = float(np.clip(action[0], -1.0, 1.0))
        control.steer = steer
        control.throttle = 0.5  # Constant throttle
        self.vehicle.apply_control(control)
        self.world.tick()

        # Compute state info
        vehicle_tf = self.vehicle.get_transform()
        vehicle_loc = vehicle_tf.location
        vehicle_rot = vehicle_tf.rotation
        waypoint = self.waypoint_list[self.current_wp_index].transform.location

        dx = waypoint.x - vehicle_loc.x
        dy = waypoint.y - vehicle_loc.y
        yaw = math.radians(vehicle_rot.yaw)
        forward_dist = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        lateral_dist = dx * math.sin(-yaw) + dy * math.cos(-yaw)
        angle_to_wp = math.atan2(lateral_dist, forward_dist)

        # Encourage aligned forward motion
        if forward_dist > 0 and abs(angle_to_wp) < 0.3:
            reward += 0.2 * forward_dist

        # Penalize spinning
        angular_z = self.vehicle.get_angular_velocity().z
        reward -= 0.2 * abs(angular_z)

        # Track travel distance
        curr_loc = self.vehicle.get_location()
        dx = curr_loc.x - self.last_location.x
        dy = curr_loc.y - self.last_location.y
        self.total_distance += math.sqrt(dx**2 + dy**2)
        self.last_location = curr_loc

        # Collision check
        if len(self.collision_hist) > 0:
            reward = -100.0
            done = True
        else:
            dist = self._distance_to_waypoint()
            if dist < self.prev_distance:
                reward += 0.5
            self.prev_distance = dist

            if dist < 1.0:
                self.current_wp_index += 1
                reward += 5.0
                if self.current_wp_index >= len(self.waypoint_list):
                    print("Reached final waypoint!")
                    reward += 100.0
                    done = True

        # Penalize drifting
        if self.prev_distance > 2.5:
            reward -= 0.5
        elif self.prev_distance > 1.5:
            reward -= 0.2

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # Compose the observation vector
        vehicle_tf = self.vehicle.get_transform()
        vehicle_loc = vehicle_tf.location
        vehicle_rot = vehicle_tf.rotation
        waypoint = self.waypoint_list[self.current_wp_index].transform.location

        dx = waypoint.x - vehicle_loc.x
        dy = waypoint.y - vehicle_loc.y
        yaw = math.radians(vehicle_rot.yaw)
        forward_dist = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        lateral_dist = dx * math.sin(-yaw) + dy * math.cos(-yaw)

        velocity = self.vehicle.get_velocity()
        vel_x = velocity.x / 10.0
        vel_y = velocity.y / 10.0
        angle_to_wp = math.atan2(lateral_dist, forward_dist) / math.pi

        forward_dist /= 10.0
        lateral_dist /= 10.0

        return np.array([forward_dist, lateral_dist, vel_x, vel_y, angle_to_wp], dtype=np.float32)

    def _distance_to_waypoint(self):
        if self.current_wp_index >= len(self.waypoint_list):
            return 0.0
        vehicle_loc = self.vehicle.get_location()
        wp_loc = self.waypoint_list[self.current_wp_index].transform.location
        return math.sqrt((vehicle_loc.x - wp_loc.x)**2 + (vehicle_loc.y - wp_loc.y)**2)

    def get_total_distance(self):
        return self.total_distance

    def close(self):
        if self.vehicle:
            self.vehicle.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
