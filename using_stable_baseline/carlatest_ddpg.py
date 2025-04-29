import os
import sys
import gym
import numpy as np
import math

# Add CARLA Python API path
script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, "carla-0.9.10-py3.7-win-amd64.egg"))
import carla

class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.load_world("Town02")
        self.map = self.world.get_map()

        self.vehicle = None
        self.collision_sensor = None
        self.collision_hist = []

        self.waypoint_list = []
        self.current_wp_index = 0

    def reset(self):
        self.episode_steps = 0

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

        self.world.tick()

        spectator = self.world.get_spectator()
        transform = self.vehicle.get_transform()
        spectator.set_transform(carla.Transform(
            transform.location + transform.get_forward_vector() * -8 + carla.Location(z=3),
            transform.rotation
        ))

        coords = [
            (-7.42, 142.24), (-7.42, 145.24), (-7.42, 148.24), (-7.42, 151.24),
            (-7.42, 154.24), (-7.42, 157.24), (-7.42, 160.24), (-7.42, 163.24),
            (-7.42, 166.24), (-7.42, 169.24), (-7.42, 172.24)
        ]

        self.waypoint_list = []
        for x, y in coords:
            wp = self.map.get_waypoint(carla.Location(x=x, y=y, z=0.5))
            self.waypoint_list.append(wp)
            self.world.debug.draw_string(
                wp.transform.location, 'O', draw_shadow=False,
                color=carla.Color(r=0, g=255, b=0), life_time=600
            )

        self.current_wp_index = 0
        self.visited_points = set()

        return self._get_obs()

    def step(self, action):
        control = carla.VehicleControl()

        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))

        if throttle < 0.5:
            throttle = 0.5

        if throttle >= brake:
            brake = 0.0
        else:
            throttle = 0.0

        control.steer = steer
        control.throttle = throttle
        control.brake = brake

        self.vehicle.apply_control(control)
        self.world.tick()

        self.episode_steps += 1

        reward, done, info = self._compute_reward()

        return self._get_obs(), reward, done, info

    def _compute_reward(self):
        reward = 0.0
        done = False
        info = {}

        if self.episode_steps >= 1000:
            reward = -500.0
            done = True
            info["event"] = "timeout"
            return reward, done, info

        if len(self.collision_hist) > 0:
            reward = -1000.0
            done = True
            info["event"] = "collision"
            return reward, done, info

        vehicle_loc = self.vehicle.get_location()
        vehicle_tf = self.vehicle.get_transform()
        yaw = math.radians(vehicle_tf.rotation.yaw)

        waypoint = self.waypoint_list[self.current_wp_index].transform.location
        dx = waypoint.x - vehicle_loc.x
        dy = waypoint.y - vehicle_loc.y

        forward_dist = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        lateral_dist = dx * math.sin(-yaw) + dy * math.cos(-yaw)

        distance = math.sqrt(dx**2 + dy**2)

        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2)

        reward += 2.0 * speed
        reward -= 5.0 * abs(lateral_dist)

        if distance < 2.0:
            if self.current_wp_index not in self.visited_points:
                reward += 100.0
                self.visited_points.add(self.current_wp_index)
            self.current_wp_index += 1
            if self.current_wp_index >= len(self.waypoint_list):
                print("Finished all waypoints!")
                reward += 500.0
                done = True
                info["event"] = "success"

        return reward, done, info

    def _get_obs(self):
        if self.current_wp_index >= len(self.waypoint_list):
            return np.zeros(5, dtype=np.float32)

        vehicle_tf = self.vehicle.get_transform()
        vehicle_loc = vehicle_tf.location
        yaw = math.radians(vehicle_tf.rotation.yaw)

        waypoint = self.waypoint_list[self.current_wp_index].transform.location
        dx = waypoint.x - vehicle_loc.x
        dy = waypoint.y - vehicle_loc.y

        forward_dist = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        lateral_dist = dx * math.sin(-yaw) + dy * math.cos(-yaw)

        velocity = self.vehicle.get_velocity()
        vel_x = velocity.x / 10.0
        vel_y = velocity.y / 10.0
        angle_to_wp = math.atan2(lateral_dist, forward_dist) / math.pi

        return np.array([forward_dist / 10.0, lateral_dist / 10.0, vel_x, vel_y, angle_to_wp], dtype=np.float32)

    def close(self):
        if self.vehicle:
            self.vehicle.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
