import carla
from sensors import SensorManager

class CarAgent:
    def __init__(self, client):
        self.client = client
        self.world = client.get_world()
        self.vehicle = None
        self.sensors = None

    def spawn_vehicle(self):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]  # Tesla Model 3

        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Attach sensors to the car
        self.sensors = SensorManager(self.vehicle)
        self.sensors.attach_camera()
        self.sensors.attach_collision_sensor()

    def drive(self, throttle=0.5, steer=0.0, brake=0.0):
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        control.brake = brake
        self.vehicle.apply_control(control)

    def destroy(self):
        if self.sensors:
            self.sensors.destroy_sensors()
        if self.vehicle:
            self.vehicle.destroy()