import carla
import numpy as np
import cv2

class SensorManager:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.sensors = []

    def attach_camera(self):
        world = self.vehicle.get_world()
        blueprint_library = world.get_blueprint_library()

        # Camera Sensor
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))  # Position relative to car
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.sensors.append(camera)

        def process_image(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
            cv2.imshow("Camera", array)
            cv2.waitKey(1)

        camera.listen(lambda image: process_image(image))

    def attach_collision_sensor(self):
        world = self.vehicle.get_world()
        blueprint_library = world.get_blueprint_library()

        # Collision Sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.sensors.append(collision_sensor)

        def collision_event(event):
            print("Collision detected!")

        collision_sensor.listen(lambda event: collision_event(event))

    def destroy_sensors(self):
        for sensor in self.sensors:
            sensor.destroy()
        self.sensors = []
