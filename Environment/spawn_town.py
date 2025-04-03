
import carla
import random
import time

def spawn_vehicles(client, num_vehicles=10):
    world = lient.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('vehicle.*')

    spawn_points = world.get_map().get_spawn_points()
    vehicles = []

    for i in range(num_vehicles):
        vehicle_bp = random.choice(vehicle_blueprints)
        spawn_point = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

        if vehicle:
            vehicles.append(vehicle)

    return vehicles

def spawn_pedestrians(client, num_pedestrians=10):
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    pedestrian_bp = blueprint_library.filter('walker.pedestrian.*')

    spawn_points = [carla.Transform(carla.Location(x=random.uniform(-50, 50),
                                                   y=random.uniform(-50, 50),
                                                   z=1)) for _ in range(num_pedestrians)]
    
    pedestrians = []
    
    for spawn_point in spawn_points:
        pedestrian = world.try_spawn_actor(random.choice(pedestrian_bp), spawn_point)
        if pedestrian:
            pedestrians.append(pedestrian)

    return pedestrians

def setup_traffic_lights(client):
    world = client.get_world()
    traffic_lights = world.get_actors().filter('traffic.traffic_light')

    for light in traffic_lights:
        light.set_state(carla.TrafficLightState.Red)  # Set all to red
        time.sleep(2)  # Change lights after 2 seconds
        light.set_state(carla.TrafficLightState.Green)  # Set all to green

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    print("Spawning vehicles...")
    vehicles = spawn_vehicles(client, num_vehicles=15)

    print("Spawning pedestrians...")
    pedestrians = spawn_pedestrians(client, num_pedestrians=10)

    print("Setting up traffic lights...")
    setup_traffic_lights(client)

    print("Simulation running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)  # Keep the simulation running
    except KeyboardInterrupt:
        print("Stopping simulation...")
        for vehicle in vehicles:
            vehicle.destroy()
        for pedestrian in pedestrians:
            pedestrian.destroy()

if __name__ == "__main__":
    main()
