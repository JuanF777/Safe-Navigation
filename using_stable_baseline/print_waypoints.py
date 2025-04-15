import sys
import os

script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, "carla-0.9.10-py3.7-win-amd64.egg"))

import carla

client = carla.Client("localhost", 2000)
client.set_timeout(5.0)
world = client.load_world("Town02")
map = world.get_map()

# Get all road waypoints, every 3 meters
waypoints = map.generate_waypoints(distance=3.0)

# Output file
output_path = os.path.join(script_dir, "waypoints_output.txt")
with open(output_path, "w") as f:
    f.write(f"Generated {len(waypoints)} waypoints.\n\n")
    for i, wp in enumerate(waypoints):
        loc = wp.transform.location
        yaw = wp.transform.rotation.yaw
        f.write(f"[{i}] x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}, yaw={yaw:.2f}\n")

print(f"✅ Waypoints written to: {output_path}")
