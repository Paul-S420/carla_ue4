import carla
import random
import time

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Retrieve the blueprint library and select a random vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = random.choice(blueprint_library.filter('vehicle.*.*'))

# Get all recommended spawn points from the map
spawn_points = world.get_map().get_spawn_points()
spawn_point = random.choice(spawn_points)

# Attempt to spawn the vehicle
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

if vehicle is not None:
    print(f"Successfully spawned {vehicle.type_id} at {spawn_point.location}")
    
    # Enable the built-in autopilot
    vehicle.set_autopilot(True)
    
    try:
        print("The vehicle is now driving. Press Ctrl+C in the terminal to exit and remove it.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nRemoving vehicle...")
        vehicle.destroy()
else:
    print("Spawn collision detected. Try running the script again.")