# import carla
# import time

# client = carla.Client('urban1.ifl.kit.edu', 2000)
# client.load_world('Town10HD_Opt')
# world = client.get_world()

# client.start_recorder('recording.log')
# time.sleep(10)
# client.stop_recorder()


import carla
import random
import time

# Replace with your server's actual IPv4 address
SERVER_IP = 'urban1.ifl.kit.edu'

# Connect to the remote server
client = carla.Client('localhost', 2000)
# client.set_timeout(20.0)
world = client.get_world()

# Start the recorder
# The file will be saved on the SERVER's hard drive
log_filename = "remote_session_01.log"
client.start_recorder(log_filename)
print(f"Recording started. The file '{log_filename}' is being saved on the server.")

# # Spawn a single vehicle to have something to record
# blueprint_library = world.get_blueprint_library()
# vehicle_bp = random.choice(blueprint_library.filter('vehicle.*.*'))
# spawn_point = random.choice(world.get_map().get_spawn_points())

# vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
# if vehicle:
#     vehicle.set_autopilot(True)
#     print("Vehicle spawned and moving.")

try:
    print("Recording in progress. Press Ctrl+C to stop.")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping recorder and cleaning up...")
    client.stop_recorder()
    print(client.show_recorder_file_info("remote_session_01.log", True))
    # if vehicle:
    #     vehicle.destroy()