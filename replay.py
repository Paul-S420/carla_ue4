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
client.replay_file("recording01.log", 10, 0, 24)
