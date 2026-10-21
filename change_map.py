import carla

# Connect to the server
client = carla.Client('urban1.ifl.kit.edu', 2000)
client.set_timeout(10.0)

# 1. List available maps
available_maps = client.get_available_maps()
print("Available maps on the server:")
for map_path in available_maps:
    print(map_path)

# 2. Get the currently active map
world = client.get_world()
current_map = world.get_map().name
print(f"\nCurrent map: {current_map}")

# 3. Load a new map
# You can use the full path or just the short name (e.g., 'Town02')
map_to_load = 'Town10HD_Opt'
print(f"Loading {map_to_load}...")

# This will block the script until the new map is fully loaded
world = client.load_world(map_to_load)
print(f"Successfully loaded: {world.get_map().name}")