import carla

client = carla.Client('urban1.ifl.kit.edu', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Get the blueprint library
bp_library = world.get_blueprint_library()

# Filter and print specific categories (e.g., vehicles, sensors)
list = sorted([bp.id for bp in bp_library.filter('vehicle.*')])
for item in list: print(item)
