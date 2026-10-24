import carla
import numpy as np
import pygame

# Connect to the remote server
client = carla.Client('urban1.ifl.kit.edu', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Initialize Pygame
pygame.init()
width, height = 1280*1.5, 720*1.5
display = pygame.display.set_mode((width, height))
pygame.display.set_caption("CARLA Remote Free Camera - Mouse Navigation")

# Retrieve spectator position
spectator = world.get_spectator()
transform = spectator.get_transform()

# Spawn RGB camera
blueprint_library = world.get_blueprint_library()
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(width))
camera_bp.set_attribute('image_size_y', str(height))
camera = world.spawn_actor(camera_bp, transform)

surface = None

def render(image):
    global surface
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

camera.listen(render)

# Control sensitivities
speed = 1.0
rot_sensitivity = 0.2
pan_sensitivity = 0.05
clock = pygame.time.Clock()

print("Controls:")
print("  Right Click + Drag  : Rotate camera view (Look around)")
print("  Left/Middle + Drag  : Pan camera (Left/Right & Up/Down)")
print("  Mouse Scroll        : Adjust movement speed")
print("  W / A / S / D       : Move Forward / Left / Backward / Right")
print("  Q / E               : Move Down / Up")
print("  Arrow Keys          : Keyboard camera rotation")

try:
    while True:
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
                
            elif event.type == pygame.MOUSEMOTION:
                # Right Click Drag -> Rotate view (Pitch / Yaw)
                if event.buttons[2]:
                    dx, dy = event.rel
                    transform.rotation.yaw += dx * rot_sensitivity
                    # Clamp pitch between -89 and 89 degrees to prevent gimbal lock
                    transform.rotation.pitch = max(-89.0, min(89.0, transform.rotation.pitch - dy * rot_sensitivity))
                
                # Left Click or Middle Click Drag -> Pan view along camera screen plane
                elif event.buttons[0] or event.buttons[1]:
                    dx, dy = event.rel
                    rt = transform.get_right_vector()
                    
                    # Shift along the camera's local right vector
                    transform.location.x -= rt.x * dx * pan_sensitivity
                    transform.location.y -= rt.y * dx * pan_sensitivity
                    # Shift along world Z axis for vertical panning
                    transform.location.z += dy * pan_sensitivity

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Scroll Up -> Increase Speed
                if event.button == 4:
                    speed = min(10.0, speed + 0.2)
                # Scroll Down -> Decrease Speed
                elif event.button == 5:
                    speed = max(0.1, speed - 0.2)

        # Keyboard continuous controls
        keys = pygame.key.get_pressed()
        
        # Keyboard Rotation
        if keys[pygame.K_LEFT]:  transform.rotation.yaw -= 2.0
        if keys[pygame.K_RIGHT]: transform.rotation.yaw += 2.0
        if keys[pygame.K_UP]:    transform.rotation.pitch = min(89.0, transform.rotation.pitch + 2.0)
        if keys[pygame.K_DOWN]:  transform.rotation.pitch = max(-89.0, transform.rotation.pitch - 2.0)
        
        # Directional Vectors
        fw = transform.get_forward_vector()
        rt = transform.get_right_vector()
        
        # Keyboard Movement
        if keys[pygame.K_w]: 
            transform.location.x += fw.x * speed
            transform.location.y += fw.y * speed
            transform.location.z += fw.z * speed
        if keys[pygame.K_s]: 
            transform.location.x -= fw.x * speed
            transform.location.y -= fw.y * speed
            transform.location.z -= fw.z * speed
        if keys[pygame.K_a]: 
            transform.location.x -= rt.x * speed
            transform.location.y -= rt.y * speed
            transform.location.z -= rt.z * speed
        if keys[pygame.K_d]: 
            transform.location.x += rt.x * speed
            transform.location.y += rt.y * speed
            transform.location.z += rt.z * speed
        if keys[pygame.K_q]: transform.location.z -= speed
        if keys[pygame.K_e]: transform.location.z += speed

        # Sync both the client camera and the server's spectator transform
        camera.set_transform(transform)
        spectator.set_transform(transform)

        if surface is not None:
            display.blit(surface, (0, 0))
            pygame.display.flip()

except KeyboardInterrupt:
    print("\nStopping stream and cleaning up...")
finally:
    camera.stop()
    camera.destroy()
    pygame.quit()