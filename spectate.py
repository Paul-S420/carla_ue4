import carla
import numpy as np
import pygame

# Connect to the remote server
client = carla.Client('urban1.ifl.kit.edu', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Initialize Pygame and Font
pygame.init()
pygame.font.init()
width, height = int(1280), int(720)
display = pygame.display.set_mode((width, height))
pygame.display.set_caption("CARLA Remote Free Camera - Multi-Modality")

font = pygame.font.SysFont(None, 20)

# Tag names
TAG_NAMES = {
    1: "Road", 2: "Sidewalk", 3: "Building", 4: "Wall", 5: "Fence",
    6: "Pole", 7: "Traffic Light", 8: "Traffic Sign", 9: "Vegetation",
    10: "Terrain", 11: "Sky", 12: "Pedestrian", 13: "Rider",
    14: "Car", 15: "Truck", 16: "Bus", 17: "Train", 18: "Motorcycle",
    19: "Bicycle", 20: "Static", 21: "Dynamic", 22: "Other",
    23: "Water", 24: "RoadLine", 25: "Ground", 26: "Bridge",
    27: "RailTrack", 28: "GuardRail"
}

# Semantic Segmentation Palette (Converted to 0-255 uint8)
RAW_PALETTE = {
    1: [128, 64, 128], 2: [232, 35, 244], 3: [70, 70, 70], 4: [156, 102, 102], 
    5: [153, 153, 190], 6: [153, 153, 153], 7: [30, 170, 250], 8: [0, 220, 220], 
    9: [35, 142, 107], 10: [152, 251, 152], 11: [180, 130, 70], 12: [60, 20, 220], 
    13: [0, 0, 255], 14: [142, 0, 0], 15: [70, 0, 0], 16: [100, 60, 0], 
    17: [100, 80, 0], 18: [230, 0, 0], 19: [32, 11, 119], 20: [160, 190, 110], 
    21: [50, 120, 170], 22: [80, 90, 55], 23: [150, 60, 45], 24: [50, 234, 157], 
    25: [81, 0, 81], 26: [100, 100, 150], 27: [140, 150, 230], 28: [180, 165, 180]
}
PALETTE_LUT = np.zeros((256, 3), dtype=np.uint8)
for tag, color in RAW_PALETTE.items():
    PALETTE_LUT[tag] = color

# Modality configuration
MODALITIES = [
    {"name": "Instance Segmentation (Highlight)", "bp": "sensor.camera.instance_segmentation"},
    {"name": "RGB", "bp": "sensor.camera.rgb"},
    {"name": "Semantic Segmentation", "bp": "sensor.camera.semantic_segmentation"}
]

current_modality = 0
camera = None
surface = None
target_tag = 14
filter_active = True
center_pixel_info = None  # Stores (R, G, B) of the center pixel
blueprint_library = world.get_blueprint_library()

spectator = world.get_spectator()
transform = spectator.get_transform()

def render(image):
    global surface, target_tag, current_modality, filter_active, center_pixel_info
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    
    # Extract center pixel for inspection (CARLA array is BGRA)
    cy, cx = image.height // 2, image.width // 2
    b, g, r, _ = array[cy, cx]
    # Cast to standard Python integers to prevent uint8 overflow on bitwise operations
    center_pixel_info = (int(r), int(g), int(b))
    
    if current_modality == 0:
        if filter_active:
            # Highlight mode (Grayscale + Red)
            semantic_tags = array[:, :, 2]
            display_array = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            display_array[:, :, 0] = semantic_tags * 7 
            display_array[:, :, 1] = semantic_tags * 7 
            display_array[:, :, 2] = semantic_tags * 7 
            
            mask = (semantic_tags == target_tag)
            display_array[mask] = [255, 50, 50]
        else:
            # Standard Instance Segmentation (Raw BGRA converted to RGB)
            display_array = array[:, :, 2::-1]
            
        surface = pygame.surfarray.make_surface(display_array.swapaxes(0, 1))
        
    elif current_modality == 1:
        # RGB
        display_array = array[:, :, 2::-1]
        surface = pygame.surfarray.make_surface(display_array.swapaxes(0, 1))
        
    elif current_modality == 2:
        # Semantic Segmentation
        semantic_tags = array[:, :, 2]
        display_array = PALETTE_LUT[semantic_tags]
        surface = pygame.surfarray.make_surface(display_array.swapaxes(0, 1))

def switch_camera():
    global camera, transform
    if camera is not None:
        camera.stop()
        camera.destroy()
        
    bp_name = MODALITIES[current_modality]["bp"]
    camera_bp = blueprint_library.find(bp_name)
    camera_bp.set_attribute('image_size_x', str(width))
    camera_bp.set_attribute('image_size_y', str(height))
    camera = world.spawn_actor(camera_bp, transform)
    camera.listen(render)

# Initial spawn
switch_camera()

speed = 1.0
rot_sensitivity = 0.2
pan_sensitivity = 0.05
clock = pygame.time.Clock()

try:
    while True:
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
                
            elif event.type == pygame.MOUSEMOTION:
                if event.buttons[2]:
                    dx, dy = event.rel
                    transform.rotation.yaw += dx * rot_sensitivity
                    transform.rotation.pitch = max(-89.0, min(89.0, transform.rotation.pitch - dy * rot_sensitivity))
                
                elif event.buttons[0] or event.buttons[1]:
                    dx, dy = event.rel
                    rt = transform.get_right_vector()
                    
                    transform.location.x -= rt.x * dx * pan_sensitivity
                    transform.location.y -= rt.y * dx * pan_sensitivity
                    transform.location.z += dy * pan_sensitivity

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    speed = min(10.0, speed + 0.2)
                elif event.button == 5:
                    speed = max(0.1, speed - 0.2)
                    
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_KP_PLUS:
                    target_tag = min(255, target_tag + 1)
                elif event.key == pygame.K_KP_MINUS:
                    target_tag = max(0, target_tag - 1)
                elif event.key == pygame.K_m:
                    current_modality = (current_modality + 1) % 3
                    switch_camera()
                elif event.key == pygame.K_f:
                    filter_active = not filter_active

        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]:  transform.rotation.yaw -= 2.0
        if keys[pygame.K_RIGHT]: transform.rotation.yaw += 2.0
        if keys[pygame.K_UP]:    transform.rotation.pitch = min(89.0, transform.rotation.pitch + 2.0)
        if keys[pygame.K_DOWN]:  transform.rotation.pitch = max(-89.0, transform.rotation.pitch - 2.0)
        
        fw = transform.get_forward_vector()
        rt = transform.get_right_vector()
        
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

        camera.set_transform(transform)
        spectator.set_transform(transform)

        if surface is not None:
            display.blit(surface, (0, 0))
            
            # --- Draw Crosshair ---
            cx, cy = width // 2, height // 2
            pygame.draw.line(display, (255, 255, 255), (cx - 10, cy), (cx + 10, cy), 2)
            pygame.draw.line(display, (255, 255, 255), (cx, cy - 10), (cx, cy + 10), 2)
            pygame.draw.circle(display, (255, 50, 50), (cx, cy), 2)
            
            # --- Top Status Bar ---
            mode_name = MODALITIES[current_modality]["name"]
            if current_modality == 0:
                category_name = TAG_NAMES.get(target_tag, "Unknown")
                status = "ON" if filter_active else "OFF"
                ui_text = f"Mode [M]: {mode_name} | Tag [+/-]: {target_tag} ({category_name}) | Filter [F]: {status}"
            else:
                ui_text = f"Mode [M]: {mode_name}"
                
            text_surface = font.render(ui_text, True, (255, 255, 255))
            bg_rect = text_surface.get_rect(topleft=(20, 20))
            pygame.draw.rect(display, (0, 0, 0), bg_rect.inflate(20, 10))
            display.blit(text_surface, (20, 20))

            # --- Target Inspection Info ---
            if center_pixel_info:
                r, g, b = center_pixel_info
                center_cat = TAG_NAMES.get(r, "Unknown")
                
                # Instance ID in CARLA is encoded in G and B channels
                instance_id = (b << 8) | g
                
                inspect_text = f"Aim: Tag {r} ({center_cat}) | G: {g} | B: {b} | (ID: {instance_id})"
                inspect_surface = font.render(inspect_text, True, (255, 255, 0))
                
                inspect_bg = inspect_surface.get_rect(topleft=(20, 65))
                pygame.draw.rect(display, (0, 0, 0), inspect_bg.inflate(20, 10))
                display.blit(inspect_surface, (20, 65))

            pygame.display.flip()

except KeyboardInterrupt:
    pass
finally:
    if camera is not None:
        camera.stop()
        camera.destroy()
    pygame.quit()