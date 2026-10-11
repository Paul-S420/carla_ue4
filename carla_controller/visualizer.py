import pygame
import numpy as np
import carla

class DisplayManager:
    def __init__(self, modalities, img_width=800, img_height=600):
        self.modalities = modalities
        self.img_width = img_width
        self.img_height = img_height
        
        pygame.display.init()
        # Dynamically widen the window to fit all active cameras side-by-side
        window_width = self.img_width * len(modalities)
        self.display = pygame.display.set_mode(
            (window_width, self.img_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CARLA Sensor Feed")

    def render(self, frame_data):
        if not self.display:
            return

        for i, modality in enumerate(self.modalities):
            if modality in frame_data:
                carla_image = frame_data[modality]
                
                # 1. Convert raw flat buffer to 3D numpy array (Height, Width, BGRA)
                array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (carla_image.height, carla_image.width, 4))
                
                # 2. Slice off the Alpha channel and reverse BGR to RGB
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                
                # 3. Swap axes from (Height, Width) to Pygame's (Width, Height) expectation
                surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
                
                # 4. Blit to the screen at the correct horizontal offset
                self.display.blit(surface, (i * self.img_width, 0))
        
        pygame.display.flip()

    def destroy(self):
        pygame.display.quit()
        self.display = None