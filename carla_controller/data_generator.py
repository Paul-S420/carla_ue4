import carla
import queue
import weakref
import os
import random
import datetime
import json

from bounding_box_utils import get_mask_bounding_boxes

fov = "90"
image_size_x = "800"
image_size_y = "600"

class DataGenerator:
    def __init__(self, world):
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.ego_vehicle = None
        self.sensors = []
        
        self.image_queue = queue.Queue()
        self.frame_buffer = {}
        
        self.active_modalities = []
        self.output_dirs = {}
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        self.current_session_dir = None

    def _setup_output_dir(self, modality_name):
        if self.current_session_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_session_dir = os.path.join(self.script_dir, "data", timestamp)
            
        path = os.path.join(self.current_session_dir, f"_out_{modality_name}")
        os.makedirs(path, exist_ok=True)
        self.output_dirs[modality_name] = path

    def spawn_ego_vehicle(self, spawn_point=None):
        if not spawn_point:
            spawn_point = random.choice(self.world.get_map().get_spawn_points())

        # Preserved from your older script: original police charger blueprint
        vehicle_bp = self.blueprint_library.filter("vehicle.dodge.charger_police")[0]
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.ego_vehicle.set_autopilot(True)

    def spawn_sensor(self, sensor_type):
        """Universally handles spawning for cameras and LiDAR, permanently ensuring instance segmentation and bboxes are active."""
        if self.ego_vehicle is None:
            print("Error: Spawn ego vehicle first.")
            return

        # Ensure instance segmentation is always spawned alongside any sensor to support permanent bounding boxes
        modalities_to_spawn = [sensor_type]
        if sensor_type != "instance_segmentation" and "instance_segmentation" not in self.active_modalities:
            modalities_to_spawn.append("instance_segmentation")
        
        # Ensure bbox output directory is initialized permanently
        if "bbox" not in self.output_dirs:
            self._setup_output_dir("bbox")

        for modality in modalities_to_spawn:
            if modality not in self.active_modalities:
                self.active_modalities.append(modality)
                self._setup_output_dir(modality)

            if modality == 'lidar':
                sensor_bp = self.blueprint_library.find('sensor.lidar.ray_cast_semantic')
                sensor_bp.set_attribute('channels', '32')
                sensor_bp.set_attribute('range', '50.0')
                sensor_bp.set_attribute('points_per_second', '100000')
                sensor_bp.set_attribute('rotation_frequency', '20')
                # sensor_bp.set_attribute('upper_fov', '20.0') # default 10.0, changed to view traffic lights
                # sensor_bp.set_attribute('lower_fov', '-20') # default -30.0, changed to view traffic lights
                
                sensor_transform = carla.Transform(carla.Location(x=1.5, z=2.6))
            else:
                sensor_bp = self.blueprint_library.find('sensor.camera.' + modality)
                sensor_bp.set_attribute('image_size_x', image_size_x)
                sensor_bp.set_attribute('image_size_y', image_size_y)
                sensor_bp.set_attribute('fov', fov)
                
                sensor_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            
            sensor = self.world.spawn_actor(sensor_bp, sensor_transform, attach_to=self.ego_vehicle)
                
            weak_self = weakref.ref(self)
            sensor.listen(lambda data, m=modality: DataGenerator._sensor_callback(weak_self, data, m))
            self.sensors.append(sensor)

    @staticmethod
    def _sensor_callback(weak_self, image, sensor_name):
        self = weak_self()
        if not self:
            return
        
        self.image_queue.put((image.frame, sensor_name, image))

    def process_data(self):
        """Called every tick. Saves data and returns the latest synced frame."""
        if not self.active_modalities:
            return None

        current_frame = None
        latest_synced_frame = None

        while not self.image_queue.empty():
            frame, sensor_name, image = self.image_queue.get()
            current_frame = frame
            
            if frame not in self.frame_buffer:
                self.frame_buffer[frame] = {}
            self.frame_buffer[frame][sensor_name] = image
            
            if len(self.frame_buffer[frame]) == len(self.active_modalities):
                
                if "semantic_segmentation" in self.frame_buffer[frame]:
                    sem_image = self.frame_buffer[frame]["semantic_segmentation"]
                    sem_image.convert(carla.ColorConverter.CityScapesPalette)
                    sem_image.save_to_disk(f"{self.output_dirs['semantic_segmentation']}/{frame:06d}.png")
                    
                if "rgb" in self.frame_buffer[frame]:
                    rgb_image = self.frame_buffer[frame]["rgb"]
                    rgb_image.save_to_disk(f"{self.output_dirs['rgb']}/{frame:06d}.png")

                if "instance_segmentation" in self.frame_buffer[frame]:
                    ins_image = self.frame_buffer[frame]["instance_segmentation"]
                    ins_image.save_to_disk(f"{self.output_dirs['instance_segmentation']}/{frame:06d}.png")
                    
                    # Bounding boxes are permanently enabled and generated from instance segmentation
                    bboxes = get_mask_bounding_boxes(ins_image)
                    with open(f"{self.output_dirs['bbox']}/{frame:06d}.json", "w") as f:
                        json.dump({"frame": frame, "boxes": bboxes}, f)

                if "lidar" in self.frame_buffer[frame]:
                    lidar_data = self.frame_buffer[frame]["lidar"]
                    lidar_data.save_to_disk(f"{self.output_dirs['lidar']}/{frame:06d}.ply")
                
                # CACHE THE FRAME DATA BEFORE DELETION
                latest_synced_frame = self.frame_buffer[frame].copy()
                del self.frame_buffer[frame]

        # Prevent memory leak (fixed line break bug)
        if current_frame is not None:
            stale_frames = [f for f in self.frame_buffer.keys() if f < current_frame - 10]
            for f in stale_frames:
                del self.frame_buffer[f]

        return latest_synced_frame

    def destroy(self):
        for sensor in self.sensors:
            sensor.stop()
            sensor.destroy()
        self.sensors.clear()
        self.active_modalities.clear()
            
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

        # Purge remaining data to ensure a clean start later
        self.frame_buffer.clear()
        while not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                break
                
        # Reset the session directory so the next start creates a new timestamp folder
        self.current_session_dir = None
        self.output_dirs.clear()