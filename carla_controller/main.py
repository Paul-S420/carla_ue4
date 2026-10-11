import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import carla
import pygame
import threading
import cmd
import sys
import queue
import time

from data_generator import DataGenerator
from traffic_generator import TrafficGenerator
from visualizer import DisplayManager
from environment_manager import Environment_manager

host_ip = 'urban1.ifl.kit.edu'
tick_duration = 0.05  # Duration of each tick in seconds

class CarlaSimulationManager:
    def __init__(self):
        self.client = carla.Client(host_ip, 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        self.running = True
        self.ego_spawned = False

        #Initialize Traffic Manager
        self.traffic_manager = self.client.get_trafficmanager(8000)
        
        # Centralized Synchronization Setup
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = tick_duration
        self.world.apply_settings(settings)

        # Synchronize Traffic Manager with the server
        self.traffic_manager.set_synchronous_mode(True)
        
        # Initialize the data collection module and command queue
        self.data_module = DataGenerator(self.world)
        self.traffic_module = TrafficGenerator(self.client, self.world, self.traffic_manager)
        self.environment_module = Environment_manager(world=self.world)
        self.command_queue = queue.Queue()

        # Display Variables
        self.display_manager = None
        self.show_display = False

        # Performance Tracking Variables
        self.is_recording = False
        self.real_start_time = 0.0
        self.tick_count = 0

    def _start_recording_timer(self):
        if not self.is_recording:
            self.is_recording = True
            self.real_start_time = time.time()
            self.tick_count = 0

    def spawn_sem_camera(self, show=False):
        self._start_recording_timer()
        if not self.ego_spawned:
            self.data_module.spawn_ego_vehicle()
            self.ego_spawned = True
        self.data_module.spawn_sensor(sensor_type="semantic_segmentation")
        self.show_display = self.show_display or show
        print("Semantic camera initialized.")

    def spawn_rgb_camera(self, show=False):
        self._start_recording_timer()
        if not self.ego_spawned:
            self.data_module.spawn_ego_vehicle()
            self.ego_spawned = True
        self.data_module.spawn_sensor(sensor_type="rgb")
        self.show_display = self.show_display or show
        print("RGB camera initialized.")

    def spawn_ins_camera(self, show=False):
        self._start_recording_timer()
        if not self.ego_spawned:
            self.data_module.spawn_ego_vehicle()
            self.ego_spawned = True
        self.data_module.spawn_sensor(sensor_type="instance_segmentation")
        self.show_display = self.show_display or show
        print("Instance segmentation camera initialized.")

    def spawn_lidar_sensor(self, show=False):
        self._start_recording_timer()
        if not self.ego_spawned:
            self.data_module.spawn_ego_vehicle()
            self.ego_spawned = True
        self.data_module.spawn_sensor('lidar')
        print("LiDAR sensor initialized.")


    def process_commands(self):
        while not self.command_queue.empty():
            command = self.command_queue.get()
            
            # 1. Handle Tuple commands (for CLI arguments)
            if isinstance(command, tuple):
                action, args = command
                if action == "spawn_traffic":
                    self.traffic_module.spawn_traffic(args['vehicles'], args['walkers'])
                    print("Spawned Vehicle IDs:")
                    print([v.id for v in self.world.get_actors().filter('vehicle.*')])
                elif action == "start_sem":
                    self.spawn_sem_camera(show=args.get('show', False))
                    if args.get('bbox'):
                        self.data_module.spawn_bbox_modality()
                elif action == "start_rgb":
                    self.spawn_rgb_camera(show=args.get('show', False))
                    if args.get('bbox'):
                        self.data_module.spawn_bbox_modality()
                elif action == "start_ins":
                    self.spawn_ins_camera(show=args.get('show', False))
                    if args.get('bbox'):
                        self.data_module.spawn_bbox_modality()
                elif action == "start_lidar":
                    self.spawn_lidar_sensor(show=args.get('show', False))
                    if args.get('bbox'):
                        self.data_module.spawn_bbox_modality()
                    print("LiDAR recording initialized.")
                elif action == "set_weather":
                    hour = args.get("hour", 12)
                    minute = args.get("minute", 0)
                    self.environment_module.set_weather(hour, minute)
                    print(f"set time to {hour}:{minute}")

            
            # 2. Handle simple string commands
            else:
                if command == "clear_traffic":
                    self.traffic_module.destroy()
                elif command == "stop":
                    self.stop()
                elif command == "exit":
                    self.destroy()

    def _print_performance_metrics(self):
        if self.is_recording:
            real_time_elapsed = time.time() - self.real_start_time
            sim_time_elapsed = self.tick_count * tick_duration
            real_fps = self.tick_count / real_time_elapsed if real_time_elapsed > 0 else 0.0
            
            print("\n--- Recording Stopped ---")
            print(f"Recorded frames: {self.tick_count}")
            print(f"Processed {sim_time_elapsed:.2f} seconds of simulated driving in {real_time_elapsed:.2f} seconds of real time.")
            print(f"Average execution speed: {real_fps:.2f} real-time frames per second.\n")
            
            self.is_recording = False
           

    def tick(self):
        self.process_commands()
        self.world.tick()
        
        if self.is_recording:
            self.tick_count += 1
        
        # Only poll pygame events if the video system is currently active
        if pygame.display.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.command_queue.put("exit")
        
        synced_data = self.data_module.process_data()
        
        if self.show_display and synced_data:
            if self.display_manager is None:
                self.display_manager = DisplayManager(self.data_module.active_modalities)
            self.display_manager.render(synced_data)

        # if self.tick_count % 20 == 0 and self.tick_count > 0:
        #     print(f"Tick {self.tick_count}: Processed {self.tick_count * tick_duration:.2f} seconds of simulation time.")

    def stop(self):
        self.data_module.destroy()
        self.ego_spawned = False
        
        if self.display_manager:
            self.display_manager.destroy()
            self.display_manager = None
        self.show_display = False
        
        if self.is_recording:
            self._print_performance_metrics()
        else:
            print("Simulation stopped. Cleaned up actors.")

    def destroy(self):
        self.running = False
        
        self._print_performance_metrics()
            
        self.data_module.destroy()
        self.traffic_module.destroy()

        # Revert Traffic Manager and World settings
        self.traffic_manager.set_synchronous_mode(False)
        self.world.apply_settings(self.original_settings)
        print("Simulation terminated. Cleaned up actors.")

        if self.display_manager:
            self.display_manager.destroy()

class SimulationCLI(cmd.Cmd):
    intro = 'Type help or ? to list commands.\n'
    prompt = '(carla-cli) '

    def __init__(self, manager):
        super().__init__()
        self.manager = manager

    def parse_args(self, arg):
        """Helper to extract 'show' and 'bbox' arguments."""
        args = arg.lower().split()
        return {"show": "show" in args, "bbox": "bbox" in args}

    def do_start_sem(self, arg):
        """Spawn ego vehicle and start semantic data collection. Usage: start_sem [show] [bbox]"""
        self.manager.command_queue.put(("start_sem", self.parse_args(arg)))

    def do_start_rgb(self, arg):
        """Spawn ego vehicle and start rgb data collection. Usage: start_rgb [show] [bbox]"""
        self.manager.command_queue.put(("start_rgb", self.parse_args(arg)))

    def do_start_ins(self, arg):
        """Spawn ego vehicle and start instance segmentation data collection. Usage: start_ins [show] [bbox]"""
        self.manager.command_queue.put(("start_ins", self.parse_args(arg)))

    def do_start_lidar(self, arg):
        """Spawn ego vehicle and start LiDAR data collection. Usage: start_lidar [bbox]"""
        self.manager.command_queue.put(("start_lidar", self.parse_args(arg)))

    def do_start_all(self, arg):
        """Spawn ego vehicle and start all data collection. Usage: start_all [bbox]"""
        options = self.parse_args(arg)
        self.manager.command_queue.put(("start_sem", options))
        self.manager.command_queue.put(("start_rgb", options))
        self.manager.command_queue.put(("start_ins", options))
        self.manager.command_queue.put(("start_lidar", options))

    def do_spawn_traffic(self, arg):
        """Spawn background traffic. Usage: spawn_traffic [vehicles] [walkers] (e.g., spawn_traffic 50 30)"""
        args = arg.split()
        try:
            num_vehicles = int(args[0]) if len(args) > 0 else 30
            num_walkers = int(args[1]) if len(args) > 1 else 30
            self.manager.command_queue.put(("spawn_traffic", {"vehicles": num_vehicles, "walkers": num_walkers}))
        except ValueError:
            print("Error: Arguments must be integers. Example: spawn_traffic 50 30")

    def do_clear_traffic(self, arg):
        """Removes all background traffic from the map."""
        self.manager.command_queue.put("clear_traffic")

    def do_set_time(self, arg):
        """Set simulated time of day. Usage: set_time [hour] [minute]"""
        args = arg.split()
        try:
            hour = int(args[0]) if len(args)>0 else 12
            minute = int(args[1]) if len(args)>1 else 0
            self.manager.command_queue.put(("set_weather", {"hour":hour, "minute":minute}))
        except ValueError:
            print("Error: Arguments must be integers. Example: set_time 04 20")

    def do_exit(self, arg):
        """Exit the simulation safely."""
        self.manager.command_queue.put("exit")
        return True

    def do_stop(self, arg):
        """Stop the simulation and clean up actors."""
        self.manager.command_queue.put("stop")
        
def cli_thread(manager):
    cli = SimulationCLI(manager)
    cli.cmdloop()

def main():
    pygame.init()
    
    manager = CarlaSimulationManager()
    
    # Start the CLI in a parallel daemon thread
    thread = threading.Thread(target=cli_thread, args=(manager,))
    thread.daemon = True
    thread.start()
    
    try:
        # Main simulation loop continuously ticks the server
        while manager.running:
            manager.tick()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Shutting down...")
    except Exception as e:
        print(f"\nCRITICAL ERROR in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # GUARANTEE cleanup runs even if the loop crashed or Ctrl+C was pressed
        if manager.running:
            manager.destroy()
            
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    main()
