import carla
import random

class TrafficGenerator:
    def __init__(self, client, world, traffic_manager):
        self.client = client
        self.world = world
        self.traffic_manager = traffic_manager
        
        self.vehicle_ids = []
        self.walker_ids = []
        self.controller_ids = []

    def spawn_traffic(self, num_vehicles, num_walkers):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

        self.spawn_vehicles(num_vehicles)
        self.spawn_walkers(num_walkers)

        self.traffic_manager.set_synchronous_mode(True)

        settings.synchronous_mode = True
        self.world.apply_settings(settings)



    def spawn_vehicles(self, num_vehicles):
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Get every vehicle in the library
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        
        # Filter out bicycles and motorcycles to prevent Traffic Manager crashes/jams
        blueprints = [bp for bp in blueprints if bp.has_attribute('number_of_wheels') and bp.get_attribute('number_of_wheels').as_int() == 4]
        
        num_vehicles = min(num_vehicles, len(spawn_points))
        
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        
        batch = []
        for spawn_point in random.sample(spawn_points, num_vehicles):
            bp = random.choice(blueprints)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            bp.set_attribute('role_name', 'autopilot')
            
            # Batch spawn and set autopilot in one atomic network command
            batch.append(SpawnActor(bp, spawn_point).then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port())))
            
        results = self.client.apply_batch_sync(batch, True) # Ticks the world to register actors
        for res in results:
            if not res.error:
                self.vehicle_ids.append(res.actor_id)
        print(f"Spawned {len(self.vehicle_ids)} background vehicles.")

    def spawn_walkers(self, num_walkers):
        SpawnActor = carla.command.SpawnActor
        walker_bps = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        
        # 1. Generate safe spawn points
        spawn_points = []
        for _ in range(num_walkers):
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point = carla.Transform(loc)
                spawn_point.location.z += 2
                spawn_points.append(spawn_point)

        # 2. Spawn Walkers
        batch = []
        walker_speeds = []
        for spawn_point in spawn_points:
            bp = random.choice(walker_bps)
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            
            if bp.has_attribute('speed'):
                speed = random.choice(bp.get_attribute('speed').recommended_values)
                walker_speeds.append(float(speed))
            else:
                walker_speeds.append(0.0)
                
            batch.append(SpawnActor(bp, spawn_point))
            
        results = self.client.apply_batch_sync(batch, True)
        
        active_walker_speeds = []
        for i, res in enumerate(results):
            if not res.error:
                self.walker_ids.append(res.actor_id)
                active_walker_speeds.append(walker_speeds[i])

        # 3. Spawn Controllers mapping strictly to successful walkers
        controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        batch = [SpawnActor(controller_bp, carla.Transform(), w_id) for w_id in self.walker_ids]
        
        results = self.client.apply_batch_sync(batch, True)
        for res in results:
            if not res.error:
                self.controller_ids.append(res.actor_id)

        # 4. Start Controllers safely
        controllers = self.world.get_actors(self.controller_ids)
        for i, controller in enumerate(controllers):
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(active_walker_speeds[i])

        # try to fix invisible pedestrians bug for LIDAR
        all_actors = self.world.get_actors()
        for actor in all_actors:
            if "walker.pedestrian." in actor.type_id:
                actor.set_collisions(True)
                actor.set_simulate_physics(True)
            
        print(f"Spawned {len(self.walker_ids)} pedestrians.")

    def destroy(self):
        """Safely removes only the background traffic spawned by this instance."""
        if not self.vehicle_ids and not self.walker_ids:
            return

        controllers = self.world.get_actors(self.controller_ids)
        for controller in controllers:
            controller.stop()
            
        commands = [carla.command.DestroyActor(x) for x in self.vehicle_ids + self.walker_ids + self.controller_ids]
        self.client.apply_batch_sync(commands, True)
        
        self.vehicle_ids.clear()
        self.walker_ids.clear()
        self.controller_ids.clear()
        print("Background traffic cleared.")