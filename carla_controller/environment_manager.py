import carla

class Environment_manager:
    def __init__(self, world):
        self.world = world
        self.weather = world.get_weather()

    def set_weather(self, hour  = 12, minute = 0):
        self.set_time(hour,minute)
        self.world.set_weather(self.weather)

    def set_time(self, hour=12, minute=0):
        """Set the time of day in the simulation."""
        time_decimal = hour + (minute / 60.0)
        
        # Sun elevation: peak at noon (+90°), below horizon at night (-90°)
        self.weather.sun_altitude_angle = 90.0 - 15.0 * abs(12.0 - time_decimal)
        
        # Sun azimuth: smooth 360° circular trajectory (15° per hour)
        self.weather.sun_azimuth_angle = (time_decimal * 15.0) % 360.0