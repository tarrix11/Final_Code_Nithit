#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""


from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import numpy as np
import numpy.random as random
import re
import sys
import weakref
import atexit 
import csv
import pandas as pd
import matplotlib.pyplot as plt 


from matplotlib.animation import FuncAnimation
import time 
# use perf time to get time elapsed ###############
# In this file, we can simulate the velocity for ecah time step and plot a graph for each throttle to combine them and get a throttle vs acceleration graph
# FREEZE GEAR CODE TO MEASURE THROTTLE ACCELERATION GRAPH
# change sync time properly for time step instead of perf time 
######################################################################### Testing PID control 
                                                                        # Reduced integral gain to eliminate oscialltion
                                                                        # Increase OIVP throttle to match real instead of Drive By Wire limits
                                                                        # Graph kinda matches the 0-20 mph of OIVP 
                                                                        # great match for both 10 and 20 mph speed steps
                                                                        # attempt to match the OIVP graphs using overlap and further PID tuning
                                                                        # changed code in the PID to adjust throttle acceleration maps and changed intiial velocity
        # freeze pid values code
        # made code more readable and easier to edit and run 
        # can run ramp csv files
        # Allign simulation start time with OVIP response delay
        # Tuning PID to match ramp response
        # Implement Brake PID 
        

file = open("Throttle_AC18_ramp3_result2.csv","w")       # saved csv file name
writer = csv.writer(file)
data = []                              # this is where the data of acceleration and time time_val gets added to
writer.writerow(["Time (s)", "Velocity (m/s)","Acceleration (m/s^2)", "Distance (m)","labeli"])
error_array = []
initial_throttle = 0.798             # carla always accelerates very fast in the beginning
time_step = 0.04       # simulation time step (Dont change because it syncs with 25 fps pygame)
initial_high_acceleration_time = 2.2
velocity_of_acc_graph_transition0 = 6.75
simulation_time_limit = 25
brake_limit = 0.23
inputs = pd.read_csv('ramp_3.csv')

target_speed_step = None         # set target speed in (m/s)   4.47- 8.94   // pass None to the step input out when you dont want to use and put in the ramp input file, code will work 
time_start_delay = 4.2      # 3.9 - 4.8
time_delay_OIVP = 0.8
P = 0.29
I = 0.05 #0.0316
D = 0.002

bP = 0.1
bI = 0.05
bD = 0.00001

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-errors
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error

#from agents.navigation.controller import VehiclePIDController 
# agents.navigation.controller is the file path to the controller file and we are importing the Class VehiclePIDController 
# from file import class 
# ok so we no longer need to import those because the Behavior_Agent itself has an import to another file which imports the PID controller 


pygame.display.set_caption("Tar CARLA Automatic Control")


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def use_throttle_PID(target_speed, current_speed,p_gain,i_gain,d_gain): # where current speed just the velocity variable in the loop
    # This is a very basic 'discrete' PID with the sampling time 'dt'
    error = target_speed - current_speed
    #print(f"error: {error}")
    error_array.append(error)           # every new added error is the last element [-1]
    #print(error_array)
    if len(error_array)>2:
        proportional = error
        integral = sum(error_array)*time_step
        derivative = (error_array[-1] - error_array[-2])/time_step

    else:
        integral = 0
        derivative = 0
        proportional = 0
       
    totalPID = p_gain*proportional + i_gain*integral + d_gain*derivative
    total_gain = np.clip(totalPID,0,1)  
    # after adding up the components multiplied by their gains 
    # we clip it so that it will never exceed a value of 1, which which will be used to multiply the max throttle to get a ratio of max throt
    # so the final throttle will always be less than 0.75 (or a valid throttle value), which is throttle used to acheive the same acceleration as the car 
   
    throttle_limit = highest_throttle_calculation(target_speed,current_speed)
    final_throttle = total_gain*throttle_limit # wait but what if i set the velocity to something else, this will have to change too , needs to be calculated from a function of target speed 

    print(f"P: {p_gain*proportional}")
    print(f"I: {i_gain*integral}")
    print(f"D: {d_gain*derivative}")
    print(f"Sum of components: {totalPID}")
    print(f"GAIN: {total_gain}")
    print(f"Throttle_limit: {throttle_limit}")
    print(f"PID Resultant throttle: {final_throttle}")                 # throt_ratio = 0.75 for this particular graph of OIVP 

    return final_throttle

def use_brake_PID(target_speed, current_speed,bp_gain,bi_gain,bd_gain): # where current speed just the velocity variable in the loop
   
    error = abs(target_speed - current_speed)
    #print(f"error: {error}")
    error_array.append(error)           # every new added error is the last element [-1]
    
    if len(error_array)>2:
        proportional = abs(error)
        integral = sum(error_array)*time_step
        derivative = (error_array[-1] - error_array[-2])/time_step
        print(f"Error: {error}")

    else:
        integral = 0
        derivative = 0
        proportional = 0
       
    totalPID = bp_gain*proportional + bi_gain*integral + bd_gain*derivative
    total_gain = np.clip(totalPID,0,1)  

    final_brake = total_gain*brake_limit # wait but what if i set the velocity to something else, this will have to change too , needs to be calculated from a function of target speed 
    print(f"Brake P: {bp_gain*proportional}")
    print(f"Brake I: {bi_gain*integral}")
    print(f"Brake D: {bd_gain*derivative}")
    print(f"Final brake ratio: {final_brake}")
    

    return final_brake


# THIS FUNCTION CALCULATES THE HIGHEST THROTTLE TO PUT INTO THE PID - CORRESPOND TO THE INCREASING LINEAR SECTION OF THE VELOCITY TIME GRAPH, before it tapers off
# SINCE WE DONT WANT TO PUT A THROTTLE THAT EXCEEDS THAT, WE ONLY REDUCE IT TO HAVE A SMOOTH DECELERATION AT THE END 
    # make this a function of throttle vs acceleration / can i get a function of set velocity vs acceleration of the OIVP ???
    # we are re-calculating highest_throt
    # calculate the acceleration_limit at each velocity from the DATASPEED Graph
    # thottle acceleration graph at (0-6.75 m/s) >> y = 3x - 0.84       // (6.25,1.03) - (0.78,1.5) g=3 c = -0.84

def highest_throttle_calculation(target_speed, current_speed):

    if current_speed < initial_high_acceleration_time:
        highest_throttle = initial_throttle

    elif current_speed >= initial_high_acceleration_time and current_speed < velocity_of_acc_graph_transition0:
        acceleration_limit = 0.072*current_speed + 1.03                 # relationship between max acceleration and current speed at  (0-6.75 m/s)
        print(f"Acceleration LIMIT: {acceleration_limit}")
        highest_throttle = (acceleration_limit + 0.92)/3
            
    elif current_speed >= velocity_of_acc_graph_transition0:
        acceleration_limit = 1.5
        highest_throttle = (acceleration_limit + 0.92)/3
    return highest_throttle






def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    #bps = world.get_blueprint_library().filter(filter)
    bps = world.get_blueprint_library().filter("lincoln")

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(2)                                  # input the generation into int()
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []






# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):                                            # we pass in OBJECT as this is the new way to make a class 
    """ Class representing the surrounding environment """        # it you to subclass built-in classes like init and dict and more flexibility
    

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self._args = args            # self._ is a convention meaning args is not intended to be used outside this class
                                     # we need to pass the args = argparse.parse_arguments so that the object knows to use the commands from the argparser
        self.world = carla_world     # This is passed in as client.get_world to actually get the world 
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        
        # CONFIGUIRE SIMULATION SETTING 
        # setting up class attributes
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get a random blueprint.
        blueprint = random.choice(get_actor_blueprints(self.world, self._actor_filter, self._actor_generation))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # if the player exists already 
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)    # try keyword will return none instead of exception if failed to execute
            
            self.modify_vehicle_physics(self.player)
           

            #VehiclePIDController(self.player,args_lateral = {'K_P': 1.0, 'K_D':0.0,'K_I': 0.0}, args_longitudinal = {'K_P': 1, 'K_D':0.0,'K_I': 0.0})
            # https://github.com/PacktPublishing/Hands-On-Vision-and-Behavior-for-Self-Driving-Cars/blob/master/Chapter10/Packt-Town04-PID.py 
            # Found a PID Controller Github for Carla
            # is this automatic control hard coded, and if so where can i find it so that I can disable it 

            # solve: edit the files in Navigation - behavior agents - run_step class






        while self.player is None:
        
            if not self.map.get_spawn_points():                                     # if not can be used to check if a string,list,array, iterables are Empty
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)

            # Spawn Actor 
            spawn_points = self.map.get_spawn_points()
            spawn_point =  spawn_points[10] 
            #spawn_point = carla.Transform(carla.Location(x=49.20,y=14.10,z= 2),carla.Rotation())
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)            
            self.modify_vehicle_physics(self.player)
            
            

        # check for sync, if not wait for client tick
        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)              # connect to attatched  actor and HUD display with self.player and self.hud
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)    # GNSS measures position, speed, time, orientation etc
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            
            # https://carla.readthedocs.io/en/latest/tuto_G_control_vehicle_physics/ 
            
            # SET DRIVING CONFIG HERE!!
            physics_control.mass = 1916                              # estimated weight of Ford Focus test vehicle (original curb weight 1716, I added 200 kg for passengers and equipment)
            physics_control.use_sweep_wheel_collision = True          # provides a more accurate and realistic model of the vehichle when the wheels hits a bump, instead of just glithcing into a certain direction
            physics_control.drag_coefficient = 0.3  
            physics_control.max_rpm = 6000 
            physics_control.gear_switch_time = 0 # estimated to match the curve of the real car, since this causes significant delay of acceleratio
            physics_control.final_ratio = 1   # we lower the final ratio  by 3 times to have less torque change and shifts in the graph 
            # transmission to wheel ratio OG is 3.2
            #print(f"final ratio wheel: {physics_control.final_ratio}")         
            actor.apply_physics_control(physics_control)

          
        except Exception:
            pass
    
    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

# mainly for quitting

class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    # Exit pygame using events
    def parse_events(self):
        for event in pygame.event.get():            # for every event that happens eg. I press a Key
            if event.type == pygame.QUIT:           # the X icon pressed
                return True                         # when parse_events is true game stops running in main while loop
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds
        global simtime 
        global simtime2
        simtime = timestamp.elapsed_seconds  # get the actual time from the simulation and subtract it by the offset
        temp = timestamp.elapsed_seconds
        simtime2 = temp

        #print(f"time Timestamp: {carla.Timestamp().elapsed_seconds}") # Why does this only output a zero 
        #print(f"FRAMES: {carla.Timestamp().frame}") # Why does this only output a zero 
        
        

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity() 
        
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1),            # this is just the display on the left side template
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

 
# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), attachment.Rigid),
            (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int_32)    # I had to downgrade the numpy version
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ============================================================================== # 10-11 on thursday 
# -- Game Loop ---------------------------------------------------------
# ============================================================================== # have a preentation about what i do to familizalie and allow them to undersatnd what im working towarards and get feedbacl , come up with the experimental plan 


def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()           # initialize pygame - has to be written everytime 
    pygame.font.init()      
    world = None
    print(2*int(args.lol)) # just trying out how to make arg-parse
    try:                    
        if args.seed:
            random.seed(args.seed)       # seed is used if we want it to keep generating the same old random number 

        client = carla.Client(args.host, args.port)     # these args are taken from the argparser on the last section
        client.set_timeout(10)                          # wait 10 seconds max for client server to secure connection

        # Set world 
        #client.load_world('Town03')
    

        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        if args.sync:                                  # the default is set to async, but for smooth repeatable simualtions gotta specify --sync in command line so that client and server have the same time step 
        #if 1 ==1:           # FORCE SYNC remove the hastag above to return to manual sync                   
            settings = sim_world.get_settings()        # we can use the world settings method to customize simulation environment 
            settings.synchronous_mode = True           # set synchronous mode = true so that the sensors and simulation are using the same point in time 
            settings.fixed_delta_seconds = 0.04        # simulation time step set to 1/0.04 = 25 steps per second #### DO NOT CHANGE THIS VALUE 0.04 SINCE IT HAS TO MATCH THE 25 FPS of Pygame
            sim_world.apply_settings(settings)         # we need to apply it back into the settings for it to be executed

            traffic_manager.set_synchronous_mode(True)  # if synchronous mode is enabled, the traffic mananger has to be set to synchronous mode too
        
        
        
        # Creating the screen
        display = pygame.display.set_mode(                      # set height and width of graphics window 
            (args.width, args.height),                          # get data from argparse set/default values, default='1280x720'
            pygame.HWSURFACE | pygame.DOUBLEBUF)                # Double buffering ensures that different parts of the graphics are not refreshed at differnt times causing flickering 
                                                                # Hardware Surface uses the video card (these two arguments ensures smooth operation)
        hud = HUD(args.width, args.height)                      # Head Up Display sizing, any object that requires argparse you also gotta pass in args 
        
        world = World(client.get_world(), hud, args)            # we connect the client to the world which is the simulation environment          
                                                                # World is a class containing the actors and all the features
                                                                # the args defined as arg = argparse.parse_args will tell the object to retreive the instruction from the argparser 
        controller = KeyboardControl(world)                     # object with the exit key functions
        
        if args.agent == "Basic":
            agent = BasicAgent(world.player, 30)                # world.player is the spawned vehicle ,30 max speed if it is set to follow speed limit    
            agent.follow_speed_limits(True)                     # Built in carla functio to follow speed limit
        elif args.agent == "Constant":
            agent = ConstantVelocityAgent(world.player, 30)     
            ground_loc = world.world.ground_projection(world.player.get_location(), 5)
            if ground_loc:
                world.player.set_location(ground_loc.location + carla.Location(z=0.01))
            agent.follow_speed_limits(True)
        elif args.agent == "Behavior":
            agent = BehaviorAgent(world.player, behavior=args.behavior)   # cautious, normal, aggressive: we can set what each does 

        # Set the agent destination
        spawn_points = world.map.get_spawn_points()
        #estination = carla.Transform(carla.Location(x=-57.20,y=14.050,z= 2), carla.Rotation())
        destination = spawn_points[11].location   # set a specific spawn point in this case point 2    #random.choice(spawn_points).location
        agent.set_destination(destination)      # set the destination
        clock = pygame.time.Clock()                 

       

        

        time_elapsed = 0    # initiate printed time 
        distance_travelled = 0
        
        ramp_speed_index = 0
           


        # Allows it to run continuously - the main game loop (so anything you want to run continuously is in here)
        while True:                         # always true unless broken
            clock.tick()                    # computes how many seconds has passed since has passed since the previous call                   
            if args.sync:
                world.world.tick()          # in sync mode, server waits for the client's tick to move on to the next time step 
            else:
                world.world.wait_for_tick()     # this makes the server wait for the client's tick in Async mode

            if controller.parse_events():           # self defined function for quitting when we click quit buttons and it gives a TRUE bool 
                return
           
            world.tick(clock)                       # a carla client signal to tell the server to process    
            world.render(display)
                       
            pygame.display.flip()                   # updates the screen

            if agent.done():                        # if agent reaches set_destination  
                if args.loop:                       # if you want the simulation to loop
                    agent.set_destination(destination)
                    world.hud.notification("Target reached", seconds=4.0)
                    print("The target has been reached, searching for another target")
                else:
                    print("The target has been reached, stopping the simulation")
                    break
            

            velocity = world.player.get_velocity()  # access Python API using .get methods 
            acceleration = world.player.get_acceleration()

            #speed_kmh = round(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2),4)  
            # basically absolute direction and convert m/s to kmh by * 3600/1000
            
            velocity_magnitude= round(math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2),4)
            acceleration_magnitude = round(math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2),4)
            distance = velocity_magnitude*time_step
            distance_travelled = distance_travelled + distance

            
            control = agent.run_step()
            # is this where the control is taken from 
            # if we print out the print(control) and print the data type it should show us whats in there
            # we will then manipulate the velocity / acceleration and the world.player.apply_control(control) will actually exeute the control that you manipulate
            

            control.manual_gear_shift = False # makes manual gear shift and applies it at every step 

            control.gear = 1
            # we have a maximum accleration graph, so we can just set it to not exceed that  # every +0.01 throttle = +1 km/h speed
            if time_elapsed > time_start_delay:  
                                            # this is just to delay the simulation to allow time for the car to drop from spawn
                print('PID applied')
             
                ramp_speed = inputs.iloc[ramp_speed_index,1]               # iloc automatically know to avoid the header
                
                ramp_array = inputs.iloc[:,1]
                ramp_array_size = len(ramp_array)
                           
                experiment_time = time_elapsed-time_start_delay 
                print(f"Experimemt Time: {experiment_time}")

                if experiment_time < time_delay_OIVP:
                    pass

                else:
                    if ramp_speed_index == ramp_array_size -1: 
                        target_speed = ramp_speed
                        print(f"max index reached")
                    else:
                        ramp_speed_index = ramp_speed_index + 1
                        print(f"index: {ramp_speed_index}")
                        print(f"max indexes: {ramp_array_size -1}")

                    if target_speed_step != None:
                        target_speed = target_speed_step
                        print('step inputted')
                    else: 
                        target_speed = ramp_speed

                    print(f"Target Speed: {target_speed}")
                    print(f"Current Speed:{velocity_magnitude}")

                    if target_speed > velocity_magnitude:
                        control.throttle = use_throttle_PID(target_speed, velocity_magnitude,P,I,D) # the numbers are P,I,D gains                                                                                               
                    else:
                        control.throttle = use_throttle_PID(target_speed, velocity_magnitude,P,I,D)
                        control.brake = use_brake_PID(target_speed,velocity_magnitude,bP,bI,bD)
                    
                    world.player.apply_control(control)  # applies the rest of the control = agent.runstep 
                
                data.append([experiment_time, velocity_magnitude, acceleration_magnitude,distance_travelled,'testing9']) # append measured data    
                print(f"Experiment Time: {experiment_time}")
                print("________________________________________________________________")
                   

            # set it trottle to 0 for certain amount of time until car dropped
            # map acceleration to throttle and see how it works 
            # recreate the graph conditions
            # maybe can plot an increasing throttle or many steps of throttle to see behavior 
            else:
                control.throttle = 0 

            time_elapsed = time_elapsed + time_step
      
            if time_elapsed <= simulation_time_limit:
                pass
            else:
                print('Simulation has reached set time limit')
                break
                
            






    finally:

        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()

        pygame.quit()       # check if world still exists and destroys it


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():                                     # this is the primary function, python doesnt have a dedicated main function, this is just convention
    """Main method"""                              # Handles all updates for HUD and ticking of agents and world 

    # Argument Parser in python allows you to add functionalities of of the command line when calling the file
    # eg: python3 file.py -a will allow you to choose which type of agent to pass into the functions (so we dont have to come back to the file to change agent type)
    argparser = argparse.ArgumentParser(                 
        description='CARLA Automatic Control Client')
    
    # argparser.add_argument(
    # '-a'                       the -extention you need to type (value is stored in args.a if the longform)
    # action = 'store_true'      the extention will perform this command without passing anything else
    # choices = ['1','2','3']    space and type this after the extention to choose more specifically what to pass
    # default = '1'              sets the default option if no extention is specified
    # help = 'select the mode'   tells you how to use the extentions 
    # mclock

    #ani = FuncAnimation(fig, BehaviorAgent.update_information(), interval = 1000)  # i put it here so the function is only called once when the behavior agent is initialized
    plt.show()

    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination (if randomized) upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic", "Constant"],
        help="select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    # im just testing my own argparser
    argparser.add_argument(
        '-x','--lol',
        choices = ['1','2','3'],
        help="type 1,2, or 3",
        default = 1)

    args = argparser.parse_args() # THIS LINE IS ESSENTIAL so that when you run your program which takes functions and classes#
                                  # we pass the parameter args to that it knows to take the input from the argparser

    args.width, args.height = [int(x) for x in args.res.split('x')] # spllit seperates string using spacebar into a list 

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':          # __name__ is a special variable, if this python file is ran directly, it sets __name__ = to __main__
    main()                          # so anything inside the if statement will run, in this case, the main() function  
                                    # this statement is just used to check if this is an imported file or is it being ran directly 
                                    # if its being imported and ran, it will not run the main() function
                                    # but anything outside the if-else will be ran anyways unless you do importedModule.main() is ran on the file that is run 

# to SAVE the data appened to the array througout the simulation, save it to the csv when the file will be closed
def save_data():                    
    for row in data:
        writer.writerow(row)

    file.close()
atexit.register(save_data)
