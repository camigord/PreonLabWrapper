import preonpy
import numpy as np
import glob
import os
from utils.utils import *
import logging

logging.basicConfig(filename='log_file.log',level=logging.DEBUG)
logging.debug('Starting Log')

'''
PreonScene interfaces our code with a PreonLab Scene. It allows us to interact with the scene by performing actions and it allows us to capture the current state of the scene.
'''
class PreonScene():
    def __init__(self, args, source_height = 10, test_scene=None):
        self.args = args
        self.min_x = args.min_x
        self.max_x = args.max_x
        self.min_y = args.min_y
        self.max_y = args.max_y
        self.max_volume = args.max_volume
        self.frames_per_action = args.frames_per_action
        self.init_height = source_height
        self.safety_boundary = args.safety_boundary

        # Loading the scene
        preonpy.show_progressbar = False            # This was required to fix some bug with preonpy back when we wrote the code
        if test_scene is not None:
            self.scene = preonpy.Scene(test_scene)
        else:
            self.scene = preonpy.Scene(args.test_path)  # Load the Scene

        # Define the volume capacity of the destination cup in milliliters
        self.cup_capacity = 280.0                   # Capacity of orange cup model when using preonlab spacing = 0.01

        # Load scene objects
        self.solver = self.scene.find_object("PreonSolver_1")
        self.sensor_cup1 = self.scene.find_object("Sensor_Cup1")    # Volume sensor around Cup1 (source)
        self.sensor_cup2 = self.scene.find_object("Sensor_Cup2")    # Volume sensor around Cup2
        self.cup1 = self.scene.find_object("Cup1")
        self.cup2 = self.scene.find_object("Cup2")
        self.Box = self.scene.find_object("BoxDomain_1")
        self.Source = self.scene.find_object("VolumeSource_1")      # Volume source
        self.transfor_group1 = self.scene.find_object("TransformGroup_1")   # Transformation groups used as reference for cups and sensors
        self.transfor_group2 = self.scene.find_object("TransformGroup_2")

        # Get simulation parameters
        self.frame_rate = self.scene.__getitem__("simulation frame rate")
        self.timestep_per_frame = 1.0 / self.frame_rate

        # Set initial volume: Here we could modify the scale of the volume source in order to have different initial volumes
        #self.Source.__setitem__("scale", [0.1, 0.1, self.init_height/100])

        # Set trajectorie points for the first 2 frames (this allows the liquid to settle down)
        init_angle = np.array(self.transfor_group1.__getitem__("euler angles"))[1]
        init_pos = np.array(self.transfor_group1.__getitem__("position"))
        init_frames = 2

        # Keep cups fixed during the first frames while the fluid is generated
        self.trajectorie_angles = self._set_keyframes([self.transfor_group1],[],0,init_angle,'euler angles theta')
        self.trajectorie_posx = self._set_keyframes([self.transfor_group1],[],0,init_pos[0],'position x')
        self.trajectorie_posy = self._set_keyframes([self.transfor_group1],[],0,init_pos[2],'position z')
        self.trajectorie_angles = self._set_keyframes([self.transfor_group1],self.trajectorie_angles,self.timestep_per_frame*init_frames,init_angle,'euler angles theta')
        self.trajectorie_posx = self._set_keyframes([self.transfor_group1],self.trajectorie_posx,self.timestep_per_frame*init_frames,init_pos[0],'position x')
        self.trajectorie_posy = self._set_keyframes([self.transfor_group1],self.trajectorie_posy,self.timestep_per_frame*init_frames,init_pos[2],'position z')

        # Run initial frames to allow liquid to settle down
        self.scene.simulate(0, init_frames)
        self.current_frame = init_frames

        # Get some values from the simulation
        self.init_particles = self.scene.get_statistic("Fluid Particles", self.scene.elapsed_time)

        if self.init_particles != self.max_volume:
            logging.warning('Init particles = ' + str(self.init_particles))

        self.cup1_size = np.round(np.array(self.sensor_cup1.__getitem__("scale"))*100,decimals=2)[[0,2]] # diameter and height in cm
        #self.cup2_size = np.round(np.array(self.cup2.__getitem__("scale"))*100,decimals=2)[[0,2]] # diameter and height in cm
        # NOTE: Using sensor around solid because mesh does not provide the scale.
        self.cup2_size = np.round(np.array(self.sensor_cup2.__getitem__("scale"))*100,decimals=2)[[0,2]] # diameter and height in cm

        # NOTE: Adding a safety margin around cup 2 in order to prevent the real robot to get too close to Cup2
        self.cup2_size[0] += self.safety_boundary * 2  # Add 2*safety_boundary to diameter which will create a safety_boundary margin to the left and right of Cup2
        self.cup2_size[1] += self.safety_boundary  # Add safety_boundary to height which will create Xcm safety margin on top of Cup2

        self.update_stats()

    def update_stats(self):
        self.cup1_pos = np.round(np.array(self.transfor_group1.__getitem__("position"))  *100, decimals = 2) # x,y,z in cm
        self.cup2_pos = np.round(np.array(self.transfor_group2.__getitem__("position"))  *100, decimals = 2) # x,y,z in cm

        self.cup1_angles = np.array(self.transfor_group1.__getitem__("euler angles"))  # phi,theta,psi      - We are interested in theta angle
        self.cup2_angles = np.array(self.transfor_group2.__getitem__("euler angles"))  # phi,theta,psi

        self.vol_cup1 = np.round(self.sensor_cup1.get_statistic("Volume", self.scene.elapsed_time) * 1000000)    # Rounded in milliliter
        self.vol_cup2 = np.round(self.sensor_cup2.get_statistic("Volume", self.scene.elapsed_time) * 1000000)    # Rounded in milliliter

        self.remaining_particles = self.scene.get_statistic("Fluid Particles", self.scene.elapsed_time)

    def get_cups_location(self):
        # We are using only 2 dimensions right now
        return self.cup1_pos[[0,2]], self.cup1_angles[1], self.cup2_pos[[0,2]], self.cup2_angles[1]

    def get_cups_dimensions(self):
        return self.cup1_size, self.cup2_size

    def get_info(self):
        return self.init_particles, self.remaining_particles, self.vol_cup1, self.cup_capacity

    def get_state(self):
        # Get and return current state
        pos_x, pos_y = self.cup1_pos[0], self.cup1_pos[2]           # remember that position 2 in vector represent z-axis (height in simulation)
        theta_angle = self.cup1_angles[1]
        poured_vol = self.vol_cup2
        spilled_vol = self.init_particles - self.remaining_particles

        # Compute distance between the cups
        dist_x = self.cup2_pos[0] - pos_x
        dist_y = (self.cup2_pos[2] + self.cup2_size[1]) - pos_y           # +cup_height in order to locate center of upper ring

        fill_level = poured_vol / self.cup_capacity                       # 0: Cup is emmpty, 1: Cup is full

        theta_angle = theta_angle % 360.0

        # Saving values without noise for visualizing performance
        clean_state = {'delta_x': dist_x,
                       'delta_y': dist_y,
                       'theta': theta_angle,
                       'fill_level': fill_level,
                       'spilled_vol': spilled_vol}

        if self.args.add_noise:
            # NOTE: Adding noise
            dist_x += np.random.normal(0.0, self.args.noise_x)
            dist_y += np.random.normal(0.0, self.args.noise_y)
            theta_angle += np.random.normal(0.0, self.args.noise_theta)
            fill_level += np.random.normal(0.0, self.args.noise_fill_level)
            theta_angle = theta_angle % 360.0

        # Normalizing
        # NOTE: if the level overshoots above cup's capacity, fill_level may become larger than 1, I assume we can neglect that.
        fill_level_norm = get_normalized(fill_level, 0.0, 1.0)
        spilled_vol_norm = get_normalized(spilled_vol, 0.0, self.max_volume)
        pos_x_norm = get_normalized(dist_x, self.args.min_x_dist, self.args.max_x_dist)
        pos_y_norm = get_normalized(dist_y, self.args.min_y_dist, self.args.max_y_dist)
        theta_angle_norm = get_normalized(theta_angle, 0.0, 360.0)

        normalized_state = {'delta_x_norm': pos_x_norm,
                            'delta_y_norm': pos_y_norm,
                            'theta_norm': theta_angle_norm,
                            'fill_level_norm': fill_level_norm,
                            'spilled_vol_norm': spilled_vol_norm,
                            'fill_level': fill_level}       # Adding fill_level without normalizing to compute filling_rate

        return normalized_state, clean_state

    def execute_action(self, vel_x, vel_y, vel_theta):
        '''
        Sets keyframes in simulation according to given action. It assumes that collision detection has already been checked.
        Velocities are given in cm or degrees per second
        '''
        # Calculates the next position of cup1 given velocities in cm/s, degree/s
        posx = (self.cup1_pos[0] + (self.timestep_per_frame * self.frames_per_action) * vel_x) / 100.0   # in meters for simulator
        posy = (self.cup1_pos[2] + (self.timestep_per_frame * self.frames_per_action) * vel_y) / 100.0
        theta = self.cup1_angles[1] + (self.timestep_per_frame * self.frames_per_action) * vel_theta

        # Estimate time when next action will be taken
        time_next = self.scene.elapsed_time + (self.timestep_per_frame * self.frames_per_action)

        # Set keyframes in order to reach desired position with provided velocities
        self.trajectorie_angles = self._set_keyframes([self.transfor_group1],self.trajectorie_angles,time_next,theta,'euler angles theta')
        self.trajectorie_posx = self._set_keyframes([self.transfor_group1],self.trajectorie_posx,time_next,posx,'position x')
        self.trajectorie_posy = self._set_keyframes([self.transfor_group1],self.trajectorie_posy,time_next,posy,'position z')

        # Simulate k frames
        self.scene.simulate(self.current_frame, self.current_frame + self.frames_per_action)
        self.current_frame += self.frames_per_action

        # Update statistics
        self.update_stats()


    def _set_keyframes(self,objects,position_keyframes,time,value,property_name):
        '''
        Append a new point to the trajectories of the given objects and set their keyframes accordingly
        '''
        # There was a problem (bug) with preonpy and it was only accepting "floats" when setting keyframes.
        position_keyframes.append((float(time),float(value),"Linear"))
        for o in objects:
            o.set_keyframes(property_name,position_keyframes)
        return position_keyframes

    def save_scene(self, path):
        self.scene.save(path)

    @property
    def current_time(self):
        return self.scene.elapsed_time
