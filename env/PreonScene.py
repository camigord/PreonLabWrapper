import preonpy
import numpy as np
import glob
import os
from utils.utils import *

class PreonScene():
    def __init__(self, args, source_height = 10):
        self.args = args
        self.min_x = args.min_x
        self.max_x = args.max_x
        self.min_y = args.min_y
        self.max_y = args.max_y
        self.max_volume = args.max_volume
        self.frames_per_action = args.frames_per_action
        self.init_height = source_height

        # Loading the scene
        preonpy.show_progressbar = False
        self.scene = preonpy.Scene(args.test_path)

        self.cup_capacity = 280.0       # Capacity of orange cup model when using preonlab spacing = 0.01

        # Load scene objects
        self.solver = self.scene.find_object("PreonSolver_1")
        self.sensor_cup1 = self.scene.find_object("Sensor_Cup1")
        self.sensor_cup2 = self.scene.find_object("Sensor_Cup2")
        self.cup1 = self.scene.find_object("Cup1")
        self.cup2 = self.scene.find_object("Cup2")
        self.Box = self.scene.find_object("BoxDomain_1")
        self.Source = self.scene.find_object("VolumeSource_1")
        self.transfor_group1 = self.scene.find_object("TransformGroup_1")
        self.transfor_group2 = self.scene.find_object("TransformGroup_2")

        # Get simulation parameters
        self.frame_rate = self.scene.__getitem__("simulation frame rate")
        self.timestep_per_frame = 1.0 / self.frame_rate

        # Set trajectorie points for the first 2 frames
        init_angle = np.array(self.transfor_group1.__getitem__("euler angles"))[1]
        init_pos = np.array(self.transfor_group1.__getitem__("position"))
        init_frames = 2

        # Set initial volume
        #self.Source.__setitem__("scale", [0.1, 0.1, self.init_height/100])

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
        self.cup1_size = np.round(np.array(self.sensor_cup1.__getitem__("scale"))*100,decimals=2)[[0,2]] # diameter and height in cm
        #self.cup2_size = np.round(np.array(self.cup2.__getitem__("scale"))*100,decimals=2)[[0,2]] # diameter and height in cm
        # NOTE: Using sensor around solid because mesh does not provide the scale.
        self.cup2_size = np.round(np.array(self.sensor_cup2.__getitem__("scale"))*100,decimals=2)[[0,2]] # diameter and height in cm

        self.update_stats()

    def update_stats(self):
        self.cup1_pos = np.round(np.array(self.transfor_group1.__getitem__("position"))  *100, decimals = 2) # x,y,z in cm
        self.cup2_pos = np.round(np.array(self.transfor_group2.__getitem__("position"))  *100, decimals = 2) # x,y,z in cm

        self.cup1_angles = np.array(self.transfor_group1.__getitem__("euler angles"))  # phi,theta,psi
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
        pos_x, pos_y = self.cup1_pos[0], self.cup1_pos[2]
        theta_angle = self.cup1_angles[1]
        poured_vol = self.vol_cup2
        spilled_vol = self.init_particles - self.remaining_particles

        theta_angle = theta_angle % 360.0

        theta_angle_norm = get_normalized(theta_angle, 0.0, 360.0)

        # NOTE: Replace position by distance between cups
        dist_x = self.cup2_pos[0] - pos_x
        dist_y = (self.cup2_pos[1] + self.cup2_size[1]) - pos_y           # +cup_height in order to locate center of upper ring

        '''min_x_dist = -2.0
        max_x_dist = 28.0
        min_y_dist = -30.0
        max_y_dist = 0.0
        '''
        min_x_dist = -5.0
        max_x_dist = 28.0
        min_y_dist = -30.0
        max_y_dist = -5.0

        pos_x_norm = get_normalized(dist_x, min_x_dist, max_x_dist)
        pos_y_norm = get_normalized(dist_y, min_y_dist, max_y_dist)

        fill_level = poured_vol / self.cup_capacity           # 0: Cup is emmpty, 1: Cup is full
        # NOTE: if the level overshoots above cup's capacity, fill_level may become larger than 1, I assume we can neglect that.
        fill_level_norm = get_normalized(fill_level, 0.0, 1.0)

        #poured_vol_norm = get_normalized(poured_vol, 0.0, self.max_volume)
        spilled_vol_norm = get_normalized(spilled_vol, 0.0, self.max_volume)
        return pos_x_norm, pos_y_norm, theta_angle_norm, fill_level_norm, spilled_vol_norm, fill_level

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
        position_keyframes.append((float(time),float(value),"Linear"))
        for o in objects:
            o.set_keyframes(property_name,position_keyframes)
        return position_keyframes

    def save_scene(self, path):
        self.scene.save(path)

    @property
    def current_time(self):
        return self.scene.elapsed_time
