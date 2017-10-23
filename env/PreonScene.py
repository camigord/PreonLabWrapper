import preonpy
import numpy as np
import glob
import os

# Coordinates of the destination container opening (x, z and radius in cm + minimum volume source height in cm)
# The minimum source volume height quarantees that initial volume is sufficient
coordinates = {'scene0': [15,10,5, 22],
               'scene1': [14,10,4, 10],
               'scene2': [13,20,3, 11],
               'scene3': [13,12,3, 7],
               'scene4': [15,14,4, 22],
               'scene5': [13.5,5,3.5, 4]}

class PreonScene():
    def __init__(self, args, source_height = 22):
        '''
        source_height represents the ammount of liquid in the first cup at the beginning of the episode. The height should be
        a value lower than 22cm still capable of filling the respective container.
        '''
        self.args = args
        self.min_x = args.min_x
        self.max_x = args.max_x
        self.min_y = args.min_y
        self.max_y = args.max_y
        self.max_volume = args.max_volume
        self.frames_per_action = 1              # How many frames to run before selecting a new action
        self.init_height = source_height

        # Selecting a random scene
        scene_paths = glob.glob(os.getcwd() + "/training_scenes/*.prscene")
        scene_paths.sort()
        scene_path = np.random.choice(scene_paths)
        scene_name = scene_path.split("/")[1].split(".")[0]

        # Loading the scene
        preonpy.show_progressbar = False
        self.scene = preonpy.Scene(scene_path)

        # Load scene objects
        self.solver = self.scene.find_object("PreonSolver_1")
        self.sensor_cup1 = self.scene.find_object("Sensor_Cup1")
        self.sensor_cup2 = self.scene.find_object("Sensor_Cup2")
        self.cup1 = self.scene.find_object("Cup1")
        self.cup2 = self.scene.find_object("Cup2")
        self.Box = self.scene.find_object("BoxDomain_1")
        self.Source = self.scene.find_object("VolumeSource_1")

        # Get simulation parameters
        self.frame_rate = self.scene.__getitem__("simulation frame rate")
        self.timestep_per_frame = 1.0 / self.frame_rate
        #self.time_step = self.solver.__getitem__("timestep")
        #self.steps_per_frame = int((1/self.frame_rate) / self.time_step)

        # Randomly relocate the destination container
        ring_coordinates = coordinates[scene_name]
        real_radius = ring_coordinates[0] - 10   # 10 cm is the distance to right boundarie where objects are originally placed

        cup2_pos = np.round(np.array(self.cup2.__getitem__("position"))  *100, decimals = 2) # x,y,z in cm
        sensor_cup2_pos = np.round(np.array(self.sensor_cup2.__getitem__("position"))  *100, decimals = 2) # x,y,z in cm
        max_delta_x = 30 - real_radius - cup2_pos[0]
        max_delta_z = 20 - ring_coordinates[1]      # Top boundary minus container's height

        delta_x = np.random.randint(0, max_delta_x+1)   # Pick a random displacement in each direction
        delta_z = np.random.randint(0, max_delta_z+1)

        new_pos = (cup2_pos + [delta_x, 0, delta_z]) / 100      # Relocate destination container and sensor volume
        cup2.__setitem__("position", new_pos)
        new_pos = (sensor_cup2_pos + [delta_x, 0, delta_z]) / 100
        sensor_cup2.__setitem__("position", new_pos)

        # Update ring coordinates
        self.ring_location = [ring_coordinates[0]+delta_x, ring_coordinates[1]+delta_z, ring_coordinates[2]]

        # Set trajectorie points for the first 2 frames
        init_angle = np.array(self.cup1.__getitem__("euler angles"))[1]
        init_pos = np.array(self.cup1.__getitem__("position"))
        init_frames = 2

        # Set initial volume
        self.Source.__setitem__("scale", [0.1, 0.1, self.init_height/100])

        self.trajectorie_angles = self._set_keyframes([self.cup1, self.sensor_cup1],[],0,init_angle,'euler angles theta')
        self.trajectorie_posx = self._set_keyframes([self.cup1, self.sensor_cup1],[],0,init_pos[0],'position x')
        self.trajectorie_posy = self._set_keyframes([self.cup1, self.sensor_cup1],[],0,init_pos[2],'position z')
        self.trajectorie_angles = self._set_keyframes([self.cup1, self.sensor_cup1],self.trajectorie_angles,self.timestep_per_frame*init_frames,init_angle,'euler angles theta')
        self.trajectorie_posx = self._set_keyframes([self.cup1, self.sensor_cup1],self.trajectorie_posx,self.timestep_per_frame*init_frames,init_pos[0],'position x')
        self.trajectorie_posy = self._set_keyframes([self.cup1, self.sensor_cup1],self.trajectorie_posy,self.timestep_per_frame*init_frames,init_pos[2],'position z')

        # Run initial frames to allow liquid to settle down
        self.scene.simulate(0, init_frames)
        self.current_frame = init_frames

        # Get some values from the simulation
        self.init_particles = self.scene.get_statistic("Fluid Particles", self.scene.elapsed_time)
        self.cup1_size = np.round(np.array(self.cup1.__getitem__("scale"))*100,decimals=2)[[0,2]] # diameter and height in cm
        self.cup2_size = np.round(np.array(self.cup2.__getitem__("scale"))*100,decimals=2)[[0,2]] # diameter and height in cm

        self.update_stats()

    def update_stats(self):
        self.cup1_pos = np.round(np.array(self.cup1.__getitem__("position"))  *100, decimals = 2) # x,y,z in cm
        self.cup2_pos = np.round(np.array(self.cup2.__getitem__("position"))  *100, decimals = 2) # x,y,z in cm

        self.cup1_angles = np.array(self.cup1.__getitem__("euler angles"))  # phi,theta,psi
        self.cup2_angles = np.array(self.cup2.__getitem__("euler angles"))  # phi,theta,psi

        self.vol_cup1 = np.round(self.sensor_cup1.get_statistic("Volume", self.scene.elapsed_time) * 1000000)    # Rounded in milliliter
        self.vol_cup2 = np.round(self.sensor_cup2.get_statistic("Volume", self.scene.elapsed_time) * 1000000)    # Rounded in milliliter

        self.remaining_particles = self.scene.get_statistic("Fluid Particles", self.scene.elapsed_time)

    def get_cups_location(self):
        # We are using only 2 dimensions right now
        return self.cup1_pos[[0,2]], self.cup1_angles[1], self.cup2_pos[[0,2]], self.cup2_angles[1]

    def get_cups_dimensions(self):
        return self.cup1_size, self.cup2_size

    def get_info(self):
        return self.init_particles, self.remaining_particles, self.vol_cup1

    def get_normalized(self,value,var_min,var_max):
        new_max = var_max + abs(var_min)
        norm_value = ((value + abs(var_min)) - (new_max / 2.0)) / (new_max / 2.0)
        return norm_value

    def get_state(self):
        # Get and return current state
        pos_x, pos_y = self.cup1_pos[0], self.cup1_pos[2]
        theta_angle = self.cup1_angles[1]
        poured_vol = self.vol_cup2
        spilled_vol = self.init_particles - self.remaining_particles

        theta_angle = theta_angle % 360.0

        theta_angle_norm = self.get_normalized(theta_angle, 0.0, 360.0)
        pos_x_norm = self.get_normalized(pos_x, self.min_x, self.max_x)
        pos_y_norm = self.get_normalized(pos_y, self.min_y, self.max_y)
        poured_vol_norm = self.get_normalized(poured_vol, 0.0, self.max_volume)
        spilled_vol_norm = self.get_normalized(spilled_vol, 0.0, self.max_volume)
        return pos_x_norm, pos_y_norm, theta_angle_norm, poured_vol_norm, spilled_vol_norm

    def execute_action(self, vel_x, vel_y, vel_theta):
        '''
        Sets keyframes in simulation according to given action. It assumes that collision detection has already been checked.
        Velocities are given in cm or degrees per second
        '''
        # Calculates the next position of cup1 given velocities in cm/s, degree/s
        posx = (self.cup1_pos[0] + self.timestep_per_frame*vel_x) / 100.0   # in meters for simulator
        posy = (self.cup1_pos[2] + self.timestep_per_frame*vel_y) / 100.0
        theta = self.cup1_angles[1] + self.timestep_per_frame*vel_theta

        # Estimate time when next action will be taken
        time_next = self.scene.elapsed_time + self.frames_per_action*self.timestep_per_frame

        # Set keyframes in order to reach desired position with provided velocities
        self.trajectorie_angles = self._set_keyframes([self.cup1, self.sensor_cup1],self.trajectorie_angles,time_next,theta,'euler angles theta')
        self.trajectorie_posx = self._set_keyframes([self.cup1, self.sensor_cup1],self.trajectorie_posx,time_next,posx,'position x')
        self.trajectorie_posy = self._set_keyframes([self.cup1, self.sensor_cup1],self.trajectorie_posy,time_next,posy,'position z')

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
