import preonpy
import numpy as np

class PreonScene():
    def __init__(self, path):
        self.path = path

        # Loading the scene
        preonpy.show_progressbar = False
        self.scene = preonpy.Scene(self.path)

        # Load scene objects
        self.solver = self.scene.find_object("PreonSolver_1")
        self.sensor_cup1 = self.scene.find_object("Sensor_Cup1")
        self.sensor_cup2 = self.scene.find_object("Sensor_Cup2")
        self.cup1 = self.scene.find_object("Cup1")
        self.cup2 = self.scene.find_object("Cup2")
        self.Box = self.scene.find_object("BoxDomain_1")

        # Get simulation parameters
        self.frame_rate = self.scene.__getitem__("simulation frame rate")
        self.timestep_per_frame = 1.0 / self.frame_rate
        #self.time_step = self.solver.__getitem__("timestep")
        #self.steps_per_frame = int((1/self.frame_rate) / self.time_step)

        # Run initial frames to allow liquid to settle down
        self.scene.simulate(0, 2)
        #self.simulate_frame(num_frames=2)

        # Get some values from the simulation
        self.init_particles = self.scene.get_statistic("Fluid Particles", self.scene.elapsed_time)
        self.cup1_size = np.round(np.array(self.cup1.__getitem__("scale"))*100,decimals=2)[[0,2]] # diameter and height in cm
        self.cup2_size = np.round(np.array(self.cup2.__getitem__("scale"))*100,decimals=2)[[0,2]] # diameter and height in cm

        self.update_stats()

        #return self.get_state(), self.get_info()

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
        return self.init_particles, self.remaining_particles, self.vol_cup1,

    def get_state(self):
        # Get and return current state
        return self.cup1_pos[0], self.cup1_pos[2], self.cup1_angles[1], self.vol_cup2, self.init_particles - self.remaining_particles

    def execute_action(self, vel_x, vel_y, vel_theta):
        # Calculates the next position of cup1 given velocities in cm/s, degree/s
        posx = self.cup1_pos[0] + self.timestep_per_frame*vel_x
        posy = self.cup1_pos[2] + self.timestep_per_frame*vel_y
        theta = self.cup1_angles[1] + self.timestep_per_frame*vel_theta

        # TODO: Need to keep history of the entire trajectorie or read it from simulation.
        # TODO: Need placeholder for X, Y, and Theta.
        # TODO: Need to have previously set keyframes when starting simulation and running 2 frames.

    '''
    def simulate_frame(self,num_frames=1):
        for frame in range(num_frames):
            # Simulates one frame considering the frame rate and the time step values defined by the scene
            for i in range(self.steps_per_frame):
                self.scene.simulate_step()
    '''
