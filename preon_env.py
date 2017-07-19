import numpy as np
import math
from PreonScene import PreonScene
from Collision.collision_aux import *

class preon_env():
    def __init__(self, path):
        self.path = path

    def reset(self):
        self.env = PreonScene(self.path)
        self.time_per_frame = self.env.timestep_per_frame

        return self.env.get_state(), self.env.get_info()

    def predict_collision(self,action):
        # Get current position of both cups
        cup1_pos, cup1_angle, cup2_pos, cup2_angle = self.env.get_cups_location()
        # Decompose action
        vel_x, vel_y, vel_theta = action
        # Estimate future position based on current pos and velocities
        cup1_pos += self.time_per_frame * np.array([vel_x, vel_y])
        cup1_angle += self.time_per_frame * vel_theta

        # Collision detection:
        # First we need the 4 points defining the vertices of the cups  (Position of cup is with respect to the center of the cup)
        cup1_size, cup2_size = self.env.get_cups_dimensions()   # Get diameter and height of cups

        # Now we need to compute the coordinates of the 4 vertices of each cup
        v_cup1 = get_vertices(cup1_pos, cup1_angle, cup1_size)
        v_cup2 = get_vertices(cup2_pos, cup2_angle, cup2_size)

        if collide(v_cup1, v_cup2):
            print('Collision!')
            return True
        else:
            return False

    def save_scene(self, path):
        self.env.save_scene(path)
