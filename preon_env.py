import numpy as np
import math
from PreonScene import PreonScene
from Collision.collision_aux import *

def get_vertices(pos, angle, size):
    # First get vertices without rotation (clockwise starting top left)
    theta = 360.0 - angle  # Angle is considered to be measured counterclockwise
    diameter, height = size[0], size[1]
    posx, posy = pos[0], pos[1]
    temp_vertices = []
    temp_vertices.append((posx - diameter/2.0, posy + height/2.0))   # Top left
    temp_vertices.append((posx + diameter/2.0, posy + height/2.0))   # Top right
    temp_vertices.append((posx + diameter/2.0, posy - height/2.0))   # Buttom right
    temp_vertices.append((posx - diameter/2.0, posy - height/2.0))   # Buttom left

    # Rotate each point according to angle
    # From theory at (https://math.stackexchange.com/questions/270194/how-to-find-the-vertices-angle-after-rotation)
    vertices = []

    for v in temp_vertices:
        x = np.round((v[0]-posx)*math.cos(math.radians(theta)) - (v[1]-posy)*math.sin(math.radians(theta)) + posx,decimals=2)
        y = np.round((v[0]-posx)*math.sin(math.radians(theta)) + (v[1]-posy)*math.cos(math.radians(theta)) + posy,decimals=2)
        vertices.append((x,y))

    return vertices


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
            print('Collision')
            return True
        else:
            return False
