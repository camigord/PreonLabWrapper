import numpy as np
from env.PreonScene import PreonScene
from utils.collision_aux import *

class preon_env():
    def __init__(self, args):
        self.path = args.path
        self.step_cost = args.step_cost
        self.collision_cost = args.collision_cost
        self.goal_reward = args.goal_reward
        self.goal_threshold = args.goal_threshold
        self.max_time = args.max_time

    def reset(self):
        self.env = PreonScene(self.path)
        self.time_per_frame = self.env.timestep_per_frame
        self.frame_rate = self.env.frame_rate
        self.max_steps = int(self.max_time * self.frame_rate)
        self.current_step = 0

        return self.env.get_state(), self.env.get_info()

    def predict_collision(self,vel_x, vel_y, vel_theta):
        # Get current position of both cups
        cup1_pos, cup1_angle, cup2_pos, cup2_angle = self.env.get_cups_location()

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
            return True
        else:
            return False

    def step(self,action, goal):
        vel_x, vel_y, vel_theta = action
        reward = None
        # Check that action does not end in collision
        if self.predict_collision(vel_x, vel_y, vel_theta) == True:
            # Collision detected!
            reward = self.collision_cost
        else:
            self.env.execute_action(vel_x, vel_y, vel_theta)

        # Get environment state
        state = self.env.get_state()

        # Determine if the episode is over
        if self.current_step == self.max_steps or self.get_elapsed_time() >= self.max_time:
            terminal = True
        else:
            terminal = False

        self.current_step += 1

        # Compute reward
        if reward is not None:
            if self.was_goal_reached(state, goal):
                reward = self.goal_reward
            else:
                reward = self.step_cost

        return state, reward, terminal, self.env.get_info()

    def was_goal_reached(self, state, goal):
        dist_to_goal = np.absolute(np.array(state[3:5]) - np.array(goal)) - self.goal_threshold
        if np.sum(np.maximum(dist_to_goal,0)) > 0:
            return False
        else:
            # Goal was reached
            return True

    def get_elapsed_time(self):
        return self.env.current_time

    def save_scene(self, path):
        self.env.save_scene(path)
