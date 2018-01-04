import numpy as np
from env.PreonScene import PreonScene
from utils.collision_aux import *
from utils.utils import *

class Preon_env():
    def __init__(self, args):
        self.args = args
        self.step_cost = args.step_cost
        self.collision_cost = args.collision_cost
        self.goal_reward = args.goal_reward
        self.goal_threshold = args.goal_threshold
        self.max_time = args.max_time

        self.min_x = args.min_x
        self.max_x = args.max_x
        self.min_y = args.min_y
        self.max_y = args.max_y
        self.max_volume = args.max_volume

    def reset(self, source_height = 10):
        self.env = PreonScene(self.args, source_height)
        self.time_per_frame = self.env.timestep_per_frame
        self.frame_rate = self.env.frame_rate
        self.max_steps = int(np.ceil(self.max_time * self.frame_rate / self.frames_per_action))
        self.current_step = 0

        state = list(self.env.get_state())

        #self.last_state = state[3:]         # Poured and spilled volumes in last step - initially it is just the starting values
        #flow_rate = 0.0

        self.last_state = state[5]
        filling_rate = 0.0
        state = state[0:3] + list([0.0, 0.0, 0.0]) + state[3:5] + [filling_rate]
        #state.append(flow_rate)
        #state.append(filling_rate)

        return state, self.env.get_info()

    def predict_collision(self, vel_x, vel_y, vel_theta):
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

    def is_out_of_range(self, vel_x, vel_y):
        cup1_pos, cup1_angle, _, _ = self.env.get_cups_location()
        # Estimate future position based on current pos and velocities
        cup1_pos += self.time_per_frame * np.array([vel_x, vel_y])

        if cup1_pos[0] > self.max_x or cup1_pos[0] < self.min_x or cup1_pos[1] > self.max_y or cup1_pos[1] < self.min_y:
            return True
        else:
            return False

    def step(self, action, goal):
        '''
        action is the position relative to current location where cup1 should end after running this frame.
        '''
        collision = False
        delta_x, delta_y, delta_theta = action

        # The required velocities in (cm/s or degree/s) are equal to the displacement time the number of frames per second
        vel_x, vel_y, vel_theta = self.frame_rate * np.array([delta_x, delta_y, delta_theta])

        reward = None

        # Check that action does not end in collision
        if self.predict_collision(vel_x, vel_y, vel_theta) == True or self.is_out_of_range(vel_x, vel_y)==True:
            # Collision detected!
            reward = self.collision_cost
            collision = True
        else:
            self.env.execute_action(vel_x, vel_y, vel_theta)


        # Get environment state
        state = list(self.env.get_state())
        filling_rate = state[5] - self.last_state
        filling_rate = get_normalized(filling_rate, 0.0, 1.0)
        self.last_state = state[5]

        # Add velocities and flow_rate to state
        state = state[0:3] + list(action) + state[3:5] + [filling_rate]

        # Determine if the episode is over -- current_step is necessary to keep track of progress even when collition is detected and simulation doesnt advance in time
        if self.current_step >= self.max_steps or self.get_elapsed_time() >= self.max_time:
            terminal = True
        else:
            terminal = False

        self.current_step += 1

        # Compute reward
        if reward is None:
            if self.was_goal_reached(state, goal):
                reward = self.goal_reward

                '''# Encourage No-op action once goal has been reached
                if abs(delta_x) < 0.001 and abs(delta_y) < 0.001 and abs(delta_theta) < 0.001:
                    reward = self.goal_reward
                    '''
            else:
                reward = self.step_cost
                # NOTE: Trying same reward for collision and step
                #reward = self.collision_cost

        return state, reward, terminal, self.env.get_info(), collision

    def was_goal_reached(self, state, goal):
        # Values are normalized, we need to convert them back
        goal = [get_denormalized(goal[0],0.0,1.0), get_denormalized(goal[1],0.0,self.max_volume)]
        current_vol_state = [get_denormalized(state[6],0.0,1.0), get_denormalized(state[7],0.0,self.max_volume)]

        dist_to_goal = np.absolute(np.array(current_vol_state) - np.array(goal))
        if dist_to_goal[0] > self.goal_threshold[0] or dist_to_goal[1] > self.goal_threshold[1]:
            return False
        else:
            # Goal was reached
            return True

    def estimate_new_reward(self,state,goal,reward):
        '''
        Estimates a new reward based on the new goal. If previous reward corresponds to a collision, returns the same reward.
         - state is the next_state after performing an action in the current transition
        '''
        # NOTE: Trying same reward for collision and step
        if reward != self.collision_cost:
            if self.was_goal_reached(state, goal):
                reward = self.goal_reward
            else:
                reward = self.step_cost
                # NOTE: Trying same reward for collision and step
                #reward = self.collision_cost
        return reward

    def get_elapsed_time(self):
        return self.env.current_time

    def save_scene(self, path):
        self.env.save_scene(path)
