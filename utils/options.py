import numpy as np
import os
import sys

# NOTE: IMPORTANT!
'''
Most of the code was written under the assumption that 1 particle in simulation represents 1ml of liquid.
Changing the particle size in simulation would require to check the code for inconsistencies, specially variables like max_volume, init_volume, capacity etc
'''

class AgentParams():  # hyperparameters for drl agents
    def __init__(self):

        # Input size of the network
        self.state_dim = 9
        self.action_dim = 3
        self.goal_dim = 2

        # hyperparameters
        self.batch_size       = 64        # batch size during training
        self.rm_size          = 100000    # memory replay maximum size
        self.min_memory_size  = self.batch_size * 5    # The minimum number of episodes in memory replay before starting training
        self.gamma            = 0.99     # Discount factor
        self.critic_lr        = 0.001    # Learning rate for critic
        self.actor_lr         = 0.0001   # Learning rate for actor

        self.tau              = 0.001    # moving average for target network

        self.valid_freq       = 25

class EnvParams():          # Settings for simulation environment
    def __init__(self):
        self.step_cost        = -0.5              # Reward when goal is not reached, but no collision happens
        self.collision_cost   = -1.0              # Reward when collision is detected
        self.goal_reward      = 0.0               # Reward when reaching the goal
        self.max_time         = 20.0              # Maximum length of an episode in seconds
        self.goal_threshold   = [0.1, 25.0]       # Max volume difference (in ml) for goal to be considered achieved

        self.max_lin_disp     = 1                 # Maximum linear displacement in cm/frame     (simulation runs at 5 frames/second by default)
        self.max_ang_disp     = 1                 # Maximum angular rotation in degrees/frame

        # Operation range:
        self.min_x            = -12.0             # Range of possible positions
        self.max_x            = 12.0              # Moving outside this range will result in collision
        self.min_y            = -5.0
        self.max_y            = 20.0

        self.max_volume       = 382.0             # Maximum initial volume in milliliters

        self.frames_per_action = 2                # How many frames to run before selecting a new action

        self.safety_boundary   = 0                # Safety boundary around destination cup to avoid collisions in real worls (in cm)

        self.test_path        = "./training_scenes/scene0_realmodels.prscene"  # Path to the PreonLab scene

        self.add_noise = True                     # Whether to add noise or not
        self.noise_x          = 0.5               # Measurement noise (std) in cm
        self.noise_y          = 0.5               # Measurement noise (std) in cm
        self.noise_theta      = 3.0               # Measurement noise (std) in degrees
        self.noise_fill_level = 0.05              # Measurement noise (std) in fill_level % (equivalent to 5mm height error with current cup and settings)

        # Normalizing values (these values change if operation range changes - min_x, min_y ... etc)
        self.min_x_dist       = -2.0              # Values used to normalize inputs into [-1,1] range. Represent expected range of measurements in cm
        self.max_x_dist       = 22.0              # Computed by analyzing initial position of cup2 and operation range of cup1 (min_x, max_x, min_y, max_y)
        self.min_y_dist       = -15.0
        self.max_y_dist       = 10.0


class Options():
    agent_params  = AgentParams()
    env_params = EnvParams()

    test_scene        = "./training_scenes/scene0_realmodels.prscene"   # This is the scene used for testing (could be different than the one for training...)
    saved_scenes_dir = '/saved_scenes/'
    summary_dir = './results/tboard_ddpg'
    save_dir = './results/model_ddpg'
