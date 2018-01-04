import numpy as np
import os
import sys

class AgentParams():  # hyperparameters for drl agents
    def __init__(self):

        # hyperparameters
        self.batch_size       = 64       # batch size during training
        self.rm_size          = 1000000   # memory replay maximum size
        self.gamma            = 0.99     # Discount factor
        self.critic_lr        = 0.001    # Learning rate for critic
        self.actor_lr         = 0.0001   # Learning rate for actor

        self.tau              = 0.001    # moving average for target network

        self.k_goals          = 4        # Number of additional goals to sample and add to replay memory
        self.valid_freq       = 100

class EnvParams():          # Settings for simulation environment
    def __init__(self):
        self.step_cost        = -0.5              # Reward when goal is not reached, but no collision happens
        self.collision_cost   = -1.0              # Reward when collision is detected
        self.goal_reward      = 0.0               # Reward when reaching the goal
        self.max_time         = 10.0              # Maximum length of an episode in seconds
        self.goal_threshold   = [0.1, 25.0]       # Max volume difference (in ml) for goal to be considered achieved

        self.max_lin_disp     = 2.0               # Maximum linear displacement in cm/frame
        self.max_ang_disp     = 10.0              # Maximum angular rotation in degrees/frame

        self.min_x            = -12.0             # Range of possible positions
        self.max_x            = 12.0              # Moving outside this range will result in collision
        self.min_y            = -5.0
        self.max_y            = 20.0
        # self.max_volume       = 468.0             # Maximum initial volume in milliliters
        self.max_volume       = 398.0             # Maximum initial volume in milliliters

        self.frames_per_action = 2                # How many frames to run before selecting a new action

        self.test_path             = "./training_scenes/scene0_realmodels.prscene"  # Path to the PreonLab scene

class Options():
    agent_params  = AgentParams()
    env_params = EnvParams()

    test_height = 10.0
    saved_scenes_dir = '/saved_scenes/'
    summary_dir = './results/tboard_ddpg'
    save_dir = './results/model_ddpg'
