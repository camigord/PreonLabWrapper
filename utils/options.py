import numpy as np
import os
import sys

class AgentParams():  # hyperparameters for drl agents
    def __init__(self):

        # hyperparameters
        self.batch_size       = 64       # batch size during training
        self.rm_size          = 1000000    # memory replay maximum size
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
        self.max_time         = 20.0              # Maximum length of an episode in seconds
        self.goal_threshold   = 15.0              # Max volume difference for goal to be considered achieved in milliliters

        # NOTE: these 2 may not be required
        self.max_lin_vel      = 10.0              # Maximum absolute linear velocity in cm/s
        self.max_ang_vel      = 45.0              # Maximum absolute angular velocity in degrees/s

        self.max_lin_disp     = 2.0               # Maximum linear displacement in cm/frame
        self.max_ang_disp     = 10.0              # Maximum angular rotation in degrees/frame

        self.min_x            = -12.0             # Range of possible positions
        self.max_x            = 12.0              # Moving outside this range will result in collision
        self.min_y            = 0.0
        self.max_y            = 20.0
        self.max_volume       = 468.0             # Maximum volume to pour in milliliters

        #self.goal_threshold   = 25.0
        self.path             = "scene1.prscene"  # Path to the PreonLab scene
        #self.path             = "scene2.prscene"    # Scene 2 starts with 364ml
        #self.path             = "scene3.prscene"    # Scene 3 starts with 208ml

class Options():
    agent_params  = AgentParams()
    env_params = EnvParams()
    train = True
    continue_training=True
    test_goal = [50, 0]
    save_scene_to_path = '/saved_scenes/scene2_test' + str(test_goal[0]) + 'ml.prscene'
    summary_dir = './new_results/tboard_ddpg'
    save_dir = './new_results/model_ddpg'
