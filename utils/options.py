from __future__ import absolute_import
from __future__ import division
import numpy as np
import os
import visdom
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from utils.util import *

class Params(object):
    def __init__(self):

        self.root_dir    = os.getcwd()

        # training signature
        self.visdom_port = 8098
        self.visualize = False          # Create Visdom instance

        # training configuration
        self.mode        = 1            # 1(train) | 2(test model_file)
        self.continue_training = True   # Continues training if a model already exists, otherwise starts from 0

        # Preonlab scene
        self.scene_path = self.root_dir + "/scene1.prscene"

        self.seed        = 123

        self.use_cuda           = False   # Not used
        self.dtype              = torch.FloatTensor

        # model files
        self.model_dir  = self.root_dir + "/models"
        self.load_model = False

        if (self.continue_training or self.mode == 2) and (os.path.exists(self.model_dir + "/actor.pkl") and os.path.exists(self.model_dir + "/critic.pkl")):
            self.load_model  = True
        elif self.mode == 2:
            prRed("Pre-Trained model does not exist, Testing aborted!!!")
            sys.exit()

        if self.visualize:
            self.vis = visdom.Visdom(port=self.visdom_port)
            # TODO: Create TMUX session and start visdom server
            #self.logger.warning("bash$: source activate pytorchenv")
            #self.logger.warning("bash$: python -m visdom.server")           # activate visdom server on bash
            #self.logger.warning("http://localhost:8097/env/" + self.refs)   # open this address on browser

class AgentParams(Params):  # hyperparameters for drl agents
    def __init__(self):
        super(AgentParams, self).__init__()

        # optimizer
        self.optim            = optim.Adam

        self.number_states    = 5        # Pos_x, pos_y, theta, poured_volume, spilled_volume

        # hyperparameters
        self.batch_size       = 128      # batch size during training
        self.rm_size          = 1000000  # memory replay maximum size
        self.gamma            = 0.98     # Discount factor
        self.lr               = 0.001    # Learning rate
        self.prate            = 0.0001   # Policy net learning rate

        self.criterion        = nn.MSELoss()

        self.tau              = 0.05     # moving average for target network
        self.epsilon          = 0.8      # Random action 20% of the time
        self.noise_std        = 0.05     # Add noise with 5% standard deviation to actions

        self.epochs           = 200      # Number of training epochs
        self.cycles           = 50       # Length of an epoch
        self.ep_per_cycle     = 16       # Number of episodes to run per cycle
        self.opt_steps        = 20       # Optimization steps after each cycle
        self.k_goals          = 4        # Number of additional goals to sample and add to replay memory

        self.validate_steps   = 2        # How many episodes to test and report performance

class EnvParams():          # Settings for simulation environment
    def __init__(self):
        self.path             = "scene1.prscene"  # Path to the PreonLab scene
        self.step_cost        = -0.5              # Reward when goal is not reached, but no collision happens
        self.collision_cost   = -1.0              # Reward when collision is detected
        self.goal_reward      = 1.0               # Reward when reaching the goal
        self.max_time         = 20.0              # Maximum length of an episode in seconds
        self.goal_threshold   = 10.0              # Max volume difference for goal to be considered achieved in milliliters

        self.max_lin_vel      = 10.0              # Maximum absolute linear velocity in cm/s
        self.max_ang_vel      = 45.0              # Maximum absolute angular velocity in degrees/s
        self.min_x            = -12.0             # Range of possible positions
        self.max_x            = 12.0              # Moving outside this range will result in collision
        self.min_y            = 0.0
        self.max_y            = 20.0
        self.max_volume       = 468.0             # Maximum volume to pour in milliliters

class Options(Params):
    agent_params  = AgentParams()
    env_params = EnvParams()
