from __future__ import absolute_import
from __future__ import division
import numpy as np
import os
import visdom

import torch
import torch.nn as nn
import torch.optim as optim

class Params(object):   # NOTE: shared across all modules
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
        self.model_name  = self.root_dir + "/models/model0.pth"

        if self.continue_training and os.path.exists(self.model_name):
            self.model_file  = self.model_name
        else:
            self.model_file  = None

        if self.mode == 2 and os.path.exists(self.model_name):
            self.model_file  = self.model_name  # NOTE: so only need to change self.mode to 2 to test the current training
        else:
            print("Pre-Trained model does not exist, Testing aborted!!!")

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
        self.optim               = optim.Adam

        # hyperparameters
        self.steps               = 20000000 # max #iterations
        self.batch_size          = 64       # batch size during training
        self.rm_size             = 1000000  # memory replay maximum size
        self.gamma               = 0.99
        self.warm_up             = 100      # Time without training but only filling memory replay
        self.lr                  = 0.0001
        self.policy_lr           = 0.0001

        self.criterion           = nn.MSELoss()

        self.tau                 = 0.001    # moving average for target network
        self.ou_theta            = 0.15     # noise theta
        self.ou_sigma            = 0.2      # noise sigma
        self.ou_mu               = 0.0      # noise mean

class Options(Params):
    agent_params  = AgentParams()
