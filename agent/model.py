import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Check this for debugging
from ipdb import set_trace as debug

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)

        self.actor_velocities = nn.Linear(64, 3)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(3e-3)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        self.actor_velocities.weight.data = fanin_init(self.actor_velocities.weight.data.size())

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        hidden_features = self.relu(self.fc3(out))

        output_vel = self.tanh(self.actor_velocities(hidden_features))
        return output_vel

class Critic(nn.Module):
    def __init__(self, nb_states):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, 64)
        self.fc2 = nn.Linear(64+3, 64)     # 3 actions
        self.fc3 = nn.Linear(64, 64)

        self.output_V = nn.Linear(64, 3)
        self.relu = nn.ReLU()
        self.init_weights(3e-3)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.output_V.weight.data = fanin_init(self.output_V.weight.data.size())

    def forward(self, xs):
        x, a = xs
        out = self.relu(self.fc1(x))
        # debug()
        state_action = torch.cat([out,a],1)
        out = self.relu(self.fc2(state_action))
        out = self.relu(self.fc3(out))

        vel_values = self.output_V(out)
        return vel_values
