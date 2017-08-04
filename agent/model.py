import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Check this for debugging
#from ipdb import set_trace as debug

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, nb_goals):
        super(Actor, self).__init__()
        self.fc1_s = nn.Linear(nb_states, 128)   # State input
        self.fc1_g = nn.Linear(nb_goals, 16)    # Goal input
        self.fc2 = nn.Linear(128 + 16, 144)
        self.fc3 = nn.Linear(144, 64)

        self.actor_velocities = nn.Linear(64, nb_actions)

        #self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.init_weights(3e-3)

    def init_weights(self, init_w):
        self.fc1_s.weight.data = fanin_init(self.fc1_s.weight.data.size())
        self.fc1_g.weight.data = fanin_init(self.fc1_g.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.actor_velocities.weight.data = fanin_init(self.actor_velocities.weight.data.size())

    def forward(self, state, goal):
        out_state = self.elu(self.fc1_s(state))
        out_goal = self.elu(self.fc1_g(goal))
        state_goal = torch.cat([out_state,out_goal],1)
        out = self.elu(self.fc2(state_goal))
        hidden_features = self.elu(self.fc3(out))

        output_vel = self.tanh(self.actor_velocities(hidden_features))
        return output_vel

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, nb_goals):
        super(Critic, self).__init__()
        self.fc1_s = nn.Linear(nb_states, 128)   # State input
        self.fc1_g = nn.Linear(nb_goals, 16)    # Goal input
        self.fc2 = nn.Linear(128+16+nb_actions, 200)
        self.fc3 = nn.Linear(200, 64)

        self.output_V = nn.Linear(64, 1)
        #self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.init_weights(3e-3)

    def init_weights(self, init_w):
        self.fc1_s.weight.data = fanin_init(self.fc1_s.weight.data.size())
        self.fc1_g.weight.data = fanin_init(self.fc1_g.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.output_V.weight.data = fanin_init(self.output_V.weight.data.size())

    def forward(self, state, goal, action):
        out_state = self.elu(self.fc1_s(state))
        out_goal = self.elu(self.fc1_g(goal))
        # debug()
        state_goal_action = torch.cat([out_state,out_goal,action],1)
        out = self.elu(self.fc2(state_goal_action))
        out = self.elu(self.fc3(out))

        vel_value = self.output_V(out)
        return vel_value
