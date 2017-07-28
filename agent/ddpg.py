from __future__ import absolute_import
from __future__ import division

import numpy as np

import torch
import torch.nn as nn

from agent.model import Actor, Critic
from utils.memory import ReplayMemory
from utils.util import *

class DDPG(object):
    def __init__(self, args):

        self.args = args
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = args.number_states
        self.num_actions = 3
        self.num_goals_feat = 2

        # Create Actor and Critic Network
        self.actor = Actor(self.nb_states, self.num_actions, self.num_goals_feat )
        self.actor_target = Actor(self.nb_states, self.num_actions,self.num_goals_feat )
        self.actor_optim  = args.optim(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.num_actions, self.num_goals_feat )
        self.critic_target = Critic(self.nb_states, self.num_actions, self.num_goals_feat )
        self.critic_optim  = args.optim(self.critic.parameters(), lr=args.lr)

        # Loads a previous model only if "continue_training" flag was set or we are in testing mode
        if self.args.load_model:
            self.load_weights(self.args.model_dir)

        hard_update(self.actor_target, self.actor)  # Make sure both networks are identical
        hard_update(self.critic_target, self.critic)

        #Create replay buffer
        self.memory = ReplayMemory(args.rm_size)

        # Hyper-parameters
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.discount = args.gamma

        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        self.epsilon = args.epsilon
        self.criterion = args.criterion

        if args.use_cuda: self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, goal_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample(self.batch_size)

        # Prepare for the target q batch
        var_goal = to_tensor(goal_batch, volatile=True)
        var_next_state = to_tensor(next_state_batch, volatile=True)
        next_q_values = self.critic_target(var_next_state, var_goal,self.actor_target(var_next_state, var_goal))
        next_q_values.volatile=False

        var_terminal = to_tensor(terminal_batch.astype(np.float))
        target_q_batch = to_tensor(reward_batch) + (self.discount * var_terminal * next_q_values)

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic(to_tensor(state_batch), to_tensor(goal_batch), to_tensor(action_batch))

        value_loss = self.criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic(to_tensor(state_batch), to_tensor(goal_batch),
            self.actor(to_tensor(state_batch), to_tensor(goal_batch))
        )

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return to_numpy(value_loss), to_numpy(policy_loss)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, goal, r_t, s_t1, done):
        if self.is_training:
            self.memory.push(self.s_t, goal, self.a_t, s_t1, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.num_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, goal, decay_epsilon=False):
        if self.is_training:
            if np.random.random_sample() > self.epsilon:    # Random action
                return self.random_action()
            else:
                action = to_numpy(self.actor(to_tensor(np.array([s_t])),to_tensor(np.array([goal])))).squeeze(0)
                action = self.add_noise_to_action(action)
                action = np.clip(action, -1., 1.)

                self.a_t = action
                return action
        else:
            # No noise or exploration during testing
            action = to_numpy(self.actor(to_tensor(np.array([s_t])),to_tensor(np.array([goal])))).squeeze(0)
            self.a_t = action
            return action

        '''
        if decay_epsilon:
            self.epsilon += self.depsilon
        '''

    def add_noise_to_action(self, action):
        '''
        Adds normal noise to each action with standard deviation equal to 5% of the total range. In this case the range is [-1,1]
        '''
        return np.random.normal(action,self.args.noise_std)

    def reset(self, obs):
        self.s_t = obs

    def load_weights(self, output):
        prGreen("Loading previous model...")
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))

    def save_model(self,output):
        torch.save(self.actor.state_dict(),'{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(),'{}/critic.pkl'.format(output))

    def seed(self,s):
        torch.manual_seed(s)
        if self.args.use_cuda:
            torch.cuda.manual_seed(s)
