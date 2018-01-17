import tensorflow as tf
import numpy as np

from networks import ActorNetwork
from utils.options import Options
from utils.utils import *

class Policy(object):

    '''
    Policy class
    - Receives a tensorflow session when initializing to restore model
    '''
    def __init__(self, sess):
        # Loading all configuration parameters
        self.opt = Options()

        # tensorflow session
        self.sess = sess

        # Define the policy network
        self.actor = ActorNetwork(self.opt.agent_params.state_dim, self.opt.agent_params.action_dim, self.opt.agent_params.goal_dim, 0.0, 1.0, self.opt.env_params)

        # Init operation and saver (to restore model)
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=5)

        self.sess.run(self.init_op)

        self.actor.set_session(self.sess)

        # Restoring previous model using path defined in Options()
        self.saver.restore(self.sess,tf.train.latest_checkpoint(self.opt.save_dir+'/'))
        self.actor.restore_params(tf.trainable_variables())

        print('***********************')
        print('Model Restored')
        print('***********************')


    '''
    set_goal allows us to define a new goal at any time
    target_goal is a list with two values: [desired_fill_level, desired_spillage].
    i.e [0.5, 0] -> poures 50% with 0ml spillage
    '''
    def set_goal(self, target_goal):
        # Normalizing values to be in range [-1, 1]
        desired_fill_level_norm = get_normalized(target_goal[0],0.0,1.0)
        desired_spilled_vol_norm = get_normalized(target_goal[1],0.0,self.opt.env_params.max_volume)
        self.goal = [desired_fill_level_norm, desired_spilled_vol_norm]

    '''
    get_output receives the current state as a list and outputs the policy action as an array
    state = [delta_x, delta_y, theta_angle, previous_action_x, previous_action_y, previous_action_theta, fill_level, spillage, filling_rate]
    action = [displacement_x, displacement_y, rotation_theta]
     the action values has already been multiplied by a scale factor defined in Options (max_lin_disp, max_ang_disp), so they represent the
     relative displacement with respect to the current location
    '''
    def get_output(self, state):
        # Normalizing state
        pos_x_norm = get_normalized(state[0], self.opt.env_params.min_x_dist, self.opt.env_params.max_x_dist)
        pos_y_norm = get_normalized(state[1], self.opt.env_params.min_y_dist, self.opt.env_params.max_y_dist)
        theta_angle_norm = get_normalized(state[2], 0.0, 360.0)
        action_x_norm = get_normalized(state[3], -self.opt.env_params.max_lin_disp, self.opt.env_params.max_lin_disp)   # state[3]
        action_y_norm = get_normalized(state[4], -self.opt.env_params.max_lin_disp, self.opt.env_params.max_lin_disp)   # state[4]
        action_theta_norm = get_normalized(state[5], -self.opt.env_params.max_ang_disp, self.opt.env_params.max_ang_disp) # state[5]
        fill_level_norm = get_normalized(state[6], 0.0, 1.0)
        spilled_vol_norm = get_normalized(state[7], 0.0, self.opt.env_params.max_volume)
        filling_rate_norm = get_normalized(state[8], 0.0, 1.0)

        state = [pos_x_norm, pos_y_norm, theta_angle_norm, action_x_norm, action_y_norm, action_theta_norm, fill_level_norm, spilled_vol_norm, filling_rate_norm]

        input_s = np.reshape(state, (1, self.actor.s_dim))
        input_g = np.reshape(self.goal, (1, self.actor.goal_dim))
        action = self.actor.predict(input_s, input_g)
        return action[0]
