import preonpy
import tensorflow as tf
import numpy as np
from env.preon_env import Preon_env
from utils.options import Options
from utils.utils import *
import tflearn
import sys
import os
import argparse

from networks import ActorNetwork, CriticNetwork
from utils.replay_buffer import ReplayBuffer
from policy import Policy

'''
/////   DDPG - DISTRIBUTED APPROACH
'''

# input flags
parser = argparse.ArgumentParser(description='Help message')
parser.add_argument('--job_name', help="Either 'ps' or 'worker'")
parser.add_argument('--task_index', type=int, default=0, help="Index of task within the job")
parser.add_argument('--restore', action='store_true', help="Restores previous model")
parser.add_argument('--testing', action='store_true', help="Runs a testing episode")
parser.add_argument('--scene_name', help="Name of the scene to be generated in output folder")
parser.add_argument('--goal', type=float, default=1.0, help="Target pouring volume [0.0,1.0]")
args = parser.parse_args()

# cluster specification
parameter_servers = ["10.8.105.176:2222"]    # 192.168.167.176


workers = ["10.8.105.176:2223",            # Oreo
           "10.5.167.17:2224"]             # GPU2
'''
workers = ["10.8.105.176:2223"]             # Oreo
'''

# ==========================
#   Training Parameters
# ==========================

opt = Options()

# Max training steps
MAX_EPISODES = 50000
# Discount factor
GAMMA = opt.agent_params.gamma

# ===========================
#   Utility Parameters
# ===========================

# Directory for storing tensorboard summary results
SUMMARY_DIR = opt.summary_dir
SAVE_DIR = opt.save_dir
RANDOM_SEED = 1256
# Size of replay buffer
BUFFER_SIZE = opt.agent_params.rm_size
MINIBATCH_SIZE = opt.agent_params.batch_size
VALID_FREQ = opt.agent_params.valid_freq

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries():
    training_summaries = []
    episode_reward = tf.Variable(0.)
    training_summaries.append(tf.summary.scalar("Reward", episode_reward))
    episode_ave_max_q = tf.Variable(0.)
    training_summaries.append(tf.summary.scalar("Qmax_Value", episode_ave_max_q))
    value_loss = tf.Variable(0.)
    training_summaries.append(tf.summary.scalar("Value_Loss", value_loss))
    collision_rate = tf.Variable(0.)
    training_summaries.append(tf.summary.scalar("Collision_Rate", collision_rate))
    avg_spillage = tf.Variable(0.)
    training_summaries.append(tf.summary.scalar("Spillage_per_Episode", avg_spillage))
    extra_goals = tf.Variable(0.)
    training_summaries.append(tf.summary.scalar("Extra_Goals", extra_goals))
    filling_rates = tf.placeholder(tf.float32)
    training_summaries.append(tf.summary.histogram("Filling_rates", filling_rates))
    training_poured_percent = tf.Variable(0.)
    training_summaries.append(tf.summary.scalar("Training_Pouring_percentage", training_poured_percent))

    train_ops = tf.summary.merge(training_summaries)

    # Validation variables
    valid_summaries = []
    valid_Reward = tf.Variable(0.)
    valid_summaries.append(tf.summary.scalar("Validation_Rewards", valid_Reward))
    valid_collision = tf.Variable(0.)
    valid_summaries.append(tf.summary.scalar("Validation_Collision_Rate", valid_collision))
    valid_poured_percent = tf.Variable(0.)
    valid_summaries.append(tf.summary.scalar("Validation_Pouring_percentage", valid_poured_percent))
    valid_spillage = tf.Variable(0.)
    valid_summaries.append(tf.summary.scalar("Validation_Spillage_ml", valid_spillage))

    valid_ops = tf.summary.merge(valid_summaries)

    valid_vars = [valid_Reward, valid_collision, valid_poured_percent, valid_spillage]
    training_vars = [episode_reward, episode_ave_max_q, value_loss, collision_rate, avg_spillage, extra_goals, filling_rates, training_poured_percent]

    return train_ops, valid_ops, training_vars, valid_vars


def generate_new_goal(args, init_volume, cup_capacity):
    fill_level = np.random.randint(1, 11) / float(10)

    # We first make sure that the goal is realistic
    while fill_level*cup_capacity > init_volume:
        fill_level = np.random.randint(1, fill_level*10) / float(10)

    desired_spilled_vol = 0.0

    desired_fill_level_norm = get_normalized(fill_level,0.0,1.0)
    desired_spilled_vol_norm = get_normalized(desired_spilled_vol,0.0,args.max_volume)
    new_goal = [desired_fill_level_norm, desired_spilled_vol_norm]
    return new_goal

def get_state_as_list(dict_state):
    '''
    Returns the state representation as a list of values taken from a dictionary
    Here we can easily define which values make part of the state
    '''
    state = []
    state.append(dict_state['delta_x_norm'])
    state.append(dict_state['delta_y_norm'])
    state.append(dict_state['theta_norm'])
    state.append(dict_state['action_x'])
    state.append(dict_state['action_y'])
    state.append(dict_state['action_angle'])
    state.append(dict_state['fill_level_norm'])
    state.append(dict_state['spilled_vol_norm'])
    state.append(dict_state['filling_rate_norm'])

    return state

# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor, critic, saver, global_step, step_op, replay_buffer):

    # Set up summary Ops
    train_ops, valid_ops, training_vars, valid_vars = build_summaries()

    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    validation_step = 0
    for i in range(MAX_EPISODES):
        current_step = sess.run(global_step)

        #init_height = np.random.randint(2, 11)
        init_height = 10
        normalized_state, clean_state, info = env.reset(init_height)
        s = get_state_as_list(normalized_state)
        goal = generate_new_goal(opt.env_params, int(env.env.init_particles), info[3])

        ep_reward = 0
        ep_ave_max_q = 0
        value_loss = 0
        ep_collisions = 0
        ep_extra_goals = 0
        filling_rates = []

        for j in range(env.max_steps+1):

            # Added exploration noise
            input_s = np.reshape(s, (1, actor.s_dim))
            input_g = np.reshape(goal, (1, actor.goal_dim))
            a = actor.predict(input_s, input_g) + (1. / (1. + current_step))

            normalized_next_state, r, terminal, info, collision, clean_state = env.step(a[0], goal)
            s2 = get_state_as_list(normalized_next_state)

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(goal, (actor.goal_dim,)),
                              np.reshape(a, (actor.a_dim,)), r, terminal, np.reshape(s2, (actor.s_dim,)))

            '''
            Hindsight experience replay
            '''
            # NOTE: Add extra experiences to memory
            # 1) Adding only those transitions where liquid was poured/spilled
            if normalized_next_state['fill_level_norm'] != normalized_state['fill_level_norm'] or normalized_next_state['spilled_vol_norm'] != normalized_state['spilled_vol_norm']:
                ep_extra_goals += 1
                new_goal = [normalized_next_state['fill_level_norm'], normalized_next_state['spilled_vol_norm']]
                new_reward = env.estimate_new_reward(normalized_next_state, new_goal, r)
                replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(new_goal, (actor.goal_dim,)),
                                  np.reshape(a, (actor.a_dim,)), new_reward, terminal, np.reshape(s2, (actor.s_dim,)))


            # Keep adding experience to the memory until there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                ''' Training process - DDPG'''
                s_batch, g_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic.predict_target(s2_batch, g_batch, actor.predict_target(s2_batch, g_batch))

                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _, v_loss = critic.train(s_batch, g_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)
                value_loss += v_loss

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch, g_batch)
                grads = critic.action_gradients(s_batch, g_batch, a_outs)

                actor.train(s_batch, g_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()
                ''' End of training '''

            s = s2
            normalized_state = normalized_next_state
            ep_reward += r
            ep_collisions += int(collision)
            filling_rates.append(clean_state['filling_rate'])

            if terminal:
                episode_spillage = clean_state['spilled_vol']

                # Compute the % of liquid which was poured
                train_poured_vol = clean_state['fill_level']
                train_poured_goal = get_denormalized(goal[0],0.0,1.0)
                train_percentage_poured = train_poured_vol*100 / float(train_poured_goal)

                summary_str = sess.run(train_ops, feed_dict={
                    training_vars[0]: ep_reward,
                    training_vars[1]: ep_ave_max_q / float(j),
                    training_vars[2]: value_loss / float(j),
                    training_vars[3]: ep_collisions / float(j),
                    training_vars[4]: episode_spillage,
                    training_vars[5]: ep_extra_goals,
                    training_vars[6]: np.array(filling_rates),
                    training_vars[7]: train_percentage_poured
                })
                writer.add_summary(summary_str, current_step)
                writer.flush()

                if i%VALID_FREQ == 0:
                    valid_terminal = False
                    valid_r = 0
                    valid_collision = 0
                    # init_height = np.random.randint(2, 11)
                    init_height = 10
                    normalized_state, clean_state, info = env.reset(init_height)
                    s = get_state_as_list(normalized_state)
                    goal = generate_new_goal(opt.env_params, int(env.env.init_particles), info[3])

                    while not valid_terminal:
                        input_s = np.reshape(s, (1, actor.s_dim))
                        input_g = np.reshape(goal, (1, actor.goal_dim))
                        a = actor.predict_target(input_s, input_g)

                        normalized_next_state, r, valid_terminal, _, collision, clean_state = env.step(a[0], goal)
                        s2 = get_state_as_list(normalized_next_state)
                        valid_r += r
                        valid_collision += int(collision)
                        s = s2

                    # Compute the % of liquid which was poured
                    valid_poured_vol = clean_state['fill_level']
                    valid_poured_goal = get_denormalized(goal[0],0.0,1.0)
                    valid_percentage_poured = valid_poured_vol*100 / float(valid_poured_goal)

                    # Compute the spillage in milliliters
                    valid_spillage = clean_state['spilled_vol']

                    summary_valid = sess.run(valid_ops, feed_dict={
                        valid_vars[0]: valid_r,
                        valid_vars[1]: valid_collision,
                        valid_vars[2]: valid_percentage_poured,
                        valid_vars[3]: valid_spillage
                    })
                    writer.add_summary(summary_valid, current_step)
                    writer.flush()

                    save_path = saver.save(sess, SAVE_DIR + "/model", global_step=global_step)
                    replay_buffer.save_pickle()
                    print('-------------------------------------')
                    print("Model saved in file: %s" % save_path)
                    print('-------------------------------------')

                    validation_step += 1

                break

        # Increase global_step
        sess.run(step_op)

def test(policy, env, test_goal, scene_name):

    policy.set_goal([test_goal, 0.0])

    init_height = opt.test_height   # Not necessary right now (determines initial volume)

    normalized_state, clean_state, info = env.reset(init_height)
    s = get_state_as_list(normalized_state)

    print('**************************************')
    print(info)
    print('**************************************')

    ep_reward = 0
    ep_collisions = 0

    action_x = []
    action_y = []
    action_theta = []
    collisions = []

    for j in range(env.max_steps+1):
        # Denormalize because policy class normalizes internally
        # Policy class normalizes internally in order to make it simple to test the model on real scenarios (no need to worry about normalizing)
        # If you change the order or the size of the state vector, make sure to adjust the following code accordingly.
        state = []
        state.append(get_denormalized(s[0], opt.env_params.min_x_dist, opt.env_params.max_x_dist))
        state.append(get_denormalized(s[1], opt.env_params.min_y_dist, opt.env_params.max_y_dist))
        state.append(get_denormalized(s[2], 0.0, 360.0))
        state.append(get_denormalized(s[3], -opt.env_params.max_lin_disp, opt.env_params.max_lin_disp))
        state.append(get_denormalized(s[4], -opt.env_params.max_lin_disp, opt.env_params.max_lin_disp))
        state.append(get_denormalized(s[5], -opt.env_params.max_ang_disp, opt.env_params.max_ang_disp))
        state.append(get_denormalized(s[6], 0.0, 1.0))
        state.append(get_denormalized(s[7], 0.0, opt.env_params.max_volume))
        state.append(get_denormalized(s[8], 0.0, 1.0))

        a = policy.get_output(state)
        action_x.append(a[0])
        action_y.append(a[1])
        action_theta.append(a[2])

        normalized_next_state, r, terminal, _, collision, clean_state = env.step(a, policy.goal)
        s2 = get_state_as_list(normalized_next_state)

        collisions.append(collision)

        s = s2
        ep_reward += r
        ep_collisions += int(collision)


        if terminal:
            episode_spillage = clean_state['spilled_vol']
            episode_filling = clean_state['fill_level']
            break

    print('----------------------------------------------------')
    print('----------------------------------------------------')
    print('Episode reward: ', ep_reward)
    print('Number of collisions: ', ep_collisions)
    print('Episode Spillage: ', episode_spillage)
    print('Episode Poured Volume: ', episode_filling)
    print('Saving scene...')
    env.save_scene(os.getcwd()+opt.saved_scenes_dir+scene_name)
    print('Completed')

    print('Action in X ', len(action_x))
    print(action_x)
    print('Action in Y ', len(action_y))
    print(action_y)
    print('Rotation ', len(action_theta))
    print(action_theta)
    print('Collision ', len(collisions))
    print(collisions)


def main(_):

    tf.reset_default_graph()
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)

    if not args.testing:   # When training

        cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})
        server = tf.train.Server(cluster, job_name=args.job_name, task_index=args.task_index)

        if args.job_name == "ps":
            server.join()
        elif args.job_name == "worker":
            with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % args.task_index,cluster=cluster)):
                is_chief = (args.task_index == 0)
                # count the number of updates
                global_step = tf.get_variable('global_step',[],initializer = tf.constant_initializer(0),trainable = False)
                step_op = global_step.assign(global_step+1)

                actor = ActorNetwork(opt)
                critic = CriticNetwork(actor.get_num_trainable_vars(), opt)

                # Initialize replay memory
                replay_buffer = ReplayBuffer(BUFFER_SIZE, SAVE_DIR, RANDOM_SEED)

                init_op = tf.global_variables_initializer()

                # Add ops to save and restore all the variables.
                saver = tf.train.Saver(max_to_keep=5)

                if is_chief and args.restore:
                    def restore_model(sess):
                        actor.set_session(sess)
                        critic.set_session(sess)
                        saver.restore(sess,tf.train.latest_checkpoint(SAVE_DIR+'/'))
                        actor.restore_params(tf.trainable_variables())
                        critic.restore_params(tf.trainable_variables())
                        print('***********************')
                        print('Model Restored')
                        print('***********************')
                        replay_buffer.load_pickle()
                        print('***********************')
                        print('Load RM: ', replay_buffer.size())
                        print('***********************')
                else:
                    def restore_model(sess):
                        actor.set_session(sess)
                        critic.set_session(sess)
                        # Initialize target network weights
                        actor.update_target_network()
                        critic.update_target_network()
                        print('***********************')
                        print('Model Initialized')
                        print('***********************')

                # with tf.Session() as sess:
                with tf.Session(server.target) as sess:
                    sess.run(init_op)
                    restore_model(sess)
                    env = Preon_env(opt.env_params)

                    train(sess, env, actor, critic, saver, global_step, step_op, replay_buffer)


    else:           # When testing

        with tf.Session() as sess:
            policy = Policy(sess)
            env = Preon_env(opt.env_params)
            test(policy, env, args.goal, args.scene_name)

if __name__ == '__main__':
    tf.app.run()
