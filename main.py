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

# ==========================
#   Training Parameters
# ==========================

opt = Options()

# Max training steps
MAX_EPISODES = 50000

# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = opt.agent_params.actor_lr
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = opt.agent_params.critic_lr
# Discount factor
GAMMA = opt.agent_params.gamma
# Soft target update param
TAU = opt.agent_params.tau

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
    training_summaries.append(tf.summary.scalar("Qmax Value", episode_ave_max_q))
    value_loss = tf.Variable(0.)
    training_summaries.append(tf.summary.scalar("Value Loss", value_loss))
    collision_rate = tf.Variable(0.)
    training_summaries.append(tf.summary.scalar("Collision Rate", collision_rate))
    avg_spillage = tf.Variable(0.)
    training_summaries.append(tf.summary.scalar("Spillage per Episode", avg_spillage))
    extra_goals = tf.Variable(0.)
    training_summaries.append(tf.summary.scalar("Extra Goals", extra_goals))
    filling_rates = tf.placeholder(tf.float32)
    training_summaries.append(tf.summary.histogram("Filling rates", filling_rates))

    train_ops = tf.summary.merge(training_summaries)

    # Validation variables
    valid_summaries = []
    valid_Reward = tf.Variable(0.)
    valid_summaries.append(tf.summary.scalar("Validation Rewards", valid_Reward))
    valid_collision = tf.Variable(0.)
    valid_summaries.append(tf.summary.scalar("Validation Collision Rate", valid_collision))
    valid_poured_percent = tf.Variable(0.)
    valid_summaries.append(tf.summary.scalar("Validation Pouring (%)", valid_poured_percent))
    valid_spillage = tf.Variable(0.)
    valid_summaries.append(tf.summary.scalar("Validation Spillage (ml)", valid_spillage))

    valid_ops = tf.summary.merge(valid_summaries)

    valid_vars = [valid_Reward, valid_collision, valid_poured_percent, valid_spillage]
    training_vars = [episode_reward, episode_ave_max_q, value_loss, collision_rate, avg_spillage, extra_goals, filling_rates]

    #summary_ops = tf.summary.merge_all()

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

'''
def generate_new_goal(args, init_volume, cup_capacity):

    possible_values = [50,90,150,180,220,270,300,350,400,450]

    max_possible_val = min(init_volume, cup_capacity)
    desired_poured_vol = np.random.choice([x for x in possible_values if x <= max_possible_val])
    desired_spilled_vol = 0.0

    desired_fill_level = desired_poured_vol / cup_capacity

    desired_fill_level_norm = get_normalized(desired_fill_level,0.0,1.0)
    desired_spilled_vol_norm = get_normalized(desired_spilled_vol,0.0,args.max_volume)
    new_goal = [desired_fill_level_norm, desired_spilled_vol_norm]
    return new_goal
'''
# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor, critic, action_dim, goal_dim, state_dim, saver, global_step, step_op):

    # Set up summary Ops
    train_ops, valid_ops, training_vars, valid_vars = build_summaries()

    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    validation_step = 0
    for i in range(MAX_EPISODES):
        current_step = sess.run(global_step)

        #init_height = np.random.randint(2, 11)
        init_height = 10
        s, info = env.reset(init_height)
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

            s2, r, terminal, info, collision = env.step(a[0], goal)

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(goal, (actor.goal_dim,)),
                              np.reshape(a, (actor.a_dim,)), r, terminal, np.reshape(s2, (actor.s_dim,)))

            '''
            Hindsight experience replay
            '''
            # NOTE: Add extra experiences to memory
            # 1) Adding only those transitions where liquid was poured/spilled
            if s2[6:8] != s[6:8]:
                ep_extra_goals += 1
                new_goal = s2[6:8]
                new_reward = env.estimate_new_reward(s2,new_goal,r)
                replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(new_goal, (actor.goal_dim,)),
                                  np.reshape(a, (actor.a_dim,)), new_reward, terminal, np.reshape(s2, (actor.s_dim,)))


            # Keep adding experience to the memory until there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
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

            s = s2
            ep_reward += r
            ep_collisions += int(collision)
            filling_rates.append(get_denormalized(s[8], 0.0, 1.0))

            if terminal:
                episode_spillage = s[7]

                summary_str = sess.run(train_ops, feed_dict={
                    training_vars[0]: ep_reward,
                    training_vars[1]: ep_ave_max_q / float(j),
                    training_vars[2]: value_loss / float(j),
                    training_vars[3]: ep_collisions / float(j),
                    training_vars[4]: episode_spillage,
                    training_vars[5]: ep_extra_goals,
                    training_vars[6]: np.array(filling_rates)
                })
                writer.add_summary(summary_str, current_step)
                writer.flush()

                if i%VALID_FREQ == 0:
                    valid_terminal = False
                    valid_r = 0
                    valid_collision = 0
                    # init_height = np.random.randint(2, 11)
                    init_height = 10
                    s, info = env.reset(init_height)
                    goal = generate_new_goal(opt.env_params, int(env.env.init_particles), info[3])

                    while not valid_terminal:
                        input_s = np.reshape(s, (1, actor.s_dim))
                        input_g = np.reshape(goal, (1, actor.goal_dim))
                        a = actor.predict_target(input_s, input_g)

                        s2, r, valid_terminal, _, collision = env.step(a[0], goal)
                        valid_r += r
                        valid_collision += int(collision)
                        s = s2

                    # Compute the % of liquid which was poured
                    valid_poured_vol = get_denormalized(s[6],0.0,1.0)
                    valid_poured_goal = get_denormalized(goal[0],0.0,1.0)
                    valid_percentage_poured = valid_poured_vol*100 / float(valid_poured_goal)

                    # Compute the spillage in milliliters
                    valid_spillage = get_denormalized(s[7],0.0,opt.env_params.max_volume)

                    summary_valid = sess.run(valid_ops, feed_dict={
                        valid_vars[0]: valid_r,
                        valid_vars[1]: valid_collision,
                        valid_vars[2]: valid_percentage_poured,
                        valid_vars[3]: valid_spillage
                    })
                    writer.add_summary(summary_valid, current_step)
                    writer.flush()

                    save_path = saver.save(sess, SAVE_DIR + "/model", global_step=global_step)
                    print('-------------------------------------')
                    print("Model saved in file: %s" % save_path)
                    print('-------------------------------------')

                    validation_step += 1

                break

        # Increase global_step
        sess.run(step_op)

def test(sess, env, actor, critic, action_dim, goal_dim, state_dim, test_goal, scene_name):

    desired_fill_level_norm = get_normalized(test_goal,0.0,1.0)
    desired_spilled_vol_norm = get_normalized(0.0,0.0,opt.env_params.max_volume)
    norm_test_goal = [desired_fill_level_norm, desired_spilled_vol_norm]

    init_height = opt.test_height

    s, info = env.reset(init_height)
    print('**************************************')
    print(info)
    print('**************************************')

    ep_reward = 0
    ep_collisions = 0

    for j in range(env.max_steps+1):
        # Added exploration noise
        input_s = np.reshape(s, (1, actor.s_dim))
        input_g = np.reshape(norm_test_goal, (1, actor.goal_dim))
        a = actor.predict(input_s, input_g)

        s2, r, terminal, info, collision = env.step(a[0], norm_test_goal)

        s = s2
        ep_reward += r
        ep_collisions += int(collision)

        spillage = get_denormalized(s[7],0.0,opt.env_params.max_volume)

        if terminal:
            episode_spillage = s[7]
            episode_filling = s[6]
            break


    print('----------------------------------------------------')
    print('----------------------------------------------------')
    print('Episode reward: ', ep_reward)
    print('Number of collisions: ', ep_collisions)
    print('Episode Spillage: ', get_denormalized(episode_spillage,0.0,opt.env_params.max_volume))
    print('Episode Poured Volume: ', get_denormalized(episode_filling,0.0,1.0))
    print('Saving scene...')
    env.save_scene(os.getcwd()+opt.saved_scenes_dir+scene_name)
    print('Completed')


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

                state_dim = 9
                action_dim = 3
                goal_dim = 2

                actor = ActorNetwork(state_dim, action_dim, goal_dim, ACTOR_LEARNING_RATE, TAU, opt.env_params)
                critic = CriticNetwork(state_dim, action_dim, goal_dim, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(), opt.env_params)

                init_op = tf.global_variables_initializer()

                # Add ops to save and restore all the variables.
                saver = tf.train.Saver(max_to_keep=5)

                if is_chief and args.restore:
                    def restore_model(sess):
                        actor.set_session(sess)
                        critic.set_session(sess)
                        saver.restore(sess,tf.train.latest_checkpoint(opt.save_dir+'/'))
                        actor.restore_params(tf.trainable_variables())
                        critic.restore_params(tf.trainable_variables())
                        print('***********************')
                        print('Model Restored')
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

                    train(sess, env, actor, critic, action_dim, goal_dim, state_dim, saver, global_step, step_op)


    else:           # When testing

        state_dim = 9
        action_dim = 3
        goal_dim = 2

        actor = ActorNetwork(state_dim, action_dim, goal_dim, ACTOR_LEARNING_RATE, TAU, opt.env_params)
        critic = CriticNetwork(state_dim, action_dim, goal_dim, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(), opt.env_params)

        init_op = tf.global_variables_initializer()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(max_to_keep=5)

        with tf.Session() as sess:
            sess.run(init_op)
            actor.set_session(sess)
            critic.set_session(sess)
            saver.restore(sess,tf.train.latest_checkpoint(opt.save_dir+'/'))
            actor.restore_params(tf.trainable_variables())
            critic.restore_params(tf.trainable_variables())
            print('***********************')
            print('Model Restored')
            print('***********************')

            env = Preon_env(opt.env_params)
            test(sess, env, actor, critic, action_dim, goal_dim, state_dim, args.goal, args.scene_name)


if __name__ == '__main__':
    tf.app.run()
