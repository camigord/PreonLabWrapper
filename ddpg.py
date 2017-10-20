import preonpy
import tensorflow as tf
import numpy as np
from env.preon_env import Preon_env
from utils.options import Options
import tflearn
import sys
import os

from networks import ActorNetwork, CriticNetwork
from replay_buffer import ReplayBuffer

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
    training_vars = [episode_reward, episode_ave_max_q, value_loss, collision_rate, avg_spillage, extra_goals]

    #summary_ops = tf.summary.merge_all()

    return train_ops, valid_ops, training_vars, valid_vars


def generate_new_goal(args, step, init_volume, validation=False):
    if validation:
        possible_values = [50,100,150,200,250,300,350,400,450]
        #desired_poured_vol = possible_values[int(step%len(possible_values))]
        desired_poured_vol = np.random.choice([x for x in possible_values if x <= init_volume])
        desired_spilled_vol = 0.0
    else:
        desired_poured_vol = np.random.randint(0,init_volume+1)
        # We can only spill so much as available liquid remains
        desired_spilled_vol = np.random.randint(0,init_volume - desired_poured_vol)

    desired_poured_vol_norm = (desired_poured_vol - args.max_volume/2.0) / (args.max_volume/2.0)
    desired_spilled_vol_norm = (desired_spilled_vol - args.max_volume/2.0) / (args.max_volume/2.0)
    new_goal = [desired_poured_vol_norm, desired_spilled_vol_norm]
    return new_goal

def get_value_in_milliliters(args, value):
    return value*(args.max_volume/2.0) + (args.max_volume/2.0)

# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor, critic, action_dim, goal_dim, state_dim):
    # Set up summary Ops
    train_ops, valid_ops, training_vars, valid_vars = build_summaries()

    global_step = tf.Variable(0, name='global_step', trainable=False)
    step_op = global_step.assign(global_step+1)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(max_to_keep=5)

    sess.run(tf.global_variables_initializer())

    if opt.continue_training:
        saver.restore(sess,tf.train.latest_checkpoint(SAVE_DIR+'/'))
        actor.restore_params(tf.trainable_variables())
        critic.restore_params(tf.trainable_variables())
        print('Model Restored')
    else:
        # Initialize target network weights
        actor.update_target_network()
        critic.update_target_network()

    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    validation_step = 0
    for i in range(MAX_EPISODES):
        current_step = sess.run(global_step)

        init_height = np.random.randint(1, 11)
        s, info = env.reset(init_height)
        goal = generate_new_goal(opt.env_params, current_step, int(env.env.init_particles), validation=True)

        ep_reward = 0
        ep_ave_max_q = 0
        value_loss = 0
        ep_collisions = 0
        ep_extra_goals = 0

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

            if terminal:
                episode_spillage = s[7]

                summary_str = sess.run(train_ops, feed_dict={
                    training_vars[0]: ep_reward,
                    training_vars[1]: ep_ave_max_q / float(j),
                    training_vars[2]: value_loss / float(j),
                    training_vars[3]: ep_collisions / float(j),
                    training_vars[4]: episode_spillage,
                    training_vars[5]: ep_extra_goals
                })
                writer.add_summary(summary_str, current_step)
                writer.flush()


                if i%VALID_FREQ == VALID_FREQ-1:
                    valid_terminal = False
                    valid_r = 0
                    valid_collision = 0
                    init_height = np.random.randint(1, 11)
                    s, info = env.reset(init_height)
                    goal = generate_new_goal(opt.env_params, validation_step, int(env.env.init_particles), validation=True)
                    while not valid_terminal:
                        input_s = np.reshape(s, (1, actor.s_dim))
                        input_g = np.reshape(goal, (1, actor.goal_dim))
                        a = actor.predict_target(input_s, input_g)

                        s2, r, valid_terminal, _, collision = env.step(a[0], goal)
                        valid_r += r
                        valid_collision += int(collision)
                        s = s2


                    # Compute the % of liquid which was poured
                    valid_poured_vol = get_value_in_milliliters(opt.env_params,s[6])
                    valid_poured_goal = get_value_in_milliliters(opt.env_params,goal[0])
                    valid_percentage_poured = valid_poured_vol*100 / float(valid_poured_goal)

                    # Compute the spillage in milliliters
                    valid_spillage = get_value_in_milliliters(opt.env_params,s[7])

                    summary_valid = sess.run(valid_ops, feed_dict={
                        valid_vars[0]: valid_r,
                        valid_vars[1]: valid_collision,
                        valid_vars[2]: valid_percentage_poured,
                        valid_vars[3]: valid_spillage
                    })
                    writer.add_summary(summary_valid, current_step)
                    writer.flush()

                    save_path = saver.save(sess, SAVE_DIR + "/model", global_step=current_step)
                    print('-------------------------------------')
                    print("Model saved in file: %s" % save_path)
                    print('-------------------------------------')

                    validation_step += 1

                break

        # Increase global_step
        sess.run(step_op)

def test(sess, env, actor, critic, action_dim, goal_dim, state_dim, test_goal):
    # Set up summary Ops
    train_ops, valid_ops, training_vars, valid_vars = build_summaries()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    step_op = global_step.assign(global_step+1)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    saver.restore(sess,tf.train.latest_checkpoint(SAVE_DIR+'/'))
    actor.restore_params(tf.trainable_variables())
    critic.restore_params(tf.trainable_variables())
    print('Model Restored')

    desired_poured_vol_norm = (test_goal[0] - opt.env_params.max_volume/2.0) / (opt.env_params.max_volume/2.0)
    desired_spilled_vol_norm = (test_goal[1] - opt.env_params.max_volume/2.0) / (opt.env_params.max_volume/2.0)
    norm_test_goal = [desired_poured_vol_norm, desired_spilled_vol_norm]

    init_height = opt.test_height
    s, info = env.reset(init_height)

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

        if terminal:
            episode_spillage = s[7]
            break

    print('----------------------------------------------------')
    print('----------------------------------------------------')
    print('Episode reward: ', ep_reward)
    print('Number of collisions: ', ep_collisions)
    print('Episode Spillage: ', get_value_in_milliliters(opt.env_params,episode_spillage))
    print('Episode Poured Volume: ', get_value_in_milliliters(opt.env_params,s[6]))
    print('Saving scene...')
    env.save_scene(os.getcwd()+opt.save_scene_to_path)
    print('Completed')

def main(_):
    '''
    previous = tf.train.import_meta_graph(SAVE_DIR + '/model.ckpt.meta')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        previous.restore(sess,tf.train.latest_checkpoint(SAVE_DIR+'/'))
        last_vars = tf.trainable_variables()
        data = sess.run(last_vars)
        print('Model Restored')
    '''
    tf.reset_default_graph()

    with tf.Session() as sess:
        env = Preon_env(opt.env_params)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        state_dim = 9
        action_dim = 3
        goal_dim = 2

        actor = ActorNetwork(sess, state_dim, action_dim, goal_dim, ACTOR_LEARNING_RATE, TAU, opt.env_params)
        critic = CriticNetwork(sess, state_dim, action_dim, goal_dim, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(), opt.env_params)

        if opt.train:
            train(sess, env, actor, critic, action_dim, goal_dim, state_dim)
        else:
            test(sess,env,actor,critic,action_dim, goal_dim, state_dim, opt.test_goal)

if __name__ == '__main__':
    tf.app.run()
