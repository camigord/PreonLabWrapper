import preonpy
import tensorflow as tf
import numpy as np
from env.preon_env import Preon_env
from utils.options import Options
from utils.utils import *
from utils.ou_noise import OUNoise
import tflearn
import sys
import os
import argparse

from networks import RecurrentActorNetwork, RecurrentCriticNetwork
from utils.replay_buffer_trace import ReplayBufferTrace
from policy import Policy


'''
/////   DDPG - DISTRIBUTED APPROACH
'''

# input flags
parser = argparse.ArgumentParser(description='Help message')
parser.add_argument('--job_name', help="Either 'ps' or 'worker'")
parser.add_argument('--task_index', type=int, default=0, help="Index of task within the job")
parser.add_argument('--restore', action='store_true', help="Restores previous model")
parser.add_argument('--new_mem_reply', action='store_false', help="If set, does not restore experience replay when restoring model")
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
    extra_trajectories = tf.Variable(0.)
    training_summaries.append(tf.summary.scalar("Extra Trajectories", extra_trajectories))
    filling_rates = tf.placeholder(tf.float32)
    training_summaries.append(tf.summary.histogram("Filling rates", filling_rates))
    training_poured_percent = tf.Variable(0.)
    training_summaries.append(tf.summary.scalar("Training Pouring (%)", training_poured_percent))

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
    training_vars = [episode_reward, episode_ave_max_q, value_loss, collision_rate, avg_spillage, extra_trajectories, filling_rates, training_poured_percent]

    #summary_ops = tf.summary.merge_all()

    return train_ops, valid_ops, training_vars, valid_vars


# ===========================
#   Generate a new normalized random goal (fill_level, spillage)
# ===========================
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


# ===========================
#   Implement Hindsight Experience Replay on trajectories
# ===========================
def augment_trajectories(replay_buffer, env, episode_trajectory, max_samples):
    # NOTE: Working with fill_level and spillage and assuming those values are stored in state[3:5]

    # Get the initial state (fill_level, spillage)
    init_state = episode_trajectory[0][0][3:5]

    # Get all the future states which are different than initial and not close enough to the goal
    possible_goals = []
    for transition in episode_trajectory:
        next_state = transition[-1][3:5]    # fill_level, spillage at next_state
        old_reward = transition[3]
        if next_state != init_state and old_reward != env.goal_reward:
            possible_goals.append(next_state)

    to_sample = min(len(possible_goals), max_samples)
    # Randomly pick new goals from possible ones
    index = np.random.choice(range(len(possible_goals)), to_sample, replace=False)
    for i in index:
        # New goal for augmented trajectory
        new_goal = possible_goals[i]

        augmented_trajectory = []
        # Go through the entire episode changing goals and rewards for the new trajectory
        for transition in episode_trajectory:
            state = transition[0]
            goal = new_goal
            action = transition[2]
            old_reward = transition[3]      # Required to check for collision
            terminal = transition[4]
            next_state = transition[5]

            dict_state_next = {'fill_level': next_state[3],
                          'spillage': next_state[4]}
            new_reward = env.estimate_new_reward(dict_state_next, new_goal, old_reward)


            temp = [np.reshape(state, (len(state),)), np.reshape(goal, (len(goal),)),
                    np.reshape(action, (len(action),)), new_reward, terminal, np.reshape(next_state, (len(next_state),))]

            augmented_trajectory.append(temp)

        # Add augmented transitions to memory
        replay_buffer.add(augmented_trajectory)

    return to_sample

# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor, critic, saver, global_step, step_op, replay_buffer):

    # Set up summary Ops
    train_ops, valid_ops, training_vars, valid_vars = build_summaries()

    # Initialize Ornstein-Uhlenbeck process for action exploration
    exploration_noise = OUNoise(actor.a_dim)

    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Used to store discount matrix
    discounting_mat_dict = {}

    validation_step = 0
    for episode in range(MAX_EPISODES):
        # Re-iniitialize the random process when an episode ends
        exploration_noise.reset()

        current_step = sess.run(global_step)

        #init_height = np.random.randint(2, 11)
        init_height = 10
        s, info = env.reset(init_height)
        goal = generate_new_goal(opt.env_params, int(env.env.init_particles), info[3])

        # NOTE: Lets remove previous action and filling rate from state, given that we have recurrency
        state = s[0:3] + s[6:8] # delta_x, delta_y, theta, fill_level, spillage

        ep_reward = 0
        ep_ave_max_q = []
        value_loss = []
        ep_collisions = 0
        filling_rates = []

        episode_trajectory = []

        for step in range(env.max_steps+1):
            if step == 0:
                init_actor_hidden_c = state_initialiser(shape=(1,actor.rnn_size),mode='g')
                init_actor_hidden_m = state_initialiser(shape=(1,actor.rnn_size),mode='g')
                actor_init_hidden_cm = (init_actor_hidden_c, init_actor_hidden_m)

            input_s = np.reshape(state, (1, 1, actor.s_dim))
            input_g = np.reshape(goal, (1, 1, actor.goal_dim))

            # Added exploration noise
            action, actor_last_hidden_cm = actor.predict(input_s, input_g,actor_init_hidden_cm)

            # Add exploration noise
            noise = exploration_noise.noise()
            action += noise
            # action += (1. / (1. + current_step))

            s2, r, terminal, info, collision = env.step(action, goal)

            # NOTE: Lets remove previous action and filling rate from state, given that we have recurrency
            next_state = s2[0:3] + s2[6:8] # delta_x, delta_y, theta, fill_level, spillage

            transition = [np.reshape(state, (actor.s_dim,)), np.reshape(goal, (actor.goal_dim,)),
                          np.reshape(action, (actor.a_dim,)), r, terminal, np.reshape(next_state, (actor.s_dim,))]

            episode_trajectory.append(transition)

            state = next_state
            s = s2                  # We keep this information for visualization purposes
            actor_init_hidden_cm = actor_last_hidden_cm
            ep_reward += r
            ep_collisions += int(collision)
            filling_rates.append(get_denormalized(s[8], 0.0, 1.0))

            # TRAINING!
            # Keep adding experience to the memory until there are at least minibatch size samples
            if replay_buffer.size() >= opt.agent_params.min_memory_size:

                # Sample from memory replay
                minibatch = replay_buffer.sample_batch(MINIBATCH_SIZE)
                try:
                    state_trace_batch = np.stack(minibatch[:,:,0].ravel()).reshape(MINIBATCH_SIZE, opt.agent_params.trace_length, actor.s_dim)
                    goal_trace_batch = np.stack(minibatch[:,:,1].ravel()).reshape(MINIBATCH_SIZE, opt.agent_params.trace_length, actor.goal_dim)
                    action_trace_batch = np.stack(minibatch[:,:,2].ravel()).reshape(MINIBATCH_SIZE,opt.agent_params.trace_length, actor.a_dim)

                    reward_trace_batch = np.stack(minibatch[:,:,3].ravel()).reshape(MINIBATCH_SIZE, opt.agent_params.trace_length, 1)
                    done_trace_batch = np.stack(minibatch[:,:,4].ravel()).reshape(MINIBATCH_SIZE, opt.agent_params.trace_length, 1)

                    next_state_batch = np.stack(minibatch[:,-1,5].ravel()).reshape(MINIBATCH_SIZE, 1, actor.s_dim)
                    next_state_trace_batch = np.concatenate([state_trace_batch, next_state_batch],axis=1)

                    # Needed when using next_state_trace_batch, because it is one value longer
                    extra_goal_batch = np.stack(minibatch[:,-1,1].ravel()).reshape(MINIBATCH_SIZE, 1, actor.goal_dim)
                    extra_goal_trace_batch = np.concatenate([goal_trace_batch, extra_goal_batch],axis=1)

                except Exception as e:
                    print(str(e))
                    raise

                # Initialize hidden states
                init_actor_hidden_batch = state_initialiser(shape=(MINIBATCH_SIZE, actor.rnn_size), mode='z')
                actor_init_h_batch = (init_actor_hidden_batch, init_actor_hidden_batch)

                init_critic_hidden_batch = state_initialiser(shape=(MINIBATCH_SIZE, actor.rnn_size), mode='z')
                critic_init_h_batch = (init_critic_hidden_batch, init_critic_hidden_batch)

                if opt.agent_params.trace_length <= opt.agent_params.opt_length:
                    target_actor_init_h_batch = actor_init_h_batch
                    target_critic_init_h_batch = critic_init_h_batch

                    update_length = opt.agent_params.trace_length
                else:
                    # If trace is longer than optimization (BPTT), we need to estimate hidden state at the beginning of optimization
                    target_actor_init_h_batch = actor.predict_target(state_trace_batch[:,:-opt.agent_params.opt_length,:],
                                                             goal_trace_batch[:,:-opt.agent_params.opt_length,:], actor_init_h_batch, mode =1)
                    target_critic_init_h_batch = critic.predict_target(state_trace_batch[:,:-opt.agent_params.opt_length,:],
                                                                   goal_trace_batch[:,:-opt.agent_params.opt_length,:],
                                                                   action_trace_batch[:,:-opt.agent_params.opt_length,:], critic_init_h_batch, mode = 1)

                    state_trace_batch = state_trace_batch[:,-opt.agent_params.opt_length:,:]
                    next_state_trace_batch = next_state_trace_batch[:,-(opt.agent_params.opt_length+1):,:]
                    extra_goal_trace_batch = extra_goal_trace_batch[:,-(opt.agent_params.opt_length+1):,:]
                    action_trace_batch = action_trace_batch[:,-opt.agent_params.opt_length:,:]
                    reward_trace_batch = reward_trace_batch[:,-opt.agent_params.opt_length:,:]
                    done_trace_batch = done_trace_batch[:,-opt.agent_params.opt_length:,:]
                    goal_trace_batch = goal_trace_batch[:,-opt.agent_params.opt_length:,:]

                    update_length = opt.agent_params.opt_length


                # Calculate targets, equivalent to: target_q = critic.predict_target(s2_batch, g_batch, actor.predict_target(s2_batch, g_batch))
                next_action_batch = actor.predict_target(next_state_trace_batch, extra_goal_trace_batch, target_actor_init_h_batch)
                next_action_trace_batch = np.concatenate([action_trace_batch, np.expand_dims(next_action_batch, axis=1)], axis=1)
                target_q_batch = critic.predict_target(next_state_trace_batch, extra_goal_trace_batch, next_action_trace_batch, target_critic_init_h_batch)

                # Mask Q values of terminal states
                target_lastQ_batch_masked = target_q_batch * (1.- done_trace_batch[:,-1])
                # Vector of rewards + lastQ
                rQ = np.concatenate([np.squeeze(reward_trace_batch[:,-update_length:],axis=-1), target_lastQ_batch_masked],axis=1)

                # Create Discount matrix (using gamma)
                try:
                    # If already defined
                    discounting_mat = discounting_mat_dict[update_length]
                except KeyError:
                    discounting_mat = np.zeros(shape=(update_length,update_length+1),dtype=np.float)
                    for i in range(update_length):
                        discounting_mat[i,:i] = 0.
                        discounting_mat[i,i:] = opt.agent_params.gamma ** np.arange(0.,-i+update_length+1)
                    discounting_mat = np.transpose(discounting_mat)
                    discounting_mat_dict[update_length] = discounting_mat

                try:
                    # Dot product between rewards and discount factors
                    y_trace_batch = np.expand_dims(np.matmul(rQ, discounting_mat), axis=-1)
                except Exception as e:
                    raise

                # Update the critic given the targets. Equivalent to critic.train(s_batch, g_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
                predicted_q_value, _, v_loss = critic.train(state_trace_batch,
                                                            goal_trace_batch,
                                                            action_trace_batch,
                                                            y_trace_batch,
                                                            update_length, critic_init_h_batch)


                # Update the actor policy using the sampled gradient
                for i in range(update_length):
                    actor_init_h_batch_trace = (np.expand_dims(actor_init_h_batch[0],axis=1), np.expand_dims(actor_init_h_batch[1],axis=1))
                    critic_init_h_batch_trace = (np.expand_dims(critic_init_h_batch[0],axis=1), np.expand_dims(critic_init_h_batch[1],axis=1))
                    if i == 0:
                        actor_init_h_batch_stack = actor_init_h_batch_trace
                        critic_init_h_batch_stack = critic_init_h_batch_trace
                    else:
                        actor_init_h_batch_stack = (np.concatenate((actor_init_h_batch_stack[0],actor_init_h_batch_trace[0]),axis=1),np.concatenate((actor_init_h_batch_stack[1],actor_init_h_batch_trace[1]),axis=1))
                        critic_init_h_batch_stack = (np.concatenate((critic_init_h_batch_stack[0],critic_init_h_batch_trace[0]),axis=1),np.concatenate((critic_init_h_batch_stack[1],critic_init_h_batch_trace[1]),axis=1))

                    action_trace_batch_for_gradients, actor_init_h_batch = actor.action_trace(np.expand_dims(state_trace_batch[:,i],1),
                                                                                              np.expand_dims(goal_trace_batch[:,i],1),
                                                                                              actor_init_h_batch)
                    critic_init_h_batch = critic.predict(np.expand_dims(state_trace_batch[:,i],1),
                                                                np.expand_dims(goal_trace_batch[:,i],1),
                                                                np.expand_dims(action_trace_batch[:,i],1), critic_init_h_batch, mode = 1)
                    if i == 0:
                        action_trace_batch_for_gradients_stack = action_trace_batch_for_gradients
                    else:
                        action_trace_batch_for_gradients_stack = np.concatenate((action_trace_batch_for_gradients_stack,action_trace_batch_for_gradients),axis=1)

                state_trace_batch_stack = np.reshape(state_trace_batch,(MINIBATCH_SIZE*update_length, 1, actor.s_dim))
                action_trace_batch_stack = np.reshape(action_trace_batch,(MINIBATCH_SIZE*update_length, 1, actor.a_dim))
                goal_trace_batch_stack = np.reshape(goal_trace_batch,(MINIBATCH_SIZE*update_length, 1, actor.goal_dim))
                action_trace_batch_for_gradients_stack = np.reshape(action_trace_batch_for_gradients_stack,(MINIBATCH_SIZE*update_length, 1, actor.a_dim))
                actor_init_h_batch_stack = (np.reshape(actor_init_h_batch_stack[0],(MINIBATCH_SIZE*update_length, actor.rnn_size)), np.reshape(actor_init_h_batch_stack[1],(MINIBATCH_SIZE*update_length, actor.rnn_size)))
                critic_init_h_batch_stack = (np.reshape(critic_init_h_batch_stack[0],(MINIBATCH_SIZE*update_length, critic.rnn_size)), np.reshape(critic_init_h_batch_stack[1],(MINIBATCH_SIZE*update_length, critic.rnn_size)))


                # Update actor - equivalent to:
                # a_outs = actor.predict(s_batch, g_batch)
                # grads = critic.action_gradients(s_batch, g_batch, a_outs)
                # actor.train(s_batch, g_batch, grads[0])

                q_gradient_trace_batch = critic.action_gradients(state_trace_batch_stack,
                                                                 goal_trace_batch_stack,
                                                                 action_trace_batch_for_gradients_stack,
                                                                 1,
                                                                 critic_init_h_batch_stack)

                # Update the actor policy using the sampled gradient:
                actor.train(state_trace_batch_stack,
                            goal_trace_batch_stack,
                            q_gradient_trace_batch,
                            1,
                            actor_init_h_batch_stack)


                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

                ep_ave_max_q.append(np.amax(predicted_q_value))
                value_loss.append(v_loss)


            if terminal:
                # Add episode transitions to memory
                replay_buffer.add(episode_trajectory)

                # Augment memory using Hindsight Experience Replay (HER)
                ep_extra_trajectories = augment_trajectories(replay_buffer, env, episode_trajectory, opt.agent_params.max_augmented_goals)

                episode_spillage = s[7]

                # Compute the % of liquid which was poured
                train_poured_vol = get_denormalized(s[6],0.0,1.0)
                train_poured_goal = get_denormalized(goal[0],0.0,1.0)
                train_percentage_poured = train_poured_vol*100 / float(train_poured_goal)

                summary_str = sess.run(train_ops, feed_dict={
                    training_vars[0]: ep_reward,
                    training_vars[1]: np.mean(ep_ave_max_q),
                    training_vars[2]: np.mean(value_loss),
                    training_vars[3]: ep_collisions / float(env.max_steps),
                    training_vars[4]: episode_spillage,
                    training_vars[5]: ep_extra_trajectories,
                    training_vars[6]: np.array(filling_rates),
                    training_vars[7]: train_percentage_poured
                })
                writer.add_summary(summary_str, current_step)
                writer.flush()

                if episode % VALID_FREQ == 0:
                    valid_terminal = False
                    valid_r = 0
                    valid_collision = 0
                    # init_height = np.random.randint(2, 11)
                    init_height = 10
                    s, info = env.reset(init_height)
                    goal = generate_new_goal(opt.env_params, int(env.env.init_particles), info[3])

                    # NOTE: Lets remove previous action and filling rate from state, given that we have recurrency
                    state = s[0:3] + s[6:8] # delta_x, delta_y, theta, fill_level, spillage

                    valid_hidden_c = state_initialiser(shape=(1,actor.rnn_size),mode='g')
                    valid_hidden_m = state_initialiser(shape=(1,actor.rnn_size),mode='g')
                    init_valid_hidden = (valid_hidden_c, valid_hidden_m)

                    while not valid_terminal:
                        input_s = np.reshape(state, (1, 1, actor.s_dim))
                        input_g = np.reshape(goal, (1, 1, actor.goal_dim))

                        action, init_valid_hidden = actor.predict(input_s, input_g, init_valid_hidden)

                        s2, r, terminal, info, collision = env.step(action, goal)

                        # NOTE: Lets remove previous action and filling rate from state, given that we have recurrency
                        next_state = s2[0:3] + s2[6:8] # delta_x, delta_y, theta, fill_level, spillage

                        valid_r += r
                        valid_collision += int(collision)
                        s = s2
                        state = next_state

                    # Compute the % of liquid which was poured
                    valid_poured_vol = get_denormalized(s[6], 0.0, 1.0)
                    valid_poured_goal = get_denormalized(goal[0], 0.0, 1.0)
                    valid_percentage_poured = valid_poured_vol*100 / float(valid_poured_goal)

                    # Compute the spillage in milliliters
                    valid_spillage = get_denormalized(s[7], 0.0, opt.env_params.max_volume)

                    summary_valid = sess.run(valid_ops, feed_dict={
                        valid_vars[0]: valid_r,
                        valid_vars[1]: valid_collision,
                        valid_vars[2]: valid_percentage_poured,
                        valid_vars[3]: valid_spillage
                    })
                    writer.add_summary(summary_valid, current_step)
                    writer.flush()

                    # Save model
                    save_path = saver.save(sess, SAVE_DIR + "/model", global_step=global_step)
                    # Save current replay memory
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

    init_height = opt.test_height

    s, info = env.reset(init_height)
    print('**************************************')
    print(info)
    print('**************************************')

    ep_reward = 0
    ep_collisions = 0

    for j in range(env.max_steps+1):
        # Denormalize because policy class normalizes internally
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

        s2, r, terminal, info, collision = env.step(a, policy.goal)

        s = s2
        ep_reward += r
        ep_collisions += int(collision)

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

                actor = RecurrentActorNetwork(opt)
                critic = RecurrentCriticNetwork(actor.get_num_trainable_vars(), opt)

                # Initialize replay memory
                replay_buffer = ReplayBufferTrace(BUFFER_SIZE, opt.agent_params.trace_length, opt.save_dir, RANDOM_SEED)

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

                        # Restore memory reply
                        if not args.new_mem_reply:
                            replay_buffer.load_pickle()
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

                    train(sess, env, actor, critic, saver, global_step, step_op)


    else:           # When testing

        with tf.Session() as sess:
            policy = Policy(sess)
            env = Preon_env(opt.env_params)
            test(policy, env, args.goal, args.scene_name)

if __name__ == '__main__':
    tf.app.run()
