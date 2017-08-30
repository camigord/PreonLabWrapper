import preonpy
import tensorflow as tf
import numpy as np
from env.preon_env import Preon_env
from utils.options import Options
import tflearn
import sys
import os

from replay_buffer import ReplayBuffer

# ==========================
#   Training Parameters
# ==========================

opt = Options()

# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 1000
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
#   Actor and Critic DNNs
# ===========================


class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, sess, state_dim, action_dim, goal_dim, learning_rate, tau, args):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.goal_dim = goal_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.args = args

        # Actor Network
        self.inputs, self.goal, self.out, self.scaled_out = self.create_actor_network()
        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_goal, self.target_out, self.target_scaled_out = self.create_actor_network()
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        # Partial derivatives of scaled_out w.r.t network_params. action_gradient holds the initial gradients for each scaled_out
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        goal = tflearn.input_data(shape=[None, self.goal_dim])

        net_state = tflearn.fully_connected(inputs, 400, activation='relu')
        net_goal = tflearn.fully_connected(goal, 200, activation='relu')

        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net_state, 300)
        t2 = tflearn.fully_connected(net_goal, 300)

        net = tflearn.activation(tf.matmul(net_state, t1.W) + tf.matmul(net_goal, t2.W) + t2.b, activation='relu')
        net = tflearn.fully_connected(net, 300, activation='relu')

        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, [self.args.max_lin_disp, self.args.max_lin_disp, self.args.max_ang_disp])
        return inputs, goal, out, scaled_out

    def train(self, inputs, goals, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.goal: goals,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs, goals):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs,
            self.goal: goals,
        })

    def predict_target(self, inputs, goals):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs,
            self.target_goal: goals,
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def restore_params(self, parameters):
        restore_network = [self.network_params[i].assign(parameters[i]) for i in range(len(self.network_params))]
        restore_target = [self.target_network_params[i].assign(parameters[i+len(self.network_params)]) for i in range(len(self.target_network_params))]
        self.sess.run([restore_network, restore_target])


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, goal_dim, learning_rate, tau, num_actor_vars, args):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.goal_dim = goal_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.args = args
        self.num_actor_vars = num_actor_vars

        # Create the critic network
        self.inputs, self.goals, self.action, self.out = self.create_critic_network()
        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_goals, self.target_action, self.target_out = self.create_critic_network()
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
        self.clipped_value = tf.clip_by_value(self.predicted_q_value, -100.0, 0.0)

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.clipped_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the network w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        goals = tflearn.input_data(shape=[None, self.goal_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])

        net_state = tflearn.fully_connected(inputs, 400, activation='relu', regularizer='L2')
        net_goal = tflearn.fully_connected(goals, 200, activation='relu', regularizer='L2')

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net_state, 300, regularizer='L2')
        t2 = tflearn.fully_connected(net_goal, 300, regularizer='L2')
        t3 = tflearn.fully_connected(action, 300, regularizer='L2')

        net = tflearn.activation(tf.matmul(net_state, t1.W) + tf.matmul(net_goal, t2.W) + \
                                 tf.matmul(action, t3.W) + t3.b, activation='relu')

        net = tflearn.fully_connected(net, 200, activation='relu', regularizer='L2')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init, regularizer='L2')
        return inputs, goals, action, out

    def train(self, inputs, goals, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize, self.loss], feed_dict={
            self.inputs: inputs,
            self.goals: goals,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, goals, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.goals: goals,
            self.action: action
        })

    def predict_target(self, inputs, goals, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_goals: goals,
            self.target_action: action
        })

    def action_gradients(self, inputs, goals, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.goals: goals,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def restore_params(self, parameters):
        restore_network = [self.network_params[i].assign(parameters[i+self.num_actor_vars]) for i in range(len(self.network_params))]
        restore_target = [self.target_network_params[i].assign(parameters[i+self.num_actor_vars+len(self.network_params)]) for i in range(len(self.target_network_params))]
        self.sess.run([restore_network, restore_target])

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


def generate_new_goal(args, step, validation=False):
    if validation:
        possible_values = [50,100,150,200,250,300,350,400,450]
        desired_poured_vol = possible_values[int(step%len(possible_values))]
        #desired_poured_vol = np.random.choice([50,100,150,200,250,300,350,400,450])
        desired_spilled_vol = 0.0
    else:
        desired_poured_vol = np.random.randint(0,args.max_volume)
        # We can only spill so much as available liquid remains
        desired_spilled_vol = np.random.randint(0,args.max_volume - desired_poured_vol)

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

        goal = generate_new_goal(opt.env_params, current_step, validation=True)
        s, info = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0
        value_loss = 0
        ep_collisions = 0
        ep_extra_goals = 0

        for j in range(MAX_EP_STEPS):

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
            if s2[6:] != s[6:]:
                ep_extra_goals += 1
                new_goal = s2[6:]
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
                episode_spillage = s[-1]

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
                    s, info = env.reset()
                    goal = generate_new_goal(opt.env_params, validation_step, validation=True)
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
                    valid_spillage = get_value_in_milliliters(opt.env_params,s[-1])

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

    s, info = env.reset()

    ep_reward = 0
    ep_collisions = 0

    for j in range(MAX_EP_STEPS):
        # Added exploration noise
        input_s = np.reshape(s, (1, actor.s_dim))
        input_g = np.reshape(norm_test_goal, (1, actor.goal_dim))
        a = actor.predict(input_s, input_g)

        s2, r, terminal, info, collision = env.step(a[0], norm_test_goal)

        s = s2
        ep_reward += r
        ep_collisions += int(collision)

        if terminal:
            episode_spillage = s[-1]
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

        state_dim = 8
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
