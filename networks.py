import tensorflow as tf
import tflearn

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -1 and 1
    """
    def __init__(self, args):
        self.s_dim = args.agent_params.state_dim
        self.a_dim = args.agent_params.action_dim
        self.goal_dim = args.agent_params.goal_dim
        self.learning_rate = args.agent_params.actor_lr
        self.tau = args.agent_params.tau
        self.max_lin_disp = args.env_params.max_lin_disp
        self.max_ang_disp = args.env_params.max_ang_disp
        self.args = args
        self.l2_reg = 1e-4
        self.std = 1e-3

        # Actor Network
        self.inputs, self.goal, self.out = self.create_actor_network('actor_network')
        #self.inputs, self.goal, self.out, self.preactivations = self.create_actor_network('actor_network')
        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_goal, self.target_out = self.create_actor_network('actor_network')
        #self.target_inputs, self.target_goal, self.target_out, _ = self.create_actor_network('target_actor')
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        # Partial derivatives of output w.r.t network_params. action_gradient holds the initial gradients for each output
        self.actor_gradients = tf.gradients(self.out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        # Regularize preactivations to prevent saturation of the output layer
        #reg_variables = tf.losses.get_regularization_losses(scope="actor_network")
        #self.regularization_loss = tf.reduce_sum(reg_variables)
        #self.preactivation_loss = tf.multiply(tf.reduce_mean(tf.square(self.preactivations)), 1e-5)
        #self.total_loss = self.regularization_loss  # + self.preactivation_loss
        # self.optimize_reg = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self, scope):
        '''
        # tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(scope):
            inputs = tf.placeholder(tf.float32, [None, self.s_dim])
            goal = tf.placeholder(tf.float32, [None, self.goal_dim])

            net_state = tf.layers.dense(inputs, 500, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(stddev=self.std),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            net_goal = tf.layers.dense(goal, 200, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(stddev=self.std),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))

            net = tf.concat ([net_state, net_goal], axis=1)

            net = tf.layers.dense(net, 400, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(stddev=self.std),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            net = tf.layers.dense(net, 300, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(stddev=self.std),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))

            # Final layer weights are init to Uniform[-3e-3, 3e-3](NOTE: Not any more)  # tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
            preactivations = tf.layers.dense(net, self.a_dim, activation=None, kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))

            out = tf.tanh(preactivations)

        return inputs, goal, out, preactivations
        '''
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        goal = tflearn.input_data(shape=[None, self.goal_dim])

        net_state = tflearn.fully_connected(inputs, 500, activation='elu', regularizer='L2')
        net_goal = tflearn.fully_connected(goal, 200, activation='elu', regularizer='L2')

        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net_state, 400, regularizer='L2')
        t2 = tflearn.fully_connected(net_goal, 400, regularizer='L2')

        net = tflearn.activation(tf.matmul(net_state, t1.W) + tf.matmul(net_goal, t2.W) + t2.b, activation='elu')
        net = tflearn.fully_connected(net, 300, activation='elu', regularizer='L2')

        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim, activation='tanh', weights_init=w_init, regularizer='L2')
        # Scale output to -action_bound to action_bound
        # scaled_out = tf.multiply(out, [self.max_lin_disp, self.max_lin_disp, self.max_ang_disp])
        return inputs, goal, out

    def set_session(self,sess):
        self.sess = sess

    def train(self, inputs, goals, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.goal: goals,
            self.action_gradient: a_gradient
        })
    '''
    def train(self, inputs, goals, a_gradient):
        _, total_loss, preac_loss = self.sess.run([self.optimize, self.total_loss, self.preactivation_loss], feed_dict={
            self.inputs: inputs,
            self.goal: goals,
            self.action_gradient: a_gradient
        })
        return total_loss, preac_loss
    '''

    def predict(self, inputs, goals):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.goal: goals,
        })

    def predict_target(self, inputs, goals):
        return self.sess.run(self.target_out, feed_dict={
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

    def __init__(self, num_actor_vars, args):
        self.s_dim = args.agent_params.state_dim
        self.a_dim = args.agent_params.action_dim
        self.goal_dim = args.agent_params.goal_dim
        self.learning_rate = args.agent_params.critic_lr
        self.tau = args.agent_params.tau
        self.args = args
        self.num_actor_vars = num_actor_vars
        self.l2_reg = 1e-4
        self.std = 1e-3

        # Create the critic network
        self.inputs, self.goals, self.action, self.out = self.create_critic_network('critic_network')
        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_goals, self.target_action, self.target_out = self.create_critic_network('target_critic')
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        self.clipped_value = tf.clip_by_value(self.predicted_q_value, -100.0, 0.0)  # Clip targets to range of possible values [-1/(1-gamma), 0]

        # Get regularization loss
        #reg_variables = tf.losses.get_regularization_losses(scope="critic_network")
        #self.regularization_loss = tf.reduce_sum(reg_variables)

        # Define loss and optimization Op
        #self.loss = tf.reduce_mean(tf.square(self.clipped_value - self.out)) + self.regularization_loss
        #self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.loss = tflearn.mean_square(self.clipped_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the network w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def set_session(self,sess):
        self.sess = sess

    def create_critic_network(self, scope):
        '''
        with tf.variable_scope(scope):
            inputs = tf.placeholder(tf.float32, [None, self.s_dim])
            goals = tf.placeholder(tf.float32, [None, self.goal_dim])
            action = tf.placeholder(tf.float32, [None, self.a_dim])

            net_state = tf.layers.dense(inputs, 500, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(stddev=self.std),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            net_goal = tf.layers.dense(goals, 200, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(stddev=self.std),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))

            net = tf.concat ([net_state, net_goal, action], axis=1)

            net = tf.layers.dense(net, 400, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(stddev=self.std),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            net = tf.layers.dense(net, 200, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(stddev=self.std),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))

            # linear layer connected to 1 output representing Q(s,a)
            # Weights are init to Uniform[-3e-3, 3e-3]      # tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
            preactivations = tf.layers.dense(net, 1, activation=None, kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))

            out = tf.tanh(preactivations)

        return inputs, goals, action, out
        '''
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        goals = tflearn.input_data(shape=[None, self.goal_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])

        net_state = tflearn.fully_connected(inputs, 500, activation='elu', regularizer='L2')
        net_goal = tflearn.fully_connected(goals, 200, activation='elu', regularizer='L2')

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net_state, 400, regularizer='L2')
        t2 = tflearn.fully_connected(net_goal, 400, regularizer='L2')
        t3 = tflearn.fully_connected(action, 400, regularizer='L2')

        net = tflearn.activation(tf.matmul(net_state, t1.W) + tf.matmul(net_goal, t2.W) + \
                                 tf.matmul(action, t3.W) + t3.b, activation='elu')

        net = tflearn.fully_connected(net, 200, activation='elu', regularizer='L2')

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
