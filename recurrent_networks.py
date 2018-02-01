import tensorflow as tf
import tflearn

# ===========================
#   Actor and Critic DNNs
# ===========================

class RecurrentActorNetwork(object):
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
        self.rnn_size = args.agent_params.rnn_size
        self.learning_rate = args.agent_params.actor_lr
        self.tau = args.agent_params.tau
        self.max_lin_disp = args.env_params.max_lin_disp
        self.max_ang_disp = args.env_params.max_ang_disp
        self.args = args

        # Actor Network
        self.inputs, self.goal, self.init_hidden_cm, self.scaled_out, self.rnn_hidden_out = self.create_actor_network('actor_rnn')
        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_goal, self.target_init_hidden_cm, self.target_scaled_out, self.target_rnn_hidden_out = self.create_actor_network('actor_target_rnn')
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient_trace_input = tf.placeholder(tf.float32, [None, None, self.a_dim])

        self.update_length = tf.placeholder(tf.int32)

        # Combine the gradients here
        # Partial derivatives of scaled_out w.r.t network_params. action_gradient holds the initial gradients for each scaled_out
        self.actor_gradients = tf.gradients(self.scaled_out[:,-self.update_length:,:], self.network_params, -self.action_gradient_trace_input)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self, scope='actor_net'):
        state_trace_input = tflearn.input_data(shape=[None, None, self.s_dim])
        goal_trace_input = tflearn.input_data(shape=[None, None, self.goal_dim])

        init_hidden_c = tflearn.input_data(shape=[None, self.rnn_size])
        init_hidden_m = tflearn.input_data(shape=[None, self.rnn_size])

        init_hidden_cm = (init_hidden_c, init_hidden_m)

        net_state = tflearn.layers.conv.conv_2d (tf.expand_dims(state_trace_input,-1), 500, [1, self.s_dim], padding='valid', activation='elu', bias=True, weights_init='uniform_scaling', regularizer='L2')
        net_goal = tflearn.layers.conv.conv_2d (tf.expand_dims(goal_trace_input,-1), 200, [1, self.goal_dim], padding='valid', activation='elu', bias=True, weights_init='uniform_scaling', regularizer='L2')

        concat_net = tf.concat([net_state, net_goal],axis=-1)
        concat_net = tf.squeeze(input=concat_net, axis=2)

        cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
        init_state = tf.nn.rnn_cell.LSTMStateTuple(init_hidden_c, init_hidden_m)
        rnn_states, rnn_hidden_cm = tf.nn.dynamic_rnn(cell, concat_net, dtype=tf.float32, initial_state=init_state, scope=scope)

        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        final = tflearn.layers.conv.conv_2d (tf.expand_dims(rnn_states,-1), self.a_dim, [1, self.rnn_size], padding='valid', activation='tanh', bias=True, weights_init=w_init, regularizer='L2')
        out = tf.squeeze(input=final, axis=2)

        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, [self.max_lin_disp, self.max_lin_disp, self.max_ang_disp])
        return state_trace_input, goal_trace_input, init_hidden_cm, scaled_out, rnn_hidden_cm

    def set_session(self,sess):
        self.sess = sess

    def train(self, inputs, goals, a_gradient, update_length, init_temporal_hidden_cm_batch):
        try:
            self.sess.run(self.optimize, feed_dict={
                self.inputs: inputs,
                self.goal: goals,
                self.action_gradient_trace_input: a_gradient,
                self.update_length: update_length,
                # NOTE: Original included action_trace
                #self.scaled_out: action_trace_batch,
                self.init_hidden_cm: init_temporal_hidden_cm_batch
            })
        except Exception as e:
            raise

    # NOTE: This is the same as calling predict with mode = 2
    def action_trace(self, input_trace_batch, goal_trace_batch, init_temporal_hidden_cm_batch):
        return self.sess.run([self.scaled_out, self.rnn_hidden_out],feed_dict={
            self.inputs: input_trace_batch,
            self.goal: goal_trace_batch,
            self.init_hidden_cm: init_temporal_hidden_cm_batch
            })

    def predict(self, input_trace_batch, goal_trace_batch, init_temporal_hidden_cm, mode=2):
        action, last_h = self.sess.run([self.scaled_out, self.rnn_hidden_out], feed_dict={
            self.inputs: input_trace_batch,
            self.goal: goal_trace_batch,
            self.init_hidden_cm: init_temporal_hidden_cm
        })
        if mode == 0:
            return action[0][0]
        elif mode == 1:
            return last_h
        elif mode == 2:
            return action[0][0], last_h

    def predict_target(self, input_trace_batch, goal_trace_batch, init_temporal_hidden_cm, mode=0):
        action_trace, last_h = self.sess.run([self.target_scaled_out, self.target_rnn_hidden_out], feed_dict={
            self.target_inputs: input_trace_batch,
            self.target_goal: goal_trace_batch,
            self.target_init_hidden_cm: init_temporal_hidden_cm
        })
        if mode == 0:
            return action_trace[:,-1,:]
        elif mode == 1:
            return last_h

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def restore_params(self, parameters):
        restore_network = [self.network_params[i].assign(parameters[i]) for i in range(len(self.network_params))]
        restore_target = [self.target_network_params[i].assign(parameters[i+len(self.network_params)]) for i in range(len(self.target_network_params))]
        self.sess.run([restore_network, restore_target])

class RecurrentCriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, num_actor_vars, args):
        self.s_dim = args.agent_params.state_dim
        self.a_dim = args.agent_params.action_dim
        self.goal_dim = args.agent_params.goal_dim
        self.rnn_size = args.agent_params.rnn_size
        self.learning_rate = args.agent_params.critic_lr
        self.tau = args.agent_params.tau
        self.args = args
        self.num_actor_vars = num_actor_vars

        # Create the critic network
        self.input_trace, self.goal_trace, self.action_trace, self.init_hidden_cm, self.out_trace, self.last_hidden_cm = self.create_critic_network('critic_rnn')
        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_input_trace, self.target_goal_trace, self.target_action_trace, self.target_init_hidden_cm, self.target_out_trace, self.target_last_hidden = self.create_critic_network('target_critic_rnn')
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value_trace = tf.placeholder(tf.float32, [None, None, 1])
        self.update_length = tf.placeholder(tf.int32)

        self.clipped_value = tf.clip_by_value(self.predicted_q_value_trace, -100.0, 0.0)  # Clip targets to range of possible values [-1/(1-gamma), 0]

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.clipped_value, self.out_trace[:,-self.update_length:,:])
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the network w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all actions except for one.
        self.action_grads = tf.gradients(self.out_trace, self.action_trace)

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def set_session(self,sess):
        self.sess = sess

    def create_critic_network(self, scope='critic_net'):
        input_trace = tflearn.input_data(shape=[None, None, self.s_dim])
        goal_trace = tflearn.input_data(shape=[None, None, self.goal_dim])
        action_trace = tflearn.input_data(shape=[None, None, self.a_dim])

        init_hidden_c = tf.placeholder("float", [None,self.rnn_size])
        init_hidden_m = tf.placeholder("float", [None,self.rnn_size])

        init_hidden_cm = (init_hidden_c, init_hidden_m)

        net_state = tflearn.layers.conv.conv_2d (tf.expand_dims(input_trace,-1), 500, [1, self.s_dim], padding='valid', activation='elu', bias=True, weights_init='uniform_scaling', regularizer='L2')
        net_goal = tflearn.layers.conv.conv_2d (tf.expand_dims(goal_trace,-1), 200, [1, self.goal_dim], padding='valid', activation='elu', bias=True, weights_init='uniform_scaling', regularizer='L2')
        net_action = tflearn.layers.conv.conv_2d (tf.expand_dims(action_trace,-1), 10, [1, self.a_dim], padding='valid', activation='elu', bias=True, weights_init='uniform_scaling', regularizer='L2')

        concat_net = tf.concat([net_state, net_goal, net_action], axis=-1)
        concat_net = tf.squeeze(input=concat_net, axis=2)

        cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
        init_state = tf.nn.rnn_cell.LSTMStateTuple(init_hidden_c, init_hidden_m)
        rnn_states, rnn_hidden_cm = tf.nn.dynamic_rnn(cell, concat_net, dtype=tf.float32, initial_state=init_state, scope=scope)


        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        final = tflearn.layers.conv.conv_2d (tf.expand_dims(rnn_states,-1), 1, [1, self.rnn_size], padding='valid', activation='tanh', bias=True, weights_init=w_init, regularizer='L2')
        out = tf.squeeze(final, axis=2)

        return input_trace, goal_trace, action_trace, init_hidden_cm, out, rnn_hidden_cm

    def train(self, input_trace, goal_trace, action_trace, predicted_q_value, update_length, init_hidden_cm):
        try:
            return self.sess.run([self.out_trace, self.optimize, self.loss], feed_dict={
                self.input_trace: input_trace,
                self.goal_trace: goal_trace,
                self.action_trace: action_trace,
                self.predicted_q_value_trace: predicted_q_value,
                self.init_hidden_cm: init_hidden_cm,
                self.update_length: update_length
            })

        except Exception as e:
            raise

    def predict(self, input_trace, goal_trace, action_trace, init_hidden_cm, mode = 2):
        q_value, last_h = self.sess.run([self.out_trace, self.last_hidden_cm], feed_dict={
            self.input_trace: input_trace,
            self.goal_trace: goal_trace,
            self.action_trace: action_trace,
            self.init_hidden_cm: init_hidden_cm
        })

        if mode == 0:
            return q_value[0][0]
        elif mode == 1:
            return last_h
        elif mode == 2:
            return q_value[0][0], last_h

    def predict_target(self, input_trace, goal_trace, action_trace, init_hidden_cm, mode = 0):
        q_value, last_h = self.sess.run([self.target_out_trace, self.target_last_hidden], feed_dict={
            self.target_input_trace: input_trace,
            self.target_goal_trace: goal_trace,
            self.target_action_trace: action_trace,
            self.target_init_hidden_cm: init_hidden_cm
        })

        if mode == 0:
            return q_value[:,-1,:]
        elif mode == 1:
            return last_h

    def action_gradients(self, input_trace, goal_trace, action_trace, update_length, init_hidden_cm):
        return self.sess.run(self.action_grads, feed_dict={
            self.input_trace: input_trace,
            self.goal_trace: goal_trace,
            self.action_trace: action_trace,
            self.init_hidden_cm: init_hidden_cm
        })[0][:,-update_length:,:]

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def restore_params(self, parameters):
        restore_network = [self.network_params[i].assign(parameters[i+self.num_actor_vars]) for i in range(len(self.network_params))]
        restore_target = [self.target_network_params[i].assign(parameters[i+self.num_actor_vars+len(self.network_params)]) for i in range(len(self.target_network_params))]
        self.sess.run([restore_network, restore_target])
