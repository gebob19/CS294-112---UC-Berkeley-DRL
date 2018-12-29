import tensorflow as tf
import numpy as np

import utils


class ModelBasedPolicy(object):

    def __init__(self,
                 env,
                 init_dataset,
                 horizon=15,
                 num_random_action_selection=4096,
                 nn_layers=1):
        self._cost_fn = env.cost_fn
        self._state_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._action_space_low = env.action_space.low
        self._action_space_high = env.action_space.high
        self._init_dataset = init_dataset
        self._horizon = horizon
        self._num_random_action_selection = num_random_action_selection
        self._nn_layers = nn_layers
        self._learning_rate = 1e-3

        self._sess, self._state_ph, self._action_ph, self._next_state_ph,\
            self._next_state_pred, self._loss, self._optimizer, self._best_action = self._setup_graph()

    def _setup_placeholders(self):
        """
            Creates the placeholders used for training, prediction, and action selection

            returns:
                state_ph: current state
                action_ph: current_action
                next_state_ph: next state

            implementation details:
                (a) the placeholders should have 2 dimensions,
                    in which the 1st dimension is variable length (i.e., None)
        """
        state_ph = tf.placeholder(tf.float64, shape=[None, self._state_dim])
        action_ph = tf.placeholder(tf.float64, shape=[None, self._action_dim])
        next_state_ph = tf.placeholder(tf.float64, shape=[None, self._state_dim])

        return state_ph, action_ph, next_state_ph

    def _dynamics_func(self, state, action, reuse):
        """
            Takes as input a state and action, and predicts the next state

            returns:
                next_state_pred: predicted next state

            implementation details (in order):
                (a) Normalize both the state and action by using the statistics of self._init_dataset and
                    the utils.normalize function
                (b) Concatenate the normalized state and action
                (c) Pass the concatenated, normalized state-action tensor through a neural network with
                    self._nn_layers number of layers using the function utils.build_mlp. The resulting output
                    is the normalized predicted difference between the next state and the current state
                (d) Unnormalize the delta state prediction, and add it to the current state in order to produce
                    the predicted next state

        """
        s_mean, s_std = tf.constant(self._init_dataset.state_mean), tf.constant(self._init_dataset.state_std)
        a_mean, a_std = tf.constant(self._init_dataset.action_mean), tf.constant(self._init_dataset.action_std)

        norm_state = utils.normalize(state, s_mean, s_std)
        norm_action = utils.normalize(action, a_mean, a_std)
        
        sa = tf.concat([norm_state, norm_action], 1)

        delta_pred = utils.build_mlp(sa,
                self._state_dim,
                'dynamics',
                n_layers=self._nn_layers,
                reuse=reuse)

        delta_pred = utils.unnormalize(delta_pred, self._init_dataset.delta_state_mean, self._init_dataset.delta_state_std)
        next_state_pred = state + delta_pred

        return next_state_pred

    def _setup_training(self, state_ph, next_state_ph, next_state_pred):
        """
            Takes as input the current state, next state, and predicted next state, and returns
            the loss and optimizer for training the dynamics model

            returns:
                loss: Scalar loss tensor
                optimizer: Operation used to perform gradient descent

            implementation details (in order):
                (a) Compute both the actual state difference and the predicted state difference
                (b) Normalize both of these state differences by using the statistics of self._init_dataset and
                    the utils.normalize function
                (c) The loss function is the mean-squared-error between the normalized state difference and
                    normalized predicted state difference
                (d) Create the optimizer by minimizing the loss using the Adam optimizer with self._learning_rate

        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        delta = state_ph - next_state_ph
        delta_pred = state_ph - next_state_pred

        d_mean, d_std = self._init_dataset.delta_state_mean, self._init_dataset.delta_state_std

        delta = utils.normalize(delta, d_mean, d_std)
        delta_pred = utils.normalize(delta_pred, d_mean, d_std)

        loss = tf.losses.mean_squared_error(delta, delta_pred)
        optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)

        return loss, optimizer

    def _setup_action_selection(self, state_ph):
        """
            Computes the best action from the current state by using randomly sampled action sequences
            to predict future states, evaluating these predictions according to a cost function,
            selecting the action sequence with the lowest cost, and returning the first action in that sequence

            returns:
                best_action: the action that minimizes the cost function (tensor with shape [self._action_dim])

            implementation details (in order):
                (a) We will assume state_ph has a batch size of 1 whenever action selection is performed
                (b) Randomly sample uniformly self._num_random_action_selection number of action sequences,
                    each of length self._horizon
                (c) Starting from the input state, unroll each action sequence using your neural network
                    dynamics model
                (d) While unrolling the action sequences, keep track of the cost of each action sequence
                    using self._cost_fn
                (e) Find the action sequence with the lowest cost, and return the first action in that sequence

            Hints:
                (i) self._cost_fn takes three arguments: states, actions, and next states. These arguments are
                    2-dimensional tensors, where the 1st dimension is the batch size and the 2nd dimension is the
                    state or action size
                (ii) You should call self._dynamics_func and self._cost_fn a total of self._horizon times
                (iii) Use tf.random_uniform(...) to generate the random action sequences

        """
        ### PROBLEM 2
        ran_sample = tf.random_uniform((self._horizon, self._num_random_action_selection, self._action_dim), minval=self._action_space_low,
                                        maxval=self._action_space_high, dtype=tf.float64)

        cost_traj = tf.zeros([self._num_random_action_selection], dtype=tf.float32)

        state_batch = tf.ones([self._num_random_action_selection, 1], dtype=tf.float64) * state_ph[0]
        
        for index in range(self._horizon):
            act_batch = ran_sample[index]
            next_state_batch = self._dynamics_func(state_batch, act_batch, True)
            cost_traj += self._cost_fn(tf.cast(state_batch, dtype=tf.float32), tf.cast(act_batch, dtype=tf.float32), tf.cast(next_state_batch, dtype=tf.float32))
            state_batch = next_state_batch

        min_cost_seq = tf.argmin(cost_traj, axis=0)     
        best_action = ran_sample[0][min_cost_seq]
        return best_action
    
    def _setup_graph(self):
        """
        Sets up the tensorflow computation graph for training, prediction, and action selection

        The variables returned will be set as class attributes (see __init__)
        """
        sess = tf.Session()

        ### PROBLEM 1
        ### YOUR CODE HERE
        state_ph, action_ph, next_state_ph = self._setup_placeholders()
        next_state_pred = self._dynamics_func(state_ph, action_ph, False)
        loss, optimizer = self._setup_training(state_ph, next_state_ph, next_state_pred)

        ### PROBLEM 2
        best_action = self._setup_action_selection(state_ph)

        sess.run(tf.global_variables_initializer())

        return sess, state_ph, action_ph, next_state_ph, \
                next_state_pred, loss, optimizer, best_action

    def train_step(self, states, actions, next_states):
        """
        Performs one step of gradient descent

        returns:
            loss: the loss from performing gradient descent
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        loss, _ = self._sess.run([self._loss, self._optimizer], feed_dict={
            self._state_ph: states,
            self._action_ph: actions,
            self._next_state_ph: next_states
        })
        return loss

    def predict(self, state, action):
        """
        Predicts the next state given the current state and action

        returns:
            next_state_pred: predicted next state

        implementation detils:
            (i) The state and action arguments are 1-dimensional vectors (NO batch dimension)
        """
        assert np.shape(state) == (self._state_dim,)
        assert np.shape(action) == (self._action_dim,)

        ### PROBLEM 1
        ### YOUR CODE HEREs
        next_state_pred = self._sess.run(self._next_state_pred, feed_dict={
            self._state_ph: [state],
            self._action_ph: [action]
        })[0]
        
        assert np.shape(next_state_pred) == (self._state_dim,)
        return next_state_pred

    def get_action(self, state):
        """
        Computes the action that minimizes the cost function given the current state

        returns:
            best_action: the best action
        """
        assert np.shape(state) == (self._state_dim,)

        ### PROBLEM 2
        ### YOUR CODE HERE
        best_action = self._sess.run(self._best_action, feed_dict={
            self._state_ph: [state]
        })

        assert np.shape(best_action) == (self._action_dim,)
        return best_action
