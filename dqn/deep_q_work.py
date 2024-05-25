import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class DeepQNetwork:

    def __init__(self, actions, n_features, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, replace_target_iter=200,
                 memory_size=3000, batch_size=32, e_greedy_increment=None, output_graph=False, double_q=True,
                 sess=None):
        tf.compat.v1.disable_eager_execution()
        self.n_actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.n_features = n_features
        self.double_q = double_q
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.learn_step_counter = 0

        self._build_net()

        t_params = tf.compat.v1.get_collection('target_net_params')
        e_params = tf.compat.v1.get_collection('eval_net_params')

        self.replace_target_op = [tf.compat.v1.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.compat.v1.Session()
            self.sess.run(tf.compat.v1.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.create_file_writer("logs/", self.sess.grah)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.cost_his = []

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action_values = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_values)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features],
                self.s: batch_memory[:, -self.n_features]
            }
        )

        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_next[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss], feed_dict={self.s: batch_memory[:, self.n_features],
                                                                             self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def _build_net(self):
        self.s = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.n_features], name='s')
        self.q_target = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, len(self.n_actions)], name='Q_target')
        with tf.compat.v1.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.compat.v1.random_normal_initializer(0, 0.3), tf.compat.v1.constant_initializer(0.1)

            # first layer
            with tf.compat.v1.variable_scope('l1'):
                w1 = tf.compat.v1.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer,
                                               collections=c_names)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer
            with tf.compat.v1.variable_scope('l2'):
                w2 = tf.compat.v1.get_variable('w2', [n_l1, len(self.n_actions)], initializer=w_initializer,
                                               collections=c_names)
                b2 = tf.compat.v1.get_variable('b2', [1, len(self.n_actions)], initializer=b_initializer,
                                               collections=c_names)
                self.q_eval = tf.nn.relu(tf.matmul(l1, w2) + b2)

        with tf.compat.v1.name_scope('loss'):
            self.loss = tf.reduce_sum(tf.compat.v1.squared_difference(self.q_target, self.q_eval))
        with tf.compat.v1.name_scope('train'):
            self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        self.s_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.n_features], name='s_')
        with tf.compat.v1.variable_scope('target_net'):
            c_names = ['target_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]

            with tf.compat.v1.variable_scope('l1'):
                w1 = tf.compat.v1.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer,
                                               collections=c_names)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.compat.v1.variable_scope('l2'):
                w2 = tf.compat.v1.get_variable('w2', [n_l1, len(self.n_actions)], initializer=w_initializer,
                                               collections=c_names)
                b2 = tf.compat.v1.get_variable('b2', [1, len(self.n_actions)], initializer=b_initializer,
                                               collections=c_names)
                self.q_next = tf.nn.relu(tf.matmul(l1, w2) + b2)

    def plot_cost(self):
        plt.plot(np.arrange(len(self.cost_his)), self.cost_his)
        plt.show()