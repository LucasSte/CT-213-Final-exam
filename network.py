import random

import tensorflow as tf
import numpy as np
from collections import deque


class AgentDoubleDQN:

    def __init__(self, state_size, action_size, batch_size):
        self.epsilon = 1.0
        self.epsilon_min = 1.0
        self.epsilon_decay = 0.99
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = 0.99
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=4098)
        self.q_net = self.make_model()
        self.t_net = self.make_model()

    def update_double_dqn(self):
        for train_grad, pred_grad in zip(self.q_net.trainable_variables,
                                         self.t_net.trainable_variables):
            pred_grad.assign(train_grad)

    def update_q_value(self, rewards, current_q_list, next_q_list, actions, done):
        current_q_list = current_q_list.numpy()
        next_max_qs = np.max(next_q_list, axis=1)
        new_qs = rewards + (np.ones(done.shape) - done) * self.gamma * next_max_qs
        for i in range(len(current_q_list)):
            current_q_list[i, actions[i]] = new_qs[i]
        return current_q_list

    def get_action(self, state):
        if np.random.random() > self.epsilon:
            _action = self.get_prediction_double_dqn(np.expand_dims(state, axis=0))
            action = np.argmax(_action)
        else:
            action = np.random.randint(0, self.action_size)
        return action

    def get_greedy_action(self, state):
        _action = self.q_net.predict(np.expand_dims(state, axis=0))
        return np.argmax(_action)

    def get_prediction_double_dqn(self, states):
        states = np.reshape(states, newshape=(states.shape[0], self.state_size))
        prediction = self.t_net(states)
        return prediction

    def predict(self, states):
        states = np.reshape(states, newshape=(states.shape[0], self.state_size))
        prediction = self.q_net(states)
        return prediction

    @tf.function
    def train_step(self, states, actions):
        with tf.GradientTape() as tape:
            predictions = self.q_net(states)
            loss = tf.keras.losses.mean_squared_error(actions, predictions)
        gradients = tape.gradient(loss, self.q_net.trainable_variables)
        gradients = [tf.clip_by_norm(gradient, 10) for gradient in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.q_net.trainable_variables))

    def make_model(self):

        inp = tf.keras.layers.Input((self.state_size, ))
        x = tf.keras.layers.Dense(128, activation='relu')(inp)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(self.action_size, activation='linear')(x)
        model = tf.keras.Model(inputs=inp, outputs=x)
        model.summary()
        return model

    def get_batch(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        current_nodes, actions, next_nodes, rewards, done = list(zip(*batch))
        return [np.stack(current_nodes), np.array(actions), np.stack(next_nodes), np.array(rewards), np.array(done)]

    def train(self):
        current_nodes, actions, next_nodes, rewards, done = self.get_batch()
        #print(current_nodes.shape, actions.shape, rewards.shape, next_nodes.shape)
        current_action_qs = self.predict(current_nodes)
        next_action_qs = self.get_prediction_double_dqn(next_nodes)
        current_action_qs = self.update_q_value(rewards, current_action_qs, next_action_qs, actions, done)
        current_nodes = np.reshape(current_nodes, newshape=(self.batch_size, self.state_size))

        self.train_step(current_nodes, current_action_qs)

    def add_memory(self, state, action, next_state, reward, done):
        self.replay_buffer.append([state, action, next_state, reward, 1 if done else 0])

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def load(self, agent, game):
        self.q_net.load_weights(agent + game)

    def save(self, agent, game):
        self.q_net.save_weights(agent + game)
