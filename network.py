import random

import tensorflow as tf
import numpy as np
from collections import deque


class DeepQnetwork:

    def __init__(self, state_size, action_size, name, batch_size):
        self.epsilon, self.epsilon_min = 1.0, 0.01
        self.epsilon_decay = 0.99
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.name = name
        self.gamma = 0.99
        self.counter = 0
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.previous_memory = deque(maxlen=4098)
        self.train_network = self.build_network()
        self.predict_network = self.build_network()

    def update_prediction_network(self):
        for train_grad, pred_grad in zip(self.train_network.trainable_variables,
                                         self.predict_network.trainable_variables):
            pred_grad.assign(train_grad)
        self.train_network.save_weights(self.name)
        #print("leveling up")

    def update_q_value(self, rewards, current_q_list, next_q_list, actions, done):
        current_q_list = current_q_list.numpy()
        next_max_qs = np.max(next_q_list, axis=1)
        new_qs = rewards + (np.ones(done.shape) - done) * self.gamma * next_max_qs
        for i in range(len(current_q_list)):
            current_q_list[i, actions[i]] = new_qs[i]
        return current_q_list

    def loss(self, ground_truth, prediction):
        loss = tf.keras.losses.mean_squared_error(ground_truth, prediction)
        return loss

    def get_action(self, state):
        if np.random.random() > self.epsilon:
            _action = self.get_prediction(np.expand_dims(state, axis=0))
            action = np.argmax(_action)
            #print(_action, action)
        else:
            action = np.random.randint(0, self.action_size)
        return action

    def get_greedy_action(self, state):
        _action = self.train_network.predict(np.expand_dims(state, axis=0))
        return np.argmax(_action)

    def get_prediction(self, states):
        # print(np.shape(states))
        states = np.reshape(states, newshape=(states.shape[0], self.state_size))
        prediction = self.predict_network(states)
        return prediction

    def predict(self, states):
        states = np.reshape(states, newshape=(states.shape[0], self.state_size))
        prediction = self.train_network(states)
        return prediction

    @tf.function
    def train_step(self, states, actions):
        with tf.GradientTape() as tape:
            predictions = self.train_network(states)
            loss = self.loss(actions, predictions)
        gradients = tape.gradient(loss, self.train_network.trainable_variables)
        gradients = [tf.clip_by_norm(gradient, 10) for gradient in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.train_network.trainable_variables))
        return loss

    def build_network(self):

        inp = tf.keras.layers.Input((self.state_size, ))
        x = tf.keras.layers.Dense(128, activation='relu')(inp)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(self.action_size, activation='linear')(x)
        model = tf.keras.Model(inputs=inp, outputs=x)
        model.summary()
        return model

    def get_batch(self):
        batch = random.sample(self.previous_memory, self.batch_size)
        current_nodes, actions, next_nodes, rewards, done = list(zip(*batch))
        return [np.stack(current_nodes), np.array(actions), np.stack(next_nodes), np.array(rewards), np.array(done)]

    def train(self):
        self.counter += 1
        current_nodes, actions, next_nodes, rewards, done = self.get_batch()
        #print(current_nodes.shape, actions.shape, rewards.shape, next_nodes.shape)
        current_action_qs = self.predict(current_nodes)
        next_action_qs = self.get_prediction(next_nodes)
        current_action_qs = self.update_q_value(rewards, current_action_qs, next_action_qs, actions, done)
        current_nodes = np.reshape(current_nodes, newshape=(self.batch_size, self.state_size))

        loss = self.train_step(current_nodes, current_action_qs)
        #print(f'loss: {loss}')

    def add_memory(self, state, action, next_state, reward, done):
        self.previous_memory.append([state, action, next_state, reward, 1 if done else 0])

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def load(self, agent, game):
        self.train_network.load_weights(agent + game)
        print("Carregado!!")

    def save(self, agent, game):
        self.train_network.save_weights(agent + game)
