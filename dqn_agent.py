from collections import deque

import matplotlib.pyplot
from tensorflow.keras import models, layers, losses, optimizers
import cv2
import numpy as np
import random


class DQNAgent:
    model = None

    def __init__(self, state_size, action_size, gamma=0.95, epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.98,
                 learning_rate=0.001, buffer_size=4098):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

    def act(self, state):
        raise NotImplemented('Do not use the superclass')

    def replay(self, batch_size):
        """
        Learns from memorized experience.
        """
        minibatch = random.sample(self.replay_buffer, batch_size)
        states = [i[0] for i in minibatch]
        targets = self.model.predict(np.array(states))
        next_states = [i[3] for i in minibatch]
        next_prediction = self.model.predict(np.array(next_states))

        for idx, item in enumerate(minibatch):
            state, action, reward, next_state, done = item
            target = targets[idx]
            if not done:
                next_state = np.reshape(next_state, [1, self.state_size])
                target[0][action] = reward + self.gamma * np.max(next_prediction[idx][0])
            else:
                target[0][action] = reward
        history = self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        return loss
        # raise NotImplemented('Do I need to specify this function for each sub-class?')

    def load(self, agent, game):
        self.model.load_weights(agent + game)

    def save(self, agent, game):
        self.model.save_weights(agent + game)

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def append_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward,
                                   next_state, done))

    def prepare_state(self, state):
        raise NotImplemented("Do not use the superclass.")


class RAMAgent(DQNAgent):

    def __init__(self, state_size, action_size, gamma=0.95, epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.98,
                 learning_rate=0.001, buffer_size=4098):
        DQNAgent.__init__(self, state_size, action_size, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate,
                          buffer_size)
        self.model = self.make_model()
        self.agentType = "RAM"

    def make_model(self):
        model = models.Sequential()
        model.add(layers.Dense(128, activation='relu', input_shape=(self.state_size,)))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss=losses.mse, optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        model.summary()
        return model

    def prepare_state(self, state):
        return np.reshape(state, (1, self.state_size))

    def act(self, input):

        input = np.reshape(input, [1, self.state_size])
        q = self.model.predict(input)
        prob = np.random.uniform(0, 1)
        if prob < self.epsilon:
            sp = np.shape(q)
            return np.random.randint(0, sp[1])
        else:
            return np.argmax(q[0, :])




class ImageAgent(DQNAgent):
    #new_image_size = (80, 105)
    new_image_size = (160, 210)
    color_space = 255  # Todo: 255 or 127?

    def __init__(self, state_size, action_size, gamma=0.95, epsilon=0.5,
                 epsilon_min=0.01, epsilon_decay=0.98, learning_rate=0.001,
                 buffer_size=4098):
        DQNAgent.__init__(self, state_size, action_size, gamma, epsilon, epsilon_min,
                          epsilon_decay, learning_rate, buffer_size)
        self.model = self.make_model()
        self.agentType = "Image"

    def make_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (8, 8), strides=(2, 2), activation='relu',
                                input_shape=(self.new_image_size[1], self.new_image_size[0], 3)))
        model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss=losses.mse, optimizer=optimizers.Adam(lr=self.learning_rate))
        model.summary()

        return model

    def prepare_state(self, state):
        #image = cv2.resize(image, self.new_image_size, interpolation=cv2.INTER_AREA)
        image = np.array(state)
        image = image.astype(float)
        image = image / self.color_space
        image = image.reshape((1, self.new_image_size[1], self.new_image_size[0], 3))
        return image

    def act(self, input):

        img = self.prepare_state(input)
        q = self.model.predict(img)
        prob = np.random.uniform(0, 1)
        if prob < self.epsilon:
            sp = np.shape(q)
            return np.random.randint(0, sp[1])
        else:
           return np.argmax(q[0, :])
