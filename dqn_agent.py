from collections import deque
from tensorflow.keras import models, layers, losses, optimizers
import cv2
import numpy as np

class DQNAgent:

    model = None

    def __init__(self, state_size, action_size, gamma=0.95, epsilon=0.5,
                 epsilon_min=0.01, epsilon_decay=0.98, learning_rate=0.001,
                 buffer_size=4098):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

    def act(self, input):
        raise NotImplemented('Do not use the superclass')

    def replay(self, batch_size):
        raise NotImplemented('Do I need to specify this function for each sub-class?')

    def append_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min


class RAMAgent(DQNAgent):

    def __init__(self, state_size, action_size, gamma=0.95, epsilon=0.5,
                 epsilon_min=0.01, epsilon_decay=0.98, learning_rate=0.001,
                 buffer_size=4098):
        DQNAgent.__init__(self, state_size, action_size, gamma, epsilon, epsilon_min,
                       epsilon_decay, learning_rate, buffer_size)

    def make_model(self):
        model = models.Sequential()
        model.add(layers.Dense(128, activation='relu', input_shape=(self.state_size, )))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss=losses.mse, optimizer=optimizers.Adam(lr=self.learning_rate))
        model.summary()
        self.model = model

    def act(self, input):

        q = self.model.predict(input)
        prob = np.random.uniform(0, 1)
        if prob < self.epsilon:
            sp = np.shape(q)
            return np.random.randint(0, sp[1])
        else:
            return np.argmax(q[0, :])


class ImageAgent(DQNAgent):

    new_image_size = (105, 80)
    color_space = 255 # TODO: 255 or 127?

    def __init__(self, state_size, action_size, gamma=0.95, epsilon=0.5,
                 epsilon_min=0.01, epsilon_decay=0.98, learning_rate=0.001,
                 buffer_size=4098):
        DQNAgent.__init__(self, state_size, action_size, gamma, epsilon, epsilon_min,
                       epsilon_decay, learning_rate, buffer_size)

    def make_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (8, 8), strides=(2, 2), activation='relu',
                                input_shape=(self.new_image_size[0], self.new_image_size[1], 3)))
        model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss=losses.mse, optimizer=optimizers.Adam(lr=self.learning_rate))
        model.summary()

    def prepare_image(self, image):
        image = cv2.resize(image, self.new_image_size, interpolation=cv2.INTER_AREA)
        image = np.array(image)
        image = image.astype(float)
        image = image / self.color_space
        return image

    def act(self, input):

        q = self.model.predict(self.prepare_image(input))
        prob = np.random.uniform(0, 1)
        if prob < self.epsilon:
            sp = np.shape(q)
            return np.random.randint(0, sp[1])
        else:
            return np.argmax(q[0, :])