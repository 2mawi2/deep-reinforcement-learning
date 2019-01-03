import gym
import random
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot


class Agent:
    def __init__(self, state_size, action_size):
        self.weight_backup = "weights.h5"
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        # NN topology = 4 - 24 - 24 - 2
        # state size = 4

        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['acc'])

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.epsilon = 0  # greedy exploitation
        return model

    def save_model(self):
        self.model.save(self.weight_backup)

    def get_best_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def save(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return

        # we take a random sample batch from the last stored states
        sample_batch = random.sample(self.memory, sample_batch_size)

        msas = np.array([])
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                # td_target is the value which we want the neural network to output
                # therefore we fit the neural net to td_target
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)

            Q = target_f[0][action]
            alpha = 1  # we set alpha to 1 -> deterministic environment
            target_f[0][action] = Q + alpha * (target - Q)

            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            if len(history.history) > 0:
                msas = np.append(msas, history.history['loss'])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return msas.mean()


class Environment:
    def __init__(self):
        gym_environment = 'CartPole-v1'
        self.sample_batch_size = 32
        self.episodes = 10000
        self.env = gym.make(gym_environment)

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = Agent(self.state_size, self.action_size)

        print(f"Environment: {gym_environment}")
        print(f"Number of states: {self.state_size}")
        print(f"Number of actions: {self.action_size}")

    def run(self):

        running_msa = []
        running_reward = []
        running_reward_avg = []
        running_msa_avg = []

        try:
            for index_episode in range(self.episodes):
                evaluate_mode = self.episodes - index_episode < 100

                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

                done = False
                index = 0
                while not done:
                    # self.env.render()

                    action = self.agent.get_best_action(state)

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])

                    if not evaluate_mode:
                        self.agent.save(state, action, reward, next_state, done)
                    state = next_state
                    index += 1

                if not evaluate_mode:
                    msa = self.agent.learn(self.sample_batch_size)
                    running_msa.append(msa)
                    # if len(running_msa) > 50:
                    #    running_msa_avg.append(np.mean(running_msa[-50:]))
                else:
                    msa = 0
                    running_msa.append(0)
                print(f'Episode {index_episode} Score: {index + 1} Epsilon: {self.agent.epsilon} MSA: {msa}')

                running_reward.append(index + 1)
                running_reward_avg.append(np.mean(running_reward[-50:]))

                if evaluate_mode:
                    self.agent.epsilon = 0

            self.agent.save_model()
        finally:
            pyplot.plot(running_reward_avg)
            pyplot.show()
            pyplot.plot(running_msa_avg)
            pyplot.show()


if __name__ == "__main__":
    environment = Environment()
    environment.run()
