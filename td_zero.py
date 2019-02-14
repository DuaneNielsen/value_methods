import gridworld
import numpy as np
from collections import namedtuple
import warnings
from statistics import mean

discount = 0.99
decay = 0.05


class Step:
    def __init__(self, state, action, reward, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.Return = None

    def __repr__(self):
        return str((self.state, self.action, self.reward, self.done, self.Return))


class Policy:
    def __init__(self, num_states, action_map):
        self.policy = np.ones((num_states, len(action_map)))
        self.action_map = action_map

    def sample(self, state):
        p = self.policy[state, :]
        return np.random.choice(self.action_map, p=p)

    def greedy(self, state):
        p = self.policy[state, :]
        return self.action_map[np.argmax(p)]


class UniformRandomPolicy(Policy):
    """ all actions equally likely"""
    def __init__(self, num_states, action_map):
        super().__init__(num_states, action_map)
        self.policy = self.policy / len(action_map)


env = gridworld.GridworldEnv()

action_map = [0, 1, 2, 3]

policy = UniformRandomPolicy(env.nS, action_map)

v = np.zeros(env.nS)
for episode in range(1000):
    observation_0 = env.reset()
    done = False
    while not done:
        action = policy.sample(observation_0)
        observation_1, reward, done, info = env.step(action)
        v1 = (discount * v[observation_1])
        v[observation_0] += decay * (reward + v1 - v[observation_0])
        observation_0 = observation_1

print(v.reshape(4, 4))
