import gridworld
import numpy as np
from collections import namedtuple
import warnings
from statistics import mean

discount = 0.99


class Step:
    def __init__(self, state, action, reward, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.Return = None

    def __repr__(self):
        return str((self.state, self.action, self.reward, self.done, self.Return))


class ReplayBuffer:
    def __init__(self):
        self.rollout = []
        self.episode = []

    def step(self, state, action, reward, done):
        self.episode.append(Step(state, action, reward, done))
        if done:
            self.rollout.append(self.episode)
            self.episode = []

    def calc_returns(self):
        for episode in self.rollout:
            Return = 0
            for step in reversed(episode):
                Return = Return * discount
                Return += step.reward
                step.Return = Return


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
buffer = ReplayBuffer()

for episode in range(10000):
    observation_0 = env.reset()
    action = policy.sample(observation_0)
    done = False
    while not done:
        observation_1, reward, done, info = env.step(action)
        buffer.step(observation_0, action, reward, done)
        observation_0 = observation_1
        action = policy.sample(observation_0)

buffer.calc_returns()

v = np.zeros(env.nS)
c = np.zeros(env.nS)
for episode in buffer.rollout:
    for step in episode:
        c[step.state] += 1
        v[step.state] += (step.Return - v[step.state]) / c[step.state]

print(v.reshape(4, 4))