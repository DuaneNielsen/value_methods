import gridworld
import numpy as np
from collections import namedtuple
import warnings
from statistics import mean

discount = 0.99
decay = 0.5


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


class Greedy(Policy):
    def __init__(self, q, action_map):
        policy_max = np.argmax(q, axis=1)
        super().__init__(q.shape[0], action_map)
        self.policy = np.zeros_like(self.policy)
        np.put_along_axis(self.policy, np.expand_dims(policy_max, axis=1), 1.0, axis=1)


class EpsilonGreedy(Policy):
    def __init__(self, q, action_map, epsilon=0.2):
        policy_max = np.argmax(q, axis=1)
        super().__init__(q.shape[0], action_map)
        self.policy = np.ones_like(self.policy) * epsilon / len(action_map)
        np.put_along_axis(self.policy, np.expand_dims(policy_max, axis=1), 1.0 - epsilon + (epsilon / len(action_map)),
                          axis=1)


env = gridworld.GridworldEnv()

action_map = [0, 1, 2, 3]

q = np.ones((env.nS, env.nA))

for episode in range(20000):
    policy = EpsilonGreedy(q, action_map)
    t_policy = Greedy(q, action_map)
    observation_0 = env.reset()

    done = False
    step = 0
    while not done and step < 50:
        action_0 = policy.sample(observation_0)
        observation_1, reward, done, info = env.step(action_0)
        action_1 = t_policy.greedy(observation_1)
        q[observation_0, action_0] += decay * (reward + discount * q[observation_1, action_1] - q[observation_0, action_0])
        observation_0 = observation_1
        step += 1

    p = np.argmax(q, axis=1)
    actions = [u'\u2191', u'\u2192', u'\u2193', u'\u2190']
    a = []
    for s in p.tolist():
        a.append(actions[s])

    #print(np.array(a).reshape(4, 4))
    #print(np.max(q, axis=1).reshape(4, 4))