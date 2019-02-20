import gridworld
import numpy as np
from collections import namedtuple
import warnings

discount = 0.99
epsilon = 0.01

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

    def action(self, state):
        p = self.policy[state, :]
        return np.random.choice(self.action_map, p=p)


class RandomPolicy(Policy):
    def __init__(self, num_states, action_map):
        super().__init__(num_states, action_map)
        self.policy = self.policy / len(action_map)


class Greedy(Policy):
    def __init__(self, q):
        policy_max = np.argmax(q, axis=1)
        super().__init__(env.nS, action_map)
        self.policy = np.zeros(env.nS, env.nA)
        np.put_along_axis(self.policy, np.expand_dims(policy_max, axis=1), 1.0, axis=1)


class EpsilonGreedy(Policy):
    def __init__(self, q, epsilon):
        policy_max = np.argmax(q, axis=1)
        super().__init__(env.nS, action_map)
        self.policy = np.ones_like(new_policy.policy) * epsilon / len(action_map)
        np.put_along_axis(self.policy, np.expand_dims(policy_max, axis=1), 1.0 - epsilon + (epsilon / len(action_map)), axis=1)


env = gridworld.GridworldEnv()

action_map = [0, 1, 2, 3]

buffer = ReplayBuffer()

episodes_first = None
q = np.zeros((env.nS, env.nA))
c = np.zeros((env.nS, env.nA))
t_policy = Greedy(q)

while True:
    b_policy = EpsilonGreedy(q, epsilon)

    for episode in range(1):
        observation_0 = env.reset()
        action = b_policy.action(observation_0)
        done = False
        while not done:
            observation_1, reward, done, info = env.step(action)
            buffer.step(observation_0, action, reward, done)
            observation_0 = observation_1
            action = b_policy.action(observation_0)

    g = 0
    w = 1

    for episode in buffer.rollout:
        for step in reversed(episode):
            g = discount * g + step.reward
            c[step.state, step.action] += w
            q[step.state, step.action] += (w / c[step.state, step.action]) * (g - q[step.state, step.action])
            t_policy = Greedy(q)
            # this is why this does not work
            if step.action != t_policy.action(step.state):
                break
            w = w / b_policy.policy[step.state, step.action]