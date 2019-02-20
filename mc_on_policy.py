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


class RandomPolicy(Policy):
    def __init__(self, num_states, action_map):
        super().__init__(num_states, action_map)
        self.policy = self.policy / len(action_map)

    def action(self, state):
        p = self.policy[state, :]
        return np.random.choice(self.action_map, p=p)

env = gridworld.GridworldEnv()

action_map = [0, 1, 2, 3]

policy = RandomPolicy(env.nS, action_map)
buffer = ReplayBuffer()

episodes_first = None

while True:
    for episode in range(1):
        observation_0 = env.reset()
        action = policy.action(observation_0)
        done = False
        while not done:
            observation_1, reward, done, info = env.step(action)
            buffer.step(observation_0, action, reward, done)
            observation_0 = observation_1
            action = policy.action(observation_0)

    buffer.calc_returns()

    for episode in buffer.rollout:
        first = np.zeros((env.nS, env.nA))
        first[:] = np.nan
        episodes = []
        for step in episode:
            if np.isnan(first[step.state, step.action]):
                first[step.state, step.action] = step.Return
        episodes.append(first)
    episodes_stack = np.stack(episodes, axis=0)
    if episodes_first is not None:
        episodes_first = np.concatenate((episodes_first, episodes_stack), axis=0)
    else:
        episodes_first = episodes_stack

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        q = np.nanmean(episodes_first, axis=0)

    v = np.diag(np.dot(q, policy.policy.T))
    print(v.reshape((4, 4)))

    # make an epsilon greedy policy
    new_policy = Policy(env.nS, action_map)
    a_star = np.argmax(q, axis=1)
    a_star = np.expand_dims(a_star, axis=1)
    policy.policy = np.ones_like(new_policy.policy) * epsilon / len(action_map)
    np.put_along_axis(policy.policy, a_star, 1 - epsilon + (epsilon / len(action_map)), axis=1)



#new_policy.policy[:, a_star] += 1 - epsilon


#print(new_policy.policy)