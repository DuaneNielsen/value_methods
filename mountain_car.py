import numpy as np
import gym
import torch
from torch import nn
from torch import optim
import random
import torch.nn.functional as F
import gridworld
from torch.utils.data import Dataset, DataLoader
import functools
import operator

discount = 0.99
decay = 0.5
step_size = 0.0001


class Step:
    def __init__(self, state, action, reward, done, state_prime):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.state_prime = state_prime

    def __repr__(self):
        return str((self.state, self.action, self.reward, self.done, self.state_prime))


class ReplayBuffer(Dataset):
    def __init__(self, transform):
        self.rollout = []
        self.episode = []
        self.flat = []
        self.finalized = False
        self.t = transform

    def step(self, state, action, reward, done, state_prime):
        self.episode.append(Step(state, action, reward, done, state_prime))
        if done:
            self.rollout.append(self.episode)
            self.episode = []

    def finalize(self):
        self.flat = functools.reduce(operator.concat, self.rollout)
        self.finalized = True

    def __getitem__(self, item):
        if not self.finalized:
            self.finalize()
        x = self.flat[item]
        return self.t(x)

    def __len__(self):
        if not self.finalized:
            self.finalize()
        return len(self.flat)


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


class EnvConfig:
    def __init__(self, env_string, input_size, prepro, transform, max_steps):
        if isinstance(env_string, str):
            self.env = gym.make(env_string)
        else:
            self.env = env_string
        self.env.nA = self.env.action_space.n
        self.env.nS = input_size
        self.env_string = env_string
        self.action_map = range(self.env.action_space.n)
        self.prepro = prepro
        self.transform = transform
        self.max_steps = max_steps


def grid(observation, insert_batch=False):
    if insert_batch:
        grid = np.zeros((1, 16))
    else:
        grid = np.zeros((observation.shape[0], 16))
    grid[:, observation] = 1.0
    return torch.from_numpy(grid).float()


def prepro(obs, insert_batch=False):
    if insert_batch:
        return torch.from_numpy(obs).float().unsqueeze(0)
    else:
        return torch.from_numpy(obs).float()


def as_tensor(step):
    return torch.from_numpy(step.state), step.action, step.reward, step.done, torch.from_numpy(step.state_prime)


def grid_to_tensor(step):
    return torch.tensor(step.state, dtype=torch.int), \
           torch.tensor(step.action, dtype=torch.int64),\
           step.reward, \
           torch.tensor(step.state_prime, dtype=torch.int)


#env = gym.make('CartPole-v0')
#env =

#config = EnvConfig('CartPole-v0', 4, prepro)
config = EnvConfig(gridworld.GridworldEnv(), 16, grid, grid_to_tensor, max_steps=20)
env = config.env


q = nn.Linear(env.nS, env.nA)
#q.load_state_dict(torch.load('mountain_car.wgt'))
optimizor = optim.SGD(q.parameters(), lr=0.001)


def greedyestimate(obs):
    est = q(obs)
    act = torch.argmax(est, dim=1)
    return act


def eps_greedy(obs, epsilon=0.1):
    est = q(obs)
    best_act = torch.argmax(est, dim=1).item()
    ep = epsilon/est.size(1)
    pr = np.ones(est.size(1))*ep
    pr[best_act] = 1 - epsilon + ep
    return np.random.choice(range(env.nA), size=obs.size(0), p=pr)




for epoch in range(1000):
    print(epoch)
    replay = ReplayBuffer(config.transform)
    for episode in range(100):
        observation_0 = env.reset()
        done = False
        step = 0
        while not done and step < config.max_steps:
            action_0 = eps_greedy(config.prepro(observation_0, insert_batch=True))[0]
            observation_1, reward, done, info = env.step(action_0)
            replay.step(observation_0, action_0, reward, done, observation_1)
            if episode % 50 == 0:
                pass
                #env.render()
            observation_0 = observation_1
            step += 1

    replay = DataLoader(replay, batch_size=len(replay), shuffle=True)

    for state, action, reward, state_prime in replay:
        optimizor.zero_grad()
        reward = reward.float()
        action_1 = greedyestimate(config.prepro(state_prime.numpy()))
        target = reward + discount * q(config.prepro(state_prime.numpy()))[torch.arange(state.size(0)), action_1]
        target = target.detach()
        predicted = q(config.prepro(state.numpy()))[torch.arange(state.size(0)), action]
        loss = (target - predicted) ** 2
        loss.mean().backward()
        optimizor.step()


        if episode % 1 == 0:
            torch.save(q.state_dict(), 'mountain_car.wgt')

        def print_q(q_func):
            q_t = np.zeros((16, 4))
            for s in range(16):
                q_t[s] = q_func(torch.tensor(grid(s, insert_batch=True), dtype=torch.float)).detach().numpy()
            p = np.argmax(q_t, axis=1)
            actions = [u'\u2191', u'\u2192', u'\u2193', u'\u2190']
            a = []
            for s in p.tolist():
                a.append(actions[s])

            print(np.array(a).reshape(4, 4))
            print(np.max(q_t, axis=1).reshape(4, 4))

    print_q(q)
