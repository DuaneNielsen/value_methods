import numpy as np
import gym
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import gridworld

discount = 0.95
decay = 0.5
step_size = 0.01

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





env = gym.make('CartPole-v0')
#env = gridworld.GridworldEnv()
env.nA = env.action_space.n
env.nS = 4
action_map = range(env.action_space.n)

q = nn.Linear(env.nS, env.nA)
#q.load_state_dict(torch.load('mountain_car.wgt'))
optimizor = optim.SGD(q.parameters(), lr=0.01)

def greedyestimate(obs):
    est = q(obs)
    act = torch.argmax(est).item()
    return act

def eps_greedy(obs, epsilon=0.1):
    est = q(obs)
    best_act = torch.argmax(est).item()
    ep = epsilon/est.size(0)
    pr = np.ones_like(est.data.numpy())*ep
    pr[best_act] = 1 - epsilon + ep
    return np.random.choice(range(env.nA), p=pr)


def grid(observation):
    grid = np.zeros(16)
    grid[observation] = 1.0
    return grid


def prepro(obs):
    return torch.from_numpy(obs).float()

for episode in range(1000):
    #policy = EpsilonGreedy(q, action_map)
    #t_policy = Greedy(q, action_map)
    observation_0 = env.reset()
    temp = 1 - (episode / 1000)
    done = False
    step = 0
    while not done:
        optimizor.zero_grad()
        action_0 = eps_greedy(prepro(observation_0), temp)
        observation_1, reward, done, info = env.step(action_0)
        env.render()
        action_1 = greedyestimate(prepro(observation_1))
        #target = decay * (reward + discount * q(observation_1)[action_1] - q(observation_0)[action_0]) + q(observation_0)[action_0]
        target = reward + discount * q(prepro(observation_1))[action_1]
        if np.argmax(observation_1) == 15:
            print(target)
        target = target.detach()
        loss = (target - q(prepro(observation_0))[action_0]) ** 2
        loss.backward()
        optimizor.step()
        observation_0 = observation_1
        step += 1

    print(episode)
    if episode % 1 == 0:
        torch.save(q.state_dict(), 'mountain_car.wgt')

    def print_q(q_func):
        q_t = np.zeros((16, 4))
        for s in range(16):
            q_t[s] = q_func(torch.tensor(grid(s), dtype=torch.float)).detach().numpy()
        p = np.argmax(q_t, axis=1)
        actions = [u'\u2191', u'\u2192', u'\u2193', u'\u2190']
        a = []
        for s in p.tolist():
            a.append(actions[s])

        print(np.array(a).reshape(4, 4))
        print(np.max(q_t, axis=1).reshape(4, 4))

    #print_q(q)
