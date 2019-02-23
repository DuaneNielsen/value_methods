import torch
from torch import nn, optim
import gym


class Config:
    def __init__(self, env, num_inputs, num_actions, prepro, discount=0.95):
        self.env = env
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.discount = discount
        self.prepro = prepro


def numpy_prepro(observation, insert_batch=False):
    s = torch.from_numpy(observation).float()
    if insert_batch:
        s = s.unsqueeze(0)
    return s


config = Config(env=gym.make('MountainCar-v0').unwrapped,
                num_inputs=2,
                num_actions=3,
                prepro=numpy_prepro)

q_f = nn.Linear(config.num_inputs, config.num_actions)
opt = optim.SGD(q_f.parameters(), lr=0.01)


def greedy(s):
    return torch.argmax(q_f(s))


def episilon_greedy(s, epsilon=0.2):
    batch_size = s.shape[0]
    e = epsilon / config.num_actions
    p = torch.ones((batch_size, config.num_actions)) * e
    q = q_f(s)
    greedy_a = torch.argmax(q)
    p[torch.arange(batch_size), greedy_a] = 1.0 - epsilon + e
    return torch.distributions.Categorical(p).sample()

episode = 0
epsilon = 0.5

while True:
    s = config.prepro(config.env.reset(), insert_batch=True)
    total_reward = 0
    done = False
    epsilon = epsilon * 0.99
    step = 0
    while not done and step < 6000:

        batch_size = s.shape[0]
        opt.zero_grad()
        action = episilon_greedy(s, epsilon)
        s_next, reward, done, info = config.env.step(action.item())
        if episode % 50 == 0 and episode != 0:
            config.env.render()
        total_reward += reward
        s_next = config.prepro(s_next, insert_batch=True)
        a_next = greedy(s)
        predicted = q_f(s)[torch.arange(batch_size), action]
        target = reward + config.discount * q_f(s_next)[torch.arange(batch_size), a_next]
        target.detach()
        loss = (target - predicted) ** 2
        loss.backward()
        opt.step()
        step += 1
        s = s_next

    episode += 1
    print(episode, total_reward, epsilon)