import gridworld
import numpy as np

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


class TDBuffer:
    def __init__(self, num_states, depth):
        self.lb = []
        self.depth = depth + 1
        self.v = np.zeros(num_states)

    def add(self, state, action, reward, done, state_prime):
        self.lb.append(Step(state, action, reward, done))
        if len(self.lb) == self.depth:
            self.update_value(state_prime)
        if done:
            while len(self.lb) > 0:
                self.update_value(state_prime)

    def update_value(self, state_prime):
        updating = self.lb[0].state
        V = self.v[state_prime]
        for step in reversed(self.lb):
            V = V * discount
            V += step.reward
        self.v[updating] += decay * (V - self.v[updating])
        self.lb.pop(0)

    def __len__(self):
        return len(self.lb)


env = gridworld.GridworldEnv()

action_map = [0, 1, 2, 3]

policy = UniformRandomPolicy(env.nS, action_map)
buffer = TDBuffer(env.nS, 3)

for episode in range(1000):
    observation_0 = env.reset()
    done = False
    while not done:
        action = policy.sample(observation_0)
        observation_1, reward, done, info = env.step(action)
        buffer.add(observation_0, action, reward, done, observation_1)
        observation_0 = observation_1

print(buffer.v.reshape(4, 4))
