import gym
import gym_minigrid
import random
import time
from gym_minigrid.wrappers import *
import numpy as np

discount = 0.99

env = gym.make('MiniGrid-Empty-6x6-v0')
obs = env.reset()


def prepro(env, obs):
    x = env.agent_pos[0]
    y = env.agent_pos[1]
    f = obs[x, y, 1]
    return x, y, f


class Policy:
    def __init__(self, x, y, f, num_actions):
        self.probs = np.ones((x, y, 4, num_actions)) / num_actions

    def actions_p(self, x, y, f):
        actions_p = self.probs[x, y, f]
        return actions_p

    def action_p(self, x, y, f, a):
        return self.probs[x, y, f, a]


policy = Policy(env.width, env.height, 4, env.action_space.n)

v = np.zeros((env.width, env.height, 4))
stm = np.ones((env.width, env.height, 4, env.width, env.height, 4))

v_old = np.copy(v)
delta = np.ones((env.width, env.height, 4)) * 0.00001

while True:
    for x in range(env.width):
        for y in range(env.height):
            for f in range(4):
                vs = 0
                for a in range(env.action_space.n):
                    env.agent_pos = (x, y)
                    env.agent_dir = f
                    obs, reward, done, info = env.step(a)
                    print(reward)
                    xpr, ypr = env.agent_pos
                    fpr = env.agent_dir
                    #print(xpr, ypr, fpr)
                    p_action = policy.action_p(x, y, f, a)
                    #print(x, y, f, xpr, ypr, fpr)
                    stp = stm[x, y, f, xpr, ypr, fpr]
                    vpr = v_old[xpr, ypr, fpr]
                    vs += p_action * (reward + discount * stp * vpr)
                v[x, y, f] = vs

    if np.all(np.absolute(v - v_old) < delta):
        break
    v_old = np.copy(v)
    #print(v)