import gym
import gridworld
import random
import time
from gym_minigrid.wrappers import *
import numpy as np

discount = 0.99

env = gridworld.GridworldEnv()
obs = env.reset()
pass

class Policy:
    def __init__(self, s_size, num_actions):
        self.probs = np.ones((s_size, num_actions)) / num_actions

    def action_p(self, s, a):
        return self.probs[s, a]


policy = Policy(env.nS, env.nA)

v = np.zeros(env.nS)
stm = np.ones((env.nS, env.nS))

v_old = np.copy(v)
delta = np.ones(env.nS) * 0.00001

while True:
    for s in range(env.nS):
        vs = 0
        for a in range(env.nA):
            state_transition_prob, s_next, reward, done = env.P[s][a][0]
            p_action = policy.action_p(s, a)
            vpr = v_old[s_next]
            vs += p_action * (reward + discount * state_transition_prob * vpr)
        v[s] = vs
    if np.all(np.absolute(v - v_old) < delta):
        break
    v_old = np.copy(v)
    print(v.reshape((4, 4)))