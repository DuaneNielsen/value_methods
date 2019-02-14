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

policy = np.ones((env.nS, env.nA)) / env.nA
policy_old = np.zeros((env.nS, env.nA))
policy_delta = np.ones((env.nS, env.nA)) * 0.00001

v = np.zeros(env.nS)
stm = np.ones((env.nS, env.nS))

v_old = np.copy(v)
delta = np.ones(env.nS) * 0.00001

def update_value():
    global v_old, v
    while True:
        for s in range(env.nS):
            vs = []
            for a in range(env.nA):
                state_transition_prob, s_next, reward, done = env.P[s][a][0]
                vpr = v_old[s_next]
                vs.append(state_transition_prob * (reward + discount * vpr))
            v[s] = max(vs)
        if np.all(np.absolute(v - v_old) < delta):
            return
        v_old = np.copy(v)


def update_policy():
    global policy_old, policy
    policy_old = np.copy(policy)
    for s in range(env.nS):
        values = [0 for a in range(env.nA)]
        for a in range(env.nA):
            state_transition_prob, s_next, reward, done = env.P[s][a][0]
            values[a] = v[s_next]
        a = np.argmax(values)
        policy[s, :] = 0.0
        policy[s, a] = 1.0


while True:
    update_value()
    update_policy()
    if np.all(np.absolute(policy - policy_old) < policy_delta):
        break


print(policy)