# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:16:01 2023

@author: furkan

Exercise 2
Frozen Lake environment
Documentation -> https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
Reasonable deterministic policy
1000 games
Plot win percentage over trailing 10 games
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

desc = ["SFFF",
        "FHFH",
        "FFFH",
        "HFFG"]

# Left = 0
# Down = 1
# Right = 2
# Up = 3

policy = {0: 1,
          1: 2,
          2: 1,
          3: 0,
          4: 1,
          6: 1,
          8: 2,
          9: 1,
          10: 1,
          13: 2,
          14: 2}

env = gym.make("FrozenLake-v1", desc=desc, map_name="4x4", is_slippery = True)
n_games = 1000
win_pct = []
scores = []

for i in range(n_games):
    terminated = False
    state, _ = env.reset()
    score = 0
    
    while not terminated:
        action = policy[state]
        state, reward, terminated, _, _ = env.step(action)
        score += reward
    scores.append(score)
    
    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)

plt.plot(win_pct)
plt.show()

