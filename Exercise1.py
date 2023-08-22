# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:39:43 2023

@author: furkan

First Exercise
0 reward per step, +1 for escaping
Agent slides
Holes (H) terminate the episode
Documentation -> https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
Random agent, 1000 games
Poor performance
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
env = gym.make("FrozenLake-v1", desc=desc, map_name="4x4", is_slippery = True)

n_games = 1000
win_pct = []
scores = []

for i in range(n_games):
    terminated = False
    state = env.reset()
    score = 0
    while not terminated:
        action = env.action_space.sample()
        state, reward, terminated, _, _ = env.step(action)
        score += reward
    scores.append(score)
    
    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)

plt.plot(win_pct)
plt.show()

