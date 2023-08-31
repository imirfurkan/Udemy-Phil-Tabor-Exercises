# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 21:01:15 2023

@author: furkan

Exercise 3
Frozen Lake environment
Documentation -> https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
Implement Q learning algorithm
Agent is a class
Decrement epsilon over time
Plot average score over 100 games
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from exercise3_class import Agent

desc = ["SFFF",
        "FHFH",
        "FFFH",
        "HFFG"]

# Left = 0
# Down = 1
# Right = 2
# Up = 3

alpha = 0.001
gamma = 0.9
epsilon_max = 1.0
epsilon_min = 0.01 
epsilon_dec = 0.9999995
n_games = 500000    


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", desc=desc, map_name="4x4", is_slippery = True)
    agent = Agent(alpha=alpha, gamma=gamma, epsilon_max=epsilon_max, epsilon_min=epsilon_min,
                  epsilon_dec=epsilon_dec,n_states=env.observation_space.n, n_actions=env.action_space.n)

    scores = []
    win_pct_list = []
    
    for i in range(n_games):
        done = False
        state, info = env.reset()
        score = 0
        while not done:    
            action = agent.choose_action(state)
            state_, reward, done, _, _ = env.step(action)
            agent.learn(state, action, reward, state_)
            score += reward
            state = state_
        scores.append(score)
        
        if i % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)
            if i % 1000 == 0:
                print("episode ", i, "win pct %.2f" % win_pct, "epsilon %.2f" % agent.epsilon)
                
    plt.plot(win_pct_list)
    plt.show()