# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 21:01:19 2023

@author: furkan
"""
import numpy as np

class Agent():
    
    def __init__(self, alpha, gamma, epsilon_max, epsilon_min, epsilon_dec, n_states, n_actions):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.numberOfStates = n_states
        self.numberOfActions = n_actions
        self.Q = {}
        self.init_Q()
        
    def init_Q(self):
        for s in range(self.numberOfStates):
            for a in range(self.numberOfActions):
                self.Q[s, a] = 0.0
                
    def choose_action(self, state): # using epsilon-greedy policy
        random_number = np.random.random() # [0, 1)
        
        if random_number < self.epsilon:
            return np.random.choice(self.numberOfActions)
        else:
            actions = np.array([self.Q[state, a] for a in range(self.numberOfActions)])
            return np.argmax(actions)
            
    def decrement_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon*self.epsilon_dec
        else:
            self.epsilon = self.epsilon_min
        
    def learn(self, state, action, reward, state_):
        actions = np.array([self.Q[state_, a] for a in range(self.numberOfActions)])
        a_max = np.argmax(actions)
        self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * self.Q[state_, a_max] - self.Q[state, action])
        self.decrement_epsilon()