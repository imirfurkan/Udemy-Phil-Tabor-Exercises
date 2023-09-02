# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 21:13:21 2023

@author: furkan

Exercise 4
Naive Deep Q Learning in Code
"""

import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
from util import plot_learning_curve

class LinearDeepQNetwork(nn.Module): # using nn.Module gives access to self.parameters() function.
    
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork, self).__init__() # initialize the parent class (nn.Module) before customizing the initialization of your own class (LinearDeepQNetwork)
        
        self.fc1 = nn.Linear(*input_dims, 128) # (*) operator unpacks tuples/lists and passes them as seperate arguments
        self.fc2 = nn.Linear(128, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device) # moves the entire model (all layers and parameters) to the specified device (GPU or CPU) for computation
        
    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)
        
        return actions
    
class Agent():
    def __init__(self, lr, gamma, epsilon, epsilon_dec, epsilon_min, n_actions, input_dims):
        self.lr = lr
        self.input_dims = input_dims
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        
        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)
        
    def choose_action(self, state):
        random_number = np.random.random() # [0, 1)
        if random_number > self.epsilon:
            state = T.tensor(state, dtype = T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item() # item converts from tensor to numpy array
        else:
            action = np.random.choice(self.action_space)
        
        return action

    def decrement_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_dec
        else:
            self.epsilon = self.epsilon_min

    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()
        states = T.tensor(state, dtype = T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype = T.float).to(self.Q.device)

# tensors are moved to the device when they are involved in operations
# that require consistency with the device where the neural network's
# parameters reside. When tensors are used purely for temporary
# computations or calculations that don't involve backpropagation,
# they may not be explicitly moved to the device.

        q_pred = self.Q.forward(states)[actions] # predicted q value for the action taken
        
        q_next = self.Q.forward(states).max()
            
        q_target = reward + self.gamma*q_next
        
        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward() # compute gradients
        self.Q.optimizer.step() # update Q network's weights
        self.decrement_epsilon()
        
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    n_games = 10000
    scores = []
    eps_history = []
    
    gamma = 0.99
    epsilon = 1.0
    epsilon_dec = 1e-5
    epsilon_min = 0.01
    agent = Agent(lr = 0.0001, gamma = gamma, epsilon = epsilon, epsilon_dec = epsilon_dec, 
                  epsilon_min = epsilon_min, input_dims = env.observation_space.shape, 
                  n_actions = env.action_space.n)
    
    for i in range(n_games):
        score = 0
        done = False
        state, _ = env.reset()
        
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, _, _ = env.step(action)
            score += reward
            agent.learn(state, action, reward, state_)
            state = state_
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print("epsiode ", i, "score %.1f avg score %.1f epsilon %.2f"
                  % (score, avg_score, agent.epsilon))
        
    filename = "cartpole_naive_dqn.png"
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)