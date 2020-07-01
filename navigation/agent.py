# -*- coding: utf-8 -*-
"""
Custom RL agent for learning how to navigate through the Unity-ML environment
provided in the project.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from navigation.q import Q
from navigation.replay_buffer import ReplayBuffer


class MainAgent:
    """
    This model contains my code for the agent to learn and be able to navigate
    through the pertinent problem.
    """

    def __init__(self, alg, state_size, action_size, seed=13, **kwargs):
        self.alg = alg
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        np.random.seed(seed)

        # extract hyperparameters for the general algorithm
        self.epsilon = kwargs.get('epsilon', 0.9)
        self.epslion_decay = kwargs.get('epsilon_decay', 0.999)
        self.epsilon_min = kwargs.get('epsilon_min', 0.1)
        self.gamma = kwargs.get('gamma', 0.9)
        self.alpha = kwargs.get('alpha', 0.1)
        self.t_freq = kwargs.get('t_freq', 10)

        # parameters for the replay buffer
        self.buffer_size = kwargs.get('buffer_size', 1E6)
        self.batch_size = kwargs.get('batch_size', 32)

        self.q = Q(alg, self.state_size, self.action_size)
        if 'dqn' in self.alg.lower():
            self.target_q = Q(alg, self.state_size, self.action_size)
            self.optimizer = optim.Adam(self.q.q.parameters(), lr=self.alpha)
            self.memory = ReplayBuffer(self.action_size, self.buffer_size,
                                       self.batch_size, seed=seed)
            self.t = 1

    def _select_random_a(self):
        """
        Select a random action.
        """
        return np.random.randint(self.action_size)

    def get_action(self, state):
        """
        Extract the action intended by the agent based on the selection
        criteria.
        """
        if self.alg == 'random':
            return self._select_random_a()
        else:
            rand_val = np.random.rand()
            if rand_val < self.epsilon:
                return self._select_random_a()
            return self.q.get_action(state)

    def compute_update(self, state, action, next_state, reward, done):
        """
        Compute the updated value for the Q-function estimate based on the
        experience tuple.
        """
        if 'dqn' not in self.alg.lower():
            curr_val = self.q.get_value(state, action)
            next_val = self.q.get_value(next_state, self.get_action(next_state))

            return curr_val + self.alpha * (
                reward + self.gamma * next_val - curr_val)
        else:
            # get target q for next state (on max) and current estimate
            target_q_vals = self.target_q.get_value(next_state)
            curr_q_est = self.q.get_value(state, action)

            # compute the error
            target_vals = reward + (self.gamma * target_q_vals * (1 - done))

            return F.mse_loss(target_vals, curr_q_est)

    def learn(self, state, action, next_state, reward, done):
        """
        """
        if 'dqn' not in self.alg.lower():
            new_value = self.compute_update(state, action, next_state,
                                            reward, done)
            self.q.update_value(state, action, new_value)
        else:
            self.memory.store_tuple(state, action, next_state, reward, done)
            if (self.t % self.t_freq) == 0 and not self.memory.is_empty():
                # extract experience tuples
                exp_tuples = self.memory.sample()
                states, actions, nexts, rewards, dones = exp_tuples

                # compute TD error
                loss = self.compute_update(states, actions, nexts,
                                                rewards, dones)
                
                # advance optimizer using loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.t += 1