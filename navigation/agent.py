# -*- coding: utf-8 -*-
"""
Custom RL agent for learning how to navigate through the Unity-ML environment
provided in the project.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import numpy as np
from navigation.q import Q


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

        self.epsilon = kwargs.get('epsilon', 0.9)
        self.epslion_decay = kwargs.get('epsilon_decay', 0.999)
        self.epsilon_min = kwargs.get('epsilon_min', 0.1)

        self.q = Q(alg, self.state_size, self.action_size)

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
        elif self.alg == 'q':
            rand_val = np.random.rand()
            if rand_val < self.epsilon:
                return self._select_random_a()
            return self.q.get_action(state)

    def compute_update(self, state, action, next_state, reward, done):
        """
        Compute the updated value for the Q-function estimate based on the
        experience tuple.
        """
        return reward

    def learn(self, state, action, next_state, reward, done):
        """
        """
        if 'dqn' not in self.alg:
            new_value = self.compute_update(state, action, next_state,
                                            reward, done)
            self.Q.update_value(state, action, new_value)

