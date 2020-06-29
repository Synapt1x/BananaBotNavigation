# -*- coding: utf-8 -*-
"""
Custom RL agent for learning how to navigate through the Unity-ML environment
provided in the project.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import numpy as np


class Q:
    """
    This code contains the functionality for initializing, deciding on actions,
    and updating the underlying Q-function representation.
    """

    def __init__(self, alg, state_size, action_size, **kwargs):
        self.alg = alg
        self.state_size = state_size
        self.action_size = action_size

        self.q = self._init_q()

    def _init_q(self):
        """
        Initialize the representation of the Q-function.
        """
        if self.alg == 'random':
            return None
        elif self.alg == 'q':
            return np.random.rand(shape=(self.state_size, self.action_size))

    def get_value(self, state, action):
        """
        """
        if self.alg == 'q':
            return q[state, action]

    def get_action(self, state):
        """
        """
        if self.alg == 'q':
            return np.argmax(q[state])

    def update_value(self, state, action, new_val):
        """
        """
        if self.alg == 'q':
            self.q[state, action] = new_val

