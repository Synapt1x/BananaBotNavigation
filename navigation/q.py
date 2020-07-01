# -*- coding: utf-8 -*-
"""
Custom RL agent for learning how to navigate through the Unity-ML environment
provided in the project.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import numpy as np
import torch
from navigation.torch_models.simple_linear import LinearModel


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
        elif self.alg == 'dqn':
            #TODO: pass parameters to architecture
            return LinearModel(state_size=self.state_size,
                               action_size=self.action_size)

    def get_value(self, state, action):
        """
        """
        if self.alg == 'q':
            return self.q[state, action]
        elif self.alg == 'dqn':
            return self.q(state).detach().gather(1, action)

    def get_action(self, state):
        """
        """
        if self.alg == 'q':
            return np.argmax(self.q[state])
        elif self.alg == 'dqn':
            #TODO: ensure this works for batches
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state)
            a_vals = self.q(state).detach()
            a_max = a_vals.argmax(-1)

            # return singular max action if not a batch
            if len(a_max.shape) == 0:
                return a_max.item()

            return a_max[0].unsqueeze(1)

    def update_q_table(self, state, action, new_val):
        """
        """
        if self.alg == 'q':
            self.q[state, action] = new_val
