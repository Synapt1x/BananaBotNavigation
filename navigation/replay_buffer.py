# -*- coding: utf-8 -*-
"""
Code implementing the functionality for a replay buffer in the DQN algorithm.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import os
import argparse

import numpy as np
import torch


class ReplayBuffer:
    """
    This class implements the functionality for storing experience tuples in a
    replay buffer to sample during learning steps in the DQN algorithm.
    """

    def __init__(self, action_size, buffer_size=1E6, batch_size=32, seed=13):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        np.random.seed(seed)  # seed for reproducibility

        # initalize device; use GPU if available
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.memory = []

    def is_empty(self):
        """
        Check to see if memory contains enough tuples for sampling.
        """
        return len(self.memory) < self.batch_size

    def store_tuple(self, state, action, next_state, reward, done):
        """
        Add the experience tuple to memory.

        Parameters
        ----------
        state: np.array/torch.Tensor
            Tensor singleton containing state information
        action: np.array/torch.Tensor
            Tensor singleton containing the action taken from state
        next_state: np.array/torch.Tensor
            Tensor singleton containing information about what state followed
            the action taken from the state provided by 'state'
        reward: np.array/torch.Tensor
            Tensor singleton containing reward information
        done: np.array/torch.Tensor
            Tensor singleton representing whether or not the episode ended after
            action was taken
        """
        # only keep the most recent tuples if memory size has been reached
        if len(self.memory) == self.buffer_size:
            self.memory = self.memory[1:]
        self.memory.append((state, action, next_state, reward, done))

    def sample(self):
        """
        Extract a random sample of tuples from memory.
        """
        random_ints = np.random.choice(len(self.memory), self.batch_size,
                                       replace=False)

        raw_sample = [self.memory[i] for i in random_ints]
        exp_batch_lists = list(zip(*raw_sample))

        exp_batch = tuple(torch.from_numpy(
            np.array(exp_batch_lists[i])).float().to(self.device)
                          for i in range(len(exp_batch_lists)))

        return exp_batch
