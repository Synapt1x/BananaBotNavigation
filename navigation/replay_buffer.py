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

    def __init__(self, action_size, buffer_size=1E6, batch_size=32,
                 priorited=False, seed=13):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.prioritized_e = prioritized_e
        self.prioritized_a = prioritized_a
        self.priorited = prioritized
        if self.prioritized:
            self.errs = []
            self.probs = []
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

    def store_tuple(self, state, action, next_state, reward, done, err=None):
        """
        Add the experience tuple to memory.
        """
        # only keep the most recent tuples if memory size has been reached
        if len(self.memory) == self.buffer_size:
            self.memory = self.memory[1:]
            if self.prioritized:
                self.errs = self.errs[1:]
                self.probs = self.probs[1:]
        if self.prioritized:
            self.memory.append((state, action, next_state, reward, done))
            self.errs.append(err)
            ind_probs = np.pow(np.array(self.errs) + self.e,
                               self.prioritized_a)
            self.probs = ind_probs / np.sum(ind_probs)
        else:
            self.memory.append((state, action, next_state, reward, done))

    def sample(self):
        """
        Extract a random sample of tuples from memory.
        """
        # randomly select either weighted by priority or uniformly
        if self.prioritized:
            random_ints = np.random.choice(len(self.memory), self.batch_size,
                                           p=self.probs)
        else:
            random_ints = np.random.choice(len(self.memory), self.batch_size,
                                           replace=False)

        # then extract batch based on random indices selected
        raw_sample = [self.memory[i] for i in random_ints]
        exp_batch_lists = list(zip(*raw_sample))

        exp_batch = tuple(torch.from_numpy(
            np.array(exp_batch_lists[i])).float().to(self.device)
                          for i in range(len(exp_batch_lists)))

        return exp_batch
