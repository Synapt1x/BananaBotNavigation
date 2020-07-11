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
                 prioritized=False, prioritized_e=0.0, prioritized_a=1.0,
                 prioritized_b=1.0, seed=13):
                 seed=13):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # set parameters for prioritized replay
        self.prioritized = prioritized
        self.prioritized_e = prioritized_e
        self.prioritized_a = prioritized_a
        self.prioritized_b = prioritized_b
        if self.prioritized:
            self.errs = []
            self.probs = []
        self.seed = seed
        np.random.seed(seed)  # seed for reproducibility

        # initalize device; use GPU if available
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.memory = []

    def to_tensor(self, data):
        """
        Convert provided data into a torch tensor as appropriate.
        """
        return torch.from_numpy(np.array(data)).float().to(self.device)

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
            ind_probs = np.power(np.array(self.errs) + self.prioritized_e,
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

        # also extract errors if using prioritized replay
        if self.prioritized:
            probs = [self.probs[i] for i in random_ints]
            weights = torch.pow(self.to_tensor(probs * len(self.memory)),
                                -self.prioritized_b)
            weights = weights / torch.max(weights)  # normalize to max w_i
        else:
            weights = None

        exp_batch = tuple(self.to_tensor(d_list) for d_list in exp_batch_lists)

        return exp_batch, weights
