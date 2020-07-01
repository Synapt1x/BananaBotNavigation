# -*- coding: utf-8 -*-
"""
Code implementing the functionality for a replay buffer in the DQN algorithm.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import os
import argparse

import numpy as np



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

        self.memory = []

    def is_empty(self):
        """
        Check to see if memory contains enough tuples for sampling.
        """
        return len(self.memory) < self.batch_size

    def store_tuple(self, state, action, next_state, reward, done):
        """
        Add the experience tuple to memory.
        """
        # only keep the most recent tuples if memory size has been reached
        if len(self.memory) == self.buffer_size:
            self.memory = self.memory[1:]
        self.memory.append((state, action, next_state, reward, done))

    def sample(self):
        """
        Extract a random sample of tuples from memory.
        """
        random_ints = np.random.randint(0, len(self.memory), self.batch_size)

        raw_sample = [self.memory[i] for i in random_ints]
        exp_batch = list(zip(*raw_sample))

        return (exp_batch[0], exp_batch[1], exp_batch[2], exp_batch[3],
                exp_batch[4])
        
