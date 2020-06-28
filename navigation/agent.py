# -*- coding: utf-8 -*-
"""
Custom RL agent for learning how to navigate through the Unity-ML environment
provided in the project.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import os
import argparse


class MainModel:
    """
    This model contains my code for the agent to learn and be able to navigate
    through the pertinent problem.
    """

    def __init__(self, alg, state_size, action_size, **kwargs):
        self.alg = alg
        self.state_size = state_size
        self.action_size = action_size

    def get_action(self, state):
        """
        Extract the action intended by the agent based on the selection
        criteria.
        """
        #TODO: currently completely random
        action = np.random.randint(self.action_size)

        return action

    def learn(self):
        """

        """
        pass
