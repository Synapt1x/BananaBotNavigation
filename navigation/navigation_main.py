# -*- coding: utf-8 -*-
"""
Main code for the navigation task.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import os
import argparse

from unityagents import UnityEnvironment
import numpy as np


class NavigationMain:
    """
    This code contains functionality for running the navigation environment and
    running the code as per training, showing navigation, and loading/saving my
    models.
    """

    def __init__(self, file_path, **model_args, **kwargs):
        self.env, self.brain_name, self.brain = self._init_env(file_path)
        self.model = self._init_model(**model_args, **kwargs)

    @staticmethod
    def _eval_state(curr_env_info):
        """
        Evaluate a provided game state.
        """
        s = env_info.vector_observations[0]
        r = env_info.rewards[0]
        d = env_info.local_done[0]

        return s, r, d

    @staticmethod
    def _print_final(score):
        """
        Helper method for printing out the state of the game after completion.
        """
        print(f"Final Score: {score}")

    def _init_env(self):
        """
        Initialize the Unity-ML Navigation environment.
        """
        env = UnityEnvironment(file_name=file_path)
        brain_name = env.brain_names[0]
        first_brain = env.brains[brain_name]

        return env, brain_name, first_brain

    def run_interaction(self, train_mode=True):
        """
        Run interaction in the Unity-ML Navigation environment.
        """
        score = 0
        try:
            # initiate interaction and learning in environment
            env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
            state = env_info.vector_observations[0]

            while True:
                # first have the agent act and evaluate state
                action = self.model.get_action(state)
                env_info = self.env.step(action)[self.brain_name]
                next_state, reward, done = self._eval_state(env_info)

                score += reward
                state = next_state

                if done:
                    break
        except KeyboardInterrupt:
            print("Exiting game gracefully...")
        finally:
            # gracefully close the env and print last score
            self.env.close()
            self._print_on_close(score)

