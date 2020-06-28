# -*- coding: utf-8 -*-
"""
Main code for the navigation task.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import os
import argparse
import time

from unityagents import UnityEnvironment
import numpy as np

from navigation.agent import MainAgent


class NavigationMain:
    """
    This code contains functionality for running the navigation environment and
    running the code as per training, showing navigation, and loading/saving my
    models.
    """

    def __init__(self, file_path, model_params, frame_time=0.075):
        self.frame_time = frame_time
        self.env, self.brain_name, self.brain = self._init_env(file_path)
        self.agent = self._init_agent(model_params)

    @staticmethod
    def _eval_state(curr_env_info):
        """
        Evaluate a provided game state.
        """
        s = curr_env_info.vector_observations[0]
        r = curr_env_info.rewards[0]
        d = curr_env_info.local_done[0]

        return s, r, d

    @staticmethod
    def _print_on_close(score):
        """
        Helper method for printing out the state of the game after completion.
        """
        print(f"Final Score: {score}")

    def _init_env(self, file_path):
        """
        Initialize the Unity-ML Navigation environment.
        """
        env = UnityEnvironment(file_name=file_path)
        brain_name = env.brain_names[0]
        first_brain = env.brains[brain_name]

        return env, brain_name, first_brain

    def _init_agent(self, model_params):
        """
        Initialize the custom model utilized by the agent.
        """
        # extract state and action information
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        state_size = len(env_info.vector_observations[0])
        action_size = self.brain.vector_action_space_size

        return MainAgent(**model_params, state_size=state_size,
                         action_size=action_size)

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
                action = self.agent.get_action(state)
                env_info = self.env.step(action)[self.brain_name]
                next_state, reward, done = self._eval_state(env_info)

                score += reward
                state = next_state

                if done:
                    break
                time.sleep(self.frame_time)
        except KeyboardInterrupt:
            print("Exiting game gracefully...")
        finally:
            # gracefully close the env and print last score
            self.env.close()
            self._print_on_close(score)

