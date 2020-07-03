# -*- coding: utf-8 -*-
"""
Main code for the navigation task.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import os
import argparse
import datetime
import time

from unityagents import UnityEnvironment
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use backend for saving plots only
from matplotlib import pyplot as plt

from navigation.agent import MainAgent


class NavigationMain:
    """
    This code contains functionality for running the navigation environment and
    running the code as per training, showing navigation, and loading/saving my
    models.
    """

    def __init__(self, file_path, model_params, frame_time=0.075,
                 max_episodes=1E5, max_iterations=1E5):
        self.frame_time = frame_time
        self.max_iterations = max_iterations
        self.max_episodes = max_episodes

        self.env, self.brain_name, self.brain = self._init_env(file_path)
        self.agent = self._init_agent(model_params)

        self.score_store = []
        self.average_scores = []

    @staticmethod
    def _eval_state(curr_env_info):
        """
        Evaluate a provided game state.
        """
        s = curr_env_info.vector_observations[0]
        r = curr_env_info.rewards[0]
        d = int(curr_env_info.local_done[0])  # convert bool to int

        return s, r, d

    @staticmethod
    def _print_progress(iteration, score_avg):
        """
        Helper method for printing out the state of the game after completion.
        """
        print(f"Average score so far: {score_avg}")

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

    def save_model(self, file_name):
        """
        Save the model to the file name specified.
        """
        self.agent.save_model(file_name)

    def load_model(self, file_name):
        """
        Load the model specified.
        """
        self.agent.load_model(file_name)

    def save_training_plot(self):
        """
        Plot training performance through episodes.
        """
        num_eval = len(self.average_scores)

        if num_eval > 100:
            # Set up plot file and directory names
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            plot_dir = os.path.join(cur_dir, os.pardir, 'output')
            cur_date = datetime.datetime.now().strftime('%Y-%m-%d')
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)
            plot_file = os.path.join(plot_dir,
                                     f'training-performance-{cur_date}.png')

            # plot and save the plot file
            fig = plt.figure(figsize=(12, 8))

            plt.plot(self.average_scores, linewidth=2)
            plt.title(f'Agent Average Score During Training', fontsize=20)

            plt.xlabel('Episode', fontsize=16)
            plt.ylabel('Average Score', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt.savefig(plot_file)

            print(f'Training plot saved to {plot_file}')
        else:
            print('Not enough average scores computed. Skipping plotting.')

    def run_episode(self, train_mode=True):
        """
        Run an episode of interaction in the Unity-ML Navigation environment.
        """
        score = 0
        iteration = 0

        # initiate interaction and learning in environment
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        state = env_info.vector_observations[0]

        while iteration < self.max_iterations:
            # first have the agent act and evaluate state
            action = self.agent.get_action(state)
            env_info = self.env.step(action)[self.brain_name]
            next_state, reward, done = self._eval_state(env_info)

            # learn from experience tuple batch
            if train_mode:
                self.agent.learn(state, action, next_state, reward, done)

            # increment score and compute average
            score += reward
            state = next_state

            # # update state of agent
            # self.agent.step()

            if done:
                break
            time.sleep(self.frame_time)

            # print average score as training progresses
            iteration += 1

        if len(self.score_store) == 1000:
            self.score_store = self.score_store[1:]
        self.score_store.append(score)
        score_avg = np.mean(self.score_store[-100:])
        self.average_scores.append(score_avg)

        return score_avg

    def train_agent(self, train_mode=True):
        """
        Train an agent by running learning episodes in the navigation task.
        """
        episode = 1
        try:
            # run episodes
            while episode < self.max_episodes:
                # set epsilon to 0 to always use Q network for actions if eval
                if not train_mode:
                    self.agent.epsilon = 0.0
                avg_after_ep = self.run_episode(train_mode=train_mode)

                print(f'* Episode {episode} completed * avg: {avg_after_ep} *')
                if train_mode:
                    self.agent.step()

                episode += 1
        except KeyboardInterrupt:
            print("Exiting learning gracefully...")
        finally:
            if train_mode:
                self.save_training_plot()
            self.env.close()
