# -*- coding: utf-8 -*-
"""
Custom RL agent for learning how to navigate through the Unity-ML environment
provided in the project.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from navigation.q import Q
from navigation.replay_buffer import ReplayBuffer


class MainAgent:
    """
    This model contains my code for the agent to learn and be able to navigate
    through the pertinent problem.
    """

    def __init__(self, alg, state_size, action_size, seed=13, **kwargs):
        self.alg = alg
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        np.random.seed(seed)

        # extract hyperparameters for the general algorithm
        self.epsilon = kwargs.get('epsilon', 0.9)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.9999)
        self.epsilon_min = kwargs.get('epsilon_min', 0.05)
        self.gamma = kwargs.get('gamma', 0.9)
        self.alpha = kwargs.get('alpha', 0.2)
        self.t_freq = kwargs.get('t_freq', 10)
        self.tau = kwargs.get('tau', 0.1)

        # parameters for the replay buffer
        self.buffer_size = kwargs.get('buffer_size', 1E6)
        self.batch_size = kwargs.get('batch_size', 32)
        self.prioritized = kwargs.get('prioritized', False)
        self.prioritized_e = kwargs.get('prioritized_e', 0.00)
        self.prioritized_a = kwargs.get('prioritized_a', 1.0)
        self.prioritized_b = kwargs.get('prioritized_b', 1.0)

        # architecture parameters for DQN
        self.inter_dims = kwargs.get('inter_dims', [64, 256])

        self.q = Q(alg, self.state_size, self.action_size, self.inter_dims)
        if 'dqn' in self.alg.lower():
            self.target_q = Q(alg, self.state_size, self.action_size,
                              self.inter_dims)
            self.optimizer = optim.Adam(self.q.q.parameters(), lr=self.alpha)
            self.memory = ReplayBuffer(self.action_size, self.buffer_size,
                                       self.batch_size, self.prioritized,
                                       self.prioritized_e, self.prioritiezd_a,
                                       self.prioritized_b, seed=seed)
            self.t = 1

    def _update_target(self):
        """
        Update target network using a soft rolling update from the primary
        network (as opposed to every N-th iteration).
        """
        # soft update target network (as in DQN mini-project)
        for t_param, q_param in zip(self.target_q.q.parameters(),
                                    self.q.q.parameters()):
            update_q = self.tau * q_param.data
            target_q = (1.0 - self.tau) * t_param.data
            t_param.data.copy_(update_q + target_q)

    def _select_random_a(self):
        """
        Select a random action.
        """
        return np.random.randint(self.action_size)

    def _extract_model_names(self, file_name):
        """
        Extract the model names for DQN.
        """
        file_split = file_name.split('.')[:-1]
        main_model_name = ''.join(file_split + ['-main.pkl'])
        target_model_name = ''.join(file_split + ['-target.pkl'])

        return main_model_name, target_model_name

    def save_model(self, file_name):
        """
        Save the agent's underlying model(s).
        """
        if self.alg.lower() == 'random':
            return None

        if 'dqn' in self.alg.lower():
            main_file, target_file = self._extract_model_names(file_name)

            self.q.save_model(main_file)
            self.target_q.save_model(target_file)

    def load_model(self, file_name):
        """
        Load the agent's underlying model(s).
        """
        if self.alg.lower() == 'random':
            return None

        if 'dqn' in self.alg.lower():
            main_file, target_file = self._extract_model_names(file_name)

            self.q.load_model(main_file)
            self.target_q.load_model(target_file)

    def get_action(self, state):
        """
        Extract the action intended by the agent based on the selection
        criteria.
        """
        if self.alg.lower() == 'random':
            return self._select_random_a()
        else:
            rand_val = np.random.rand()
            if rand_val < self.epsilon:
                return self._select_random_a()
            return self.q.get_action(state)

    def compute_update(self, state, action, next_state, reward, done):
        """
        Compute the updated value for the Q-function estimate based on the
        experience tuple.
        """
        if self.alg.lower() == 'random':
            return 0.0

        if 'dqn' not in self.alg.lower():
            curr_val = self.q.get_value(state, action)
            next_val = self.q.get_value(next_state, self.get_action(next_state))

            return curr_val + self.alpha * (
                reward + self.gamma * next_val - curr_val)
        else:
            # get target q for next state (on max) and current estimate
            if 'ddqn' in self.alg.lower():
                max_action = self.q.get_action(next_state)
                target_q_vals = self.target_q.get_value(next_state, max_action)
            else:
                target_q_vals = self.target_q.get_value(next_state)
            curr_q_est = self.q.get_value(state, action)

            # compute the error
            target_vals = reward + (self.gamma * target_q_vals * (1 - done))

            loss = F.mse_loss(target_vals, curr_q_est)

            return loss

    def learn(self, state, action, next_state, reward, done):
        """
        """
        if self.alg.lower() == 'random':
            return None

        if 'dqn' not in self.alg.lower():
            new_value = self.compute_update(state, action, next_state,
                                            reward, done)
            self.q.update_value(state, action, new_value)
        else:
            loss = self.compute_update(state, action, next_state, reward, done)

            # store experience tuple into replay buffer
            if self.prioritized:
                self.memory.store_tuple(state, action, next_state, reward, done)
            else:
                self.memory.store_tuple(state, action, next_state, reward, done,
                                        loss.data)

            if not self.memory.is_empty():
                # extract experience tuples
                exp_tuples = self.memory.sample()
                states, actions, nexts, rewards, dones = exp_tuples

                # compute TD error
                loss = self.compute_update(states, actions, nexts,
                                           rewards, dones)

                # advance optimizer using loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (self.t % self.t_freq) == 0:
                    self._update_target()

            self.t += 1

    def step(self):
        """
        Update state of the agent and take a step through the learning process
        to reflect experiences have been acquired and/or learned from.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)