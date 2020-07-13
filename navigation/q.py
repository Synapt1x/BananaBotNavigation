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
from navigation.torch_models.dueling_network import DuelingNetwork


class Q:
    """
    This code contains the functionality for initializing, deciding on actions,
    and updating the underlying Q-function representation.
    """

    def __init__(self, alg, state_size, action_size, inter_dims=None, **kwargs):
        self.alg = alg
        self.state_size = state_size
        self.action_size = action_size
        if inter_dims is None:
            inter_dims = [64, 256]
        self.inter_dims = inter_dims

        # initalize device; use GPU if available
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.q = self._init_q()

    def _init_q(self):
        """
        Initialize the representation of the Q-function.
        """
        if self.alg.lower() == 'random':
            return None
        elif self.alg.lower() == 'q':
            return np.random.rand(shape=(self.state_size, self.action_size))
        elif self.alg.lower() == 'dqn' or self.alg.lower() == 'ddqn':
            return LinearModel(state_size=self.state_size,
                               action_size=self.action_size,
                               inter_dims=self.inter_dims).to(self.device)
        elif 'dueling' in self.alg.lower():
            return DuelingNetwork(state_size=self.state_size,
                                  action_size=self.action_size,
                                  inter_dims=self.inter_dims).to(self.device)

    def save_model(self, file_name):
        """
        Save the underlying model.

        Parameters
        ----------
        file_name: str
            File name to which the agent will be saved for future use.
        """
        torch.save(self.q.state_dict(), file_name)

    def load_model(self, file_name):
        """
        Load the parameters for the underlying model.

        Parameters
        ----------
        file_name: str
            File name from which the agent will be loaded.
        """
        if self.device.type == 'cpu':
            self.q.load_state_dict(torch.load(file_name,
                                              map_location=self.device.type))
        else:
            self.q.load_state_dict(torch.load(file_name))
        self.q.eval()

    def get_value(self, state, action=None):
        """
        Extract a value from the provided state. If an action is provided then
        extract the values for the specified actions from the Q-function
        representation, otherwise use a max over actions.

        Parameters
        ----------
        state: np.array/torch.Tensor
            Array or Tensor singleton or batch containing state information
            either in the shape (1, 37) or (batch_size, 37)
        action: int/torch.Tensor
            An integer specifying an action to extract value from from
            Q(s=state, a=action), or a batch of values to extract from
            Q(s=states, a=actions).

        Returns
        -------
        np.array/torch.Tensor
            Values of the relevant actions for the provided state
        """
        if 'dqn' not in self.alg.lower():
            if action is None:
                return np.max(self.q[state])
            return self.q[state, action]
        else:
            if action is None:
                return self.q(state).detach().max(-1)[0]
            return self.q(state).gather(
                1, action.view(-1, 1).long()).squeeze(-1)

    def get_action(self, state, in_train=True):
        """
        Extract the action that is maximal for a provided state. If the model is
        in train mode, then this method also ensures the network is put into
        train() mode following extracting actions in eval() mode.

        Parameters
        ----------
        state: np.array/torch.Tensor
            Array or Tensor singleton or batch containing state information
            either in the shape (1, 37) or (batch_size, 37)
        in_train: bool
            Flag to indicate whether or not the agent will be training or just
            running inference.

        Returns
        -------
        int/torch.Tensor
            Integer or torch.Tensor batch of integers indicating the actions to
            be selected based on state provided.
        """
        if 'dqn' not in self.alg.lower():
            return np.argmax(self.q[state])
        else:
            # convert to tensfor if array is provided
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).to(self.device)

            # set model to eval mode and determine max action
            self.q.eval()
            with torch.no_grad():
                a_vals = self.q(state)
                a_max = a_vals.argmax(-1)

            # reset to train if model is not meant to stay in eval mode
            if in_train:
                self.q.train()

            # return singular max action if not a batch
            if len(a_max.shape) == 0 or (len(a_max)) == 1:
                return a_max.item()

            if len(a_max.shape) == 1:
                return a_max.unsqueeze(1)

            return a_max[0].unsqueeze(1)

    def update_q_table(self, state, action, new_val):
        """
        Update Q-function representation if using a Q-table. This is not used by
        default since the default algorithm is DQN.

        state: np.array/torch.Tensor
            Array or Tensor singleton or batch containing state information
            either in the shape (1, 37) or (batch_size, 37)
        action: int
            An integer specifying an action to extract value from from
            Q(s=state, a=action).
        new_val: float
            Float value that represents the new value for a Q-table value.
        """
        self.q[state, action] = new_val
