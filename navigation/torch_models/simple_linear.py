# -*- coding: utf-8 -*-
"""
Simple torch model that is a set of fully connected layers.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import numpy as np
import torch
from torch import nn


class LinearModel(nn.Module):
    """
    Torch model containing a set of dense fully connected layers.
    """

    def __init__(self, state_size, action_size, inter_dims=None, seed=13):
        if inter_dims is None:
            self.inter_dims = [64, 256]
        else:
            self.inter_dims = inter_dims

        self.state_size = state_size
        self.action_size = action_size
        
        # set the seed
        self.seed = seed
        torch.manual_seed(self.seed)

        super(LinearModel, self).__init__()

        # initialize the architecture
        self._init_model()

    def _init_model(self):
        """
        Define the architecture and all layers in the model.
        """
        self.input = nn.Linear(self.state_size, self.inter_dims[0])
        hidden_layers = []

        for dim_i, hidden_dim in enumerate(self.inter_dims[1:]):
            prev_dim = self.inter_dims[dim_i]
            hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
        
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output = nn.Linear(self.inter_dims[-1], self.action_size)

    def forward(self, state):
        """
        Define the forward-pass for data through the model.
        """
        data_x = torch.relu(self.input(state.float()))
        for layer in self.hidden_layers:
            data_x = torch.relu(layer(data_x))
        action_values = torch.relu(self.output(data_x)) 

        return action_values