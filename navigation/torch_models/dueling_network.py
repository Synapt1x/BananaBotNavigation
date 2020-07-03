# -*- coding: utf-8 -*-
"""
A dueling network with V and A streams in a dueling head.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import numpy as np
import torch
from torch import nn


class DuelingNetwork(nn.Module):
    """
    Torch model containing a set of fully connected linear layers and a split
    stream dueling head for advantage and value estimates.
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

        super(DuelingNetwork, self).__init__()

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

        # init a stream
        self.a_stream_in = nn.Linear(self.inter_dims[-1],
                                     self.inter_dims[-1])
        self.a_stream_out = nn.Linear(self.inter_dims[-1], self.action_size)

        # init v stream
        self.v_stream_in = nn.Linear(self.inter_dims[-1],
                                     self.inter_dims[-1])
        self.v_stream_out = nn.Linear(self.inter_dims[-1], 1)

    def forward(self, state):
        """
        Define the forward-pass for data through the model.
        """
        data_x = torch.relu(self.input(state.float()))
        for layer in self.hidden_layers:
            data_x = torch.relu(layer(data_x))

        # pass data through a stream to get advantage values
        a_data = torch.relu(self.a_stream_in(data_x))
        a_data = self.a_stream_out(a_data)
        avg_a_vals = a_data.mean(dim=-1).repeat(self.action_size, 1).transpose(
            0, 1)
        advantage_vals = a_data - avg_a_vals

        # pass same data through v stream to get value estimates
        v_data = torch.relu(self.v_stream_in(data_x))
        v_data = (self.v_stream_out(v_data)
        value_vals = v_data.repeat(1, self.action_size)

        action_values = advantage_vals + value_vals

        return action_values