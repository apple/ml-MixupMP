#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import torch 
import torch.nn as nn 
import numpy as np


class CNN(nn.Module):
    def __init__(self, input_dim, input_channels, num_classes, hidden_channels=[6, 16], hidden_fc_layers=[120, 84],
                 activation=nn.ReLU(), kernel_size=5, bias=True, dropout_prob=None, use_batchnorm=False):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.input_dim = input_dim
        self.output_dim = num_classes
        self.input_channels = input_channels

        linear_input = input_dim
        for dim in hidden_channels:
            self.conv_layers.append(nn.Conv2d(input_channels, dim, kernel_size, bias=bias))
            input_channels = dim
            linear_input = int(np.floor(linear_input - kernel_size + 1))
            if activation is not None:
                assert isinstance(activation, nn.Module), \
                    "Activation must be a torch.nn.modules.Module."
                self.conv_layers.append(activation)
            self.conv_layers.append(nn.MaxPool2d(2, 2))
            linear_input = int(np.floor(linear_input / 2))
            if dropout_prob is not None:
                assert dropout_prob < 1
                assert dropout_prob >= 0
                self.conv_layers.append(nn.Dropout(dropout_prob))
            if use_batchnorm:
                self.conv_layers.append(nn.BatchNorm1d(input_dim))

        linear_input = linear_input * linear_input * input_channels
        for dim in hidden_fc_layers:
            self.fc_layers.append(nn.Linear(linear_input, dim, bias=bias))
            linear_input = dim
            if activation is not None:
                assert isinstance(activation, nn.Module), \
                    "Activation must be a torch.nn.modules.Module."
                self.fc_layers.append(activation)
            if dropout_prob is not None:
                assert dropout_prob < 1
                assert dropout_prob >= 0
                self.fc_layers.append(nn.Dropout(dropout_prob))
            if use_batchnorm:
                self.fc_layers.append(nn.BatchNorm1d(input_dim))
        self.fc_layers.append(nn.Linear(linear_input, num_classes, bias=bias))


    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        
        x = torch.flatten(x, 1)
        for layer in self.fc_layers:
            x = layer(x)
        return x


    def save(self, path):
        """
        Save state dict of model
        :param path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        """
        Load model from state dict
        :param path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path, map_location=device))
