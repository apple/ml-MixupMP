#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import torch 
import torch.nn as nn 


class FFNN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_layers=[], activation=nn.ReLU(),
                 dropout_prob=None, use_batchnorm=False, bias=True):
        super(FFNN, self).__init__()
        #self.layers = nn.ModuleList()
        self.layers = [] 
        self.input_dim = input_dim
        self.output_dim = output_dim
        for dim in hidden_layers:
            self.layers.append(nn.Linear(input_dim, dim, bias=True))
            input_dim = dim  # For the next layer
            if activation is not None:
                assert isinstance(activation, nn.Module), \
                    "Activation must be a torch.nn.modules.Module."
                self.layers.append(activation)
            if dropout_prob is not None:
                assert dropout_prob < 1
                assert dropout_prob >= 0
                self.layers.append(nn.Dropout(dropout_prob))
            if use_batchnorm:
                self.layers.append(nn.BatchNorm1d(input_dim))
        self.layers.append(nn.Linear(input_dim, output_dim, bias=bias))
        
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        #for layer in self.layers:
        #    x = layer(x)
        x = self.layers(x)
        x = x.squeeze(-1)
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
