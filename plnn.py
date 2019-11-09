""" Wrappers for building piecewise linear nets
    (generally we'll only worry about ReLU nets here)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

class PLNN(nn.Module):
    def __init__(self, layer_sizes=None, bias=True, dtype=torch.float32):
        super(PLNN, self).__init__()
        self.layer_sizes = layer_sizes
        self.dtype = dtype
        self.fcs = []
        self.bias = bias
        self.net = self.build_network(layer_sizes)

    def build_network(self, layer_sizes):
        layers = OrderedDict()

        num = 1
        for size_pair in zip(layer_sizes, layer_sizes[1:]):
            size, next_size = size_pair
            layer = nn.Linear(size, next_size, bias=self.bias).type(self.dtype)
            layers[str(num)] = layer
            self.fcs.append(layer)
            num = num + 1
            layers[str(num)] = nn.ReLU()
            num = num + 1
        del layers[str(num-1)]      # No ReLU for the last layer
        net = nn.Sequential(layers).type(self.dtype)
        return net

    def get_parameters(self):
        params = []
        for fc in self.fcs:
            fc_params = [elem for elem in fc.parameters()]
            for param in fc_params:
                params.append(param)
        return params


    def forward(self, x, return_preacts=False):
        """ Standard forward method for ReLU net
            If return_preacts is True, then we collect and return the
            values after applying each linear layer
        """
        x = x.view(-1, self.fcs[0].in_features)
        preacts = []
        for i, fc in enumerate(self.fcs):
            if i == len(self.fcs) - 1:
                x = fc(x)
                if return_preacts:
                    preacts.append(x)
                    return preacts
                return x
            x = fc(x)
            preacts.append(torch.clone(x))
            x = F.relu(x)
