""" Wrapper for building networks with general activation functions
    (right now constrained to have same activation function throughout)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt 
import utilities as utils
import gurobipy as gb
import copy
import re


class GenNet(nn.Module):
    """ Wrapper for general networks. 
        Right now we can handle the following:
        Linear Layers: Linear, Conv2d, MaxPool2d, AvgPool2d
        Nonlinearities: ReLU, LeakyRELU
    """ 
    SUPPORTED_LINS = [nn.Linear, nn.Conv2d, nn.ConvTranspose2d] # L
    SUPPORTED_NONLINS = [nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh] # R
    SUPPORTED_POOLS = [nn.MaxPool2d, nn.AvgPool2d, nn.Flatten] # P
    def __init__(self, net, dtype=torch.float):
        """ Constructor for GenNets 
        ARGS:
            sequential : nn.Sequential object - layers we include 
                         this sequential must match the regex:
                         /^(LR(P)?)+L$/ 
            dtype: torch datatype
        """
        super(GenNet, self).__init__()
        self.net = net
        self.dtype = dtype 
        self.shapes = None
        self._support_check()
        self._setup()

    def __getitem__(self, idx):
        return self.net[idx]

    def _setup(self):
        """ Sets up some attributes we want to know computed off the 
            sequential unit 
        """
        pass 
        # Um... idk what needs to go here exactly, but we need some stuff 




    def _support_check(self):
        """ Checks the sequential to be  """
        string_sub = {}
        for module_list, c in [(self.SUPPORTED_LINS, 'L'), 
                          (self.SUPPORTED_NONLINS, 'R'),
                          (self.SUPPORTED_POOLS, 'P')]:
            for _type in module_list:
                string_sub[_type] = c
        string_seq = ''.join([string_sub[module.__class__] 
                              for module in self.net])
        rematch = re.compile(r'^(LR(P)?)+LR?$')
        assert rematch.match(string_seq) is not None

    @classmethod
    def fc_net(cls, layer_sizes, nonlinearity=nn.ReLU, dtype=torch.float):
        seq_list = []

        for i in range(len(layer_sizes) - 1):
            in_dim, out_dim = layer_sizes[i], layer_sizes[i + 1]
            seq_list.append(nn.Linear(in_dim, out_dim).to(dtype))
            if i < len(layer_sizes) - 2:
                seq_list.append(nonlinearity())

        net = nn.Sequential(*seq_list)
        return cls(net, dtype=dtype)

    def set_shapes(self, x):
        # Given an input to the net will go through and set shapes
        # self.shapes[i] refers to the input of the i^{th} layer!
        # We'll always omit the first index 
        shapes = []
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                x = x.view(-1, layer.in_features)           
                
            if isinstance(layer, nn.ConvTranspose2d):
                numex = x.shape[0] 
                pix = x.numel() / (numex * layer.in_channels)
                hw = round(math.sqrt(pix))            
                x = x.view(-1, layer.in_channels, hw, hw)     
            shapes.append(x.shape[1:])
            x = layer(x) 
        shapes.append(x.shape[1:])
        self.shapes = shapes

    def partial_forward(self, x, start_idx):
        for layer in self.net[start_idx:]:
            x = layer(x)
        return x

    def forward(self, x):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                x = x.view(-1, layer.in_features)

            if isinstance(layer, nn.ConvTranspose2d):
                numex = x.shape[0] 
                pix = x.numel() / (numex * layer.in_channels)
                hw = round(math.sqrt(pix))            
                x = x.view(-1, layer.in_channels, hw, hw)                

            x = layer(x)
        return x


    def tensorfy_clone(self, x, requires_grad=False):
        """ Clones whatever x is into a tensor with self's datatype """
        if isinstance(x, torch.Tensor):
            return x.clone().detach()\
                    .type(self.dtype).requires_grad_(requires_grad)
        else:
            return torch.tensor(x, dtype=self.dtype, 
                                requires_grad=requires_grad)


    def display_decision_bounds(self, x_range, y_range, density, figsize=(8,8)):
        """ For 2d-input networks, will use EricWong-esque code to 
            build an axes object and plot decision boundaries 
        ARGS:
            x_range  : pair of floats (lo, hi) - denotes x range of the grid
            y_range  : pair of floats (lo, hi) - denotes y range of the grid 
            density : int - number of x,y coords to check 
            figsize : tuple - for custom figure sizes 
        RETURNS:
            ax object
        """
        # Right now only works for functions mapping R2->R2
        assert self.net[-1].out_features == 2# and self.layer_sizes[-1] == 2

        # Compute the grid points
        x_lo, x_hi = x_range
        y_lo, y_hi = y_range
        XX, YY = np.meshgrid(np.linspace(x_lo, x_hi, density), 
                             np.linspace(y_lo, y_hi, density))
        X0 = torch.Tensor(np.stack([np.ravel(XX), np.ravel(YY)]).T)
        y0 = self(X0)
        ZZ = (y0[:,0] - y0[:,1]).view(density, density).data.numpy()

        # Build plot and plot gridpoints
        fig, ax = plt.subplots(figsize=figsize)
        ax.contourf(XX, YY, -ZZ, cmap='coolwarm', 
                    levels=np.linspace(-1000, 1000, 3))
        ax.axis([x_lo, x_hi, y_lo, y_hi])
        return ax       