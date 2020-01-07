""" Wrappers for building piecewise linear nets
    (generally we'll only worry about ReLU nets here)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

class ReLUNet(nn.Module):
    def __init__(self, layer_sizes=None, bias=True, dtype=torch.float32):
        super(ReLUNet, self).__init__()
        self.layer_sizes = layer_sizes
        self.dtype = dtype
        self.fcs = []
        self.bias = bias
        self.net = self.build_network(layer_sizes)

    def tensorfy_clone(self, x, requires_grad=False):
        """ Clones whatever x is into a tensor with self's datatype """
        if isinstance(x, torch.Tensor):
            return x.clone().detach()\
                    .type(self.dtype).requires_grad_(requires_grad)
        else:
            return torch.tensor(x, dtype=self.dtype, 
                                requires_grad=requires_grad)

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


    def get_grad_at_point(self, x, c_vector):
        """ Simple helper method to get the gradient of a single input point
            (specifically, gets the gradient of <c_vector, f(x)> wrt x)
        ARGS:
            x: tensor [1xINPUT-SHAPE]
            c_vector: tensor [OUTPUT_DIM] 
        """
        c_vector = self.tensorfy_clone(c_vector)
        x = self.tensorfy_clone(x, requires_grad=True)
        output = self.forward(x).mv(c_vector).sum()
        output.backward()
        return x.grad



    def random_max_grad(self, hyperbox, c_vector, num_random, pnorm=1):
        """ Takes a bunch of random points within the hyperbox and 
            computes the gradient magnitude at each. Records the 
            maximum normed gradient (and point)
        ARGS:
            hyperbox: HyperBox object
            c_vector: tensor [OUTPUT_DIM]
            num_random : int - number of random points 
            pnorm : int, float - Lp norm to take of gradients
        RETURNS:
            {norm, point, grad} that maximizes grad norm 
        """

        max_norm, max_point, max_grad = -1, None, None 
        c_vector = self.tensorfy_clone(c_vector)
        for i in range(num_random):
            random_input = hyperbox.random_point(tensor_or_np='tensor')
            grad = self.get_grad_at_point(random_input, c_vector)
            grad_norm = grad.norm(p=pnorm)
            if grad_norm > max_norm:
                max_norm = grad_norm
                max_point = random_input
                max_grad = grad 
        return {'norm': max_norm,
                'point': max_point,
                'grad': max_grad}
