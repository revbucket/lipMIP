""" Wrappers for building piecewise linear nets
    (generally we'll only worry about ReLU nets here)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt 


class ReLUNet(nn.Module):
    def __init__(self, layer_sizes=None, bias=True, dtype=torch.float32,
                 manual_net=None):
        super(ReLUNet, self).__init__()
        self.layer_sizes = layer_sizes
        self.dtype = dtype
        self.fcs = []
        self.bias = bias
        if manual_net is None:
            self.net = self.build_network(layer_sizes)
        else:
            self.net = manual_net


    @classmethod
    def from_sequential(self, sequential):
        fcs = [_ for _ in sequential if isinstance(_, nn.Linear)]
        layer_sizes = [fcs[0].in_features]
        for fc in fcs:
            layer_sizes.append(fc.out_features)
        bias = any(fc.bias for fc in fcs)
        dtype = fcs[0].dytpe

        new_net = ReLUNet(layer_sizes=layer_sizes, bias=bias, 
                          dtype=dtype, manual_net=sequential)
        new_net.fcs = fcs
        return new_net


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


    def _check_valid_split_parameters(self, split_parameters):
        if split_parameters.manual_splits is None:
            return True 
        else:
            return sum(split_parameters.manual_splits) == len(self.fcs) - 1

    def make_subnetworks(self, split_parameters):
        """ Takes in a SplitParameters object and generates multiple ReLUNets
            in a list. The idea is that if a neural net is denoted as 
            {L: linear layer, R: relu layer}, a ReluNet looks like 
            LRLRLRLRLRL
            So if there are N hidden layers (R's in the above), then we split 
            into chunks like
            - [LR, LR, LR, LR, LRL]
            (where the last one has a linear layer at the output)

        ARGS:
            split_parameters: SplitParameters object which describes how to 
                              split this neural net 
        RETURNS:
            list of ReLUNets, where all except the last one are of the form 
            regex: /(LR)+I/  where {L: linear, R: Relu, I: Identity}
            and the last one is of form /(LR)+L/
        """
        # setup
        assert self._check_valid_split_parameters(split_parameters)
        split_size_list = []
        num_hidden_units = len(self.fcs) - 1

        # Build the split size list
        if split_parameters.num_splits is not None:
            num_splits = split_parameters.num_splits
            split_size = math.ceil(num_hidden_units / float(num_splits))
            split_size_list = [split_size for _ in range(num_splits -1)]
            split_size_list.append(num_hidden_units - sum(split_size_list))

        elif split_pararameters.every_x is not None:
            every_x = int(split_pararameters.every_x)
            split_size_list = [every_x for _ in 
                               range(num_hidden_units // every_x)]
            if num_hidden_units % every_x != 0:
                split_size_list.append(num_hidden_units % every_x)

        else:
            split_size_list = split_parameters.manual_splits

        # Now collect the subnetworks
        start_idx = 0
        subseqs = []
        for subnet_no, size in enumerate(split_size_list):
            end_idx = start_idx + size * 2
            if subnet_no == len(subnetworks) - 1:
                end_idx +=1
            new_seq = self.net[start_idx:end_idx]
            if subnet_no < len(subnetworks) - 1:
                new_seq = utils.seq_append(new_seq, nn.Identity())
            subseqs.append(new_seq)
            start_idx = end_idx

        return [ReLUNet.from_sequential(seq) for seq in subseqs]


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
        assert self.layer_sizes[0] == 2 and self.layer_sizes[-1] == 2

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
