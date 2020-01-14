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
        self.num_relus = len(layer_sizes) - 2
        self.bias = bias
        if manual_net is None:
            self.net = self.build_network(layer_sizes)
        else:
            self.net = manual_net



    @classmethod
    def from_sequential(cls, sequential):
        fcs = [_ for _ in sequential if isinstance(_, nn.Linear, nn.Identity)]
        layer_sizes = [fcs[0].in_features]
        for fc in fcs:
            layer_sizes.append(fc.out_features)
        bias = any(fc.bias for fc in fcs)
        dtype = fcs[0].dytpe

        new_net = cls(layer_sizes=layer_sizes, bias=bias, 
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

    def get_ith_hidden_unit(self, i):
        """ Returns the i^th hidden unit, which, is a pair of 
            (nn.Linear, nn.ReLU) objects, starting at index 0. 
        """
        seq_start = 2 * i
        seq_out = self.net[seq_start: seq_start + 2]
        return (seq_out[0], seq_out[1])


class SubReLUNet(ReLUNet):
    """ Expansion pack for ReLUNet which has some extra methods and 
        attributes to aid with construction of split/partitioned subnetworks
    """

    def __init__(self, layer_sizes=None, bias=True, dtype=torch.float32,
                 manual_net=None, target_units=None, parent_units=None,
                 parent_net=None):
        super(SubReLUNet, self).__init__(layer_sizes=layer_sizes, bias=bias,
                                         dtype=dtype, manual_net=manual_net)

        self.target_units = target_units
        self.parent_units = parent_units
        self.parent_net = parent_net

    @classmethod
    def _check_valid_split_parameters(self, split_parameters):
        if split_parameters.manual_splits is None:
            return True
        else:
            return sum(split_parameters.manual_splits) == len(self.fcs) - 1

    @classmethod
    def split_network(cls, network, split_parameters):
        """ Takes in a ReLUNet, SplitParameters object and generates 
            multiple ReLUNets in a list. The idea is that if a neural net is 
            denoted as {L: linear layer, R: relu layer}, a ReluNet looks like 
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
        assert cls._check_valid_split_parameters(split_parameters)
        split_size_list = []
        num_hidden_units = len(network.fcs) - 1

        # Build the split size list and hidden unit indices
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

        # And then collect the "hidden units": the (LR) blocks
        is_relu = lambda lay: isinstance(lay, nn.ReLU)
        hidden_units = utils.partition_by_suffix(network.net, is_relu)


        # Now collect the subnetworks and 'target units'
        start_idx = 0
        target_idxs = []
        parent_idxs = []
        for subnet_no, size in enumerate(split_size_list):
            end_idx = start_idx + size 
            target = [start_idx, end_idx]
            if split_parameters.lookback > 0:
                if target_start < 0:
                    target_start = 0
                start_idx = max([0, start_idx - split_pararameters.lookback])
            if split_parameters.lookahead > 0:
                end_idx = min([len(hidden_units), 
                               end_idx + split_parameters.lookahead])
            target = [abs(_ - start_idx) for _ in target]
            target_idxs.append(target)
            parent_idxs.append((start_idx, end_idx))

        # And append one unit to each of these suffixes
        last_subseq_idx = len(subseqs) - 1
        output = []
        for i, (parent, target) in enumerate(zip(parent_idxs, target_idxs)): 
            # Make the sequential
            subseq = network.net[parent[0]:parent[1]]
            if i == last_subseq_idx:
                suffix_unit = network.net[-1]
            else:
                suffix_unit = nn.Identity()
            seq = utils.flatten_list(subseq) + [suffix_unit]

            # Resolve the 'target_units':
            target_units = split_pararameters.cast_targets(target)
            new_net = SubReLUNet.from_sequential(seq)
            new_net.target_units = target_units
            new_net.parent_units = parent
            new_net.parent_net = network
            output.append(new_net)

        return output
