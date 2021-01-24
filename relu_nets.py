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
import utilities as utils
import gurobipy as gb
import copy

class ReLUNet(nn.Module):
    def __init__(self, layer_sizes=None, bias=True, dtype=torch.float32,
                 manual_net=None):
        super(ReLUNet, self).__init__()
        self.layer_sizes = layer_sizes
        self.dtype = dtype
        self.fcs = []
        self.num_relus = len(layer_sizes) - 2
        self.num_layers = len(layer_sizes) - 1 
        self.bias = bias
        if manual_net is None:
            self.net = self.build_network(layer_sizes)
        else:
            self.net = manual_net
            self._adjust_fcs()

    @classmethod
    def from_sequential(cls, sequential):
        fcs = [_ for _ in sequential if isinstance(_, (nn.Linear, nn.Identity))]
        layer_sizes = [fcs[0].in_features]
        for fc in fcs:
            layer_sizes.append(fc.out_features)
        bias = any(fc.bias is not None for fc in fcs)
        dtype = fcs[0].weight.dtype

        new_net = cls(layer_sizes=layer_sizes, bias=bias, 
                      dtype=dtype, manual_net=sequential)
        new_net.fcs = fcs
        return new_net

    def _adjust_fcs(self):
        """ Collects FC's from self.net and modifies self.fcs """
        fcs = [_ for _ in self.net if isinstance(_, (nn.Linear, nn.Identity))]
        self.fcs = fcs

    def re_init_weights(self):
        self.net = self.build_network(self.layer_sizes)

    def clone(self):
        """ Returns a deepcopy of this object """
        return self.from_sequential(copy.deepcopy(self.net))

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
        self.fcs = []
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

    def classify_np(self, x):
        tens_x = torch.Tensor(x)
        return self.forward(tens_x).max(1)[1].item()

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
        assert self.layer_sizes[0] == 2# and self.layer_sizes[-1] == 2

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

        BEHAVIOR: 
            If my relunet looks like (LR)* L_fin
            Then I want self.this(self.num_layers) -> (L_fin, None)
                        self.this(-1) -> (L_fin, None)
                        self.this(i) -> (L_i, f.relu)
        """

        if i in [self.num_layers -1, -1]:
            return self.fcs[-1], None
        else:
            return self.fcs[i], F.relu

    def get_sign_configs(self, x):
        preacts = [utils.as_numpy(_.squeeze()) for _ in 
                   self(x.view(-1), return_preacts=True)[:-1]]
        return [_ > 0 for _ in preacts]


    def polytope_from_signs(self, signs):
        """ Returns a 'polytope' of the form (A, b) where 
            A is an (m,n)-numpy array and (b) is a numpy vector 
            and this is the set of points 
            {x | Ax <= b} defined by the sign config provided
        ARGS:
            signs: list of numpy boolean arrays corresponding to 
                   which ReLU's are on 
        RETURNS: 
        """
        configs = [torch.tensor(sign).type(self.dtype) for sign in signs]
        lambdas = [torch.diag(config) for config in configs]
        js = [torch.diag(-2 * config + 1) for config in configs]
        # Compute Z_k = W_k * x + b_k for each layer
        wks = [self.fcs[0].weight]
        bks = [self.fcs[0].bias]
        for (i, fc) in enumerate(self.fcs[1:]):
            current_wk = wks[-1]
            current_bk = bks[-1]
            current_lambda = lambdas[i]
            precompute = fc.weight.matmul(current_lambda)
            wks.append(precompute.matmul(current_wk))
            bks.append(precompute.matmul(current_bk) + fc.bias)

        a_stack = []
        b_stack = []
        for j, wk, bk in zip(js, wks, bks):
            a_stack.append(j.matmul(wk))
            b_stack.append(-j.matmul(bk))

        polytope_A = utils.as_numpy(torch.cat(a_stack, dim=0))
        polytope_b = utils.as_numpy(torch.cat(b_stack, dim=0))
        return utils.Polytope(polytope_A, polytope_b)


    def find_feasible_from_signs(self, sign_configs, input_hbox=None):
        """ Finds a feasible differentiable point that has the given 
            ReLU configs. 
        """
        # First check shapes are okay:
        assert len(sign_configs) == self.num_relus
        assert all([len(sign_configs[i]) == self.layer_sizes[i + 1] 
                    for i in range(self.num_relus)])
        # Then build a gurobi model and add constraints for each layer
        with utils.silent():
            model = gb.Model() 

        # Add input keys:
        input_key = 'input'
        input_namer = utils.build_var_namer(input_key)
        input_vars = []
        for i in range(self.layer_sizes[0]):
            if input_hbox is not None:
                lb, ub = input_hbox[i]
            else:
                lb, ub = -gb.GRB.INFINITY, gb.GRB.INFINITY
            input_vars.append(model.addVar(lb=lb, ub=ub, name=input_namer(i)))

        slack_var = model.addVar(lb=0, name='slack')

        # And then iteratively add layers
        lin_vars = input_vars
        for i in range(self.num_relus):
            lin_vars = self._add_layer_to_gurobi_model(i, model, lin_vars, 
                                                       slack_var, sign_configs[i])

        # Add the objective to maximize and then solve
        model.setObjective(slack_var, gb.GRB.MAXIMIZE)
        model.update()
        model.optimize()

        # And handle the outputs
        if model.Status in [3, 4]:
            return None
        else:
            return {'slack': model.getObjective().getValue(),
                    'x': np.array([v.X for v in input_vars]),
                    'model': model}

    def _add_layer_to_gurobi_model(self, layer_num, model, lin_vars, slack_var,
                                   layer_signs):
        """ Adds new variables and new constraints """

        weight = utils.as_numpy(self.fcs[layer_num].weight)
        bias = utils.as_numpy(self.fcs[layer_num].bias)
        output_vars = []
        for i, row in enumerate(weight):
            if layer_signs[i] == True:
                var = model.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)
                model.addConstr(var == gb.LinExpr(row, lin_vars) + bias[i])
                model.addConstr(gb.LinExpr(row, lin_vars) + bias[i] - slack_var >= 0.0)
            else:
                var = model.addVar(lb=0.0, ub=0.0)
                model.addConstr(gb.LinExpr(row, lin_vars) + bias[i] + slack_var <= 0.0)
            output_vars.append(var)
        return output_vars

    def get_layer_num_from_rev(self, rev_layer_num):
        return self.num_layers - rev_layer_num


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



class ConvNet(ReLUNet):
    def __init__(self, sequential, input_chw, dtype=torch.float):
        """ Wrapper for convolutional network:
            can handle Linear, ReLU, and Conv2D layers
        ARGS:
            sequential - list or nn.Sequential of only Linear+conv2D layers
                         (ReLU's and flattenings are added implicitly)
            input_chw - tuple(int) - triple of input channels, height, width
            dtype - torch.float/torch.double
        """
        if isinstance(sequential, list):
            sequential = nn.Sequential(*sequential)
        for layer in sequential:
            if dtype == torch.float:
                layer.float()
            else:
                layer.double()
        assert all(isinstance(layer, (nn.Linear, nn.Conv2d))
                   for el in sequential)
        super(ConvNet, self).__init__(layer_sizes=sequential, dtype=dtype, 
                                      manual_net=sequential)
        self.num_relus = len(self.net) - 1
        self.shapes = None
        self.input_chw = input_chw
        self._setup_shapes()

    def _setup_shapes(self):
        """Keeps track of all the chw's given the input chws for convs """
        shapes = [self.input_chw]
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                shapes.append(utils.conv2d_counter(shapes[-1], layer))
            elif isinstance(layer, nn.Linear):
                shapes.append(layer.out_features)
        self.shapes = shapes

    def get_ith_input_shape(self, i):
        """ Gets the shape of the tensor that gets fed into the 
            i^th linear/conv layer 
        """
        return self.shapes[i]


    def get_parameters(self):
        return [el.parameters() for el in self.net]

    def forward(self, x, return_preacts=False):
        """ Standard forward method for ConvNets. 
        If return_preacts is True, then we collect and return the 
        values before applying each ReLU layer 
        """
        assert x.shape[1:] == self.input_chw

        if isinstance(self.net[0], nn.Linear):
            x = x.view(-1, self.net[0].in_features)

        preacts = []
        for i, layer in enumerate(self.net[:-1]):
            x = layer(x)
            if return_preacts:
                preacts.append(torch.clone(x))
            x = F.relu(x) 
            if (isinstance(self.net[i + 1], nn.Linear) and 
                isinstance(layer, nn.Conv2d)):
                x = x.view(x.size(0), -1)
        x = self.net[-1](x)
        if return_preacts:
            return preacts + [x]
        else:
            return x

    def get_ith_hidden_unit(self, i):
        """ Returns the i^th hidden unit, which, is a pair of 
            (nn.Linear/nn.Conv2d, nn.ReLU) objects, starting at index 0. 
        """
        linear = self.net[i]         
        if 0 <= i < self.num_layers:            
            nonlin = F.relu 
        else:
            nonlin = None
        return (linear, nonlin)


