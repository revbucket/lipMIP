""" Analogous to the hyperbox.py/zonotope.py files, but for linear bounds """

import numpy 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import copy 
import numpy as np 
import numbers 
import utilities as utils 
import gurobipy as gb

from hyperbox import Domain, Hyperbox, BooleanHyperbox

def posmap(C, A_lo, A_hi):
    # Returns C^+A_hi + C^-A_lo 
    return torch.relu(C) @ A_hi - torch.relu(-C) @ A_lo

def negmap(C, A_lo, A_hi):
    # Returns C^-A_hi + C^-A_lo 
    return -torch.relu(-C) @ A_hi + torch.relu(C) @ A_lo

class LinearBounds(Domain): 
    """ Sets like {z | Ax + b <= z <= Cx+d, x in X}
    """
    def __init__(self, dimension, base_set, lb_A=None, lb_b=None, 
                 ub_A=None, ub_b=None, shape=None):
        self.dimension = dimension
        self.base_set = base_set 

        self.lb_A = lb_A
        self.lb_b = lb_b
        self.ub_A = ub_A 
        self.ub_b = ub_b
        self.shape = shape

        self.coord_lbs = None 
        self.coord_ubs = None 


    @classmethod 
    def from_hyperbox(cls, hyperbox):
        dim = hyperbox.dimension 
        eyedim = torch.eye(dim)
        zdim = torch.zeros(dim)
        new_LB=  cls(dimension=dim, 
                   base_set=hyperbox, 
                   lb_A=eyedim, 
                   lb_b=zdim, 
                   ub_A=eyedim, 
                   ub_b=zdim, 
                   shape=hyperbox.shape)
        new_LB._set_lbs_ubs() 
        return new_LB

    @classmethod 
    def from_vector(cls, vec):
        return cls.from_hyperbox(Hyperbox.from_vector(vec))

    @classmethod 
    def from_zonotope(cls, zono):
        gen_shape = zono.generator.shape
        linf_ball = Hyperbox.build_linf_ball(torch.zeros(gen_shape[1]), 1)
        new_LB = cls(zono.dimension, 
                   base_set=linf_ball, 
                   lb_A=zono.generator,
                   lb_b=zono.center, 
                   ub_A=zono.generator, 
                   ub_b=zono.center, 
                   shape=zono.shape)
        new_LB._set_lbs_ubs() 
        return new_LB

    @classmethod
    def cast(cls, obj):
        if isinstance(obj, Hyperbox):
            return cls.from_hyperbox(obj) 
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return cls.from_vector(obj) 
        elif isinstance(obj, Zonotope):
            return cls.from_zonotope(obj)

    def _set_lbs_ubs(self): 
        # Batch linear programs to solve upper/lower bounds
        if not isinstance(self.base_set, Hyperbox): 
            raise NotImplementedError("Later")

        center = self.base_set.center 
        rad_vec = self.base_set.radius 

        self.coord_ubs = (self.ub_A @ center + self.ub_b + 
                          (self.ub_A * rad_vec).abs().sum(dim=1))
        self.coord_lbs = (self.lb_A @ center + self.lb_b - 
                          (self.lb_A * rad_vec).abs().sum(dim=1))

    def set_2dshape(self, shape):
        self.shape = shape 


    def project_2d(self, dir_matrix):
        """ Projects this object onto the 2 provided directions, 
            can then be used to draw the shape 
        """
        lin = nn.Linear(self.dimension, 2, bias=False)
        lin.weight.data = dir_matrix
        return self.map_linear(lin)


    def draw_2d_boundary(self, num_points):
        """ For 2d sets, will draw them by rayshooting along coordinates
        ARGS: 
            num_points : int - number of points to check 
        RETURNS: 
            tensor of shape [num_points, 2] which outlines the boundary
        """
        range_matrix = torch.arange(num_points + 1) / float(num_points) * (2 * np.pi)
        cos_els = range_matrix.cos() 
        sin_els = range_matrix.sin() 

        dir_matrix = torch.stack([cos_els, sin_els]).T 

        # Now need to rayshoot along each of these directions and report the z*
        # i.e., argmax_z <c, z>
        # This means finding each x* for each direction 
        # And then pushing to the direction of each hyperbox at each 

        # So first finding each x^* 

        dir_A = (torch.relu(dir_matrix) @ self.ub_A) -\
                (torch.relu(-dir_matrix) @ self.lb_A)
        # Dir (n, 2)  |  A (2, dim)    | 
        # so Dir_A (n, dim)   and then we have 
        if not isinstance(self.base_set, Hyperbox):
            raise NotImplementedError("ONLY FOR HYPERBOXES NOW") 

        #argmax a^Tx, x in Hyperbox is sign of each coord and then map to 
        # coords 
        argmax_xs = self.base_set.center + self.base_set.radius * dir_A.sign()
        points = []
        for i, argmax_x in enumerate(argmax_xs):
            # the j^th coordinate is based on dir_matrix.sign 
            box_lo = self.lb_A @ argmax_x + self.lb_b
            box_hi = self.ub_A @ argmax_x + self.ub_b
            new_point = box_lo + torch.relu(dir_matrix[i].sign()) * (box_hi - box_lo) 
            points.append(new_point)
        return torch.stack(points)

    def as_hyperbox(self):
        if self.coord_lbs is None or self.coord_ubs is None:
            self._set_lbs_ubs()
        twocol = torch.stack([self.coord_lbs, self.coord_ubs]).T
        box_out = Hyperbox.from_twocol(twocol)
        box_out.set_2dshape(self.shape)
        return box_out        

    def random_point(self, num_points=1, tensor_or_np='tensor', 
                     requires_grad=False):
        # First sample from x, and then sample for each box 

        xs = self.base_set.random_point(num_points=num_points, 
                                        tensor_or_np=tensor_or_np, 
                                        requires_grad=False)

        # Now for each x, need to build a hyperbox and sample within that 
        #A :  i -> o  (o x i), and b is in R o 
        #points : i x n ... this can yield o x n for each 
        # so A @ xs.T is (o, i), (i, n) = (o, n) 
        low_bounds = self.lb_A @ xs.T + self.lb_b.unsqueeze(1)
        hi_bounds = self.ub_A @ xs.T + self.ub_b.unsqueeze(1)

        samples = low_bounds + (hi_bounds - low_bounds) * torch.rand_like(low_bounds)

        if tensor_or_np == 'tensor':
            return samples.T.data.requires_grad_(requires_grad)
        else:
            return utils.as_numpy(samples.T)

    def contains(self, point): 
        """ Assumes we can encode the base set as a gurobi model
        ARGS:
            point - tensor or numpy of size shape.dimension 
        """
        x_namer = utils.build_var_namer('x')
        z_namer = utils.build_var_namer('z')
        if isinstance(self.base_set, Hyperbox):
        	model = gb.Model() 
        	model.setParam('OutputFlag', False)
        	hyperbox_dim = self.base_set.dimension
        	x_vars = []
        	z_vars = []
        	for i in range(hyperbox_dim):
        		x_vars.append(model.addVar(lb=self.base_set.box_low[i], 
        					 			   ub=self.base_set.box_hi[i], 
			          					   name=x_namer(i)))
        	for j in range(self.dimension):
        		z_vars.append(model.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, 
        								   name=z_namer(j)))
        		model.addConstr(z_vars[-1] <= gb.LinExpr(self.ub_A[j], x_vars) + self.ub_b[j])
        		model.addConstr(z_vars[-1] >= gb.LinExpr(self.lb_A[j], x_vars) + self.lb_b[j])
        		model.addConstr(z_vars[-1] == point[j])
        	model.update() 
        	model.optimize() 

        	return model.Status not in [3, 4]


    def map_layer_forward(self, network, i, abstract_params=None):
        layer = network.net[i]
        if isinstance(layer, nn.Linear):
            return self.map_linear(layer, forward=True)
        elif isinstance(layer, nn.Conv2d):
            return self.map_conv2d(network, i, forward=True)
        elif isinstance(layer, nn.ReLU):
            return self.map_relu(**(abstract_params or {}))
        elif isinstance(layer, nn.LeakyReLU):
            return self.map_leaky_relu(layer, **(abstract_params or {}))
        elif isinstance(layer, (nn.Tanh, nn.Sigmoid)):
            raise NotImplementedError("NOT YET SUPPORTED")
        else:
            raise NotImplementedError("unknown layer type", layer) 

    def map_layer_backward(self, network, i, grad_bound, abstract_params):
        layer = network.net[-(i + 1)]
        forward_idx = len(network.net) - i
        if isinstance(layer, nn.Linear):
            return self.map_linear(layer, forward=False)
        elif isinstance(layer, nn.Conv2d):
            return self.map_conv2d(network, forward_idx, forward=False)
        elif isinstance(grad_bound, BooleanHyperbox):
            if isinstance(layer, nn.ReLU):
                return self.map_switch(grad_bound)
            elif isinstance(layer, nn.LeakyReLU):
                return self.map_leaky_switch(layer, grad_bound)
            else:
                pass
        elif isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid)):
            return self.map_elementwise_mult(grad_bound)
        else:
            raise NotImplementedError("unknown layer type", layer)



    def map_linear(self, linear, forward=True):
        """ Takes in a torch.Linear operator and maps this object through 
            the linear map (forward is x->Wx +b, backward is x->W^Tx)
        ARGS:
            linear : nn.Linear object 
            forward: boolean - if False, we map this backward 
        """
        assert isinstance(linear, nn.Linear)
        dtype = linear.weight.dtype

        if forward: 
            new_dimension = linear.out_features
            W = linear.weight 
        else:
            new_dimension = linear.in_features
            W = linear.weight.T 

        lb_A = negmap(W, self.lb_A, self.ub_A) 
        lb_b = negmap(W, self.lb_b, self.ub_b) 
        ub_A = posmap(W, self.lb_A, self.ub_A) 
        ub_b = posmap(W, self.lb_b, self.ub_b)

        if (forward is True) and (linear.bias is not None): 
            lb_b = lb_b + linear.bias              
            ub_b = ub_b + linear.bias

        new_LB = LinearBounds(dimension=new_dimension, base_set=self.base_set,
                              lb_A=lb_A, lb_b=lb_b, ub_A=ub_A, ub_b=ub_b)
        new_LB._set_lbs_ubs()
        return new_LB

    def map_relu(self, alpha=None):
        # Need to build a scalar to multiply each LB/UB by and a new bias to add 
        lbs, ubs = self.coord_lbs, self.coord_ubs
        # By default have the "always on case"
        ub_mult = torch.ones_like(ubs)
        lb_mult = torch.ones_like(ubs)

        ub_scalar = torch.zeros_like(ubs)
        lb_scalar = torch.zeros_like(ubs)

        # Handle the 'off' cases
        off_indices = ubs <= 0
        ub_mult[off_indices] = 0 
        lb_mult[off_indices] = 0 

        # Handle uncertain relus 
        uncertain_idxs = (ubs * lbs) < 0
        all_ub_mults = ubs / (ubs - lbs)
        all_ub_scalars = -(ubs * lbs) / (ubs - lbs)


        ub_mult[uncertain_idxs] = all_ub_mults[uncertain_idxs]
        lb_mult[uncertain_idxs] = all_ub_mults[uncertain_idxs] # ZONOTOPE???

        ub_scalar[uncertain_idxs] = all_ub_scalars[uncertain_idxs]

        new_lb_A = lb_mult.unsqueeze(1) * self.lb_A
        new_lb_b = lb_mult * self.lb_b 
        new_ub_A = ub_mult.unsqueeze(1) * self.ub_A 
        new_ub_b = ub_mult * self.ub_b + ub_scalar

        new_LB = LinearBounds(self.dimension, self.base_set, 
                              lb_A=new_lb_A, lb_b=new_lb_b,
                              ub_A=new_ub_A, ub_b=new_ub_b, 
                              shape=self.shape)
        new_LB._set_lbs_ubs()
        return new_LB


    def map_leaky_relu(self, layer, alpha=0):
        # Need to build a scalar to multiply each LB/UB by and a new bias to add 
        lbs, ubs = self.coord_lbs, self.coord_ubs
        # By default have the "always on case"
        ub_mult = torch.ones_like(ubs)
        lb_mult = torch.ones_like(ubs)

        ub_scalar = torch.zeros_like(ubs)
        lb_scalar = torch.zeros_like(ubs)

        # Handle the 'off' cases
        off_indices = ubs <= 0
        ub_mult[off_indices] = layer.negative_slope
        lb_mult[off_indices] = layer.negative_slope

        # Handle uncertain relus 
        uncertain_idxs = (ubs * lbs) < 0
        all_ub_mults = (ubs - layer.negative_slope * lbs) / (ubs - lbs)
        all_ub_scalars = (layer.negative_slope - 1) * (ubs * lbs) / (ubs - lbs)

        if alpha is None:
            all_lb_mults = torch.ones_like(ubs) * layer.negative_slope
            all_lb_mults[ubs > lbs.abs()] = 1
        else: 
            raise NotImplementedError("No adaptive setting")

        ub_mult[uncertain_idxs] = all_ub_mults[uncertain_idxs]
        lb_mult[uncertain_idxs] = all_lb_mults[uncertain_idxs]

        ub_scalar[uncertain_idxs] = all_ub_scalars[uncertain_idxs]

        new_lb_A = lb_mult.unsqueeze(1) * self.lb_A
        new_lb_b = lb_mult * self.lb_b 
        new_ub_A = ub_mult.unsqueeze(1) * self.ub_A 
        new_ub_b = ub_mult * self.ub_b + ub_scalar

        new_LB = LinearBounds(self.dimension, self.base_set, 
                              lb_A=new_lb_A, lb_b=new_lb_b,
                              ub_A=new_ub_A, ub_b=new_ub_b, 
                              shape=self.shape)
        new_LB._set_lbs_ubs()
        return new_LB


    def map_elementwise_mult(self, grad_bounds):
        """ Maps the bounds by elementwise multiplication from 
            the bounds in grad_bounds. 
        ARGS:
            grad_bounds : a Hyperbox/BooleanHyperbox
        """ 
        grad_bounds = grad_bounds.as_hyperbox()
        grad_low, grad_hi = grad_bounds.box_low, grad_bounds.box_hi

        lbs, ubs = self.coord_lbs, self.coord_ubs 

        ub_mult = torch.zeros_like(ubs)
        lb_mult = torch.zeros_like(ubs) 

        ub_scalar = torch.zeros_like(ubs) 
        lb_scalar = torch.zeros_like(ubs) 

        # Handle the 'positive' cases 
        pos_idxs = lbs >= 0 

        ub_mult[pos_idxs] = grad_hi[pos_idxs] 
        lb_mult[pos_idxs] = grad_low[pos_idxs] 

        # And the 'negative' cases 
        neg_idxs = ubs < 0 
        ub_mult[neg_idxs] = grad_low[neg_idxs] 
        lb_mult[neg_idxs] = grad_hi[neg_idxs]

        # And the other cases 
        q = (lbs * ubs) < 0
        ub_mult[q] = (ubs[q] * grad_hi[q] - lbs[q] * grad_low[q]) / (ubs[q] - lbs[q])
        lb_mult[q] = (ubs[q] * grad_low[q] - lbs[q] * grad_hi[q]) / (ubs[q] - lbs[q])

        ub_scalar[q] = ubs[q] * grad_hi[q] - ub_mult[q] * ubs[q]
        lb_scalar[q] = ubs[q] * grad_low[q] - lb_mult[q] * ubs[q]

        new_lb_A = lb_mult.unsqueeze(1) * self.lb_A
        new_lb_b = lb_mult * self.lb_b + lb_scalar
        new_ub_A = ub_mult.unsqueeze(1) * self.ub_A
        new_ub_b = ub_mult * self.ub_b + ub_scalar

        new_LB = LinearBounds(self.dimension, self.base_set, 
                              lb_A=new_lb_A, lb_b=new_lb_b,
                              ub_A=new_ub_A, ub_b=new_ub_b, 
                              shape=self.shape)
        new_LB._set_lbs_ubs()
        return new_LB



