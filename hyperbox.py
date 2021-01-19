import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import numbers
import utilities as utils


class Domain:
    pass


class LinfBallFactory(object):
    """ Class to easily generate l_inf balls with fixed radius and global bounds
        (but no center)
    """
    def __init__(self, dimension, radius, global_lo=None, global_hi=None):
        self.dimension = dimension
        self.radius = radius
        self.global_lo = global_lo
        self.global_hi = global_hi

    def __call__(self, center):
        return Hyperbox.build_linf_ball(center, self.radius, 
                                        global_lo=self.global_lo,
                                        global_hi=self.global_hi)



class Hyperbox(Domain):
    def __init__(self, dimension):
        self.dimension = dimension

        # in the l_inf ball case
        self.center = None
        self.radius = None

        self.box_low = None # ARRAY!
        self.box_hi = None # ARRAY!

        self.is_vector = False
        self.shape = None # tuple of 2d-shape (for convs), possibly None

    def __iter__(self):
        """ Iterates over twocol version of [box_low, box_high] """
        for el in self.as_twocol():
            yield el

    def __getitem__(self, idx):
        return (self.box_low[idx], self.box_hi[idx])

    # CONSTRUCTOR OVERVIEW: 
    def as_dict(self):
        return {'dimension':                self.dimension,
                'center':                   self.center,
                'radius':                   self.radius,
                'box_low':                  self.box_low,
                'box_hi':                   self.box_hi,
                'is_vector':                self.is_vector,
                'shape':                    self.shape}


    @classmethod
    def from_dict(cls, saved_dict):
        domain = cls(saved_dict['dimension'])
        for s in ['center', 'radius', 'box_low', 'box_hi', 'is_vector',
                  'shape']:
            setattr(domain, s, saved_dict.get(s, None))
        domain._fixup()
        return domain


    @classmethod
    def from_twocol(cls, twocol):
        """ Given a numpy array of shape (m, 2), creates an m-dimensional 
            hyperbox 
        ARGS:
            twocol: np array w/ shape (m, 2)
        RETURNS: 
            instance of Hyperbox
        """
        dimension = twocol.shape[0]
        center = (twocol[:, 0] + twocol[:, 1]) / 2.0
        radius =  torch.max(abs(center - twocol[:, 0]), 
                            abs(center - twocol[:, 1]))
        hbox_out = Hyperbox.from_dict({'dimension': dimension, 
                                       'center': center, 
                                       'radius': radius, 
                                       'box_low': twocol[:, 0], 
                                       'box_hi':  twocol[:, 1],
                                       'is_vector': False})
        hbox_out._fixup()
        return hbox_out


    @classmethod 
    def from_midpoint_radii(cls, midpoint, radii, shape=None):
        """ Takes in two numpy ndarrays and builds a new Hyperbox object
        ARGS:
            midpoint : np.ndarray describing the center of a hyperbox
            radii : np.ndarray describing the coordinate-wise range
                e.g. H_i in [midpoint_i - radii_i, midpoint_i + radii_i]
        RETURNS:
            hyperbox object 
        """
        new_hbox = Hyperbox(len(midpoint))
        new_hbox.box_low = midpoint - radii 
        new_hbox.box_hi = midpoint + radii 
        new_hbox.shape = shape
        new_hbox._fixup()
        return new_hbox

    @classmethod
    def from_vector(cls, c):
        """ Takes in a single numpy array and denotes this as the 
            hyperbox containing that point 
        """
        c = utils.tensorfy(c)
        new_hbox = cls.from_dict({'center': c, 'radius': 0.0, 
                                  'box_low': c, 'box_hi': c,
                                  'dimension': len(c), 'is_vector': True})
        new_hbox._fixup()
        return new_hbox


    # ==============================================================
    # =           Forward facing methods                           =
    # ==============================================================

    @classmethod
    def build_unit_hypercube(cls, dim):
        return cls.build_linf_ball(np.ones(dim) * 0.5, 0.5)

    @classmethod
    def build_linf_ball(cls, x, radius, global_lo=None, global_hi=None):
        """ Case we mostly care about -- builds an L_infinity ball centered
            at x with specified radius and also intersects with hyperbox
            with specified global lo and hi bounds
        ARGS:
            x: np.Array or torch.Tensor - center of the linf ball
            radius: float - size of L_inf ball
            global_lo: float or np.Array/torch.Tensor (like x) -
                       lower bounds if there's a domain we care about too
            global_hi : float or np.Array/torch.Tensor (like x) - upper bounds
                        if there's a domain we care about too
        RETURNS:
            Domain object
        """
        x_tensor = utils.tensorfy(x)
        shape = tuple(x_tensor.shape)
        x_tensor = x_tensor.view(-1)
        domain = cls(len(x_tensor))
        domain.center = x_tensor
        domain.radius = radius
        domain.set_2dshape(shape)
        domain._fixup()
        return domain

    def get_center(self):
        if self.center is not None:
            return self.center
        return (self.box_low + self.box_high)/2.0

    def set_2dshape(self, shape):
        self.shape = shape

    def as_hyperbox(self):
        return self 

        
    def random_point(self, num_points=1, tensor_or_np='tensor', 
                     requires_grad=False):
        """ Returns a uniformly random point in this hyperbox
        ARGS:
            num_points: int - number of output points to return
            tensor_or_np: string ['np' or 'tensor'] - decides whether to
                          return a torch.Tensor or a numpy array
        RETURNS:
            (numpy array | tensor) of w/ shape (num_points, self.x.shape[0])
        """

        assert tensor_or_np in ['np', 'tensor']
        diameter = (self.box_hi - self.box_low)
        rands = torch.rand_like(self.center.expand(num_points, self.dimension))
        rand_points = rands * diameter + self.box_low

        if tensor_or_np == 'tensor':
            return rand_points.requires_grad_(requires_grad)
        else:
            return utils.as_numpy(rand_points)

    def as_twocol(self, tensor_or_np='tensor'):
        twocol = torch.stack([self.box_low, self.box_hi]).T 
        if tensor_or_np == 'tensor':
            return twocol
        else:
            return twocol.numpy()


    def map_layer_forward(self, network, i, abstract_params=None):
        layer = network.net[i]
        if isinstance(layer, nn.Linear):
            return self.map_linear(layer, forward=True)
        elif isinstance(layer, nn.Conv2d):
            return self.map_conv2d(network, i, forward=True)
        elif isinstance(layer, nn.ReLU):
            return self.map_relu()
        elif isinstance(layer, nn.LeakyReLU):
            return self.map_leaky_relu(layer)
        else:
            raise NotImplementedError("unknown layer type", layer)


    def map_layer_backward(self, network, i, grad_bound, abstract_params):
        layer = network.net[-(i + 1)]
        forward_idx = len(network.net) - i
        if isinstance(layer, nn.Linear):
            return self.map_linear(layer, forward=False)
        elif isinstance(layer, nn.Conv2d):
            return self.map_conv2d(network, forward_idx, forward=False)
        elif isinstance(layer, nn.ReLU):
            return self.map_switch(grad_bound)
        elif isinstance(layer, nn.LeakyReLU):
            return self.map_leaky_switch(layer, grad_bound)
        else:
            raise NotImplementedError("unknown layer type", layer)


    def map_genlin(self, linear_layer, network, layer_num, forward=True):
        if isinstance(linear_layer, nn.Linear):
            return self.map_linear(linear_layer, forward=forward)
        elif isinstance(linear_layer, nn.Conv2d):
            return self.map_conv2d(network, layer_num, forward=forward)
        else:
            raise NotImplementedError("Unknown linear layer", linear_layer)


    def map_linear(self, linear, forward=True):
        """ Takes in a torch.Linear operator and maps this object through 
            the linear map (either forward or backward)
        ARGS:
            linear : nn.Linear object - 
            forward: boolean - if False, we map this 'backward' as if we
                      were doing backprop
        """
        assert isinstance(linear, nn.Linear)
        midpoint = (self.box_hi + self.box_low) / 2.0
        radii = (self.box_hi - self.box_low) / 2.0
        dtype = linear.weight.dtype
        midpoint = utils.tensorfy(midpoint)
        radii = utils.tensorfy(radii)
        if forward:
            new_midpoint = linear(midpoint)
            new_radii = F.linear(radii, torch.abs(linear.weight), None)
        else:
            new_midpoint = F.linear(midpoint, linear.weight.T, None)
            new_radii = F.linear(radii, linear.weight.T.abs())
        return Hyperbox.from_midpoint_radii(new_midpoint, new_radii)# ._dilate()

    def map_conv2d(self, network, index, forward=True):
        # Set shapes -- these are dependent upon direction
        input_shape = network.get_ith_input_shape(index)
        output_shape = network.get_ith_input_shape(index + 1)
        if not forward:
            input_shape, output_shape = output_shape, input_shape

        conv2d = network.get_ith_hidden_unit(index)[0]
        assert isinstance(conv2d, nn.Conv2d)
        dtype = conv2d.weight.dtype
        midpoint = self.center.view((1,) + input_shape)
        radii = self.radius.view((1,) + input_shape)

        if forward:
            new_midpoint = conv2d(midpoint).view(-1)
            new_radii = utils.conv2d_mod(radii, conv2d, bias=False, 
                                         abs_kernel=True).view(-1)
        else:
            # Cheat and use torch autograd to do this for me
            mid_in = torch.zeros((1,) + output_shape, requires_grad=True)
            mid_out = (conv2d(mid_in) * midpoint).sum()
            new_midpoint = torch.autograd.grad(mid_out, mid_in)[0].view(-1)


            rad_in = torch.zeros((1,) + output_shape, requires_grad=True)
            rad_out = utils.conv2d_mod(rad_in, conv2d, abs_kernel=True)
            new_radii = torch.autograd.grad((rad_out * radii).sum(), 
                                            rad_in)[0].view(-1)

        hbox_out = Hyperbox.from_midpoint_radii(new_midpoint, new_radii,
                                            shape=output_shape)
        return hbox_out

    def map_nonlin(self, nonlin):
        if nonlin == F.relu: 
            return self.map_relu()
        else: 
            return None # 

    def map_relu(self, **pf_kwargs):
        """ Returns the hyperbox attained by mapping this hyperbox through 
            elementwise ReLU operators
        """
        twocol = self.as_twocol(tensor_or_np='tensor')
        new_bounds = torch.max(twocol, torch.zeros_like(twocol))
        box_out = Hyperbox.from_twocol(new_bounds)
        box_out._fixup()
        box_out.shape = self.shape
        return box_out # ._dilate()

    def map_leaky_relu(self, layer, **pf_kwargs):
        twocol = self.as_twocol(tensor_or_np='tensor')
        box_out = Hyperbox.from_twocol(layer(twocol))
        box_out._fixup()
        box_out.shape = self.shape
        return box_out

    def map_nonlin_backwards(self, nonlin_obj, grad_bound):
        if nonlin_obj == F.relu:
            if isinstance(grad_bound, BooleanHyperbox):
                return self.map_switch(grad_bound)
        elif nonlin_obj == None:
            return self
        else:
            raise NotImplementedError("ONLY RELU SUPPORTED")

    def map_switch(self, bool_box):
        return bool_box.map_switch(self)#._dilate()

    def map_leaky_switch(self, layer, bool_box):
        return bool_box.map_switch(self, layer.negative_slope)


    def encode_as_gurobi_model(self, squire, key):
        model = squire.model 
        namer = utils.build_var_namer(key)
        gb_vars = []
        for i, (lb, ub) in enumerate(self):
            gb_vars.append(model.addVar(lb=lb, ub=ub, name=namer(i)))
        squire.set_vars(key, gb_vars)
        squire.update()
        return gb_vars

    def contains(self, point):
        """ Returns True if the provided point is in the hyperbox 
        If point is a [N x dim] tensor, it returns the boolean array of 
        this being true for all points
        """
        point = utils.tensorfy(point)
        if point.dim() == 1:
            point = point.view(1, -1)
        lo_true = (point >= self.box_low.expand_as(point)).all(dim=1)
        hi_true = (point <= self.box_hi.expand_as(point)).all(dim=1)
        truths = lo_true & hi_true
        if truths.numel == 1:
            return truths.item()
        return truths

    def as_boolean_hbox(self, params=None):
        return BooleanHyperbox.from_hyperbox(self)

    def _dilate(self, eps=1e-6):
        print("_DILATE", eps)
        self.radius += eps 
        self._fixup
        return self

    @classmethod
    def cast(cls, obj):
        """ Casts hyperboxes, zonotopes, vectors as a hyperbox
            (smallest bounding hyperbox in the case of zonos) """

        if isinstance(obj, cls):
            return obj 
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return cls.from_vector(obj)
        else:
            return obj.as_hyperbox()


    def maximize_norm(self, norm='l1'):
        """ Maximizes the l1/linf norm of the hyperbox 
        ARGS:
            norm : str - either 'l1' or 'linf', decides which norm we maximize 
        RETURNS:
            float - maximum norm of the hyperbox 
        """

        assert norm in ['l1', 'linf']
        abs_twocol = self.as_twocol().abs()
        if norm == 'l1': 
            return abs_twocol.max(1)[0].sum().item()
        else:
            return abs_twocol.max().item()


    # ==========================================================================
    # =           Helper methods                                               =
    # ==========================================================================

    def _fixup(self):
        if self.center is None:
            self.center = (self.box_low + self.box_hi) / 2.0 
            self.radius = self.box_hi - self.center
        else:
            self.box_low = self.center - self.radius
            self.box_hi = self.center + self.radius

        if isinstance(self.radius, numbers.Number):
            self.radius = torch.ones_like(self.center) * self.radius

        self.box_low = self.box_low.data
        self.box_hi = self.box_hi.data
        self.center = self.center.data
        self.radius = self.radius.data


    def _add_box_bound(self, val, lo_or_hi='lo'):
        """ Adds lower bound box constraints
        ARGS:
            val: float or torch.tensor(self.dimension) -- defines the 
                 coordinatewise bounds
            lo_or_hi: string ('lo' or 'hi') -- defines if these are lower or 
                      upper bounds
        RETURNS:
            None
        """
        if isinstance(val, numbers.Real):
            val = self._number_to_arr(val)

        attr, comp = {'lo': ('box_low', torch.max),
                      'hi': ('box_hi', torch.min)}[lo_or_hi]

        if getattr(self, attr) is None:
            setattr(self, attr, val)
        else:
            setattr(self, attr, comp(getattr(self, attr), val))
        return None


    def _number_to_arr(self, number_val):
        """ Converts float to array of dimension self.dimension """
        assert isinstance(number_val, numbers.Real)
        return torch.ones_like(self.center) * number_val




class BooleanHyperbox:
    """ Way to represent a vector of {-1, ?, 1} as a boolean 
        hyperbox. e.g., [-1, ?] = {(-1, -1), (-1, +1)}
    """
    @classmethod
    def relu_grad(cls, obj, params):
        return obj.as_boolean_hbox(params)

    @classmethod
    def from_hyperbox(cls, hbox):
        """ Takes a hyperbox and represents the orthants it resides in
        """
        values = torch.zeros(hbox.dimension, dtype=torch.int8)
        values[hbox.box_low > 0] = 1
        values[hbox.box_hi < 0] = -1
        return BooleanHyperbox(values)

    @classmethod
    def from_zonotope(cls, zonotope):
        """ Takes a zonotope and represents the orthants in resides in """
        values = torch.zeros(zonotope.dimension, dtype=torch.int8)
        values[zonotope.lbs > 0] = 1
        values[zonotope.ubs < 0] = -1
        return BooleanHyperbox(values)


    def __init__(self, values):
        """ Values gets stored as its numpy array of type np.int8 
            where all values are -1, 0, 1 (0 <=> ? <=> {-1, +1})
        """
        self.values = utils.tensorfy(values).type(torch.int8)
        self.dimension = len(self.values)

    def __getitem__(self, idx):
        return self.values[idx]

    def __iter__(self):
        for value in self.values:
            yield value

    def map_switch(self, hyperbox, leaky_value=0.0):
        """ Maps a hyperbox through elementwise switch operators
            where the switch values are self.values. 
        In 1-d switch works like this: given interval I and booleanbox a
        SWITCH(I, a): = (0.,0.)                        if (a == -1)
                        I                            if (a == +1)
                        (min(I[0], 0.), max(I[1], 0.)) if (a == 0)
        [CAVEAT: if leaky_value != 0, replace 0.^ with leaky_value]
        ARGS:
            hyperbox: hyperbox governing inputs to switch layer 
            leaky_value : negative slope for a leaky ReLU
        RETURNS: 
            hyperbox with element-wise switch's applied
        """
        eps = 1e-7
        switch_off = self.values < 0
        switch_on = self.values > 0
        switch_q = self.values == 0

        # On case by default
        new_lows = torch.clone(hyperbox.box_low)
        new_highs = torch.clone(hyperbox.box_hi)

        # Handle the off case
        new_lows[switch_off] *= leaky_value
        new_highs[switch_off] *= leaky_value

        # Handle the uncertain case
        new_lows[switch_q & (hyperbox.box_low > 0)] *= leaky_value
        new_highs[switch_q & (hyperbox.box_hi < 0)] *= leaky_value


        # Dilate just a little bit for safety 
        new_lows -= eps
        new_highs += eps 
        # And combine to make a new hyperbox
        box_out = Hyperbox.from_twocol(torch.stack([new_lows, new_highs]).T)
        box_out.shape = hyperbox.shape
        return box_out

    def map_leaky_switch(self, hyperbox, leaky_relu):
        """ Maps a hyperbox through elementwise leaky-switch operators
        In 1-d, leaky switch works like this: given interval I and boolbox a,
        (let r be the slope of the negative part)
        LEAKYSWITCH(I, a) := (r, r)                             if (a == -1)
                             I                                  if (a == +1)
                             (min(I[0], r), max(I[1], r))       if (a == 0)
        ARGS:
            hyperbox: hyperbox governing inputs to leaky-switch layer 
        RETURNS:
            hyperbox with element-wise switch's applied
        """
        eps = 1e-8


    def zero_val(self):
        # Returns a boolean hbox with all values set to zero 
        return BooleanHyperbox(torch.zeros_like(self.values))
        




