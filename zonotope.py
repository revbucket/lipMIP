""" Analogous to the hyperbox.py file, but for zonotopes """ 

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


class Zonotope(Domain):
    def __init__(self, dimension, 
                 center=None,
                 generator=None,
                 lbs=None,
                 ubs=None, 
                 shape=None):
        self.dimension = dimension 
        self.center = center # numpy ARRAY
        self.generator = generator # numpy 2D Array (matrix)
        self.lbs = lbs # Array
        self.ubs = ubs # Array 
        self.shape = shape # tuple of 2d-shape (for convs), possibly None
        self._set_lbs_ubs()

    def __getitem__(self, idx):
        return self.lbs[idx], self.ubs[idx]

    @classmethod
    def as_zonotope(cls, abstract_object):
        """ Takes in either a zonotope or hyperbox and returns an equivalent 
            zonotope
        """
        if isinstance(abstract_object, Zonotope):
            return abstract_object
        else:
            return cls.from_hyperbox(abstract_object)

    @classmethod
    def from_hyperbox(cls, hyperbox):
        """ Takes in a Hyperbox object and returns an equivalent zonotope """
        generator = torch.diag(hyperbox.radius)

        return cls(hyperbox.dimension,
                   center=hyperbox.center,
                   generator=generator,
                   lbs=hyperbox.box_low,
                   ubs=hyperbox.box_hi, 
                   shape=hyperbox.shape)

    @classmethod
    def from_vector(cls, vec):
        """ Takes in a vector and makes a hyperbox """
        return cls.from_hyperbox(Hyperbox.from_vector(vec))

    @classmethod
    def cast(cls, obj):
        """ Casts hyperboxes, zonopes, vectors as a zonotope """
        if isinstance(obj, Hyperbox):
            return cls.from_hyperbox(obj)
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return cls.from_vector(obj)
        elif isinstance(obj, cls):
            return obj
        else:
            return obj.as_zonotope()            


    def set_2dshape(self, shape):
        self.shape = shape

    def random_point(self, num_points=1, tensor_or_np='tensor', 
                     requires_grad=False):
        assert tensor_or_np in ['np', 'tensor']
        shape = (num_points, self.generator.shape[1])
        rand = torch.rand((num_points, self.generator.shape[1])) *2 - 1
        rand.type(self.center.dtype)

        points = self.y(rand)

        if tensor_or_np == 'tensor':
            return points.data.requires_grad_(requires_grad)
        else:
            return utils.as_numpy(rand_points)

    def y(self, y_tensor): 
        """ Returns a tensor of points in R^n given their generating ys 
        ARGS: 
            y_tensor: tensor shape (k, m) or (m)
        RETURNS:
            tensor of center + E @ y_tensor
        """
        return self.center + (self.generator @ y_tensor.unsqueeze(-1)).squeeze(-1)


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
        else:
            raise NotImplementedError("unknown layer type", layer)

    def map_layer_backward(self, network, i, grad_bound, abstract_params=None):
        layer = network.net[-(i + 1)]
        forward_idx = len(network.net) - 1
        if isinstance(layer, nn.Linear):
            return self.map_linear(layer, forward=False)
        elif isinstance(layer, nn.Conv2d):
            return self.map_conv2d(network, forward_idx, forward=False)
        elif isinstance(layer, nn.ReLU):
            return self.map_switch(grad_bound, **(abstract_params or {}))
        elif isinstance(layer, nn.LeakyReLU):
            return self.map_leaky_switch(layer, grad_bound, 
                                         **(abstract_params or {}))
        else:
            return NotImplementedError("Unknown layer type", layer)


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
        dtype = linear.weight.dtype 

        if forward:
            new_dimension = linear.out_features
            new_center = linear(self.center)
            new_generator = linear.weight.mm(self.generator)
        else:
            new_dimension = linear.in_features
            new_center = linear.weight.T.mv(self.center)
            new_generator = linear.weight.T.mm(self.generator)
        # Return new zonotope
        new_zono = Zonotope(dimension=new_dimension,
                            center=new_center,
                            generator=new_generator)
        new_zono._set_lbs_ubs()
        return new_zono #.pca_reduction()

    def map_conv2d(self, network, index, forward=True):
        # Set shapes -- these are dependent upon direction
        input_shape = network.get_ith_input_shape(index)
        output_shape = network.get_ith_input_shape(index + 1)
        if not forward:
            input_shape, output_shape = output_shape, input_shape


        conv2d = network.get_ith_hidden_unit(index)[0]
        # Strategy to do forward pass is to map each element of generator
        # through conv without bias (and center gets the bias)
        center = self.center.view((1,) + input_shape)
        generator = self.generator.T.view((-1,) + input_shape)
        gen_cols = self.generator.shape[1]

        if forward:
            new_center = conv2d(center).view(-1)
            new_generator = utils.conv2d_mod(generator, conv2d, bias=False,
                                             abs_kernel=False)
            new_generator = new_generator.view((gen_cols,) + (-1,)).T

        else:
            # Cheat and use torch autograd to do this for me 
            center_in = torch.zeros((1,) + output_shape, requires_grad=True)
            center_out = (conv2d(center_in) * center).sum() 
            new_center = torch.autograd.grad(center_out, center_in)[0].view(-1)

            gen_in = torch.zeros((gen_cols,) + output_shape, 
                                 requires_grad=True)
            gen_out = utils.conv2d_mod(gen_in, conv2d, bias=False, 
                                       abs_kernel=False)
            new_gen = torch.autograd.grad((gen_out * generator).sum(), gen_in)[0]
            new_generator = new_gen.view((gen_cols, -1)).T

        new_zono = Zonotope(dimension=new_center.numel(), center=new_center, 
                            generator=new_generator, shape=output_shape)
        new_zono._set_lbs_ubs()
        return new_zono

    def map_nonlin(self, nonlin):
        if nonlin == F.relu: 
            return self.map_relu()
        else: 
            return None # 


    def map_relu(self, transformer='deep', add_new_cols=True):

        single_method = {'box': self.zbox_single,
                         'diag': self.zdiag_single,
                         'switch': self.zswitch_single,
                         'smooth': self.zsmooth_single,
                         'deep': self.deepz_single}[transformer]
        single_outputs = [single_method(i) for i in range(self.dimension)]
        return self._apply_single_outputs(single_outputs, 
                                          add_new_cols=add_new_cols)

    def map_leaky_relu(self, layer, transformer='deep', add_new_cols=True):
        single_method = {'box': self.zbox_single_leaky,
                         'diag': self.zdiag_single_leaky,
                         'switch': self.zswitch_single_leaky,
                         'deep': self.deepz_single_leaky}[transformer]
        single_outputs = [single_method(i, layer) for i in 
                          range(self.dimension)]
        return self._apply_single_outputs(single_outputs, 
                                          add_new_cols=add_new_cols)


    def map_nonlin_backwards(self, nonlin_obj, grad_bound):
        if nonlin_obj == F.relu:
            if isinstance(grad_bound, BooleanHyperbox):
                return self.map_switch(grad_bound)
        elif nonlin_obj == None:
            return self
        else:
            raise NotImplementedError("ONLY RELU SUPPORTED")


    def map_switch(self, bool_box, transformer='deep', add_new_cols=True):
        """ Returns a new zonotope corresponding to a switch function applied 
            to all elements in self, with the given boolean-hyperbox 
        """
        single_method = {'box': self.sBox_single, 
                         'diag': self.sDiag_single,
                         'deep': self.deepS_single}[transformer]
        single_outputs = [single_method(i, bool_box) 
                          for i in range(self.dimension)]
        return self._apply_single_outputs(single_outputs, 
                                          add_new_cols=add_new_cols)


    def map_leaky_switch(self, layer, bool_box, transformer='deep', 
                         add_new_cols=True):
        """ Returns a new zonotope corresponding to a leaky switch function 
            applied to all elements in self, with the given boolean-hyperbox 
        """

        single_outputs = [self.deepS_single_leaky(i, layer, bool_box) 
                          for i in range(self.dimension)]
        return self._apply_single_outputs(single_outputs, 
                                          add_new_cols=add_new_cols)


    def _apply_single_outputs(self, single_outputs, add_new_cols=True):
        new_center = torch.tensor([_[0] for _ in single_outputs])
        new_generator = torch.stack([_[1] for _ in single_outputs])
        new_cols = [_[2] for _ in single_outputs if _[2] is not None]
        if len(new_cols) > 0 and add_new_cols:
            new_generator = torch.cat([new_generator, 
                                       torch.stack(new_cols).T], dim=1)
        new_zono = Zonotope(dimension=self.dimension,
                            center=new_center,
                            generator=new_generator,
                            shape=self.shape)
        new_zono._set_lbs_ubs()
        return new_zono


    def _set_lbs_ubs(self):
        """ Takes in a Zonotope object without self.lbs, self.ubs set
            and modifies these attributes 
        """
        if self.center is None or self.generator is None:
            return 
        radii = torch.abs(self.generator).sum(1)
        self.lbs = self.center - radii
        self.ubs = self.center + radii 

    def as_hyperbox(self):
        if self.lbs is None or self.ubs is None:
            self._set_lbs_ubs()
        twocol = torch.stack([self.lbs, self.ubs]).T
        box_out = Hyperbox.from_twocol(twocol)
        box_out.set_2dshape(self.shape)
        return box_out

    def pca_reduction(self):
        """ PCA order reduction technique from 
            'Methods for order reduction of zonotopes' (Kopetzki et al)
        """
        # 1) center the generator and compute SVD
        column_mean = torch.mean(self.generator, dim=1, keepdim=True)
        center_gen = self.generator - column_mean
        U, S, V = torch.svd(center_gen, some=True)

        # 2) Map zonotope through the U, and turn into a box 
        new_center = U.T.mv(self.center)
        new_generator = U.T.mm(self.generator)
        intermed_zono = Zonotope(dimension=self.dimension, 
                                 center=new_center,
                                 generator=new_generator, 
                                 shape=self.shape)
        new_zono = Zonotope.from_hyperbox(intermed_zono.as_hyperbox())

        # 3) Go from box back to rotated space
        final_center = U.mv(new_zono.center)
        final_generator = U.mm(new_zono.generator)
        final_zono = Zonotope(dimension=self.dimension,
                              center=final_center,
                              generator=final_generator,
                              shape=self.shape)
        final_zono._set_lbs_ubs()
        return final_zono

    def as_boolean_hbox(self, params=None):
        return BooleanHyperbox.from_zonotope(self)

    def contains(self, points):
        """ runs .contains_point(...) for every point in points """
        if points.dim() == 1:
            return [self.contains_point(points)]
        else:
            return [self.contains_point(_) for _ in points]


    def contains_point(self, point, indices=None):
        """ Takes in a numpy array of length self.dimension and returns True/False
            depending if point is contained in the zonotope.
            We naively just solve this with a linear program 
        ARGS:
            point: np.array - point to check membership of 
            indices: if not None, only looks at the subset of indices
        RETURNS:
            boolean
        """
        eps = 1e-6
        if indices is None:
            indices = range(self.dimension)
        with utils.silent():
            model = gb.Model() 
        model.setParam('OutputFlag', False)

        gb_vars = [model.addVar(lb=-1.0, ub=1.0) 
                   for i in range(self.generator.shape[1])]
        for i, idx in enumerate(indices):
            model.addConstr(point[i].item() - self.center[idx].item() >= 
                            gb.LinExpr(self.generator[idx], gb_vars) - eps)
            model.addConstr(point[i].item() - self.center[idx].item() <=
                            gb.LinExpr(self.generator[idx], gb_vars) + eps)            

        model.update()
        model.optimize()
        return model.Status not in [3, 4]


    def maximize_l1_norm_mip(self, verbose=False, num_threads=2):
        """ naive gurobi technique to maximize the l1 norm of this zonotope
        RETURNS:
            opt_val - float, optimal objective value
         """
        model = self._build_l1_mip_model(verbose=verbose, 
                                         num_threads=num_threads)
        model.optimize()
        return model.ObjBound

    def maximize_l1_norm_lp(self, verbose=False, num_threads=2):
        model = self._build_l1_mip_model(verbose=verbose,
                                         num_threads=num_threads)
        for var in model.getVars():
            if var.VType == gb.GRB.BINARY:
                var.VType = gb.GRB.CONTINUOUS
                var.LB = 0.0 
                var.UB = 1.0
        model.update()
        model.optimize()
        return model.ObjBound

    def _build_l1_mip_model(self, verbose=False, num_threads=2):
        with utils.silent():
            model = gb.Model() 

        tolerance = 1e-6
        if not verbose: 
            model.setParam('OutputFlag', False)
        model.setParam('Threads', num_threads)
        gen_vars = [model.addVar(lb=-1, ub=1, name='y_%08d' % i)
                    for i in range(self.generator.shape[1])]

        # Now add variables for each coordinate
        x_vars = []
        for i, gen_row in enumerate(self.generator):
            x_vars.append(model.addVar(lb=self.lbs[i], ub=self.ubs[i], 
                                       name='x_%08d' % i))
            model.addConstr(gb.LinExpr(gen_row, gen_vars) + self.center[i] == x_vars[-1])

        # Now consider the absolute value of each x_var 
        t_vars = []
        for i, x_var in enumerate(x_vars):
            if self.ubs[i] <= 0:
                t_vars.append(model.addVar(lb=0.0, ub=abs(self.lbs[i]) + tolerance))
                model.addConstr(t_vars[i] == -x_vars[i])
            elif self.lbs[i] >= 0:
                t_vars.append(model.addVar(lb=0.0, ub=abs(self.ubs[i]) + tolerance))
                model.addConstr(t_vars[i] == x_vars[i])
            else:
                t_vars.append(model.addVar(lb=0.0, 
                                           ub=max([self.ubs[i], -self.lbs[i]])))
                bin_var = model.addVar(lb=0, ub=1, vtype=gb.GRB.BINARY)                
                t_var = t_vars[-1]
                x_var = x_vars[i]
                lb, ub = self.lbs[i], self.ubs[i]
                model.addConstr(t_var >= x_var)
                model.addConstr(t_var >= -x_var)
                model.addConstr(t_var <= x_var - 2 *  lb * (1 - bin_var))
                model.addConstr(t_var <= -x_var + 2 * ub * bin_var)                

        model.setObjective(sum(t_vars), gb.GRB.MAXIMIZE)
        model.update()
        return model     


    def maximize_linf_norm(self):
        """ Returns maximal l_inf norm of any point inside this zono 
        """
        return max([max(abs(self.lbs)), max(abs(self.ubs))])


    def check_orthants(self, coords, orthant_list=None):
        """ Checks which orthants are feasible in this zonotope 
        ARGS:
            coords: int[], which coordinate indices we consider the 
                    restriction to 
            orthant_list: None or list of possible orthants (like 
                             ['00', '01', '11,...] 
        RETURNS:
            dict with all orthants checked, and False if infeasible and 
            a proof of orthant intersection otherwise 
        """
        # Generate orthants to check: 
        def generate_power_set(n, init_list=None):
            if init_list is None:
                init_list = ['0', '1']
            if n == 1:
                return init_list
            return generate_power_set(n - 1, [_ + b for _ in init_list 
                                                    for b in ['0', '1']])

        if orthant_list is None:
            orthant_list = generate_power_set(len(coords))

        # Now check each orthant 
        model = None 
        output_dict = {} 
        for orthant in orthant_list: 
            output, model = self._check_single_orthant(coords, orthant, 
                                                       model=model)
            output_dict[orthant] = output

        return output_dict

    def _check_single_orthant(self, coords, single_orthant, model=None):
        """ Checks feasibility of a single orthant wrt some coords
            of the zonotope 
        ARGS:
            coords: int[] list of coordinates to check only 
            single_orthant: binary string that corresponds to which orthants
                            to check
            model: gurobi model (if it exists), only needs to be changed
                   if a new model is proposed 
        """
        assert len(coords) == len(single_orthant)
        x_namer = utils.build_var_namer('x')
        z_constr_namer = utils.build_var_namer('z_constr')
        # Build model if doesn't exist
        if model is None:
            with utils.silent():
                model = gb.Model() 
                model.setParam('OutputFlag', False)
            # Add variables for x, y 
            y_vars = [model.addVar(lb=-1, ub=1) 
                      for _ in range(self.generator.shape[1])]
            x_vars = [model.addVar(lb=self.lbs[i]-1, ub=self.ubs[i] + 1, name=x_namer(i))
                      for i in coords]

            # Add linExpressions for each x,y 
            for index, i in enumerate(coords):
                model.addConstr(self.center[i].item() + gb.LinExpr(self.generator[i], y_vars) ==\
                                x_vars[index])
            z_var = model.addVar(lb=0.0, name='z')
            model.setObjective(z_var, gb.GRB.MAXIMIZE)

        else: # otherwise, just remove the z-constraints
            x_vars = [model.getVarByName(x_namer(i)) for i in coords]
            z_var = model.getVarByName('z')
            for i in coords:
                model.remove(model.getConstrByName(z_constr_namer(i)))
        model.update()

        # Now add new z constraints and optimize
        for index, i in enumerate(coords):
            sign_var = 1
            if single_orthant[index] == '0':
                sign_var = -1
            model.addConstr(z_var <= sign_var * x_vars[index],
                            name=z_constr_namer(i))
        model.setObjective(z_var, gb.GRB.MAXIMIZE)            
        model.update()
        model.optimize()

        # Process the optimization output:
        if model.Status == 2:
            output = (float(model.objVal), np.array([_.X for _ in x_vars]))
        else:
            output = False
        return output, model 


    def encode_as_gurobi_model(self, squire, key):
        model = squire.model 
        namer = utils.build_var_namer(key)
        gb_vars = []
        raise NotImplementedError("Build this later!")
    

    # =======================================================
    # =           Single ReLU Transformer Methods           =
    # =======================================================
    # Each function here returns a (center_coord, gen_row, gen_col)
    # - center_coord is just a float for the i^th coord of new center
    # - gen_row is a new generator row (none if unchanged)
    # - gen_col is a new generator col (none if not needed)

    def _z_known(self, i):
        if self.lbs[i] >= 0:
            return (self.center[i], self.generator[i], None)
        if self.ubs[i] <= 0:
            return (0, torch.zeros(self.generator.shape[1]), None)

    def zbox_single(self, i):
        if self.lbs[i] * self.ubs[i] >= 0:
            return self._z_known(i)

        center_coord = self.ubs[i] / 2.0
        gen_row = torch.zeros(self.generator.shape[1])
        gen_col = torch.zeros(self.dimension)
        gen_col[i] = self.ubs[i] / 2.0

        return (center_coord, gen_row, gen_col)

    def zdiag_single(self, i):
        if self.lbs[i] * self.ubs[i] >= 0:
            return self._z_known(i)        

        center_coord = self.center[i] - self.lbs[i] / 2.0 
        gen_row = self.generator[i]
        gen_col = torch.zeros(self.dimension)
        gen_col[i] = -self.lbs[i] / 2.0
        return (center_coord, gen_row, gen_col)


    def zswitch_single(self, i):
        if abs(self.lbs[i]) > abs(self.ubs[i]):
            return self.zbox_single(i)
        else:
            return self.zdiag_single(i)

    def zsmooth_single(self, i):
        if self.lbs[i] * self.ubs[i] >= 0:
            return self._z_known(i)

        abs_range = abs(self.lbs[0]) + abs(self.ubs[0])
        zbox_c, zbox_row, zbox_col = self.zbox_single(i)
        zbox_weight = abs(self.lbs[0]) / abs_range

        zdiag_c, zdiag_row, zdiag_col = self.zdiag_single(i)
        zdiag_weight = abs(self.ubs[0]) / abs_range

        # compute center
        new_center = zbox_c * zbox_weight + zdiag_c * zdiag_weight

        # compute row
        new_row = zbox_row * zbox_weight + zdiag_row * zdiag_weight

        # compute col 
        if (zbox_col is None) and (zdiag_col is None):
            new_col = None 
        else:
            if zbox_col is None:
                zbox_col = torch.zeros(self.dimension)
            if zdiag_col is None:
                zdiag_col = torch.zeros(self.dimension)
            new_col = zbox_col * zbox_weight + zdiag_col * zdiag_weight

        return (new_center, new_row, new_col)


    def deepz_single(self, i):
        if self.lbs[i] * self.ubs[i] >= 0:
            return self._z_known(i)
        lambda_ = self.ubs[i] / (self.ubs[i] - self.lbs[i])
        mu_ = -1 * self.ubs[i] * self.lbs[i] / (self.ubs[i] - self.lbs[i])
        new_center = lambda_ * self.ubs[i] / 2.0
        # new_center = self.center[i] * lambda_ + mu_ 
        new_row = self.generator[i] * lambda_
        new_col = torch.zeros(self.dimension)
        new_col[i] = mu_/2.0 +1e-6
        return (new_center, new_row, new_col)

    # ======  End of Single ReLU Transformer Methods  =======

    # =======================================================
    # =           Single LEAKY ReLU Transformer Methods     =
    # =======================================================
    # Each function here returns a (center_coord, gen_row, gen_col)
    # - center_coord is just a float for the i^th coord of new center
    # - gen_row is a new generator row (none if unchanged)
    # - gen_col is a new generator col (none if not needed)
    # All of these take args (i, layer)
    # where i is the coordinate index and layer is the LeakyReLU instance 
    def _z_known_leaky(self, i, layer):
        if self.lbs[i] >= 0:
            return (self.center[i], self.generator[i], None)
        if self.ubs[i] <= 0:
            neg = layer.negative_slope
            return (self.center[i] * neg, self.generator[i] * neg, None)

    def zbox_single_leaky(self, i, layer):
        if self.lbs[i] * self.ubs[i] >= 0:
            return self._z_known_leaky(i, layer)
        neg = layer.negative_slope
        full_range = self.ubs[i] + neg * self.lbs[i]
        center_coord = full_range / 2.0 
        gen_row = torch.zeros(self.generator.shape[1])
        gen_col = torch.zeros(self.dimension)
        gen_col[i] = full_range / 2.0 

        retun (center_coord, gen_row, gen_col)

    def zdiag_single_leaky(self, i, layer):
        if self.lbs[i] * self.ubs[i] >= 0:
            return self._z_known_leaky(i, layer)

        neg = layer.negative_slope 
        assert neg < 1.0 # Assumption: leaky relus need slope < 1
        center_coord = self.center[i] + (neg - 1) * self.lbs[i] / 2.0
        gen_row = self.generator[i] 
        gen_col = torch.zeros(self.dimension)
        gen_col[i] = (neg - 1) * self.lbs[i] / 2.0
        return (center_coord, gen_row, gen_col)

    def zswitch_single_leaky(self, i, layer):
        if abs(self.lbs[i]) > abs(self.ubs[i]):
            return self.zbox_single_leaky(i, layer)
        else:
            return self.zdiag_single_leaky(i, layer)

    def deepz_single_leaky(self, i, layer):
        if self.lbs[i] * self.ubs[i] >= 0:
            return self._z_known_leaky(i, layer)

        neg = layer.negative_slope
        assert neg < 1.0
        # Need to establish:
        # 1) slope 
        # 2) y_max (new y range)
        # 3) new center 
        u, l = self.ubs[i], self.lbs[i]
        lambda_ = (u - neg * l) / (u - l)
        ymax = (1 - neg) * (-u * l) / (u - l)

        center_coord = (u * u - neg * l * l) / (2 * (u - l))
        gen_row = self.generator[i] * lambda_
        gen_col = torch.zeros(self.dimension)
        gen_col[i] = ymax / 2.0 

        return (center_coord, gen_row, gen_col)

    # ======  End of Single LEAKY ReLU Transformer Methods  =======

    # ===========================================
    # =           Single SWITCH ReLU            =
    # ===========================================
    # Each function here returns a (center_coord, gen_row, gen_col)
    # - center_coord is just a float for the i^th coord of new center
    # - gen_row is a new generator row (none if unchanged)
    # - gen_col is a new generator col (none if not needed)
    def _s_known(self, i, bool_box, leaky=None):
        if bool_box[i] == 1:
            return (self.center[i], self.generator[i], None)

        if bool_box[i] == -1:
            return (0.0, torch.zeros(self.generator.shape[1]), None)


    def sBox_single(self, i, bool_box):
        if bool_box[i] != 0:
            return self._s_known(i, bool_box)

        gen_row = torch.zeros(self.generator.shape[1])
        gen_col = torch.zeros(self.dimension)
        if self.lbs[i] >= 0:
            center_coord = self.ubs[i] / 2.0
            gen_col[i] = self.ubs[i] / 2.0
        elif self.ubs[i] <= 0:
            center_coord = self.lbs[i] / 2.0 
            gen_col[i] = -self.lbs[i] / 2.0
        else:
            center_coord = (self.ubs[i] + self.lbs[i]) / 2.0 
            gen_col[i] = (self.ubs[i] - self.lbs[i]) / 2.0

        return (center_coord, gen_row, gen_col)

    def sDiag_single(self, i, bool_box):
        if bool_box[i] != 0:
            return self._s_known(i, bool_box)

        gen_row = self.generator[i]
        gen_col = torch.zeros(self.dimension)

        if self.lbs[i] >= 0: 
            center_coord = self.center[i] - self.ubs[i] / 2.0 
            gen_col[i] = self.ubs[i] / 2.0
        elif self.ubs[i] <= 0: 
            center_coord = self.center[i] - self.lbs[i] / 2.0 
            gen_col[i] = -self.lbs[i] / 2.0
        else:
            center_coord = self.center[i] - (self.ubs[i] + self.lbs[i]) / 2.0 
            gen_col[i] = (self.ubs[i] - self.lbs[i]) / 2.0

        return (center_coord, gen_row, gen_col)

    def deepS_single(self, i, bool_box):
        if bool_box[i] != 0:
            return self._s_known(i, bool_box)
        if self.ubs[i] * self.lbs[i] >= 0:
            return self.sBox_single(i, bool_box)

        range_ = self.ubs[i] - self.lbs[i]
        gen_col = torch.zeros(self.dimension)
        if self.ubs[i] >= -self.lbs[i]:
            lambda_ = -self.lbs[i] / range_
            center_coord = (lambda_ * self.center[i] + self.ubs[i] / 2.0 +
                            self.ubs[i] *self.lbs[i] / range_)
            gen_row = lambda_ * self.generator[i]
            gen_col[i] = self.ubs[i] / 2.0
        else:
            lambda_ = self.ubs[i] / range_
            center_coord = (lambda_ * self.center[i] + self.lbs[i] / 2.0 -
                            self.ubs[i] * self.lbs[i] / range_)
            gen_row = lambda_ * self.generator[i] 
            gen_col[i] = -self.lbs[i] / 2.0 


        return (center_coord, gen_row, gen_col)

    # =======================================================
    # =           Single LEAKY SWITCH Transformer Methods     =
    # =======================================================
    # Each function here returns a (center_coord, gen_row, gen_col)
    # - center_coord is just a float for the i^th coord of new center
    # - gen_row is a new generator row (none if unchanged)
    # - gen_col is a new generator col (none if not needed)
    # All of these take args (i, layer, bool_box)
    # where i is the coordinate index and layer is the LeakyReLU instance 

    def _s_known_leaky(self, i, layer, bool_box):
        mult = 1.0 
        if bool_box[i] == -1:
            mult = layer.negative_slope

        return (mult * self.center[i], mult * self.generator[i], None)


    def deepS_single_leaky(self, i, layer, bool_box):
        # If this LeakyReLU is always OFF of ON -- everything becomes constant
        if bool_box[i] != 0:
            return self._s_known_leaky(i, layer, bool_box)

        neg = layer.negative_slope
        assert neg < 1
        l = self.lbs[i]
        u = self.ubs[i]
        med = (l + u) / 2.0
        ymax = (1 - neg) * max([abs(l), abs(u)])
        gen_col = torch.zeros(self.dimension)
        gen_col[i] = ymax / 2.0


        # essentially four cases 
        # Case 1: both l, u >=0 
        #         + slope is neg
        #         + center passes through (u, (1+neg) * u / 2)

        # Case 2: both l, u <= 0
        #         + slope is neg 
        #         + center passes through (l, (1+neg) * l / 2)

        if l * u >= 0: # cases 1 and 2 here
            lambda_ = neg
            gen_row = self.generator[i] * lambda_
            if u > 0:
                center_coord = (1 + neg) * u / 2 + lambda_ * (l - u) / 2.0
            else:
                center_coord = (1 + neg) * l / 2 + lambda_ * (u - l) / 2.0                

        # Cases 3 and 4: l < 0 < u
        # Case 3 occurs when u > |l|
        #        + max-vert is (1-neg) * u
        #        + slope is min([u-neg*l, neg*u-l])/(u-l)
        #        + center passes through (u, (1+neg) * u / 2)

        #Case 4: u < |l|
        #        + max-vert is (1-neg) * |l|
        #        + slope is min([u-neg*l, neg*u-l]) / (u-l)
        #        + center passes through (l, (1+neg) * l / 2)
        else:
            lambda_ = min([u - neg * l, neg * u - l]) / (u - l)
            gen_row = lambda_ * self.generator[i]
            if u > abs(l):
                cx, cy = (u, (1 + neg) * u / 2.0)
            else:
                cx, cy = (l, (1 + neg) * l / 2.0)
            center_coord = cy + lambda_ * (med - cx)


        return (center_coord, gen_row, gen_col)
