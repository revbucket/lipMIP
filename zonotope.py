""" Analogous to the hyperbox.py file, but for zonotopes """ 

import numpy 
import torch 
import torch.nn as nn 
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
                 ubs=None):
        self.dimension = dimension 
        self.center = center # numpy ARRAY
        self.generator = generator # numpy 2D Array (matrix)
        self.lbs = lbs # Array
        self.ubs = ubs # Array 


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
        generator = np.diag(hyperbox.radius)

        return cls(hyperbox.dimension,
                   center=hyperbox.center,
                   generator=generator,
                   lbs=hyperbox.box_low,
                   ubs=hyperbox.box_hi)


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

        center = torch.tensor(self.center, dtype=dtype)
        generator = torch.tensor(self.generator, dtype=dtype)
        if forward: 
            new_dimension = linear.out_features
            new_center = utils.as_numpy(linear(center))
            new_generator = utils.as_numpy(linear.weight.mm(generator))
        else:
            new_dimension = linear.in_features
            new_center = utils.as_numpy(linear.weight.T.mv(center))
            new_generator = utils.as_numpy(linear.weight.T.mm(generator))
        # Return new zonotope
        new_zono = Zonotope(dimension=new_dimension,
                            center=new_center,
                            generator=new_generator)
        new_zono._set_lbs_ubs()
        return new_zono

    def map_conv2d(self, network, index, forward=True):
        conv2d = network.get_ith_hidden_unit(index)
        assert isintsance(conv2d, nn.Conv2d)

        dtype = conv2d.dtype

        center = torch.tensor(self.center, dtype=dtype).view()
        pass


    def map_relu(self, transformer='deep'):

        single_method = {'box': self.zbox_single,
                         'diag': self.zdiag_single,
                         'switch': self.zswitch_single,
                         'smooth': self.zsmooth_single,
                         'deep': self.deepz_single}[transformer]
        single_outputs = [single_method(i) for i in range(self.dimension)]
        return self._apply_single_outputs(single_outputs)



    def map_switch(self, bool_box, transformer='deep'):
        """ Returns a new zonotope corresponding to a switch function applied 
            to all elements in self, with the given boolean-hyperbox 
        """
        single_method = {'box': self.sBox_single, 
                         'diag': self.sDiag_single,
                         'deep': self.deepS_single}[transformer]
        single_outputs = [single_method(i, bool_box) 
                          for i in range(self.dimension)]
        return self._apply_single_outputs(single_outputs)


    def _apply_single_outputs(self, single_outputs):
        new_center = np.array([_[0] for _ in single_outputs])
        new_generator = np.vstack([_[1] for _ in single_outputs])
        new_cols = [_[2] for _ in single_outputs if _[2] is not None]
        if len(new_cols) > 0:
            new_generator = np.hstack([new_generator, np.vstack(new_cols).T])
        new_zono = Zonotope(dimension=self.dimension,
                            center=new_center,
                            generator=new_generator)
        new_zono._set_lbs_ubs()
        return new_zono


    def _set_lbs_ubs(self):
        """ Takes in a Zonotope object without self.lbs, self.ubs set
            and modifies these attributes 
        """
        radii = np.abs(self.generator).sum(1)
        self.lbs = self.center - radii
        self.ubs = self.center + radii 


    def as_hyperbox(self):
        twocol = np.vstack([self.lbs, self.ubs]).T
        return Hyperbox.from_twocol(twocol)

    def as_boolean_hbox(self):
        return BooleanHyperbox.from_zonotope(self)

    def contains(self, point):
        """ Takes in a numpy array of length self.dimension and returns True/False
            depending if point is contained in the zonotope.
            We naively just solve this with a linear program 
        ARGS:
            point: np.array - point to check membership of 
        RETURNS:
            boolean
        """
        with utils.silent():
            model = gb.Model() 
        model.setParam('OutputFlag', False)

        gb_vars = [model.addVar(lb=-1.0, ub=1.0) 
                   for i in range(self.generator.shape[1])]
        for i in range(self.dimension):
            model.addConstr(point[i] - self.center[i] == 
                            gb.LinExpr(self.generator[i], gb_vars))

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
                t_vars.append(model.addVar(lb=0.0, ub=-self.lbs[i]))
                model.addConstr(t_vars[i] == x_vars[i])
            elif self.lbs[i] >= 0:
                t_vars.append(model.addVar(lb=0.0, ub=self.ubs[i]))
                model.addConstr(t_vars[i] == x_vars[i])
            else:
                t_vars.append(model.addVar(lb=0.0, 
                                           ub=max([self.ubs[i], -self.lbs[i]])))
                bin_var = model.addVar(lb=0, ub=1, vtype=gb.GRB.BINARY)                
                t_var = t_vars[-1]
                x_var = x_vars[i]
                lb, ub = self.lbs[i], self.ubs[i]
                model.addConstr(t_var >= x_var - tolerance)
                model.addConstr(t_var >= -x_var - tolerance)
                model.addConstr(t_var <= x_var - 2 *  lb * (1 - bin_var) + tolerance)
                model.addConstr(t_var <= -x_var + 2 * ub * bin_var + tolerance)                

        model.setObjective(sum(t_vars), gb.GRB.MAXIMIZE)
        model.update()
        return model        

    def maximize_linf_norm(self):
        """ Returns maximal l_inf norm of any point inside this zono 
        """
        return max([max(abs(self.lbs)), max(abs(self.ubs))])

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
            return (0, np.zeros(self.generator.shape[1]), None)

    def zbox_single(self, i):
        if self.lbs[i] * self.ubs[i] >= 0:
            return self._z_known(i)

        center_coord = self.ubs[i] / 2.0
        gen_row = np.zeros(self.generator.shape[1])
        gen_col = np.zeros(self.dimension)
        gen_col[i] = self.ubs[i] / 2.0

        return (center_coord, gen_row, gen_col)

    def zdiag_single(self, i):
        if self.lbs[i] * self.ubs[i] >= 0:
            return self._z_known(i)        

        center_coord = self.center[i] - self.lbs[i] / 2.0 
        gen_row = self.generator[i]
        gen_col = np.zeros(self.dimension)
        gen_col[i] = -self.lbs[i] / 2.0


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
                zbox_col = np.zeros(self.dimension)
            if zdiag_col is None:
                zdiag_col = np.zeros(self.dimension)
            new_col = zbox_col * zbox_weight + zdiag_col * zdiag_weight

        return (new_center, new_row, new_col)


    def deepz_single(self, i):
        if self.lbs[i] * self.ubs[i] >= 0:
            return self._z_known(i)

        lambda_ = self.ubs[i] / (self.ubs[i] - self.lbs[i])
        mu_ = -1 * self.ubs[i] * self.lbs[i] / (self.ubs[i] - self.lbs[i])
        new_center = self.center[i] * lambda_ + mu_
        new_row = self.generator[i] * lambda_
        new_col = np.zeros(self.dimension)
        new_col[i] = lambda_ * self.center[i] + mu_ +1e-6
        return (new_center, new_row, new_col)

    # ======  End of Single ReLU Transformer Methods  =======


    # ===========================================
    # =           Single SWITCH ReLU            =
    # ===========================================
    # Each function here returns a (center_coord, gen_row, gen_col)
    # - center_coord is just a float for the i^th coord of new center
    # - gen_row is a new generator row (none if unchanged)
    # - gen_col is a new generator col (none if not needed)
    def _s_known(self, i, bool_box):
        if bool_box[i] == 1:
            return (self.center[i], self.generator[i], None)

        if bool_box[i] == -1:
            return (0, np.zeros(self.generator.shape[1]), None)


    def sBox_single(self, i, bool_box):
        if bool_box[i] != 0:
            return self._s_known(i, bool_box)

        gen_row = np.zeros(self.generator.shape[1])
        gen_col = np.zeros(self.dimension)
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
        gen_col = np.zeros(self.dimension)

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
        gen_col = np.zeros(self.dimension)        
        if self.ubs[i] >= -self.lbs[i]:
            lambda_ = -self.lbs[i] / range_
            center_coord = (lambda_ * self.center[i] + self.ubs[i] / 2.0 +
                            + self.ubs[i] *self.lbs[i] / range_)
            gen_row = lambda_ * self.generator[i]
            gen_col[i] = self.ubs[i] / 2.0
        else:
            lambda_ = self.ubs[i] / range_
            center_coord = (lambda_ * self.center[i] + self.lbs[i] / 2.0 -
                            self.ubs[i] * self.lbs[i] / range_)
            gen_row = lambda_ * self.generator[i] 
            gen_col[i] = -self.lbs[i] / 2.0 


        return (center_coord, gen_row, gen_col)







