""" Main file to contain lipschitz maximization """

import numpy as np
import gurobipy as gb
import utilities as utils
from hyperbox import Hyperbox, LinfBallFactory, Domain
from pre_activation_bounds import PreactivationBounds
import time
import pprint
import re
from interval_analysis import AbstractNN, VALID_PREACTS
import torch.nn as nn
import bound_prop as bp
from general_net import GenNet
import torch

"""

   Lipschitz Maximization as a mixed-integer-program

   Ultimately we'll want to maximize ||grad(f)||_1 over
   either the entire domain or some specified input domain

   We'll do this in three steps:
   1) Encode the neural network forward pass with linear
      constraints and 1 0/1 integer per ReLU
   2) Encode the gradient as n more variables with integral
      constraints controlling the on/off-ness
   3) Take the backprop values and generate the desired norm

   Throughout, we'll store variables in a GurobiSquire object
   for easy access

"""




# =======================================================
# =           MAIN SOLVER BLOCK                         =
# =======================================================

class LipMIP(utils.ParameterObject):
    """ Object that holds ALL the parameters for a lipschitz 
        problem, but doesn't solve it until we tell it to.
    Specifically, we solve :
    max          ||J(x)c||_dualNorm == ||LRLRLRLRLC||_dualNorm
    x in domain

    where J(x) is the jacobian for the neural net

    List of Parameters to solve Lipschitz problem:

    """

    def __init__(self, network, abstract_params, primal_norm='linf', 
                 verbose=False, timeout=None, mip_gap=None, 
                 num_threads=None):
        """
        - network : GenNet object we're solving for. Should pass structural
                    checks
        - abstract_params: AbstractParams instance - contains parameters for 
                           how to do the forward/backward bound propagation 
        - primal_norm : str in {'linf', 'l1'} - determines which norm we 
                        maximize at the end. 
                        e.g. primal_norm 'linf' means we maximize l1 norm
        - verbose : bool - do we print things during the run? 
        - timeout : float or None - max number of seconds we'll run MIP for 
        - mip_gap : float or None - gap at which we terminate the MIP 
        - num_threads : int or None - number of threads Gurobi uses     
        """
        assert isinstance(network, GenNet)
        self.network = network 

        assert isinstance(abstract_params, bp.AbstractParams)
        self.abstract_params = abstract_params

        assert primal_norm in ['linf', 'l1'] 
        self.primal_norm = primal_norm

        self.verbose = verbose
        self.timeout = timeout
        self.mip_gap = mip_gap
        self.num_threads = num_threads

    def _attach_result(self, result):
        self.result = result
        self.value = result.value 
        self.compute_time = result.compute_time

    def build_gurobi_squire(self, input_range, backward_range):
        """ Builds the gurobi squire """
        network = self.network
        abstract = bp.AbstractNN2(self.network)

        # First compute the forward/backward activations:
        forward, backward = abstract.get_both_bounds(self.abstract_params, 
                                                     input_range, 
                                                     backward_range)
        squire = self.build_gurobi_model(forward, backward)
        return squire, forward, backward

    def compute_max_lipschitz(self, input_range, backward_range):
        """ Computes the maximum lipschitz constant with a fixed
            domain already set.

        ARGS:
            input_range : some abstract domain - range of inputs we're 
                          restricting lipschitz computation to. Usually this
                          is some l_p ball around a fixed input, could be 
                          the whole domain tho
            backward_range: some abstract domain - range of c-vectors we use 
                            to compute max lipschitz for. For scalar valued 
                            networks, this can just be a vector
        RETURNS:
            LipMipResult instance
        """
        timer = utils.Timer()
        squire, forw, back = self.build_gurobi_squire(input_range, 
                                                      backward_range)
        model = squire.model
        if not self.verbose:
            model.setParam('OutputFlag', False)
        if self.timeout is not None:
            model.setParam('TimeLimit', self.timeout)
        if self.mip_gap is not None:
            model.setParam('MIPGap', self.mip_gap)
        model.setParam('Threads', getattr(self, 'num_threads') or 4)

        model.optimize()
        if model.Status in [3, 4]:
            print("INFEASIBLE")

        runtime = timer.stop()
        x_vars = squire.get_vars('x')
        value = model.ObjBound #model.getObjective().getValue()
        best_x = np.array([v.X for v in x_vars])
        result = LipResult(network=self.network, 
                           abstract_params=self.abstract_params, 
                           primal_norm=self.primal_norm,
                           forward_bounds=forw, 
                           backward_bounds=back,
                           compute_time=runtime,
                           squire=squire,
                           model=model, 
                           best_x=best_x,
                           value=value)

        self._attach_result(result)
        return result
        # ======  End of MAIN SOLVER BLOCK                =======


    def build_gurobi_model(self, forward_bounds, backward_bounds):
        """ Builds the gurobi squire instance holding the details to 
            run the MIP 
        ARGS:
            forward_bounds : bp.BoundPropForward - instance describing sound
                             approximations to each outputted layer of the NN
            backward_bounds : bp.BoundPropBackward - instance describing 
                              sound apporx to each output layer in the 
                              backward pass (chain rule) of the NN 
        """
        with utils.silent():
            model = gb.Model() 

        # Build the placeholder squire
        squire = GurobiSquire(self.network, model, 
                              forward_bounds, backward_bounds)

        # Iteratively build up the constraints and set objective
        squire.build_input_constraints(forward_bounds, 'x')
        squire.build_forward_pass_constraints()
        squire.build_input_constraints(backward_bounds, 'c_vec')
        squire.build_backward_pass_constraints()
        squire.build_objective(self.primal_norm)
        model.update()

        return squire


# ==============================================================
# =           Result Object of Lipschitz Computation           =
# ==============================================================
class LipResult:
    def __init__(self, network, abstract_params, primal_norm, 
                 forward_bounds, backward_bounds, compute_time, 
                 squire, model, best_x, value):
        self.network = network 
        self.abstract_params = abstract_params
        self.primal_norm = primal_norm
        self.forward_bounds = forward_bounds
        self.backward_bounds = backward_bounds
        self.compute_time = compute_time 
        self.squire = squire
        self.model = model 
        self.best_x = best_x
        self.value = value

    def print_result(self, name='', param_list=None):
        if param_list is None:
            param_list = ['value', 'best_x', 'compute_time']
        print("LipMIP Output for network: ", name)
        for attr in param_list:
            print("\t %s" % attr, getattr(self, attr))


# ==============================================================
# =           Build Gurobi Model for Lipschitz Comp.           =
# ==============================================================

class GurobiSquire():
    # SOME FIXED STRINGS
    INPUT_NAME = 'x'
    LOGITS_NAME = 'logits'
    C_VEC_NAME = 'c_vec'
    GRAD_NAME = 'gradients'
    FORWARD_CONT_VARS = 'layer_%s_post'
    INT_VARS = 'int_%s_post'
    BACKWARD_CONT_VARS = 'backpass_%s_post'
    ABS_INT_VARS = 'abs_ints'
    ABS_CONT_VARS = 'abs_conts'

    def __init__(self, network, model, forward_bounds, backward_bounds):
        self.var_dict = {} 
        self.network = network
        self.model = model 
        self.forward_bounds = forward_bounds
        self.backward_bounds = backward_bounds

    # -----------  Variable Getter/Setters  -----------

    def get_vars(self, name):
        return self.var_dict[name]

    def get_nonlin_vars(self, layer_num):
        return self.var_dict[self.INT_VARS % layer_num]

    def set_vars(self, name, var_list):
        self.var_dict[name] = var_list

    def var_lengths(self):
        # Debug tool
        len_dict = {k: len(v) for k, v in self.var_dict.items()}
        pprint.pprint(len_dict)

    # -----------  Pre-bound getter/setters  -----------

    def get_ith_relu_box(self, i):
        # Returns Hyperbox bounding inputs to i^th relu layer
        return self.forward_bounds.get_forward_box(i)


    def get_ith_switch_box(self, i):
        # Returns BooleanHyperbox bounding values of i^th relu's int vars
        return self.backward_bounds.grad_ranges[i]

    def get_ith_backward_box(self, i):
        # Returns Hyperbox bounding inputs to i^th (forward index!) backswitch
        return self.backward_bounds.get_backward_box(i)

    # -----------  Other auxiliary methods   -----------

    def update(self):
        self.model.update()

    def _check_feas(self):
        self.model.setObjective(0)
        self.update()        
        self.model.optimize()
        return self.model.Status

    def clone(self):
        """ Makes a copy of this object, but not a deepcopy:
        Specifically, keeps the same preact_object (and network/hyperbox objects)
        but makes a new gurobi model and updates the var dict
        """
        self.model.update()
        model_clone = self.model.copy()
        new_squire = GurobiSquire(model_clone)
        for k, v in self.var_dict.items():
            if isinstance(v, list):
                new_v = [model_clone.getVarByName(_.varName) for _ in v]
            if isinstance(v, dict):
                new_v = {k2: model_clone.getVarByName(v2.varName) 
                         for k2, v2 in v.items()}
            new_squire.set_vars(k, new_v)
        return new_squire


    def get_var_at_point(self, x, var_key=None):
        """ Computes the variables prefixed by var_key at a given point.
            This does not modify any state (makes a copy of the gurobi model
            prior to changing it)
        ARGS:
            x: np.array - np array of the requisite input shape 
            reset: bool - if True, we remove the input constraints and update 
        RETURNS:
            var (as a numpy array), but also modifies the model
        """
        var_key = var_key or self.LOGITS_NAME
        model = self.model.copy()
        x_var_namer = utils.build_var_namer(self.INPUT_NAME)
        constr_namer = lambda x_var_name: 'fixed::' + x_var_name
        for i in range(len(self.get_vars(self.INPUT_NAME))):
            x_var = model.getVarByName(x_var_namer(i))
            constr_name = constr_namer(x_var.varName)
            constr = model.getConstrByName(constr_name)
            if constr is not None:
                model.remove(constr)
            model.addConstr(x_var == x[i], name=constr_name)

        model.setObjective(0)
        model.update()
        model.optimize()
        output_namer = utils.build_var_namer(var_key)
        output_num = len(self.get_vars(var_key))
        return [model.getVarByName(output_namer(i)).X 
                for i in range(output_num)]

    def get_grad_at_point(self, x):
        """ Gets gradient at point x"""
        return self.get_var_at_point(x, self.GRAD_NAME)


    def get_logits_at_point(self, x):
        """ Gets logit at point x """
        return self.get_var_at_point(x, self.LOGITS_NAME)


    def lp_ify_model(self, tighter_relu=False):
        """ Converts this model to a linear program. 
            If tighter_relu is True, we add convex upper envelope constraints 
            for all ReLU's, otherwise we just change binary variables to 
            continous ones. 
        RETURNS:
            gurobi model object (does not change self at all)
        """
        self.model.update()
        model_clone = self.model.copy()
        for var in model_clone.getVars():
            if var.VType == gb.GRB.BINARY:
                var.VType = gb.GRB.CONTINUOUS
                var.LB = 0.0
                var.UB = 1.0 
        model_clone.update()

        if not tighter_relu: # If we don't do the tight relu
            return model_clone


        # For each ReLU variable, collect it's pre/post inputs and the
        # bounds
        relu_regex = r'^relu_\d+$'
        for key in self.var_dict:
            if re.match(relu_regex, key) is None:
                continue 
            suffix = key.split('_')[1]
            bounds = self.get_ith_relu_box(int(suffix) - 1)
            pre_relu_namer = utils.build_var_namer('fc_%s_pre' % suffix)
            post_relu_namer = utils.build_var_namer('fc_%s_post' % suffix)
            for idx in self.var_dict[key]:
                pre_var = model_clone.getVarByName(pre_relu_namer(idx))
                post_var = model_clone.getVarByName(post_relu_namer(idx))
                lo, hi = bounds[idx]
                pre_var.LB = lo 
                pre_var.UB = hi
                assert (lo < 0 < hi)
                model_clone.addConstr(post_var <= hi * pre_var / (hi - lo) 
                                                  - hi * lo / (hi - lo) )
        model_clone.update()
        return model_clone


    def build_input_constraints(self, bound_object, var_key):
        var_namer = utils.build_var_namer(var_key)
        model = self.model 
        input_range = bound_object.input_range
        if utils.arraylike(bound_object.input_range):
            input_range = Hyperbox.from_vector(input_range)
        input_range.encode_as_gurobi_model(self, var_key)


    def build_forward_pass_constraints(self):
        seq = self.network.net
        for i, layer in enumerate(seq):
            input_name = self.FORWARD_CONT_VARS % i
            output_name = self.FORWARD_CONT_VARS % (i + 1)
            sign_name = self.INT_VARS % i
            if i == 0:
                input_name = self.INPUT_NAME
            if i == len(seq) - 1:
                output_name = self.LOGITS_NAME

            if isinstance(layer, nn.Linear):
                self.add_linear_mip(i, input_name, output_name)
            elif isinstance(layer, nn.ReLU):
                self.add_relu_mip(i, input_name, output_name, sign_name)
            elif isinstance(layer, nn.LeakyReLU):
                self.add_leaky_relu_mip(i, input_name, output_name, sign_name)
            else:
                raise NotImplementedError("Unsupported Layer Type", layer)

            self.update() 



    def build_backward_pass_constraints(self):
        self.model.setObjective(0.0)
        seq = self.network.net 
        for i in range(len(seq) -1, -1, -1):
            layer = seq[i]
            input_name = self.BACKWARD_CONT_VARS % i 
            output_name = self.BACKWARD_CONT_VARS % (i - 1)
            sign_name = self.INT_VARS % i
            if i == len(seq) - 1:
                input_name = self.C_VEC_NAME
            if i == 0:
                output_name = self.GRAD_NAME

            if isinstance(layer, nn.Linear):
                self.add_backward_linear_mip(i, input_name, output_name)
            elif isinstance(layer, nn.ReLU):
                self.add_backward_relu_mip(i, input_name, output_name, 
                                           sign_name)
            elif isinstance(layer, nn.LeakyReLU):
                self.add_backward_leaky_relu_mip(i, input_name, output_name, 
                                                 sign_name)
            else: 
                raise NotImplementedError("Unsupported Layer Type", layer)

            self.update()

    def build_objective(self, primal_norm):
        assert primal_norm in ['linf', 'l1']
        # In either case, we need to add the abs(...) operator to grad vars
        box_bounds = self.get_ith_backward_box(-1) # Grad range
        self.add_abs_layer(self.GRAD_NAME, box_bounds, self.ABS_CONT_VARS, 
                           self.ABS_INT_VARS)

        if primal_norm == 'linf':
            self.set_l1_objective(self.ABS_CONT_VARS)
        else:
            self.set_linf_objective(self.ABS_CONT_VARS)

    def set_l1_objective(self, abs_name):
        obj_var = self.model.addVar(name='l1_obj')
        abs_vars = self.get_vars(abs_name)

        self.model.addConstr(sum(abs_vars) == obj_var)
        self.model.setObjective(obj_var, gb.GRB.MAXIMIZE)
        self.update()

    def set_linf_objective(self, abs_name):
        raise NotImplementedError("Do this later...")

    def add_linear_mip(self, layer_no, input_name, output_name):
        seq = self.network.net
        layer = seq[layer_no]
        model = self.model 

        weight = utils.as_numpy(layer.weight)
        if layer.bias is not None:
            bias = utils.as_numpy(layer.bias)
        else:
            bias = np.zeros(layer.out_features)

        input_vars = self.get_vars(input_name)
        var_namer = utils.build_var_namer(output_name)
        output_vars = [model.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, 
                                    name=var_namer(i)) 
                       for i in range(layer.out_features)]
        self.set_vars(output_name, output_vars)

        model.addConstrs((output_vars[i] == gb.LinExpr(weight[i], input_vars)
                                            + bias[i])
                         for i in range(layer.out_features))
        model.update() 
        return 

    def add_relu_mip(self, layer_no, input_name, output_name, sign_name):
        tolerance = 1e-8
        seq = self.network.net 
        layer = seq[layer_no] 
        model = self.model 
        post_relu_vars = [] 
        relu_vars = {} # Keyed by neuron # (int) 
        cont_namer = utils.build_var_namer(output_name)
        int_namer = utils.build_var_namer(sign_name)

        input_range = self.get_ith_relu_box(layer_no)
        for i, (low, high) in enumerate(input_range):
            pre_relu = self.get_vars(input_name)[i]

            cont_name = cont_namer(i)
            int_name = int_namer(i) 
            if high <= 0:  # ReLU always off
                post_relu_vars.append(model.addVar(lb=0.0, ub=0.0, name=cont_name))
            elif low >= 0: # ReLU always on
                post_relu = model.addVar(lb=low, ub=high, name=cont_namer(i))
                post_relu_vars.append(post_relu)
                model.addConstr(post_relu == pre_relu)
            else: # ReLU may be either on or off
                post_relu = model.addVar(lb=0.0, ub=high, name=cont_namer(i))
                post_relu_vars.append(post_relu)
                relu_var = model.addVar(lb=0, ub=1, vtype=gb.GRB.BINARY,
                                        name=int_namer(i))
                relu_vars[i] = relu_var
                # Add 4 costraints:


                model.addConstr(post_relu >= 0) # relu(x) >= 0
                model.addConstr(post_relu >= pre_relu)  # relu(x) >= x

                model.addConstr(post_relu <= high * relu_var)
                model.addConstr(post_relu <= pre_relu - low * (1 - relu_var))


        model.update()
        self.var_dict[output_name] = post_relu_vars
        self.var_dict[sign_name] = relu_vars



    def add_leaky_relu_mip(self, layer_no, input_name, output_name, sign_name):
        seq = self.network.net 
        layer = seq[layer_no] 
        neg = layer.negative_slope
        model = self.model 
        post_relu_vars = [] 
        relu_vars = {} # Keyed by neuron # (int) 
        cont_namer = utils.build_var_namer(output_name)
        int_namer = utils.build_var_namer(sign_name)

        input_range = self.get_ith_relu_box(layer_no)
        for i, (low, high) in enumerate(input_range):
            pre_relu = self.get_vars(input_name)[i]

            cont_name = cont_namer(i)
            int_name = int_namer(i) 
            if high <= 0:  # ReLU always off
                post_relu = model.addVar(lb=neg * low, ub=neg * high, 
                                         name=cont_name)
                post_relu_vars.append(post_relu)
                model.addConstr(post_relu == pre_relu * neg)

            elif low >= 0: # ReLU always on
                post_relu = model.addVar(lb=low, ub=high, name=cont_namer(i))
                post_relu_vars.append(post_relu)
                model.addConstr(post_relu == pre_relu)
            else: # ReLU may be either on or off
                post_relu = model.addVar(lb=neg * low, ub=high, 
                                         name=cont_namer(i))
                post_relu_vars.append(post_relu)
                relu_var = model.addVar(lb=0, ub=1, vtype=gb.GRB.BINARY,
                                        name=int_namer(i))
                relu_vars[i] = relu_var
                # Add 4 constraints:
                model.addConstr(post_relu >= neg * pre_relu) # relu(x) >= 0
                model.addConstr(post_relu >= pre_relu)  # relu(x) >= x

                model.addConstr(post_relu <= neg * pre_relu +
                                             high * relu_var * (1 - neg))
                model.addConstr(post_relu <= pre_relu -
                                             low * (1 - relu_var) * (1 - neg))


        model.update()
        self.var_dict[output_name] = post_relu_vars
        self.var_dict[sign_name] = relu_vars


    def add_backward_linear_mip(self, layer_no, input_name, output_name):
        seq = self.network.net
        layer = seq[layer_no]
        model = self.model

        output_vars = []
        output_var_namer = utils.build_var_namer(output_name)
        weight = layer.weight.T

        input_vars = self.get_vars(input_name)

        output_vars = [model.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY,
                                    name=output_var_namer(i))
                       for i in range(layer.in_features)]
        self.set_vars(output_name, output_vars)
        model.addConstrs((output_vars[i] == gb.LinExpr(weight[i], input_vars))
                          for i in range(layer.in_features))
        model.update()
        return


    def add_backward_relu_mip(self, layer_no, input_name, output_name,
                              sign_name):
        seq = self.network.net 
        layer = seq[layer_no] 
        model = self.model 

        # Get namers and setup data structures to hold new vars
        output_namer = utils.build_var_namer(output_name)
        output_vars = []
        sign_namer = utils.build_var_namer(sign_name)

        # Get [l,u] for all switch inputs, and gradient ranges
        backbox = self.get_ith_backward_box(layer_no)     
        switchbox = self.get_ith_switch_box(layer_no)

        # And then can now go through loops:
        # First adding variables:

        for idx, val in enumerate(switchbox):
            name = output_namer(idx)
            if val < 0:                 # Switch is always off 
                lb = ub = 0.0 
            elif val > 0:               # Switch is always on 
                lb, ub = backbox[idx] 
            else:                       # Switch uncertain 
                lb = min([0, backbox[idx][0]])
                ub = max([0, backbox[idx][1]])

            output_vars.append(model.addVar(lb=lb, ub=ub, name=name))
        self.set_vars(output_name, output_vars)

        # And now add constraints 

        relu_vars = self.get_nonlin_vars(layer_no)
        input_vars = self.get_vars(input_name)

        for i, val in enumerate(switchbox):
            relu_var = relu_vars.get(i) 
            input_var = input_vars[i]
            output_var = output_vars[i]

            if val < 0: # Switch off, so do nothing
                continue 
            if val > 0: # Switch on, so equality constraint 
                model.addConstr(input_var == output_var)
                continue 
            else:
                # In this case, relu is uncertain and need 4 constraints
                bp_lo, bp_hi = backbox[i]
                lo_bar = min([bp_lo, 0])
                hi_bar = max([bp_hi, 0])
                not_relu = (1 - relu_var)

                model.addConstr(output_var <= input_var - lo_bar * not_relu)
                model.addConstr(output_var >= input_var - hi_bar * not_relu) 
                model.addConstr(output_var >= lo_bar * relu_var)
                model.addConstr(output_var <= hi_bar * relu_var)
        model.update()
        self.set_vars(output_name, output_vars)


    def add_backward_leaky_relu_mip(self, layer_no, input_name, output_name,
                                    sign_name):
        eps = 1e-7
        seq = self.network.net 
        layer = seq[layer_no] 
        neg = layer.negative_slope
        model = self.model 

        # Get namers and setup data structures to hold new vars
        output_namer = utils.build_var_namer(output_name)
        output_vars = []
        sign_namer = utils.build_var_namer(sign_name)
        sign_names = {}

        # Get [l,u] for all switch inputs, and gradient ranges
        backbox = self.get_ith_backward_box(layer_no)     # CHECK OBOE
        relu_vars = self.get_nonlin_vars(layer_no)
        input_vars = self.get_vars(input_name)
        switchbox = self.get_ith_switch_box(layer_no) # CHECK OBOE 

        uncertain_idxs = []
        for idx, bool_val in enumerate(switchbox):
            # If bool value is -1 (and always off)
            l, u = backbox[idx]
            u = u * 1000
            l = l * 1000
            input_var = input_vars[idx]
            if bool_val < 0: # Switch is always off 
                output_var = model.addVar(lb=-gb.GRB.INFINITY,
                                          name=output_namer(idx))
                model.addConstr(output_var == neg * input_var)
            elif bool_val > 0: # Switch is always on 
                output_var = model.addVar(lb=-gb.GRB.INFINITY, name=output_namer(idx))
                model.addConstr(output_var == input_var)
            else:
                lb = min([l, neg * l])
                output_var = model.addVar(lb=-gb.GRB.INFINITY, 
                                          ub=gb.GRB.INFINITY,
                                          name=output_namer(idx))
                uncertain_idxs.append(idx) # add constraints later
            #feas_check = self._check_feas()
            #print("FIRST:", (layer_no, idx), feas_check)
            output_vars.append(output_var)
        for idx in uncertain_idxs:
            l, u = backbox[idx]
            u *= 1000
            l *= 1000
            input_var = input_vars[idx] 
            output_var = output_vars[idx]
            relu_var = relu_vars[idx] 
            not_relu = (1 - relu_var)

            if u <= 0:
                model.addConstr(output_var >= input_var)
                model.addConstr(output_var <= neg * input_var)
                model.addConstr(output_var <= input_var - 
                                              (1 - neg) * l * not_relu)
                model.addConstr(output_var >= neg * input_var +
                                              (1 - neg) * l * relu_var)
            elif l >= 0:
                model.addConstr(output_var <= input_var)
                model.addConstr(output_var >= neg * input_var)
                model.addConstr(output_var >= input_var - 
                                              (1 - neg) * u * not_relu)
                model.addConstr(output_var <= neg * input_var + 
                                              (1 - neg) * u * relu_var)
            else:
                l_gap = (1 - neg) * l
                u_gap = (1 - neg) * u
                model.addConstr(output_var <= input_var - l_gap * not_relu)
                model.addConstr(output_var >= input_var - u_gap * not_relu)
                model.addConstr(output_var <= neg * input_var + u_gap * relu_var)
                model.addConstr(output_var >= neg * input_var + l_gap * relu_var)
            #feas_check = self._check_feas()
            #print("SECOND:", (layer_no, idx), (l,u,), feas_check)
        model.update()
        self.set_vars(output_name, output_vars)


    def add_abs_layer(self, input_name, input_bounds, abs_name, sign_name):
        tolerance = 1e-6
        model = self.model
        input_vars = self.get_vars(input_name)
        output_namer = utils.build_var_namer(abs_name)
        sign_namer = utils.build_var_namer(sign_name)
        output_vars = []
        sign_vars = {}

        for i, input_var in enumerate(input_vars):
            l, u = input_bounds[i]
            l -= tolerance 
            u += tolerance
            output_name = output_namer(i)
            if l >= 0:
                output_var = model.addVar(lb=l, name=output_name)
                model.addConstr(output_var == input_var)
            elif u <= 0:
                output_var = model.addVar(lb=l, name=output_name)
                model.addConstr(output_var == -input_var)
            else:
                output_var = model.addVar(lb=-tolerance, 
                                          ub=max([abs(l), u]) + tolerance,
                                          name=output_name)
                sign_var = model.addVar(lb=0, ub=1, vtype=gb.GRB.BINARY,
                                        name=sign_namer(i))
                not_sign = (1 - sign_var)
                model.addConstr(output_var >= input_var - tolerance)
                model.addConstr(output_var >= -input_var - tolerance)
                model.addConstr(output_var <= input_var - 2 * l * sign_var + mmtolerance)
                model.addConstr(output_var <= -input_var + 2 * u * not_sign + tolerance)
                sign_vars[i] = sign_var
            output_vars.append(output_var)
        model.update()
        self.set_vars(abs_name, output_vars)
        self.set_vars(sign_name, sign_vars)



class TestLipMIP:
    """ Class to help run tests on LipMIP """
    def __init__(self, network, input_range=None, backward_range=None):
        self.network = network 
        self.abstract_params = bp.AbstractParams.hyperbox_params()
        if input_range is None:
            input_dimension = network.net[0].in_features
            input_range = Hyperbox.build_unit_hypercube(input_dimension)
        self.input_range = input_range

        if backward_range is None:
            output_dimension = network.net[-1].out_features
            backward_range = 2 * torch.rand(output_dimension) - 1
        self.backward_range = backward_range 

    def run_tests(self):
        for method in [self.test_forward_correct, 
                       self.test_backward_correct,
                       self.test_lipmip_realizable]:
            try:
                method()
            except AssertionError as err:
                print(err)


    def test_forward_correct(self, num_points=100):
        """ Test case 1:
            Pick a bunch of random points, and for each of these:
            - fix the input in gurobi model and evaluate the logits (gurobi)
            - compare to the output given by torch  (pytorch) 
            - assert that the max coordinate-wise deviation is < 
              210 * gurobi_tolerance = 1e-5
        """
        # JUST BUILD THE FORWARD PASS ONLY ()
        ann = bp.AbstractNN2(self.network)
        forw, back = ann.get_both_bounds(self.abstract_params, 
                                         self.input_range, 
                                         self.backward_range)

        model = gb.Model()
        model.setParam('OutputFlag', False)
        squire = GurobiSquire(self.network, model, forw, back)
        squire.build_input_constraints(forw, 'x')
        squire.build_forward_pass_constraints() 


        rand_points = self.input_range.random_point(num_points)
        torch_outputs = self.network(rand_points).detach().numpy()

        gurobi_outputs = []
        for point in rand_points:
            gurobi_outputs.append(np.array(squire.get_var_at_point(point)))
        gurobi_outputs = np.stack(gurobi_outputs)


        max_dev = np.abs(torch_outputs - gurobi_outputs).max()
        if max_dev < 1e-5:
            print("PASSED FORWARD CHECKS!")
        else:
            raise AssertionError("FAILED FORWARD CHECKS!", max_dev)


    def test_backward_correct(self, num_points=100):
        """ Test case 2: 
            Pick a bunch of random input points and for each of these:
            - fix the input in the gurobi model and evaluate the gradient
              (with respect to the backward range) (gurobi)
            - compare to the output given by torch (pytorch)
            - assert that the max coordinate-wise deviation is <1e-5
        """
        
        ann = bp.AbstractNN2(self.network)
        forw, back = ann.get_both_bounds(self.abstract_params,
                                         self.input_range, 
                                         self.backward_range)

        model = gb.Model()
        model.setParam('OutputFlag', False)
        squire = GurobiSquire(self.network, model, forw, back)
        squire.build_input_constraints(forw, 'x')
        squire.build_forward_pass_constraints()
        squire.build_input_constraints(back, 'c_vec')
        squire.build_backward_pass_constraints()

        rand_points = self.input_range.random_point(num_points, 
                                                    requires_grad=True).float()
        cvec = utils.tensorfy(self.backward_range)
        (self.network(rand_points) @ cvec).sum().backward()
        torch_outputs = rand_points.grad.numpy()

        gurobi_outputs = [] 
        for point in rand_points:
            gurobi_outputs.append(np.array(squire.get_grad_at_point(point)))
        gurobi_outputs = np.stack(gurobi_outputs)

        max_dev = np.abs(torch_outputs - gurobi_outputs).max()
        if max_dev < 1e-5:
            print("PASSED BACKWARD CHECKS")
        else:
            raise AssertionError("FAILED BACKWARD CHECKS!", max_dev)


    def test_lipmip_realizable(self, num_points=1000):
        """ Test case 3:
            - Run LipMIP on the input range and get the right answer input/val
            - Pick a bunch of random points in a neighborhood of the right input
            - Evaluate gradient of these random points and assert that the 
              LipVal is attainable nearby
        """

        lipmip_obj = LipMIP(self.network, self.abstract_params, verbose=False, 
                            num_threads=1)
        result = lipmip_obj.compute_max_lipschitz(self.input_range, 
                                                  self.backward_range)

        best_x_box = Hyperbox.build_linf_ball(result.best_x, radius=0.001)

        rand_points = best_x_box.random_point(num_points, requires_grad=True)
        rand_points = rand_points.float()

        cvec = utils.tensorfy(self.backward_range)
        (self.network(rand_points) @ cvec).sum().backward()
        torch_output = rand_points.grad.abs().sum(1).max()

        max_dev = np.abs(torch_output - result.value)
        if max_dev < 1e-4:
            print("PASSED REALIZABLE CHECK!")
        else:
            raise AssertionError("FAILED REALIZABLE CHECK!",  max_dev)


