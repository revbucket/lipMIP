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
    - network : ReLUNet object we're solving for. Should have a sequential 
                like r/(LR)+L/ 
    - domain : Hyperbox object describing the 
    - c_vector: network has output that is a vector, so this means we 
                care about the function <c, f(.)>. If not None, needs to be 
                a domain
    - lp : primal Norm we care about. e.g., for
           |<c, f(x)- f(y)>| <= L ||x-y||
            lp refers to             ^this norm

    - preact: either string 'ia', or PreactivationBounds object corresponding 
              to this network. If 'ia', then we compute a PreactivationBounds
                  object using interval analysis when we solve
    - target_units : if not None, is the (start_idx, end_idx) of the (LR) 
                     units for which we compute the index. Standard subseq 
                     notation where 
                         start_idx: first index included 
                         end_idx: first index NOT included
    - verbose: prints Gurobi outputs if True
    - timeout: stops early (will replace with stopping criteria later)
    """

    VALID_PREACTS = VALID_PREACTS

    def __init__(self, network, domain, c_vector, primal_norm='linf', 
                 preact='naive_ia', verbose=False, 
                 timeout=None, mip_gap=None, num_threads=None):
        init_kwargs = {k: v for k, v in locals().items()
                       if k not in ['self', '__class__']}
        assert (utils.arraylike(c_vector) or 
                (c_vector in ['l1Ball1', 'crossLipschitz', 'targetCrossLipschitz', 
                              'trueCrossLipschitz', 'trueTargetCrossLipschitz']))
        super(LipMIP, self).__init__(**init_kwargs)
        self.result = None

    def _self_attach_result(self, result):
        self.result = result
        self.value = result.value 
        self.compute_time = result.compute_time

    def build_gurobi_squire(self):
        """ Builds the gurobi squire """
        network = self.network
        assert self.primal_norm in ['linf', 'l1'] # Meaning we want max ||grad(f)||_*
        assert (self.preact in self.VALID_PREACTS or
                isinstance(self.preact, AbstractNN))

        timer = utils.Timer()
        # Step 1: Build the pre-ReLU and pre-switch hyperboxes
        if not isinstance(self.preact, AbstractNN):
            pre_bounds = AbstractNN(self.network, self.domain, self.c_vector)
            pre_bounds.compute_forward(technique=self.preact)
            pre_bounds.compute_backward(technique=self.preact)
        else:
            pre_bounds = self.preact
        squire = build_gurobi_model(network, pre_bounds, self.primal_norm,
                                   verbose=self.verbose)

        return squire, timer

    def compute_max_lipschitz(self):
        """ Computes the maximum lipschitz constant with a fixed
            domain already set.

            Returns the maximum lipschitz constant and the point that
            attains it
        """
        squire, timer = self.build_gurobi_squire()
        model = squire.model

        if self.timeout is not None:
            model.setParam('TimeLimit', self.timeout)
        if self.mip_gap is not None:
            model.setParam('MIPGap', self.mip_gap)

        model.setParam('Threads', getattr(self, 'num_threads') or 4)

        model.optimize()
        if model.Status in [3, 4]:
            print("INFEASIBLE")

        if model.Status in [9]:
            print("TIME LIMIT") 
            result = LipResult(self.network, self.c_vector, 
                               value=model.objBound, 
                               compute_time = timer.stop(),
                               domain=self.domain, squire=squire, 
                               model=model)
            self._self_attach_result(result)
            return result

        # HANDLE TIMEOUT VALUE

        runtime = timer.stop()
        x_vars = squire.get_vars('x')
        value = model.ObjBound #model.getObjective().getValue()
        best_x = np.array([v.X for v in x_vars])
        best_sign_config = squire.get_sign_configs()
        result = LipResult(self.network, self.c_vector, value=value,
                           model=model, compute_time=runtime, 
                           preacts=squire.pre_bounds, best_x=best_x, 
                           domain=self.domain, squire=squire, 
                           sign_config=best_sign_config)

        self._self_attach_result(result)
        return result
        # ======  End of MAIN SOLVER BLOCK                =======



# ==============================================================
# =         Result Object                                      =
# ==============================================================

class LipResult:
    """ Handy object to store the values of a LipMIP run """
    ATTRS = set(['network', 'c_vector', 'value', 'squire', 'model',
                 'compute_time', 'preacts', 'best_x', 'domain', 'label',
                 'sign_config'])
    MIN_ATTRS = set(['c_vector', 'value', 'compute_time', 'best_x', 'domain',
                     'sign_config'])
    def __init__(self, network, c_vector, value=None, squire=None, 
                 model=None, compute_time=None, preacts=None, best_x=None,
                 domain=None, label=None, sign_config=None):
        for attr in self.ATTRS:
            setattr(self, attr, vars().get(attr))

    def as_dict(self):
        return {k: getattr(self, k) for k in self.ATTRS
                if getattr(self, k, None) is not None}

    def __repr__(self):
        output = 'LipMIP Result: \n'
        if getattr(self, 'label', None) is not None:
            output += '\tLabel: %s\n' % self.label
        output += '\tValue %.03f\n' % self.value 
        output += '\tRuntime %.03f' % self.compute_time
        return output

    def attach_label(self, label):
        """ cute way to attach a label to a result """
        self.label = label


    def shrink(self):
        """ Removes some of the unnecessary attributes so this is easier 
            to store 
        """
        for del_el in (self.ATTRS - self.MIN_ATTRS):
            delattr(self, del_el)
        return self




# ==============================================================
# =           Build Gurobi Model for Lipschitz Comp.           =
# ==============================================================

class GurobiSquire():
    def __init__(self, network, model, pre_bounds=None):
        self.var_dict = {}
        self.network = network
        self.model = model
        self.pre_bounds = pre_bounds

    # -----------  Variable Getter/Setters  -----------

    def get_vars(self, name):
        return self.var_dict[name]

    def set_vars(self, name, var_list):
        self.var_dict[name] = var_list

    def var_lengths(self):
        # Debug tool
        len_dict = {k: len(v) for k, v in self.var_dict.items()}
        pprint.pprint(len_dict)

    # -----------  Pre-bound getter/setters  -----------

    def set_pre_bounds(self, pre_bounds):
        self.pre_bounds = pre_bounds

    def get_ith_relu_box(self, i):
        # Returns Hyperbox bounding inputs to i^th relu layer
        return self.pre_bounds.get_forward_box(i)

    def get_ith_switch_box(self, i):
        # Returns BooleanHyperbox bounding values of i^th relu's int vars
        return self.pre_bounds.get_forward_switch(i)

    def get_ith_backward_box(self, i):
        # Returns Hyperbox bounding inputs to i^th (forward index!) backswitch
        if i > 0:
            return self.pre_bounds.get_backward_box(i, forward_idx=True)
        else: 
            return self.pre_bounds.gradient_range

    # -----------  Other auxiliary methods   -----------

    def update(self):
        self.model.update()

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


    def get_var_at_point(self, x, var_key):
        """ Computes the variables prefixed by var_key at a given point.
            This does not modify any state (makes a copy of the gurobi model
            prior to changing it)
        ARGS:
            x: np.array - np array of the requisite input shape 
            reset: bool - if True, we remove the input constraints and update 
        RETURNS:
            var (as a numpy array), but also modifies the model
        """

        model = self.model.copy()
        x_var_namer = utils.build_var_namer('x')
        constr_namer = lambda x_var_name: 'fixed::' + x_var_name
        for i in range(len(self.get_vars('x'))):
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
        return self.get_var_at_point(x, 'gradient')


    def get_logits_at_point(self, x):
        """ Gets logit at point x """
        return self.get_var_at_point(x, 'logits')


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


    def get_sign_configs(self):
        """ Collects numpy arrays of booleans that correspond to the 
            sign configurations of a solved gurobi model
        """
        num_relus = self.network.num_relus
        tolerance = 1e-8
        output = []
        for layer_idx in range(1, num_relus + 1):
            pre_key = 'fc_%s_pre' % layer_idx
            relu_key = 'relu_%s' % layer_idx
            pre_vars = self.get_vars(pre_key)
            relu_vars = self.get_vars(relu_key)
            layer_vals = []
            for j, el in enumerate(pre_vars):
                if el.X > tolerance:
                    layer_vals.append(True)
                elif el.X < -tolerance:
                    layer_vals.append(False)
                else:
                    layer_vals.append(relu_vars[j].X > 0.5)
            output.append(np.array(layer_vals))
        return output




def build_gurobi_model(network, pre_bounds, lp, verbose=False):

    # -- hush mode
    with utils.silent(): # ain't nobody tryna hear about your gurobi license
        model = gb.Model()  
    if not verbose:
        model.setParam('OutputFlag', False)     


    squire = GurobiSquire(network, model, pre_bounds=pre_bounds)


    # -- Actually build the gurobi model now
    build_input_constraints(squire, 'x')
    build_forward_pass_constraints(squire)
    build_back_pass_constraints(squire)
    build_objective(squire, lp)
    model.update()

    # -- return everything we want
    return squire


def build_forward_pass_constraints(gurobi_squire):
    relunet = gurobi_squire.network
    for i, fc_layer in enumerate(relunet.fcs[:-1]):
        if i == 0:
            input_name = 'x'
        else:
            input_name = 'fc_%s_post' % i

        pre_relu_name = 'fc_%s_pre' % (i + 1)
        post_relu_name = 'fc_%s_post' % (i + 1)
        relu_name = 'relu_%s' % (i+ 1)
        add_linear_layer_mip(i, gurobi_squire,
                             input_name, pre_relu_name)
        add_relu_layer_mip(i, gurobi_squire,
                           pre_relu_name, relu_name, post_relu_name)

    if isinstance(relunet.fcs[-1], nn.Linear):
        output_var_name = 'logits'
        add_linear_layer_mip(len(relunet.fcs) - 1, 
                             gurobi_squire, post_relu_name, output_var_name)
    gurobi_squire.update()


def build_back_pass_constraints(gurobi_squire):
    """ For relunet like f(x) = c R_l-1(L_l-1 ... R0(L0x))
        which has l units of Linear->ReLu
        and we only want to encode backprop up to (and including) 
        the target_units[0]'th one of them

        So we need to encode [LRLRLRC], 
    """
    relunet = gurobi_squire.network
    if getattr(relunet, 'target_units', None) is None:
        stop_idx = 0
    else:
        # need to backprop to include first of the 
        stop_idx = relunet.target_units[0]


    # Need to include how many lambdas?
    # should run through this loop (num_relus - target_units[0]) times
    for i in range(relunet.num_relus, stop_idx, -1):
        linear_in_key = 'bp_%s_postswitch' % (i + 1)
        linear_out_key = 'bp_%s_preswitch' % i
        switch_out_key = 'bp_%s_postswitch' % i
        relu_key = 'relu_%s' % i
        if i == relunet.num_relus:
            add_first_backprop_layer(gurobi_squire,
                                     'c_vector', linear_out_key)
        else:
            add_backprop_linear_layer(i, gurobi_squire,
                                      linear_in_key, linear_out_key)
        add_backprop_switch_layer_mip(i, gurobi_squire,
                                      linear_out_key, relu_key, switch_out_key)
        # TODO: ENCODE DIRECTION VECTOR TO IMPLICITLY TAKE NORMS

    # And the final layer
    add_backprop_linear_layer(stop_idx, gurobi_squire,
                              switch_out_key, 'gradient')
    gurobi_squire.update()


def build_objective(gurobi_squire, lp):
    gradient_key = 'gradient'
    abs_sign_key = 'abs_sign'
    abs_grad_key = 'abs_grad'
    grad_bounds = gurobi_squire.pre_bounds.gradient_range
    add_abs_layer(gurobi_squire, grad_bounds,
                  gradient_key, abs_sign_key, abs_grad_key)
    if lp == 'linf':
        set_l1_objective(gurobi_squire, abs_grad_key)
    elif lp == 'l1':
        set_linf_objective(gurobi_squire, grad_bounds, abs_grad_key, 'abs_max')
    gurobi_squire.update()

# ======  End of Build Gurobi Model for Lipschitz Comp.  =======




# =========================================================================
# =                       LAYERWISE HELPERS                               =
# =========================================================================

def build_input_constraints(squire, var_key):
    # If domain is a hyperbox, can cover with lb/ub in var constructor
    var_namer = utils.build_var_namer(var_key)
    model = squire.model
    input_domain = squire.pre_bounds.input_domain
    input_vars = []
    if isinstance(input_domain, Hyperbox):
        for i, (lb, ub) in enumerate(input_domain):
            input_vars.append(model.addVar(lb=lb, ub=ub, name=var_namer(i)))
    else:
        raise NotImplementedError("Only hyperboxes allowed for now!")
    model.update()
    squire.set_vars(var_key, input_vars)



def add_linear_layer_mip(layer_no, squire,
                         input_key, output_key):
    network = squire.network
    model = squire.model
    fc_layer = network.fcs[layer_no]
    fc_weight =  utils.as_numpy(fc_layer.weight)
    if network.bias:
        fc_bias = utils.as_numpy(fc_layer.bias)
    else:
        fc_bias = np.zeros(fc_layer.out_features)
    input_vars = squire.get_vars(input_key)
    var_namer = utils.build_var_namer(output_key)
    pre_relu_vars = [model.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY,
                                  name=var_namer(i))
                     for i in range(fc_layer.out_features)]
    squire.set_vars(output_key, pre_relu_vars)
    model.addConstrs((pre_relu_vars[i] == gb.LinExpr(fc_weight[i], input_vars) + fc_bias[i])
                     for i in range(fc_layer.out_features))
    model.update()
    return


def add_relu_layer_mip(layer_no, squire, input_key,
                       sign_key, output_key):
    network = squire.network
    model = squire.model
    post_relu_vars = []
    relu_vars = {} # keyed by neuron # (int)
    post_relu_namer = utils.build_var_namer(output_key)
    relu_namer = utils.build_var_namer(sign_key)
    input_box = squire.get_ith_relu_box(layer_no)
    for i, (low, high) in enumerate(input_box):
        post_relu_name = post_relu_namer(i)
        relu_name = relu_namer(i)
        if high <= 0:
            post_relu_vars.append(model.addVar(lb=0.0, ub=0.0,
                                               name=post_relu_name))
        else:
            pre_relu = squire.get_vars(input_key)[i]
            post_relu_vars.append(model.addVar(lb=low, ub=high,
                                               name=post_relu_name))
            post_relu = post_relu_vars[-1]
            if low >= 0:
                model.addConstr(post_relu == pre_relu)
                continue
            else:
                relu_var = model.addVar(lb=0.0, ub=1.0, vtype=gb.GRB.BINARY,
                                        name=relu_name)
                relu_vars[i] = relu_var

            # relu(x) >= 0 and relu(x) >= x
            model.addConstr(post_relu >= 0.0)
            model.addConstr(post_relu >= pre_relu)

            # relu(x) <= u * a
            model.addConstr(post_relu <= high * relu_var)

            # relu(x) <= pre_relu - l(1-a)
            model.addConstr(post_relu <= pre_relu - low * (1 - relu_var))

    model.update()
    squire.var_dict[output_key] = post_relu_vars
    squire.var_dict[sign_key] = relu_vars


def add_first_backprop_layer(squire, input_key, output_key):
    """ Encodes the backprop of the first linear layer.
        All the variables will be constant, and dependent upon the
        c_vector
    """
    network = squire.network
    model = squire.model
    backprop_vars = squire.pre_bounds.gurobi_backprop_domain(squire, input_key)

    output_vars = []
    output_var_namer = utils.build_var_namer(output_key)

    if isinstance(network.fcs[-1], nn.Linear):
        weight = utils.as_numpy(network.fcs[-1].weight).T
        for i in range(network.fcs[-1].in_features):
            output_vars.append(model.addVar(lb=-gb.GRB.INFINITY, 
                                            ub=gb.GRB.INFINITY,
                                            name=output_var_namer(i)))
            model.addConstr(output_vars[i] == gb.LinExpr(weight[i], backprop_vars))
    else:
        for i in range(len(backprop_vars)):
            output_vars.append(model.addVar(lb=-gb.GRB.INFINITY, 
                                            ub=gb.GRB.INFINITY,
                                            name=output_var_namer(i)))
            model.addConstr(output_vars[i] == backprop_vars[i])
    squire.set_vars(output_key, output_vars)
    model.update()


def add_backprop_linear_layer(layer_no, squire,
                              input_key, output_key):
    """ Encodes the backprop version of a linear layer """
    network = squire.network
    model = squire.model
    output_vars = []
    output_var_namer = utils.build_var_namer(output_key)
    fc_layer = network.fcs[layer_no]
    fc_weight = utils.as_numpy(fc_layer.weight)

    backprop_bounds = squire.get_ith_backward_box(layer_no)
    input_vars = squire.get_vars(input_key)
    for i in range(fc_layer.in_features):
        output_var = model.addVar(lb=backprop_bounds[i][0],
                                  ub=backprop_bounds[i][1],
                                  name=output_var_namer(i))
        weight_col = fc_weight[:, i]
        model.addConstr(output_var == gb.LinExpr(weight_col, input_vars))
        output_vars.append(output_var)
    model.update()
    squire.set_vars(output_key, output_vars)



def add_backprop_switch_layer_mip(layer_no, squire,
                                  input_key, relu_key, output_key):
    """
    Encodes the funtion
    output_key[i] := 0            if sign_key[i] == 0
                     input_key[i] if sign_key != 0
    where 0 <= input_key[i] <= squire.backprop_pos_bounds[i]
    """
    network = squire.network
    model = squire.model
    switchbox = squire.get_ith_switch_box(layer_no - 1)
    switch_inbox = squire.get_ith_backward_box(layer_no)
    post_switch_vars = []
    post_switch_namer = utils.build_var_namer(output_key)
    post_switch_namer = utils.build_var_namer(output_key)
    # First add variables
    for idx, val in enumerate(switchbox):
        name = post_switch_namer(idx)
        if val < 0:
            # Switch is always off
            lb = ub = 0.0
        elif val > 0:
            # Switch is always on
            lb, ub = switch_inbox[idx]
        else:
            # Switch uncertain
            lb = min([0, switch_inbox[idx][0]])
            ub = max([0, switch_inbox[idx][1]])
        post_switch_vars.append(model.addVar(lb=lb, ub=ub, name=name))
    squire.set_vars(output_key, post_switch_vars)


    # And then add constraints
    relu_vars = squire.get_vars(relu_key)
    pre_switch_vars = squire.get_vars(input_key)
    for idx, val in enumerate(switchbox):
        relu_var        = relu_vars.get(idx)
        pre_switch_var  = pre_switch_vars[idx]
        post_switch_var = post_switch_vars[idx]
        if val < 0:
            continue
        elif val > 0:
            model.addConstr(post_switch_var == pre_switch_var)
            continue
        else:
            # In this case, the relu is uncertain and we need to encode 
            # 4 constraints. This depends on the backprop low or high though
            bp_lo, bp_hi = switch_inbox[idx]
            lo_bar = min([bp_lo, 0])
            hi_bar = max([bp_hi, 0])
            not_relu_var = (1 - relu_var)
            model.addConstr(post_switch_var <= pre_switch_var - 
                                                lo_bar * not_relu_var)
            model.addConstr(post_switch_var >= pre_switch_var - 
                                               hi_bar * not_relu_var)
            model.addConstr(post_switch_var >= lo_bar * relu_var)
            model.addConstr(post_switch_var <= hi_bar * relu_var)
    model.update()




def add_abs_layer(squire, hyperbox_bounds, 
                  input_key, sign_key, output_key):
    """ Encodes the absolute value as gurobi models:
        - creates variables keyed by output_key that are the 
              absolute value of input_key (sign_key are integer variables 
          to control the nonconvexity of input_key)
        - conservative sign bounds based on preact 0th layer backprop 
          bounds control signs
    """
    network = squire.network
    model = squire.model

    tolerance = 1e-6
    input_vars = squire.get_vars(input_key)
    output_namer = utils.build_var_namer(output_key)
    sign_namer = utils.build_var_namer(sign_key)
    output_vars = []
    sign_vars = {}


    for i, input_var in enumerate(input_vars):
        lb, ub = hyperbox_bounds[i]
        output_name = output_namer(i)
        # always positive
        if lb >= 0:
            output_var = model.addVar(lb=-tolerance, name=output_name)
            model.addConstr(output_var == input_var)
        # always negative
        elif ub <= 0:
            output_var = model.addVar(lb=-tolerance, name=output_name)
            model.addConstr(output_var == -input_var)
        # could be positive or negative
        else:
            output_var = model.addVar(lb=- tolerance, 
                                      ub=max([abs(lb), ub]) + tolerance, 
                                      name=output_name)
            sign_var = model.addVar(lb=0, ub=1, vtype=gb.GRB.BINARY, 
                                    name=sign_namer(i))
            #model.addConstr(output_var >= 0)
            model.addConstr(output_var >= input_var - tolerance)
            model.addConstr(output_var >= -input_var - tolerance)
            model.addConstr(output_var <= input_var - 2 *  lb * (1 - sign_var) + tolerance)
            model.addConstr(output_var <= -input_var + 2 * ub * sign_var + tolerance)
            sign_vars[i] = sign_var
        output_vars.append(output_var)
    model.update()
    squire.set_vars(output_key, output_vars)
    squire.set_vars(sign_key, sign_vars)

def add_abs_layer_relu(squire, hyperbox_bounds, 
                       input_key, sign_key, output_key):
    network = squire.network
    model = squire.model

    input_vars = squire.get_vars(input_key)
    output_namer = utils.build_var_namer(output_key)
    pos_sign_namer = utils.build_var_namer(sign_key + 'POS')
    neg_sign_namer = utils.build_var_namer(sign_key + 'NEG')    
    output_vars = [] 
    sign_vars = {} 

    for i, input_var in enumerate(input_vars):
        lb, ub = hyperbox_bounds[i]
        output_name = output_namer(i)
        if lb >= 0:
            output_var = model.addVar(lb=0, name=output_name)
            model.addConstr(output_var == input_var)
        elif ub <= 0:
            output_var = model.addVar(lb=0, name=output_name)
            model.addConstr(output_var == -input_var)
        else:
            pos_sign = model.addVar(lb=0, ub=1, vtype=gb.GRB.BINARY,
                                    name=pos_sign_namer(i))
            neg_sign = model.addVar(lb=0, ub=1, vtype=gb.GRB.BINARY,
                                    name=neg_sign_namer(i))

            # ADD TWO RELU CONSTRAINTS 
            # -- positive relu constraint           
            pos_term = model.addVar(lb=0)
            model.addConstr(pos_term >= 0)
            model.addConstr(pos_term >= input_var)
            model.addConstr(pos_term <= ub * pos_sign)
            model.addConstr(pos_term <= input_var - lb * (1 - pos_sign))

            neg_term = model.addVar(lb=0)
            # range is [-ub, -lb]
            model.addConstr(neg_term >= 0)
            model.addConstr(neg_term >= -input_var)
            model.addConstr(neg_term <= -lb * neg_sign)
            model.addConstr(neg_term <= -input_var + ub * (1 - neg_sign))       

            output_var = model.addVar(lb=0, name=output_name)
            model.addConstr(output_var == neg_term + pos_term)
        output_vars.append(output_var)
    model.update()
    squire.set_vars(output_key, output_vars)


def set_linf_objective(squire, box_range, abs_key, maxint_key):
    """ Sets the objective for the MAX of the abs_keys where 
        box_range is a hyperbox for the max of these variables before 
        the absolute value is applied 
    ARGS:
        squire: gurobi squire object which holds the model
        box_range : Hyperbox bounding the values of abs_key variables before 
                    the 
        abs_key : string that points to the continuous variables that 
                  represent the absolute value of some other variable
        maxint_key : string that will refer to the integer variables names

    """
    model = squire.model
    abs_vars = squire.get_vars(abs_key)
    ubs = np.maximum(box_range.box_hi, abs(box_range.box_low))
    lbs = np.maximum(box_range.box_low, 0)
    l_max = max(lbs)
    relevant_idxs = [_ for _ in range(len(ubs)) if _ >= l_max]

    top_two = sorted(relevant_idxs, key=lambda el: -ubs[el])[:2]
    max_var = model.addVar(lb=l_max, ub=ubs[top_two[0]])
    maxint_namer = utils.build_var_namer(maxint_key)
    maxint_vars = {}

    if len(relevant_idxs) == 1:
        print("ONLY 1 THING TO MAXIMIZE")
        model.addConstr(max_var == abs_vars[relevant_idxs[0]])
        model.setObjective(max_var, gb.GRB.MAXIMIZE)
        squire.update()
    else:
        for idx in relevant_idxs:
            if idx == top_two[0]:
                u_max = ubs[top_two[1]]
            else:
                u_max = ubs[top_two[0]]
            maxint_var = model.addVar(lb=0, ub=1, vtype=gb.GRB.BINARY,
                                      name=maxint_namer(idx))
            maxint_vars[idx] = maxint_var
            model.addConstr(max_var >= abs_vars[idx])
            model.addConstr(max_var <= abs_vars[idx] + 
                                       (1 - maxint_var) * (u_max - lbs[idx]))
        model.addConstr(1 == sum(list(maxint_vars.values())))

    model.setObjective(max_var, gb.GRB.MAXIMIZE)
    squire.set_vars(maxint_key, maxint_vars)
    squire.update()


def set_l1_objective(squire, abs_key):
    """ Sets the objective for the sum of the abs_keys
        (absolute value has already been encoded by abs_layer)
    """
    abs_vars = squire.get_vars(abs_key)
    namer = utils.build_var_namer('l1_obj')
    obj_var = squire.model.addVar(name=namer(0))
    squire.model.addConstr(sum(abs_vars) == obj_var)
    squire.model.setObjective(obj_var, gb.GRB.MAXIMIZE)
    squire.set_vars('l1_obj', [obj_var])
    squire.update()



# ======  End of             LAYERWISE HELPERS                      =======


def naive_mip(relu_net, c_vec, primal_norm='linf', verbose=False, num_threads=2):
    """ Does the most naive MIP possible -- all sign configurations attainable"""

    with utils.silent():
        model = gb.Model()
    model.setParam('OutputFlag', verbose)
    model.setParam('Threads', num_threads)

    # Now do layerwise helpers in the backwards direction only 
    input_dim = relu_net.layer_sizes[0]
    output_dim = relu_net.layer_sizes[-1]

    # Get preact bounds (globally)
    input_domain = Hyperbox.build_unit_hypercube(input_dim)
    ai_box = AbstractNN(relu_net, input_domain, c_vec)
    ai_box.compute_forward()
    ai_box.backward_domains = {k: v.zero_val() for k,v
                               in ai_box.backward_domains.items()}
    ai_box.compute_backward()

    # Build squire so we can use existing tools 
    squire = GurobiSquire(relu_net, model, pre_bounds=ai_box)
    # Add dummy ReLU constraints 
    for layer_no in range(1, relu_net.num_relus + 1):
        name = 'relu_%s' % layer_no 
        namer = utils.build_var_namer(name)
        relu_vars = {i: model.addVar(vtype=gb.GRB.BINARY, name=namer(i))
                     for i in range(relu_net.layer_sizes[layer_no])}
        squire.set_vars(name, relu_vars)

    build_back_pass_constraints(squire)
    build_objective(squire, primal_norm)

    return squire