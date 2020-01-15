""" Main file to contain lipschitz maximization """

import numpy as np
import gurobipy as gb
import utilities as utils
from hyperbox import Hyperbox, LinfBallFactory, Domain
from pre_activation_bounds import PreactivationBounds
import time
import pprint
import re
from interval_analysis import HBoxIA
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

class LipParameters(utils.ParameterObject):
	"""
	Holds SOME parameters for LipMIP computation. Does not hold ALL the 
	parameters, but holds some so we are a little cleaner for some 

	- c_vector: network has output that is a vector, so this means we 
				care about the function <c, f(.)> 
	- lp : primal Norm we care about. e.g., for
		   |<c, f(x)- f(y)>| <= L ||x-y||
			lp refers to             ^this norm
	- preact_method: which abstraction pushforward operators we'll use to 
					 generate pre-relu bounds and pre-switch bounds.
					 'ia' := interval analysis 
	- verbose: prints Gurobi outputs if True
	- timeout: stops early (will replace with stopping criteria later)

	"""
	def __init__(self, domain, c_vector, lp='linf', preact_method='ia',
				 verbose=False, timeout=None):
		init_args = {k: v for k, v in locals().items() 
					 if k not in ['self', '__class__']}
		super(LipParameters, self).__init__(**init_args)


	def change_domain(self, new_domain):
		return self.change_attrs(**{'domain': new_domain})


	def change_c_vector(self, new_c_vector):
		return self.change_attrs(**{'c_vector': new_c_vector})



class LipProblem(utils.ParameterObject):
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

	VALID_PREACTS = ['naive_ia'] # add improved_ia, LP

	def __init__(self, network, domain, c_vector, lp='linf', 
				 preact='naive_ia', verbose=False, 
				 timeout=None):
		init_kwargs = {k: v for k, v in locals().items()
   					   if k not in ['self', '__class__']}
		super(LipProblem, self).__init__(**init_kwargs)



	def compute_max_lipschitz(self):
		""" Computes the maximum lipschitz constant with a fixed
			domain already set.

			Returns the maximum lipschitz constant and the point that
			attains it
		"""
		network = self.network
		assert self.lp == 'linf' # Meaning we want max ||grad(f)||_1
		assert (self.preact in self.VALID_PREACTS or
				isinstance(self.preact, HBoxIA))

		timer = utils.Timer()
		# Step 1: Build the pre-ReLU and pre-switch hyperboxes
		if not isinstance(self.preact, HBoxIA):
			pre_bounds = HBoxIA(self.network, self.domain, self.c_vector)
			pre_bounds.compute_forward(technique=self.preact)
			pre_bounds.compute_backward(technique=self.preact)
		else:
			pre_bounds = self.preact

		squire, model, preacts = build_gurobi_model(network, pre_bounds,
													self.lp, 
													verbose=self.verbose)


		if params.timeout is not None:
			model.setParam('TimeLimit', params.timeout)

		model.optimize()
		if model.Status == 3:
			print("INFEASIBLE")

		runtime = timer.stop()
		x_vars = squire.get_vars('x')
		value = model.getObjective().getValue()
		best_x = np.array([v.X for v in x_vars])
		result = LipMIPResult(network, params.c_vector, value=value, model=model,
							  runtime=runtime, preacts=preacts,
							  best_x=best_x, domain=params.domain, squire=squire)
		return result

		# ======  End of MAIN SOLVER BLOCK                =======



# ==============================================================
# =         Result Object                                      =
# ==============================================================

class LipMIPResult:
	""" Handy object to store the values of a LipMIP run """
	ATTRS = set(['network', 'c_vector', 'value', 'squire', 'model',
    			 'runtime', 'preacts', 'best_x', 'domain', 'label'])
	MIN_ATTRS = set(['c_vector', 'value', 'runtime', 'best_x', 'domain'])
	def __init__(self, network, c_vector, value=None, squire=None, 
				 model=None, runtime=None, preacts=None, best_x=None,
				 domain=None, label=None):
		for attr in self.ATTRS:
			setattr(self, attr, vars().get(attr))

	def as_dict(self):
		return {k: getattr(self, k) for k in self.ATTRS
				if getattrs(self, k) is not None}

	def __dict__(self):
		return self.as_dict() 

	def __repr__(self):
		output = 'LipMIP Result: \n'
		if self.label is not None:
			output += '\tLabel: %s\n' % self.label
		output += '\tValue %.03f\n' % self.value 
		output += '\tRuntime %.03f' % self.runtime
		return output

	def attach_label(self, label):
		""" cute way to attach a label to a result """
		self.label = label


	def shrink(self):
		""" Removes some of the unnecessary attributes so this is easier 
			to store 
		"""
		for del_el in (ATTRS - MIN_ATTRS):
			delattr(self, del_el)





# ==============================================================
# =           Build Gurobi Model for Lipschitz Comp.           =
# ==============================================================

class GurobiSquire():
	def __init__(self, model, pre_bounds=None):
		self.var_dict = {}
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
		return self.pre_bounds.get_backward_box(i, forward_idx=True)

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


	def get_grad_at_point(self, x, reset=False):
		""" Computes the gradient at a given point.
			Note -- this modifies the constraints, 
					optimizes the linear program,
			IT DOES NOT RETURN IT TO ORIGINAL STATE (unless reset=True)
		ARGS:
			x: np.array - np array of the requisite input shape 
			reset: bool - if True, we remove the input constraints and update 
		RETURNS:
			grad (as a numpy array), but also modifies the model
		"""
		constr_namer = lambda x_var_name: 'fixed::' + x_var_name
		for i, x_var in enumerate(self.get_vars('x')):
			constr_name = constr_namer(x_var.varName)
			constr = self.model.getConstrByName(constr_name)
			if constr is not None:
				self.model.remove(constr)
			self.model.addConstr(x_var == x[i], name=constr_name)

		self.model.setObjective(0)
		self.model.update()
		self.model.optimize()
		gradient = [_.X for _ in self.get_vars('gradient')]
		if reset:
			for i, x_var in enumerate(self.get_vars('x')):
				self.model.remove(self.model.getConstrByName(constr_namer(x_var.varName)))

			self.model.update()
		return gradient


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
			bounds = self.get_preact_bounds(int(suffix) - 1, two_col=True)
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


def build_gurobi_model(network, pre_bounds, lp, verbose=False):

	# -- hush mode
	with utils.silent(): # ain't nobody tryna hear about your gurobi license
		model = gb.Model()	
	if not verbose:
		model.setParam('OutputFlag', False)		


	squire = GurobiSquire(model, pre_bounds=pre_bounds)


	# -- Actually build the gurobi model now
	build_input_constraints(squire, 'x')
	build_forward_pass_constraints(network, squire)
	build_back_pass_constraints(network, squire)
	build_objective(network, squire, lp)
	model.update()

	# -- return everything we want
	return squire, model, preact_object


def build_forward_pass_constraints(relunet, gurobi_squire):

	for i, fc_layer in enumerate(relunet.fcs[:-1]):
		if i == 0:
			input_name = 'x'
		else:
			input_name = 'fc_%s_post' % i

		pre_relu_name = 'fc_%s_pre' % (i + 1)
		post_relu_name = 'fc_%s_post' % (i + 1)
		relu_name = 'relu_%s' % (i+ 1)
		add_linear_layer_mip(relunet, i, gurobi_squire, 
							 input_name, pre_relu_name)
		add_relu_layer_mip(relunet, i, gurobi_squire,
						   pre_relu_name, relu_name, post_relu_name)

	if isinstance(relunet.fcs[-1], nn.Linear):
		output_var_name = 'logits'
		add_linear_layer_mip(relunet, len(relunet.fcs) - 1, gurobi_model,
							 gurobi_squire, post_relu_name, output_var_name)
	gurobi_model.update()


def build_back_pass_constraints(relunet, gurobi_squire):
	""" For relunet like f(x) = c R_l-1(L_l-1 ... R0(L0x))
		which has l units of Linear->ReLu
		and we only want to encode backprop up to (and including) 
		the target_units[0]'th one of them

		So we need to encode [LRLRLRC], 
		and we'll 
	"""

	if relunet.target_units is None:
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
			add_first_backprop_layer(relunet, gurobi_squire,
									 'c_vector', linear_out_key)
		else:
			add_backprop_linear_layer(relunet, i, gurobi_squire,
									  linear_in_key, linear_out_key)
		add_backprop_switch_layer_mip(relunet, i, gurobi_squire,
									  linear_out_key, relu_key, switch_out_key)
		# TODO: ENCODE DIRECTION VECTOR TO IMPLICITLY TAKE NORMS

	# And the final layer
	final_output_key ='gradient'
	add_backprop_linear_layer(relunet, stop_idx, gurobi_squire,
							  switch_out_key, final_output_key)
	gurobi_model.update()


def build_objective(relunet, gurobi_squire, lp):
	gradient_key = 'gradient'
	if lp == 'linf':
		abs_sign_key = 'abs_sign'
		abs_grad_key = 'abs_grad'
		add_abs_layer(relunet, gurobi_model, gurobi_squire,
					  gradient_key, abs_sign_key, abs_grad_key)
		set_l1_objective(gurobi_model, gurobi_squire, abs_grad_key)
	gurobi_model.update()


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
	if isinstance(domain, Hyperbox):
		box_low, box_hi = domain.box_low, domain.box_hi
		for i in range(len(box_low)):
			input_vars.append(model.addVar(lb=box_low[i], ub=box_hi[i],
										   name=var_namer(i)))
	else:
		raise NotImplementedError("Only hyperboxes allowed for now!")
	model.update()
	squire.set_vars(var_key, input_vars)



def add_linear_layer_mip(network, layer_no, squire,
						 input_key, output_key):
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


def add_relu_layer_mip(network, layer_no, squire, input_key,
					   sign_key, output_key):
	post_relu_vars = []
	relu_vars = {} # keyed by neuron # (int)
	post_relu_namer = utils.build_var_namer(output_key)
	relu_namer = utils.build_var_namer(sign_key)
	input_box = squire.get_ith_relu_box(layer_no)
	for i, (low, high) in enumerate(preact_bounds):
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


def add_first_backprop_layer(network, squire, input_key, output_key):
	""" Encodes the backprop of the first linear layer.
		All the variables will be constant, and dependent upon the
		c_vector
	"""
	output_vars = []
	output_var_namer = utils.build_var_namer(output_key)
	backprop_vars = squire.pre_bounds.backprop_domain.encode_as_gurobi_model()

	output_vars.append(model.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY,
									name=output_var_namer(i)))
	squire.set_vars(output_key, output_vars)

	if isinstance(network.fcs[-1], nn.Linear):
		weight = utils.as_numpy(network.fcs[-1].weight).T
		for i in range(len(output_vars)):
			model.addConstr(output_vars[i] == LinExpr(backprop_vars, weight[i]))

	elif isinstance(network.fcs[-1], nn.Identity):
		# Just add v_i = v_i^+ - v_i^- constraints
		# for v_i^+, v_i^- in range [0.0, 1.0]
		for i in range(len(output_vars)):
			model.addConstr(output_vars[i] == backprop_vars[i])
	model.update()


def add_backprop_linear_layer(network, layer_no, model, squire,
							  input_key, output_key):
	""" Encodes the backprop version of a linear layer """
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



def add_backprop_switch_layer_mip(network, layer_no, squire,
    							  input_key, relu_key, output_key):
	"""
	Encodes the funtion
	output_key[i] := 0            if sign_key[i] == 0
				     input_key[i] if sign_key != 0
	where 0 <= input_key[i] <= squire.backprop_pos_bounds[i]
	"""
	raise RefactorError # CLEAN UP SWITCH FUNCTION CASES
	switchbox = squire.get_ith_switch_box(layer_no)	
	switch_inbox = squire.get_ith_backward_box(layer_no)
	post_switch_vars = []
	post_switch_namer = utils.build_var_namer(output_key)

	# First add variables
	for idx, val in enumerate(switchbox):
		post_switch_name = post_relu_namer(idx)
		if val < 0:
			lb = ub = 0.0 

		elif val > 0:
			# Switch is always on
			lb, ub = switch_inbox[idx]
		else:
			lb = min([0, switch_inbox[idx][0]])
			ub = max([0, switch_inbox[idx][1]])
		post_switch_vars.append(model.addVar(lb=lb, ub=ub, name=name))
	squire.set_vars(output_key, post_switch_vars)		


	# And then add constraints
	relu_vars = squire.get_vars(relu_key)
	for idx, val in enumerate(switchbox):
		relu_var 		= relu_vars[idx]
		pre_switch_var  = pre_switch_vars[i]		
		post_switch_var = post_switch_vars
		if val < 0:
			continue
		elif val > 0:
			model.addConstr(post_switch_var == pre_switch_var)
			continue
		else:
			# In this case, the relu is uncertain and we need to encode 
			# 4 constraints. This depends on the backprop low or high though
			bp_lo, bp_hi = switch_inbox[idx]
			if backprop_hi < 0:
				model.addConstr(post_switch_var >= bp_lo * relu_var)
				model.addConstr(post_switch_var <= 0)
				model.addConstr(post_switch_var >= pre_switch_var)
				model.addConstr(post_switch_var <= pre_switch_var - bp + 
												   bp_lo * relu_var)
			elif backprop_lo > 0:
				model.addConstr(post_switch_var >= 0)
				model.addConstr(post_switch_var <= bp_hi * relu_var)
				model.addConstr(post_switch_var <= pre_switch_var)
				model.addConstr(post_switch_var >= pre_switch_var - bp_hi + 
												   bp_hi * relu_var)
			else:
				model.addConstr(post_switch_var >= bp_lo * relu_var) 
				model.addConstr(post_switch_var <= bp_hi * relu_var)
				model.addConstr(post_switch_var >= pre_switch_var - bp_hi + 
												   bp_hi * relu_var)
				model.addConstr(post_switch_var <= pre_switch_var - bp_lo + 
												   bp_lo * relu_var)				

	model.update()




def add_abs_layer(network, model, squire, input_key, 
						 sign_key, output_key):
	""" Encodes the absolute value as gurobi models:
		- creates variables keyed by output_key that are the 
			  absolute value of input_key (sign_key are integer variables 
		  to control the nonconvexity of input_key)
		- conservative sign bounds based on preact 0th layer backprop 
		  bounds control signs
	""" 

	tolerance = 1e-8
	input_vars = squire.get_vars(input_key)
	output_namer = utils.build_var_namer(output_key)
	sign_namer = utils.build_var_namer(sign_key)
	output_vars = []
	sign_vars = {}
	grad_bounds = squire.get_backprop_bounds(0, two_col=True)
	for i, input_var in enumerate(input_vars):
		grad_lo, grad_hi = grad_bounds[i]
		output_name = output_namer(i)
		# always positive
		if grad_lo >= 0:
			output_var = model.addVar(lb=-tolerance, name=output_name)
			model.addConstr(output_var == input_var)
		# always negative
		elif grad_hi <= 0:
			output_var = model.addVar(lb=-tolerance, name=output_name)
			model.addConstr(output_var == -input_var)
		# could be positive or negative
		else:
			output_var = model.addVar(lb=- tolerance, 
									  ub=max([abs(grad_lo), grad_hi]) + tolerance, 
									  name=output_name)
			sign_var = model.addVar(lb=0, ub=1, vtype=gb.GRB.BINARY, 
									name=sign_namer(i))
			#model.addConstr(output_var >= 0)
			model.addConstr(output_var >= input_var - tolerance)
			model.addConstr(output_var >= -input_var - tolerance)
			model.addConstr(output_var <= input_var - 2 *  grad_lo * (1 - sign_var) + tolerance)
			model.addConstr(output_var <= -input_var + 2 * grad_hi * sign_var + tolerance)
			sign_vars[i] = sign_var
		output_vars.append(output_var)
	model.update()
	squire.set_vars(output_key, output_vars)
	squire.set_vars(sign_key, sign_vars)




def set_l1_objective(model, squire, abs_key):
	""" Sets the objective for the sum of the abs_keys
		(absolute value has already been encoded by abs_layer)
	"""
	abs_vars = squire.get_vars(abs_key)
	model.setObjective(sum(abs_vars), gb.GRB.MAXIMIZE)
	model.update()



# ======  End of             LAYERWISE HELPERS                      =======
