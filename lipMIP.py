""" Main file to contain lipschitz maximization """

import numpy as np
import gurobipy as gb
import utils
from hyperbox import Hyperbox

"""

   Lipschitz Maximization as a mixed-integer-program

   Ultimately we'll want to maximize ||grad(f)|| over
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

def compute_max_lipschitz(network, domain, l_p, c_vector,
						  preact_method='ia', verbose=False,
						  timeout=None):
	""" Computes the maximum lipschitz constant with a fixed
		domain already set.

		Returns the maximum lipschitz constant and the point that
		attains it
	"""
	assert l_p == 'l_inf' # Meaning we want max ||grad(f)||_1
	start = time.time()

	squire, model = build_gurobi_model(network, domain, l_p,
									   c_vector, preact_method=preact_method,
									   verbose=verbose)

	if timeout is not None:
		model.setParam('TimeLimit', timeout)

	model.optimize()
	if model.Status == 3:
		print("INFEASIBLE")

	end = time.time()
	x_vars = squire.get_vars('x')
	return {'lipschitz': model.getObjective().getValue(),
			'runtime': end - start,
			'best_x': np.array([v.X for v in x_vars])

# ======  End of MAIN SOLVER BLOCK                =======





# ==============================================================
# =           Build Gurobi Model for Lipschitz Comp.           =
# ==============================================================

class GurobiSquire:
	def __init__(self):
		self.var_dict = {}

	def get_vars(self, name):
		return self.var_dict[name]

	def set_vars(self, name, var_list):
		self.var_dict[name] = var_list

	def set_preact_object(self, preact_object):
		self.preact_object = preact_object

	def get_preact_bounds(self, i, two_col):
		return self.preact_object.get_ith_layer_bounds(i, two_col)

	def get_backprop_bounds(self, i, two_col):
		return self.preact_object.get_ith_layer_backprop_bounds(i, two_col)




def build_gurobi_model(network, domain, l_p, c_vector,
					   preact_method='ia', verbose=False):

	# Build model + squire
	squire = GurobiSquire()
	model = gb.Model()


	# Set some parameters
	if not verbose:
		model.setParam('OutputFlag', False)


	# Do all the setup work
	if preact_method == 'ia':
		preacts = PreactivationBounds.naive_ia_from_hyperbox(network, domain)
	else:
		raise NotImplementedError("OTHER PREACT METHODS TBD")

	preacts.backprop_bounds(c_vector)
	squire.set_preact_object(preacts)

	build_input_constraints(network, model, squire, domain, 'x')
	build_forward_pass_constraints(network, model, squire)
	build_back_pass_constraints(network, model, squire)
	build_objective(network, model, squire, l_p)


	# Return the squire and model
	model.update()
	return squire, model



def build_forward_pass_constraints(plnn_instance, gurobi_model,
								   gurobi_squire):

	for i, fc_layer in enumerate(network.fcs[:-1]):
		if i == 0:
			input_name = 'x'
		else:
			input_name = 'fc_%s_post' % i

		pre_relu_name = 'fc_%s_pre' % (i + 1)
		post_relu_name = 'fc_%s_post' % (i + 1)
		relu_name = 'relu_%s' % (i+ 1)
		add_linear_layer_mip(plnn_instance, i, gurobi_model,
							 gurobi_squire, input_name, pre_relu_name)
		add_relu_layer_mip(plnn_instance, i, gurobi_model, gurobi_squire,
						   pre_relu_name, relu_name, post_relu_name)

	output_var_name = 'logits'
	add_linear_layer_mip(network, len(network.fcs) - 1, gurobi_model,
						 gurobi_squire, post_relu_name, output_var_name)
	model.update()


def build_back_pass_constraints(plnn_instance, gurobi_model,
								gurobi_squire, c_vector):
	for i in range(len(plnn_instance.fcs) - 1, -1, -1)
		linear_in_key = 'bp_%s_postswitch' % (i + 1)
		linear_out_key = 'bp_%s_preswitch' % i
		switch_out_key = 'bp_%s_postswitch' % i
		relu_key = 'relu_%s' % i
		if i == 0:
			add_first_backprop_layer(plnn_instance, gurobi_model,
									 linear_out_key, c_vector)

		else:
			add_backprop_linear_layer(plnn_instance, i, gurobi_model,
									  gurobi_squire, linear_in_key,
									  linear_out_key)
		add_backprop_switch_layer_mip(plnn_instance, i, gurobi_model,
									  gurobi_squire, linear_out_key,
									  relu_key, switch_out_key)

	# And the final layer
	final_output_key ='gradient'
	add_backprop_linear_layer(plnn_instance, 0, gurobi_model, gurobi_squire,
							  switch_out_key, final_output_key)
	model.update()


def build_objective(plnn_instance, gurobi_model,
					gurobi_squire, l_p):
	gradient_key =' gradient'
	if l_p == 'l_inf':
		abs_sign_key = 'abs_sign'
		abs_grad_key = 'abs_grad'
		add_abs_layer(plnn_instance, gurobi_model, gurobi_squire,
					  gradient_key, abs_sign_key, abs_grad_key)
		set_l1_objective(gurobi_model, gurobi_squire, abs_grad_key)
	model.update()

# ======  End of Build Gurobi Model for Lipschitz Comp.  =======



# =========================================================================
# =                       LAYERWISE HELPERS                               =
# =========================================================================

def add_input_constraints(network, model, squire, domain, var_key):

	# If domain is a hyperbox, don't need to add any Gurobi constraints
	var_namer = utils.build_var_namer(var_key)
	input_vars = []
	if isinstance(domain, Hyperbox):
		box_low, box_hi = domain.box_low, domain.box_hi
		for i in range(len(box_low)):
			input_vars.append(model.addVar(lb=box_low[i], ub=box_hi[i],
										   name=var_namer(i))
	else:
		raise NotImplementedError("Only hyperboxes allowed for now!")
	model.update()
	squire.set_vars(var_key, input_vars)



def add_linear_layer_mip(network, layer_no, model, squire,
						 input_key, output_key):
	fc_layer = network.fcs[layer_no]
	fc_weight =  utils.as_numpy(fc_layer.weight)
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


def add_relu_layer_mip(network, layer_no, model, squire, input_key,
					   sign_key, output_key):
	post_relu_vars = []
	relu_vars = {} # keyed by neuron # (int)
	post_relu_namer = utils.build_var_namer(output_key)
	relu_namer = utils.build_var_namer(sign_key)
	preact_bounds = squire.get_preact_bounds(layer_no, two_col=True)

	for i, (low, high) in enumerate(preact_bounds):
		post_relu_name = post_relu_namer(i)
		relu_name = relu_namer(i)
		if high <= 0:
			post_relu_vars.append(model,addVar(lb=0.0, ub=0.0,
											   name=post_relu_name))
		else:
			pre_relu = squire.var_dict[input_key][i]
			post_relu_vars.append(model.addVar(lb=low, ub=high,
											   name=post_relu_name))
			post_relu = post_relu_vars[-1]
			if low >= 0:
				model.addConstr(post_relu == pre_relu)
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


def add_first_backprop_layer(network, model, squire, output_key, c_vector):
	""" Encodes the backprop of the first linear layer.
		All the variables will be constant, and dependent upon the
		c_vector
	"""
	output_vars = []
	output_var_namer = utils.var_namer(output_key)

	weight = utils.as_numpy(model.fcs[-1].weight)
	dotted = utils.as_numpy(c_vector).dot(weight)
	for i, el in enumerate(dotted):
		output_vars.append(model.addVar(lb=el, ub=el,
						   name=output_var_namer(i)))

	model.update()
	squire.set_vars(output_key, output_vars)



def add_backprop_linear_layer(network, layer_no, model, squire,
							  input_key, output_key):
	""" Encodes the backprop version of a linear layer """

	output_vars = []
	output_vars_namer = utils.var_namer(output_key)
	fc_layer = network.fcs[layer_no]
	fc_wight = fc_layer.weight

	backprop_bounds = squire.get_backprop_bounds(layer_no)
	input_vars = squire.get_vars(input_key)

	for i in range(len(fc_layer.in_features)):
		output_var = model.addVar(lb=backprop_bounds[i][0],
								  ub=backprop_bounds[i][1],
								  name=output_var_namer(i))
		weight_col = fc.weight[:, i]
		model.addConstr(output_var == gb.LinExpr(weight_col, input_vars))
		output_vars.append(output_var)
	model.update()
	squire.set_vars(output_key, output_vars)



def add_backprop_switch_layer_mip(network, layer_no, model, squire,
    							  input_key, relu_key, output_key):
	"""
	Encodes the funtion
	output_key[i] := 0            if sign_key[i] == 0
				     input_key[i] if sign_key != 0
	where 0 <= input_key[i] <= squire.backprop_pos_bounds[i]
	"""
	post_switch_vars = []
	post_switch_namer = utils.build_var_namer(output_key)
	preact_bounds = squire.get_preact_bounds(layer_no)
	backprop_bounds = squire.get_backprop_bounds(layer_no)
	pre_switch_vars = squire.get_vars(input_key)
	relu_vars = squire.get_vars(relu_key) # binary variables

	for i, (low, high) in enumerate(preact_bounds):
		post_switch_name = post_switch_namer(i)
		pre_switch_var = pre_switch_vars[i]
		backprop_lo, backprop_hi = backprop_pos_bounds[i]
		# Handle corner cases where no relu_var exists
		if high <= 0:
			post_switch_var = model.addVar(lb=0.0, ub=0.0,
										   name=post_switch_name)
		elif low >= 0:
			post_switch_var = model.addVar(lb=backprop_lo, ub=backprop_hi,
										   name=post_switch_name)
			model.addConstr(post_switch_var == pre_switch_var)
		else:
			relu_var = relu_vars[i]
			post_switch_var = model.addVar(lb=backprop_lo, ub=backprop_hi,
										   name=post_switch_name)
			model.addConstr(post_switch_var >= 0)
			model.addConstr(post_switch_var <= pre_switch_var)
			model.addConstr(post_switch_var <= backprop_ub * relu_var)
			model.addConstr(post_switch_var >= pre_switch_var - \
											   backprop_ub * (1 - relu_var))
		post_switch_vars.append(post_switch_var)

	model.update()
	squire.set_vars(output_key, post_switch_vars)



def add_abs_layer(network, model, squire, input_key,
				  sign_key, output_key):
	""" Encodes the function abs(neg[i] + pos[i]) """

	input_vars = squire.get_vars(input_key)
	abs_var_namer = utils.var_namer(output_key)
	sign_var_namer = utils.var_namer(sign_key)
	grad_bounds = squire.get_backprop_bounds(0)

	for i, input_vars in enumerate(input_vars):
		grad_lo, grad_hi = grad_bounds[i]

		abs_var = model.addVar(lb=0.0, ub=max([-grad_low, grad_hi]),
							   name=abs_var_namer(i))
		sign_var = model.addVar(lb=0, ub=1, vtype=gb.GRB.BINARY,
								name=sign_var_namer(i))
		model.addConstr(abs_var >= input_var)
		model.addConstr(abs_var >= -input_var)
		model.addConstr(abs_var <= input_var + (1-sign_var) * (-grad_lo))
		model.addConstr(abs_var <= input_var + sign_var * grad_hi)

		sign_vars.append(sign_var)
		abs_vars.append(abs_var)

	model.update()
	squire.set_vars(output_key, abs_vars)
	squire.set_vars(sign_key, sign_vars)


def set_l1_objective(model, squire, abs_key):
	""" Sets the objective for the sum of the abs_keys
		(absolute value has already been encoded by abs_layer)
	"""
	abs_vars = squire.get_vars(abs_key)
	model.setObjective(-1 * (sum(abs_vars)))
	model.update()


# ======  End of             LAYERWISE HELPERS                      =======
