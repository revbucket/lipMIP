""" Techniques to pass domains through the operation of a neural net """
import numpy as np
from hyperbox import Hyperbox, BooleanHyperbox
from zonotope import Zonotope
from l1_balls import L1Ball
import utilities as utils 
import gurobipy as gb
import torch 
import torch.nn as nn

global VALID_PREACTS
VALID_PREACTS = set(['naive_ia', 'zonotope:deep', 'zonotope:smooth',
						'zonotope:box', 'zonotope:switch', 
						'zonotope:diag']) 

class AbstractNN(object):
	""" Class that holds all information about pushing abstract domains through 
		a neural network (and does backpropagation too).
		We need a clear discussion about terminology/indices:
		- Our neural net matches Regex: r/(LR)+L/ (going left to right)
		  where {L: Linear layer, R: ReLU layer}. We'll typically only care 
		  about networks with scalar outputs, so we'll want to dot-product
		  the output with a vector to get a scalar. The choice of vector 
		  is handled by the backprop_domain argument
		- the i^th Hidden Unit is the i^{th} (LR) block, starting with 0
		- An input domain is a hyperbox over the input to the neural net 
		  that we're pushing through the network 
		- the i^th forward_domain is the abstract domain denoting the input to the 
		  i^th ReLU, 
		  e.g. 
		  	f(x) = L2( ReLU( L1( ReLU( L0(x)))))
  		  				    ^         ^
		  				    |		  0th forward_box
		  				    |
		  					1st forward_box

		- the backprop_domain is EITHER a 1-dimensional tensor OR a Domain 
		  representing a tensor that the output vector can be.
		- Backprop is represented by "Switches" where a switch function 
		  takes in an abstract domain and a BooleanHyperbox and maps to another 
		  abstract domain of the same type.
		- the i^th backward_box is the abstract domain denoting the input to the 
		  i^th switch
		  		f(x) = c^T L2( ReLU1( L1( ReLU0( L0(x)))))
		  Nabla f(x) = L0^T( Switch0( L1^T( Switch1( L2C, a1)), a0))
								     ^              ^
								     |              1st backward_domain
								     |
								     0^th backward_domain
	"""

	def __init__(self, network, input_domain, backprop_domain):
		self.network = network
		self.input_domain = input_domain

		self.c_vector = backprop_domain # HACKETY HACK
		if utils.arraylike(backprop_domain):
			backprop_domain = Hyperbox.from_vector(backprop_domain)
		elif backprop_domain in ['l1Ball1', 'crossLipschitz', 'targetCrossLipschitz',
							     'trueCrossLipschitz', 'trueTargetCrossLipschitz']:
			output_dim = network.layer_sizes[-1]
			if backprop_domain == 'l1Ball1':
				radius = 1.0
			else:
				radius = 2.0
			backprop_domain = Hyperbox.build_linf_ball(torch.zeros(output_dim), 
													   radius)


		self.backprop_domain = backprop_domain

		self.forward_domains = {}
		self.forward_switches = {}
		self.backward_domains = {}

		self.output_range = None
		self.gradient_range = None

		self.forward_computed = False
		self.backward_computed = False


	def compute_forward(self, technique='naive_ia'):
		assert technique in VALID_PREACTS

		if technique == 'naive_ia':
			domain = 'hyperbox'
			pf_kwargs = {}
		elif technique.startswith('zonotope'):
			domain = 'zonotope'
			pf_kwargs = {'transformer': technique.split(':')[1]}

		linear_outs, final_out  = self.forward_pushforward(self.input_domain,
   									self.network, domain=domain, **pf_kwargs)

		self.forward_domains = {i: linear_outs[i]
							  for i in range(len(linear_outs))}
		self.forward_switches = {k: v.as_boolean_hbox() 
								 for k, v in self.forward_domains.items()}
		self.output_range = final_out
		self.forward_computed = True


	def compute_backward(self, technique='naive_ia'):
		assert technique in VALID_PREACTS
		assert self.forward_computed
		if technique == 'naive_ia':
			domain = 'hyperbox'
			pf_kwargs = {}
		elif technique.startswith('zonotope'):
			domain = 'zonotope'
			pf_kwargs = {'transformer': technique.split(':')[1]}
		linear_outs, final_out = self.backward_pushforward(self.backprop_domain,
											self.network, self.forward_switches, 
											domain=domain, **pf_kwargs)
		self.backward_domains = {i: el for i, el in enumerate(linear_outs)}
		self.gradient_range = final_out
		self.backward_computed = True


	def get_forward_box(self, i):
		# Returns the input range to the i^th RELU
		return self.forward_domains[i]

	def get_forward_switch(self, i):
		# Returns the possible configurations of the i^th RELU
		return self.forward_switches[i]

	def get_backward_box(self, i, forward_idx=False):
		# Returns the input range to the i^th SWITCH
		if forward_idx:
			i = self.idx_flip(i)
		return self.backward_domains[i]



	def idx_flip(self, i):
		""" Takes either the forward indices and makes them backward indices
			or vice versa 
		"""
		return self.network.num_relus - i


	def slice_from_subnet(self, subnetwork):
		""" Takes in a SubReLUNet and which has access to the parent_range 
			and returns a sliced/reindexed object
		"""
		assert self.forward_computed and self.backward_computed
		assert subnetwork.parent_net == self.network
		start, stop = subnetwork.parent_units

		# --- step 1: make the new object with the right constructor args
		# 		Input domain is the previous layer's ReLU out (LRLRLR)
		if start == 0:
			input_domain = self.input_domain 
		else:
			input_domain = self.get_forward_box(start - 1).map_relu()

		# 		Backprop domain is the previous layer's out
		if paret_units[1] == self.network.num_relus:
			backprop_domain = self.backprop_domain
		else:
			backprop_domain = self.get_backward_box(self.idx_flip(stop))

		new_obj = AbstractNN(subnetwork, input_domain, backprop_domain)


		# --- step 2: manually set the right attributes here
		# 		Set the hyperboxes with the right (sliced!) indexing
		new_forward_boxes = {}
		new_forward_switches = {} 
		new_backward_boxes = {}

		for forward_idx in range(start, stop):
			new_forward_idx = forward_idx - start
			back_idx = self.idx_flip(forward_idx)
			new_back_idx = back_idx + end

			new_forward_boxes[new_forward_idx] = self.get_forward_box(forward_idx)
			new_forward_switches[new_forward_idx] = self.get_forward_switch(forward_idx)
			new_backward_boxes[new_back_idx] = self.get_backward_box(back_idx)

		new_obj.forward_boxes = new_forward_boxes
		new_obj.forward_switches = new_forward_switches
		new_obj.backward_boxes = new_backward_boxes
		new_obj.forward_computed = True 
		new_obj.backward_computed = True 

		#		And set the output_range, gradient_ranges
		if end == self.num_relus:
			new_obj.output_range = self.output_range
		else:
			new_obj.output_range = self.get_forward_box(end - 1).map_relu()

		if start == 0:
			new_obj.gradient_range = self.gradient_range
		else:
			new_obj.gradient_range = self.get_backward_box(self.flip_idx(start), 
														   forward=False)

		return new_obj

	def gurobi_backprop_domain(self, squire, key):
		""" Adds variables representing feasible points in the 
			backprop_domain to the gurobi model. These are based on the
			c_vector and not the backprop domain
		ARGS:
			squire : gurobiSquire object - holds the model 
			key: string - key for the new variables to be added
		RETURNS:
			gurobipy Variables[] - list of variables added to gurobi
		"""
		VALID_C_NAMES = ['crossLipschitz', # m-choose-2, convex hull w/ simplex
						 'targetCrossLipschitz', # m-1, convex hull w/simplex
						 'trueCrossLipschitz', # m-choose-2, MIP
						 'trueTargetCrossLipschitz', #m-1, MIP
						 'l1Ball1' # C can be in the l1 ball of norm 1
						 ]
		assert utils.arraylike(self.c_vector) or self.c_vector in VALID_C_NAMES
		model = squire.model
		namer = utils.build_var_namer(key)

		# HANDLE HYPERBOX CASE 
		if isinstance(self.c_vector, Hyperbox):
			return self.c_vector.encode_as_gurobi_model(squire, key)


		# HANDLE FIXED C-VECTOR CASE
		gb_vars = []
		if utils.arraylike(self.c_vector):
			for i, el in enumerate(self.c_vector):
				gb_vars.append(model.addVar(lb=el, ub=el, name=namer(i)))
			squire.set_vars(key, gb_vars)
			squire.update()
			return gb_vars


		# HANDLE STRING CASES (CROSS LIPSCHITZ, l1ball)

		output_dim = self.network.layer_sizes[-1]
		if self.c_vector == 'l1Ball1':
			l1_ball = L1Ball.make_unit_ball(output_dim)
			l1_ball.encode_as_gurobi_model(squire, key)
			return squire.get_vars(key)

		if self.c_vector == 'crossLipschitz':
			# --- HANDLE CROSS LIPSCHITZ CASE			
			gb_vars = [model.addVar(lb=-1.0, ub=1.0, name=namer(i)) 
					   for i in range(output_dim)]			
			pos_vars = [model.addVar(lb=0.0, ub=1.0) for i in range(output_dim)]
			neg_vars = [model.addVar(lb=0.0, ub=1.0) for i in range(output_dim)]
			model.addConstr(sum(pos_vars) <= 1)
			model.addConstr(sum(neg_vars) <= 1)
			model.addConstr(sum(neg_vars) <= sum(pos_vars))

		if self.c_vector =='trueCrossLipschitz':
			# --- HANDLE TRUE CROSS LIPSCHITZ CASE 
			gb_vars = [model.addVar(lb=-1.0, ub=1.0, name=namer(i)) 
					   for i in range(output_dim)]						
			pos_vars = [model.addVar(lb=0, ub=1, vtype=gb.GRB.BINARY) 
						for i in range(output_dim)]
			neg_vars = [model.addVar(lb=0, ub=1, vtype=gb.GRB.BINARY)
						for i in range(output_dim)]
			model.addConstr(sum(pos_vars) == 1)
			model.addConstr(sum(neg_vars) == 1)
			for i in range(output_dim):
				model.addConstr(gb_vars[i] ==pos_vars[i] - neg_vars[i])

		if self.c_vector == 'targetCrossLipschitz':
			network = squire.network
			center = squire.pre_bounds.input_domain.get_center()
			label = network.classify_np(center)

			label_less_vars = []
			for i in range(output_dim):
				if i == label:
					gb_vars.append(model.addVar(lb=1.0,ub=1.0, name=namer(i)))
				else:
					new_var = model.addVar(lb=-1.0, ub=0.0, name=namer(i))
					label_less_vars.append(new_var)
					gb_vars.append(new_var)
			model.addConstr(sum(label_less_vars) >= -1.0)

		if self.c_vector == 'trueTargetCrossLipschitz':
			network = squire.network
			center = squire.pre_bounds.input_domain.get_center()
			label = network.classify_np(center)
			int_vars = []
			for i in range(output_dim):
				if i == label:
					gb_vars.append(model.addVar(lb=1.0,ub=1.0, name=namer(i)))
				else:
					gb_vars.append(model.addVar(lb=-1.0, ub=0.0, name=namer(i)))
					int_vars.append(model.addVar(lb=0, ub=1, vtype=gb.GRB.BINARY))
					model.addConstr(gb_vars[-1] == -int_vars[-1])
			model.addConstr(sum(int_vars) <= 1)
		squire.set_vars(key, gb_vars)
		squire.update()
		return gb_vars


	# ===================================================================
	# =           Zonotope Abstract Interpration Techniques             =
	# ===================================================================
	@classmethod
	def forward_pushforward(cls, input_ai, network, domain=None,
							**pushforward_kwargs):
		""" Computes the full forward pass of a neural network.
			Returns a list of abstract domains
		ARGS:
			input_ai: Hyperbox or Zonotope that bounds inputs to the first
					  linear layer 
			network : ReLUNet object 
			domain: either 'hyperbox' or 'zonotope'
			pushforward_kwargs : extra kwargs to be sent to the map_relu method
		RETURNS:
			(linear_outs, final_output)
			- linear_outs[i]: is the abstract domain denoting the range of 
							  inputs to the i^th ReLU 
			- final_output: abstract domain representing the output after the 
							final linear layer
		"""

		assert domain in ['hyperbox', 'zonotope']

		if domain == 'zonotope':
			input_ai = Zonotope.as_zonotope(input_ai)

		linear_outs = []
		relu_out = input_ai 
		for i in range(network.num_relus):
			hidden_unit = network.get_ith_hidden_unit(i)
			linear_out, relu_out = cls._forward_pushforward_layer(
											relu_out, network, i,
					      					**pushforward_kwargs)
			linear_outs.append(linear_out)
		final_output = cls._forward_pushforward_layer(relu_out, network, -1,
							 						  **pushforward_kwargs)[0]

		return linear_outs, final_output


	@classmethod
	def _forward_pushforward_layer(cls, input_ai, network, index, **pf_kwargs):
		""" Takes in an abstract_domain and a hidden_unit and outputs two new 
			abstract domains, representing pushing the object through the layer
		ARGS:
			input_ai: Hyperbox or Zonotope - represents input to this hyperbox 
			network:  ReLUNet or ConvNet object 
			index :   which element of the sequential we are pushing forward 
		RETURNS:
			(linear_out, relu_out), two hyperboxes where 
			Linear(input_hbox) is a subset of linear_out 
			and
			ReLU(Linear(input_hbox)) is a subset of relu_out
		"""
		hidden_unit = network.get_ith_hidden_unit(index)
		if isinstance(hidden_unit[0], nn.Linear):
			out = input_ai.map_linear(hidden_unit[0], forward=True)
		elif isinstance(hidden_unit[0], nn.Conv2d):
			out = input_ai.map_conv2d(network, index, forward=True)
		else:
			return NotImplementedError("Linear + Conv2D only!")
		relu_out = out.map_relu()
		return (out, relu_out)


	@classmethod
	def backward_pushforward(cls, input_ai, network, forward_switches, 
							 domain=None, **pushforward_kwargs):
		""" Compute the full backwardpass of a neural network
			Returns a list of abstract domains 
		ARGS:
			input_ai : Hyperbox or Zonotope that bounds inputs to the gradient
			network: ReLUNet object 
			domain: either 'hyperbox' or 'zonotope'
			pushforward_kwargs : extra_kwargs to be sent to the map_relu method 
		RETURNS:
			(linear_outs, final_output)
		"""
		assert domain in ['hyperbox', 'zonotope']
		if domain == 'zonotope':
			input_ai = Zonotope.as_zonotope(input_ai)

		# Handle the FINAL linear layer (going forward)
		final_layer = network.fcs[-1]
		assert isinstance(final_layer, nn.Linear)
		layer_outs = [input_ai.map_linear(final_layer, forward=False)]

		# Now do all the hidden units
		for i in range(network.num_relus - 1, -1, -1):
			layer_out = cls._backward_pushforward_layer(layer_outs[-1],
		  	  									        network, i,
				  									    forward_switches[i],
				  									    **pushforward_kwargs)
			layer_outs.append(layer_out[0])

		return (layer_outs[:-1], layer_outs[-1])


	@classmethod
	def _backward_pushforward_layer(cls, input_ai, network, index, switch_box,
									**pf_kwargs):
		""" Takes in an abstract element and hidden unit (LR) and outputs two 
			new abstract elements. First we map through the backwards ReLU 
			(switch) using the input_ai and switch_box, then we map through 
			the backwards linear layer 
		ARGS:
			input_ai : Zonotope or Hyperbox containing real inputs to switch 
					   layer
			network : ReLUNet or ConvNet object 
			index : int - FORWARD index of which linear/conv operator we
						  are mapping through
			switch_box: BooleanHyperbox representing status of gates for 
						switch layer 
		RETURNS:
			(layer_out, switch_out), two abstract_elements where 
			switch_out is a superset of input_ai mapped through any 
					   valid switch in switch_box, and the transpose of the 
					   linear layer 
			layer_out is a superset of the inputs in input_ai mapped through 
					   the switch and then the linear/conv2d layer 
		"""
		layer, _ = network.get_ith_hidden_unit(index)
		switch_out = input_ai.map_switch(switch_box, **pf_kwargs)
		if isinstance(layer, nn.Linear):
			layer_out = switch_out.map_linear(layer, forward=False)
		elif isinstance(layer, nn.Conv2d):
			layer_out = switch_out.map_conv2d(network, index, forward=False)
		else:
			return NotImplementedError("Linear + Conv2D only!")		
		return (layer_out, switch_out)


