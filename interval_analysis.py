""" Techniques to pass domains through the operation of a neural net """
import numpy as np
from hyperbox import Hyperbox, BooleanHyperbox
from l1_balls import L1Ball
import utilities as utils 
import gurobipy as gb


class HBoxIA(object):
	""" Class that holds all information about pushing Hyperboxes through 
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
		- the i^th forward_box is the Hyperbox denoting the input to the 
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
		  takes in a Hyperbox and a BooleanHyperbox and maps to another 
		  Hyperbox.
		- the i^th backward_box is the Hyperbox denoting the input to the 
		  i^th switch
		  		f(x) = c^T L2( ReLU1( L1( ReLU0( L0(x)))))
		  Nabla f(x) = L0^T( Switch0( L1^T( Switch1( L2C, a1)), a0))
								     ^              ^
								     |              1st backward_box
								     |
								     0^th backward_box
	"""
	VALID_TECHNIQUES = set(['naive_ia']) 

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
			backprop_domain = Hyperbox.build_linf_ball(np.zeros(output_dim), radius)


		self.backprop_domain = backprop_domain

		self.forward_boxes = {}
		self.forward_switches = {}
		self.backward_boxes = {}

		self.output_range = None
		self.gradient_range = None

		self.forward_computed = False
		self.backward_computed = False


	def compute_forward(self, technique='naive_ia'):
		assert technique in self.VALID_TECHNIQUES

		if technique == 'naive_ia':
			linear_outs, final_output = self.forward_nia(self.input_domain,
														 self.network)
			self.forward_boxes = {i: linear_outs[i] 
								  for i in range(len(linear_outs))}
			self.forward_switches = {k: BooleanHyperbox.from_hyperbox(v)
									 for k,v in self.forward_boxes.items()}
			self.output_range = final_output

		self.forward_computed = True


	def compute_backward(self, technique='naive_ia'):
		assert technique in self.VALID_TECHNIQUES
		assert self.forward_computed

		if technique == 'naive_ia':
			linear_outs, final_output = self.backward_nia(self.backprop_domain, 
														  self.network, 
														  self.forward_switches)
			self.backward_boxes = {i: linear_outs[i] for i in 
								   range(len(linear_outs))}
			self.gradient_range = final_output

		self.backward_computed = True


	def get_forward_box(self, i):
		# Returns the input range to the i^th RELU
		return self.forward_boxes[i]

	def get_forward_switch(self, i):
		# Returns the possible configurations of the i^th RELU
		return self.forward_switches[i]

	def get_backward_box(self, i, forward_idx=False):
		# Returns the input range to the i^th SWITCH
		if forward_idx:
			i = self.idx_flip(i)
		return self.backward_boxes[i]



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

		new_obj = HBoxIA(subnetwork, input_domain, backprop_domain)


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
	# =           Naive Interval Analysis Techniques                    =
	# ===================================================================
	@classmethod
	def forward_nia(cls, input_hbox, network):
		""" Computes the full forward pass of a neural network. 
			Returns a list of hyperboxes where the i^th element in the 
			hyperbox indices 
		ARGS:
			input_hbox: Hyperbox that bounds inputs to the first linear layer
			network: ReLUNet object 
		RETURNS:
			(linear_outs, final_output)
			- linear_outs[i]:  is the hyperbox denoting the range of 
			   				   inputs to the i^th ReLU 
			- final_output : Hyperbox representing the output after the 
						     final layer 
		"""

		linear_outs = []
		relu_out = input_hbox
		for i in range(network.num_relus):
			hidden_unit = network.get_ith_hidden_unit(i)
			linear_out, relu_out = cls._forward_nia_layer(relu_out, hidden_unit)
			linear_outs.append(linear_out)


		final_output = cls._forward_nia_layer(relu_out, (network.fcs[-1],))[0]
		return linear_outs, final_output		

	@classmethod 
	def _forward_nia_layer(cls, input_hbox, hidden_unit):
		""" Takes in a hyperbox and a hidden_unit and outputs two new 
			hyperboxes, representing pushing the object through the hyperbox
		ARGS:
			input_hbox: Hyperbox object - represents input to this hyperbox 
			hidden_unit: tuple like (nn.Linear, nn.ReLU) - represents the 
						 operators we wish to map this hyperbox through
		RETURNS:
			(linear_out, relu_out), two hyperboxes where 
			Linear(input_hbox) is a subset of linear_out 
			and 
			ReLU(Linear(input_hbox)) is a subset of relu_out
		"""	
		linear_out = input_hbox.map_linear(hidden_unit[0])
		relu_out = linear_out.map_relu()
		return (linear_out, relu_out)


	@classmethod
	def backward_nia(cls, input_hbox, network, forward_switches):
		""" Computes backwards naive_interval analysis starting 
			from an input hyperbox and forward switches
		ARGS:
			input_hbox: Hyperbox - range of inputs for the final 
						layer the reluNet can be
			network: ReLUNet - object to backprop against
			forward_switches: dict - maps integers to BooleanHyperbox 
							  instances representing the ReLUswitch 
							  positions
		RETURNS:
			(linear_outs, final_output):
			- linear_outs[i]:  is the hyperbox denoting the range of 
			   				   inputs to the i^th switch
			- final_output : Hyperbox representing the output after the 
						     final linear layer
		"""
		linear_outs = []
		switch_out = input_hbox

		# Handle the FINAL linear layer (going forward)
		final_linear = network.fcs[-1]
		linear_outs.append(input_hbox.map_linear(final_linear, forward=False))

		# Now do all the hidden units
		for i in range(network.num_relus - 1, -1, -1):
			hidden_unit = network.get_ith_hidden_unit(i)			
			layer_out = cls._backward_nia_layer(linear_outs[-1],
  	  									        hidden_unit, 
		  									    forward_switches[i])
			linear_outs.append(layer_out[0])

		return (linear_outs[:-1], linear_outs[-1])


	@classmethod 
	def _backward_nia_layer(cls, input_hbox, hidden_unit, switch_box):
		""" Takes in a hyperbox and a hidden_unit (LR) and outputs two new 
			hyperboxes. First we map through the backwards relu (switch)
			using the input hbox and switch box. Then we map through the 
			backwards linear layer. 
		ARGS:
			input_hbox: Hyperbox representing real inputs to the switch layer 
			hidden_unit: tuple like (nn.Linear, nn.ReLU) - represents the 
						 FORWARD direction of the operators we map the 
						 input_hbox through 
			switch_box: BooleanHyperbox representing status of gates for 
						switch layer 
		RETURNS:
			(linear_out, switch_out), two hyperboxes where 
			linear_out is a superset of input_hbox mapped through any 
					   valid switch in switch_box, and the transpose of the 
					   linear layer 
			switch_out is a superset of the inputs in input_hbox mapped
					   by any valid switch combo in switch_box
		"""
		linear, _ = hidden_unit
		switch_out = switch_box.map_switch(input_hbox)
		linear_out = switch_out.map_linear(linear, forward=False)
		return (linear_out, switch_out)



