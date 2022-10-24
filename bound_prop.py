""" Bound propagation techniques """
import numpy as np
from hyperbox import Hyperbox, BooleanHyperbox
from zonotope import Zonotope
from polytope import Polytope
from l1_balls import L1Ball
from linear_bounds import LinearBounds
import utilities as utils 
import gurobipy as gb
import torch 
import torch.nn as nn
from relu_nets import ReLUNet


"""
Basic module instructions: 

Forward passes: 
- need to specify:
	+ network 
	+ input range 
	+ abstract domain to use to represent real ranges
		~ params for abstract domain
	+ abstract domain to use to represent gradient ranges
		~ params here 


Backward passes: 
	+ network 
	+ input range (cvec range)
	+ abstract domain to use to represent real ranges
		~ params for abstract domain
	+ abstract domain to use to represent gradient ranges
		~ params here 

Interface to do both of them together: 
	+ network 
	+ input range 
	+ cvec range 
"""
class AbstractParams(utils.ParameterObject):
	def __init__(self, forward_domain, forward_params, backward_domain, 
				 backward_params, grad_domain, grad_params):
		init_args = {k: v for k, v in locals().items() 
					 if k not in ['self', '__class__']}
		super(AbstractParams, self).__init__(**init_args)	

	@classmethod
	def hyperbox_params(cls):
		return cls(Hyperbox, None, Hyperbox, None, Hyperbox, None)

	@classmethod
	def basic_zono(cls):
		return cls(Zonotope, None, Zonotope, None, Hyperbox, None)

	@classmethod 
	def basic_lb(cls):
		return cls(LinearBounds, None, LinearBounds, None, Hyperbox, None)
	

class AbstractNN2(object):
	VALID_DOMAINS = [Hyperbox, Zonotope, Polytope, LinearBounds]

	HYPERBOX_DEFAULT = {}
	ZONOTOPE_DEFAULT = {'relu_forward': 'deepZ',
	 					'relu_backward': 'smooth'
	 				   }
	POLYTOPE_DEFAULT = {}
	LINEAR_BOUND_DEFAULT = {}
	""" Object that computes boundProp objects  """
	def __init__(self, network):
		self.network = network 

	@classmethod
	def _get_default_params(self, abstract_domain):
		""" Builds default params for abstract domains """
		return {Hyperbox: self.HYPERBOX_DEFAULT, 
				Zonotope: self.ZONOTOPE_DEFAULT,
				Polytope: self.POLYTOPE_DEFAULT,
				LinearBounds: self.LINEAR_BOUND_DEFAULT}[abstract_domain]

	def get_both_bounds(self, abstract_params, input_range, backward_range):
		""" Does the full forward/backward bound with the specified 
			params to do each (encapsulated in the Abstractparams obj)
		"""
		p = abstract_params # shorthand
		f_bound = self.get_forward_bounds(p.forward_domain, input_range, 
	   									  abstract_params=p.forward_params)

		b_bound = self.get_backward_bounds(p.backward_domain, p.grad_domain, 
 										   backward_range, f_bound, 
										   abstract_params=p.backward_params,
										   grad_params=p.grad_params)
		return (f_bound, b_bound)


	def get_forward_bounds(self, abstract_domain, input_range, 
   					  	   abstract_params=None):
		assert abstract_domain in self.VALID_DOMAINS

		default_params = self._get_default_params(abstract_domain)
		default_params.update(abstract_params or {})

		# First convert input range into the abstract_domain 
		cast_input_range = abstract_domain.cast(input_range)

		# And then do the bound propagation for each layer:
		forward_outs = self.forward_pushforward(self.network, abstract_domain, 
												cast_input_range, 
												abstract_params)

		# Finally make the output object to report 
		return BoundPropForward(self.network, abstract_domain, input_range, 
								forward_outs, abstract_params)


	@classmethod
	def forward_pushforward(cls, network, abstract_domain, input_ai,
							abstract_params):
		""" Does the pushforward for all layers of a neural network. 
			Will return a list of abstract domains for the PRE-ACTIVATION BOUNDS
		ARGS:
			network : ReLUNet (or other) - network to push bounds through 
			abstract_domain : class in of VALID_DOMAINS - domain to push through 
			input_ai : element of <abstract_domain> - input range to be pushed 
					   through 
		RETURNS:
			BoundProp object
		"""
		ai_outs = []
		working_ai = input_ai

		# Protocol here: (DECOUPLING LINEAR AND NONLINEAR LAYERS NOW)
		for i in range(len(network.net)):
			working_ai = working_ai.map_layer_forward(network, i, abstract_params)
			ai_outs.append(working_ai)
		return ai_outs


	def get_backward_bounds(self, abstract_domain, grad_domain, 
					        input_range,  forward_bounds, 
					        abstract_params=None, 
					   	    grad_params=None):
		assert abstract_domain in self.VALID_DOMAINS

		default_params = self._get_default_params(abstract_domain)
		default_params.update(abstract_params or {})

		# First convert input range into the abstract_domain 
		cast_input_range = abstract_domain.cast(input_range)

		# Get gradient ranges for forward_bounds 
		grad_ranges = forward_bounds.gradient_ranges(grad_domain,
													 grad_params)

		# Do bound propagation in the backward direction 
		backward_outs = self.backward_pushforward(self.network, abstract_domain, 
												  cast_input_range, grad_ranges, 
												  abstract_params)

		# Finally make the output object
		return BoundPropBackward(self.network, abstract_domain, input_range, 
								 forward_bounds, backward_outs, 
								 abstract_params=abstract_params,
								 grad_ranges=grad_ranges)



	@classmethod 
	def backward_pushforward(cls, network, abstract_domain, input_ai, 
							 grad_bounds, abstract_params):
		""" Does the pushforward (BACKWARD DIRECTION) for all layers of a 
			neural network. 
		ARGS:
			network : ReLUNet (or other) - network to push bounds through 
			abstract_domain : class in of VALID_DOMAINS - domain to push through 
			input_ai : element of <abstract_domain> - input range to be pushed 
					   through 			
		"""
		ai_outs = [] 
		working_ai = input_ai 
		for i in range(len(network.net)):
			grad_bound = grad_bounds[-(i + 1)]
			working_ai = working_ai.map_layer_backward(network, i, grad_bound, 
													   abstract_params)
			ai_outs.append(working_ai)
		return ai_outs


class BoundPropForward(object):
	""" Object that holds info about bound propagation -- is built by 
		the AbstractNN class. Mostly just holds info
	"""
	def __init__(self, network, abstract_domain, input_range, layer_ranges, 
				 abstract_params=None):
		""" 
		ARGS:	
			network : ReLUNet - object we propagate bounds over 
					  (TODO: extend to other nonlinearities) 
			abstract_domain : <abstract_class> - class like Zonotope, HyperBox, 
							  Polytope that we use to propagate 
			input_range : ??? - set that the input can take 
			layer_ranges : <abstract_class>[] - list of forward bound props
			abstract_params : dict -> dict with extra params to describe 
								  how the bound propagations were performed 
		"""
		self.network = network 
		self.abstract_domain = abstract_domain 
		self.input_range = input_range 
		self.layer_ranges = layer_ranges
		self.abstract_params = abstract_params
		self.output_range = layer_ranges[-1]

	def gradient_ranges(self, grad_domain, grad_params):
		""" Gets the gradient ranges for the specified grad_domain 
			and grad_params 
		"""
		grad_ranges = [] # None's where
		for i, layer in enumerate(self.network.net, -1):
			if isinstance(layer, (nn.ReLU, nn.LeakyReLU)):
				grad_ranges.append(grad_domain.relu_grad(self.layer_ranges[i], 
														 grad_params))
			elif isinstance(layer, (nn.Sigmoid, nn.Tanh)):
				grad_ranges.append(grad_domain.smooth_grad(self.layer_ranges[i], 
														   layer))
			else:
				grad_ranges.append(None)
		return grad_ranges


	def get_forward_box(self, i):
		if i == 0:
			return self.input_range.as_hyperbox()
		else:
			return self.layer_ranges[i - 1].as_hyperbox()



class BoundPropBackward(object):
	""" Object that holds info about bound propagation for gradient info 
		-- is built by the AbstractNN class, and just holds info 
	"""
	def __init__(self, network, abstract_domain, input_range, forward_bounds, 
				 backward_bounds, abstract_params=None, 
				 grad_ranges=None):
		""" 
		ARGS:	
			network : ReLUNet - object we propagate bounds over 
					  (TODO: extend to other nonlinearities) 
			abstract_domain : <abstract_class> - class like Zonotope, HyperBox, 
							  Polytope that we use to propagate 
			input_range : ??? - set that the input can take 
			forward_bounds : BoundPropForward object - contains info from 
							 the forward pass of bound propagation 
			backward_bounds : <abstract_domain>[] - list of partial gradient 
							  ranges 
			abstract_params : dict -> dict with extra params to describe 
								  how the bound propagations were performed 		
		"""
		self.network = network
		self.abstract_domain = abstract_domain
		self.input_range = input_range
		self.forward_bounds = forward_bounds
		self.backward_bounds = backward_bounds
		self.abstract_params = abstract_params
		self.grad_ranges = grad_ranges

		self.output_range = backward_bounds[-1]

	def get_backward_box(self, i):
		""" Returns the range of backprop input to the i^th layer 
		ARGS:
			i: int - forward index of layer we want inputs to.
				(e.g. 0 here corresponds to the ranges into the first linear layer)
		RETURNS:
			Hyperbox
		"""
		if i == len(self.network.net) - 1:
			if utils.arraylike(self.input_range):
				return Hyperbox.from_vector(self.input_range)
		else:
			forward_idx = len(self.network.net) - i - 2
			return self.backward_bounds[forward_idx].as_hyperbox()





class TestBoundProp:
	def __init__(self, network, params, input_range, output_range):
		""" Test suite for bound propagation testing """
		self.network = network
		forward, backward = AbstractNN2(network).get_both_bounds(params, 
															     input_range, 
															     output_range)
		self.forward = forward 
		self.backward = backward 

	def check_forward_backward_all(self, num_points=1000):
		""" Checks the forward/backward bounds being correct for 
			every layer 
		"""
		forw_errs, back_errs = [], []
		for i, _ in enumerate(self.network.net):
			out = self.check_forward_backward_idx(i, num_points=num_points, 
												  do_asserts=False)
			if out[0] > 0:
				forw_errs.append((i, out[0]))
			if out[1] > 0:
				back_errs.append((i, out[1]))


		all_check = (len(forw_errs) + len(back_errs) == 0)
		if all_check:
			print("ALL INDICES PASSED CHECKS! ")
		else:
			if len(forw_errs) > 0:
				print("FORWARD ERRORS:", forw_errs)
			if len(back_errs) > 0:
				print("BACKWARD ERRORS:", back_errs)

		forw_errs, back_errs


	def check_forward_backward_idx(self, start_idx, num_points=1000, 
								   do_asserts=True):
		""" Checks the forward and backward bounds being correct 
			based on random points
		- Pick an index and only consider network[idx:]
		- Pick a bunch of random points in the input to the idx'th layer 
		- Pass each point through the suffix of the network and assert it 
		  is in the right range 
		- assert the gradient (of the output wrt the idx'th input) lies 
		  in the idx'th backprop range 
		"""

		if start_idx == 0:
			input_range = self.forward.input_range
			output_range = self.backward.output_range 
		else:
			input_range = self.forward.layer_ranges[start_idx - 1]
			output_range = self.backward.backward_bounds[-(start_idx + 1)]


		rand_points = input_range.random_point(num_points, requires_grad=True)
		forward_outs = self.network.partial_forward(rand_points, start_idx)
		y = (forward_outs @ self.backward.input_range).sum() 
		y.backward() 
		backward_outs = rand_points.grad 

		forw_errs = num_points - sum(self.forward.output_range.contains(forward_outs))
		back_errs = num_points - sum(output_range.contains(backward_outs))

		if do_asserts:
			assert forw_errs == 0
			assert back_errs == 0
		return forw_errs, back_errs

