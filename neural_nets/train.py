""" File to hold training techniques for 
	-- general purpose training 
	-- adversarial training 
	-- helpful training techniques for ablation tests 
"""
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
import torch.autograd as autograd

import pickle
from abc import ABC, abstractmethod


# ===========================================================================
# =           General purpose training                                      =
# ===========================================================================


def training_loop(network, trainset, valset, num_epochs, optimizer=None, 
				  loss_functional=None, test_after_epoch=True):
	""" Trains a network over the trainset with given loss and optimizer 
	ARGS:
		network: nn.module to be trained 
		trainset: torch.utils.data.DataLoader object of the training data 
		valset: torch.utils.data.DataLoader object of the validation data 
		optimizer: None or torch.optim object - if None will be 
				   initialized to optim.Adam object with default parameters
		loss_function: None or loss object - if None will default to 
					   crossEntropyLoss (see LossObject at bottom of this )
		test_after_epoch: bool - if True, we test top1 accuracy after each 
								 epoch 
	RETURNS:
		None, but modifies network parameters 
	"""

	if optimizer is None:
		optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=0)
	if loss_functional is None:
		loss_functional = LossFunctional(network, 
 										 regularizers=[XEntropyReg(network)])

	for epoch_no in range(num_epochs):
		for examples, labels in trainset:
			optimizer.zero_grad()			
			loss = loss_functional(examples, labels)			
			loss.backward() 
			optimizer.step() 
		if test_after_epoch:
			test_acc = test_validation(network, valset)
			test_acc_str = '| Accuracy: %.02f' % (test_acc * 100)
		else: 
			test_acc_str = ''
		print(("Epoch %02d " % epoch_no) + test_acc_str)


def test_validation(network, valset, loss_functional=None):
	""" Report the top1 accuracy of network over the 
		if loss_functional is None: 
			Returns top1 accuracy in range [0.0, 1.0]
		else:
			Returns average loss value over valset
		(Assumes devices are already set)
	"""
	total = 0
	count_value = 0
	for examples, labels in valset:
		total += labels.numel()
		if loss_functional is None:
			count_value += (network(examples).max(1)[1] == labels).sum().item()
		else:
			count_value += loss_functional(examples, labels).data.item()
	return float(count_value) / total


def train_cacher(saved_name, network=None, trainset=None, valset=None, num_epochs=None, 
				 loss_functional=None, test_after_epoch=True):
	""" Convenient helper function to try and load a neural_net with a 
		given name, and if this doesn't exist, train one with the specified params
		and then save it
	ARGS:
		saved_name: (str) if not None, must end with .pkl - is the name of the 
					pickled object

	RETURNS:
		ReLUNet object that's been trained or pre-loaded

	"""
	try:
		trained_net = pickle.load(open(saved_name, 'rb'))
		print("Successfully loaded: " + saved_name)
		return trained_net
	except:
		print("Training network: " + saved_name)
		training_loop(network, trainset, valset, num_epochs, 
   					  loss_functional=loss_functional,
					  test_after_epoch=test_after_epoch)
		save_file = open(saved_name, 'wb')
		pickle.dump(network, save_file)
		return network


# ===========================================================================
# =           Regularization Objects                                        =
# ===========================================================================


class LossFunctional:
	""" Main class to be used as a loss-object. Composed of many 
		Regularizer objects which return scalar loss values.
		self.forward(...) returns the weighted sum of the regularizers
	"""	
	def __init__(self, network, regularizers=None):	
		self.network = network
		if regularizers is None:
			regularizers = []
		self.regularizers = regularizers 

	def __repr__(self):
		if len(self.regularizers) == 0:
			return '<Empty Loss Functional>'
		regs_reprs = [_.__repr__() + ',' for _ in self.regularizers]
		output_str = '<LossFunctional: [\n' + '\n'.join(regs_reprs) + '\n]>'
		return output_str		

	def __call__(self, examples, labels):
		return self.forward(examples, labels)		


	def forward(self, examples, labels):
		# naively:
		# return sum([reg.forward(examples, labels) 
		#             for reg in self.regularizers])

		# slightly more cleverly:		
		if any(reg.requires_ff for reg in self.regularizers):
			outputs = self.network(examples)
			losses = sum([reg.forward(examples, labels, outputs=outputs) 
						 for reg in self.regularizers])

		losses = [reg.forward(examples, labels) for reg in self.regularizers]
		return sum(losses)


	def attach_regularizer(self, regularizer_obj):
		self.regularizers.append(regularizer_obj)


class Regularizer(ABC):
	def __init__(self, scalar=1.0):
		self.scalar = scalar 
		self.requires_ff = None
		pass

	@abstractmethod
	def forward(self, examples, labels, outputs=None):
		pass


class XEntropyReg(Regularizer):
	def __init__(self, network, scalar=1.0):		
		super(XEntropyReg, self).__init__(scalar)
		self.network = network
		self.requires_ff = True

	def __repr__(self):
		return 'XEntropy: (scalar: %.02e)' % self.scalar

	def forward(self, examples, labels, outputs=None):
		if outputs is None:
			outputs = self.network(examples)
		return self.scalar * nn.CrossEntropyLoss()(outputs, labels)


class L1WeightReg(Regularizer):
	def __init__(self, network, scalar=1.0):
		super(L1WeightReg, self).__init__(scalar)

		self.network = network 
		self.requires_ff = False 

	def __repr__(self):
		return 'L1Reg: (scalar: %.02e)' % self.scalar

	def forward(self, examples, labels, outputs=None):
		""" Just return the l1 norm of the weight matrices """
		l1_weight = lambda fc: fc.weight.norm(1)
		return self.scalar * sum(l1_weight(fc) for fc in self.network.fcs)


class ReluStability(Regularizer):

	def __init__(self, network, scalar=1.0, l_inf_radius=None, 
				 global_lo=None, global_hi=None):
		super(ReluStability, self).__init__(scalar)

		self.network = network 
		self.requires_ff = False
		self.l_inf_radius = l_inf_radius
		self.global_lo = global_lo
		self.global_hi = global_hi

	def __repr__(self):
		return 'ReluStability: (scalar: %.02e), (radius: %.02f)' %\
			   (self.scalar, self.l_inf_radius)

	def forward(self, examples, labels, outputs=None):
		# First compute naive IA 
		naive_ia_bounds = self._naive_ia(examples)

		# And then compute element-wise relu-stability losses
		element_rs = [-torch.tanh(1.0 + _[:,:,0] * _[:,:,1]).sum()
					  for _ in naive_ia_bounds]
		return self.scalar * sum(element_rs)


	def _naive_ia(self, examples):
	    """ Useful for ReLU stability -- computes naive interval analysis for an 
	        entire batch of examples
	    ARGS:
	        examples: Tensor (N, C, H, W) - examples for upper/lower bounds we compute 
	    RETURNS:
	        bounds : tensor[], which is a list of length #hiddenLayers, where each 
	                 element is a (N, #HiddenUnits, 2) tensor
	    """
	    N = examples.shape[0]
	    reshaped_ex = examples.view(N, -1)
	    preact_bounds = []
	    postact_bounds = []
	    layer_0 = torch.stack([reshaped_ex -self.l_inf_radius, 
	                           reshaped_ex + self.l_inf_radius], dim=-1)
	    if (self.global_lo is not None) or (self.global_hi is not None):
	    	layer_0 = torch.clamp(layer_0, self.global_lo, self.global_hi)

	    postact_bounds.append(layer_0)

	    for i, fc in enumerate(self.network.fcs[:-1]):
	        input_bounds = postact_bounds[-1]
	        input_lows = input_bounds[:, :, 0]
	        input_highs = input_bounds[:, :, 1]
	        input_mids = (input_lows + input_highs) / 2.0
	        input_range = (input_highs - input_lows) / 2.0


	        new_mids = fc.forward(input_mids)
	        new_range = input_range.matmul(fc.weight.abs().t())
	        preact_lows = new_mids - new_range 
	        preact_highs = new_mids + new_range 
	        preact_bounds.append(torch.stack([preact_lows, preact_highs], dim=-1))
	        postact_bounds.append(F.relu(preact_bounds[-1]))

	    return preact_bounds



class LipschitzReg(Regularizer):
	""" Lipschitz regularization taken from this paper: 
		https://arxiv.org/pdf/1808.09540.pdf
		--- computes either the batchwise average or max gradient 
		    norm depending on the tv_or_max parameter. 
		    (loss is standard CrossEntropyLoss )
	"""
	def __init__(self, network, scalar=1.0, tv_or_max='tv', lp_norm=1):
		super(LipschitzReg, self).__init__(scalar)
		self.network = network
		self.requires_ff = False # Got to do a custom FF here


		assert tv_or_max in ['tv', 'max']
		self.tv_or_max = tv_or_max
		self.lp_norm = lp_norm

	def __repr__(self):
		prefix = ['TV', 'Max'][self.tv_or_max == 'max']
		return "%sLipschitzReg: (scalar: %.02e), (lp: %s)" %\
				 (prefix, self.scalar, self.lp_norm)

	def forward(self, examples, labels, outputs=None):
		# copy the examples to enforce gradients 
		N = examples.shape[0]
		new_ex = self.network.tensorfy_clone(examples, requires_grad=True)
		outputs = self.network(new_ex)
		loss = nn.CrossEntropyLoss()(outputs, labels)
		grads = autograd.grad(loss, new_ex, create_graph=True)[0]

		grad_norms = grads.view(N, -1).norm(p=self.lp_norm, dim=1)
		if self.tv_or_max == 'tv':
			reg_quantity = torch.mean(grad_norms)
		else:
			reg_quantity = torch.max(grad_norms)
		return self.scalar * reg_quantity


class GradientStability(ReluStability):
	def __init__(self, network, scalar=1.0, l_inf_radius=None,
				 global_lo=None, global_hi=None, c_vector=None):
	super(GradientStability, self).__init__(network, scalar, l_inf_radius,
											global_lo=None, global_hi=None)
	self.c_vector = c_vector

	def __repr__(self):
		return 'ReluStability: (scalar: %.02e), (radius: %.02f)' %\
			   (self.scalar, self.l_inf_radius)	

	def forward(self, examples, labels, outputs=None):
		# First compute gradient upper/lower bounds 
		stability_relus = self._stable_relus(examples)
		grad_bounds = self._batch_fast_lip(stability_relus)

		# Then compute element-wise stability costs 
		element_muls = grad_bounds[:,0,:] * grad_bounds[:,1,:]

		element_rs = -torch.tanh(1.0 + element_muls).sum()
		return self.scalar * element_rs


	def _stable_relus(self, examples):
		""" Generates stable/unstable relus for the set of examples
		ARGS: 
			examples: tensor (N,C,H,W) - tensor of images representing minibatch 
		RETURNS:
			stability_relus: tensor[] - where the i'th element is a tensor 
							 with shape (N, #Neurons) and elements are integers in 
							 {-1, 0, 1} for -1 meaning always off, +1 meaning always on
							 and 0 for unstable 
		"""
		ia_bounds = self._naive_ia(examples)

		stability_relus = []
		for ia_bound in ia_bounds:
			layer_stability = torch.zeros_like(ia_bound[:,:,0]).long() 
			layer_stability += (ia_bound[:,:,0] > 0).long()
			layer_stability -= (ia_bound[:,:,1] < 0).long()
			stability_relus.append(layer_stability)
		return stability_relus


	def _batch_fast_lip(self, stability_relus):
		""" Performs tensor version of fast-lip using stability_relus 
		ARGS:
			stability_relus: see output of self._stable_relus(...)
		RETURNS:
		"""
		N = stability_relus[0].shape[0]
		if self.c_vector is not None:
			num_outputs = 1 
			matrix_or_vec = 'vec'
		else:
			num_outputs = self.network.fcs[-1].out_features
			matrix_or_vec = 'matrix'
		make_mask = lambda s: s.unsqueeze(1).expand(N, 2, s.shape[2:])
		# Next compute the jacobian based on fast-lip techniquesn
		# --- set up savable state
		post_acts = [] # has shape (N, 2, # neurons, num_outputs)

		# --- loop for all but the last FC layer
		loop_iter = enumerate(zip(stability_list[::-1], self.network.fcs[:1:-1]))
		for i, (relu_stability, fc_layer) in loop_iter:
			# --- Handle the linear layer 
			weight = fc_layer.weight
			if i == 1:
				if self.c_vector is not None:
					# If we multiply output by a vector so it's real-valued
					weight = self.c_vector.view(1, -1).mm(weight)
				pre_act = weight.unsqueeze(0).unsqueeze(0)\
							    .expand((N, 2) + weight.shape)
			else:
				# interval analysis 
				pre_act = ia_mm(weight.t(), post_acts[-1], 1, 
								matrix_or_vec=matrix_or_vec)

			# --- Handle the ReLU 
			# relu_stability has shape (N, # neurons)
			# pre_act has shape (N, 2, #neurons, # outputs)
			off_neurons = (relu_stability < 0).unsqueeze(1)
			off_neurons = off_neurons.expand(N, 2, off_neurons[2:])
			uncertain_neurons = (relu_stability == 0).expand(N, off_neurons[2:])

			pos_los = (pre_act[:,0,:] > 0)
			neg_his = (pre_act[:,1,:] < 0)
			lo_uncertain = pos_los * uncertain_neurons
			hi_uncertain = neg_his * uncertain_neurons
			uncertain_mask = torch.stack([lo_uncertain, hi_uncertain], dim=1)
			pre_act[off_neurons] = 0.0
			pre_act[uncertain_mask] = 0.0
			post_acts.append(pre_act)

		# --- now handle the final (first) FC layer 
		weight = self.network.fcs[0].weight
		return ia_mm(weight.t(), post_acts[-1], 1, matrix_or_vec=matrix_or_vec)