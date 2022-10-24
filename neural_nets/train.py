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
from . import adv_attacks as aa
import pickle
from abc import ABC, abstractmethod
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'..'))

from utilities import ParameterObject, cudafy, cpufy

# ===========================================================================
# =           General purpose training                                      =
# ===========================================================================

class TrainParameters(ParameterObject):
	def __init__(self, trainset, valset, num_epochs, optimizer=None,
				 loss_functional=None, test_after_epoch=True):
		"""
		ARGS:
		trainset: torch.utils.data.DataLoader object of the training data
		valset: torch.utils.data.DataLoader object of the validation data
		optimizer: None or torch.optim object - if None will be
				   initialized to optim.Adam object with default parameters
		loss_function: None or loss object - if None will default to
					   crossEntropyLoss (see LossObject at bottom of this )
		test_after_epoch: bool - if True, we test top1 accuracy after each
								 epoch. If an int, we test top1 accuracy 
								 after each <int> epochs
		"""

		init_args = {k: v for k, v in locals().items() 
					 if k not in ['self', '__class__']}
		super(TrainParameters, self).__init__(**init_args)

	def cuda(self):
		""" If the trainset/valset are lists of tensors, this pushes them 
			to cuda
		"""
		for attr in 'trainset', 'valset':
			if isinstance(getattr(self, attr), list):
				setattr(self, attr, [cudafy(_) for _ in getattr(self, attr)])
		return

	def cpu(self):
		""" If the trainset are lists of tensors, this makes sure they end up
			back on the cpu 
		"""
		for attr in 'trainset', 'valset':
			if isinstance(getattr(self, attr), list):
				setattr(self, attr, [cpufy(_) for _ in getattr(self, attr)])
		return



def training_loop(network, train_params, epoch_start_no=0, 
				  use_cuda=False, epoch_callback=None):
	""" Trains a network over the trainset with given loss and optimizer
	ARGS:
		network: ReLUNet - network to train
		train_params: TrainParameters - parameters object governing training
		epoch_start_no: int - number to start epochs at for print purposes
		use_cuda: bool - if True, we use CUDA to train the net. Everything 
						 gets returned on CPU

		epoch_callback: if not none, is a function that takes in arguments
					    {'network': network, 'epoch_no': epoch_no}

	RETURNS:
		None, but modifies network parameters
	"""
	use_cuda = torch.cuda.is_available() and use_cuda
	if use_cuda: 
		network.cuda()
		train_params.cuda()
	else:
		network.cpu()
		train_params.cpu()

	# Unpack the parameter object
	if train_params.optimizer is None:
		optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=0)
	else:
		optimizer = train_params.optimizer 

	if train_params.test_after_epoch is False:
		test_after_epoch = float('inf')
	else:
		test_after_epoch = int(train_params.test_after_epoch)

	if train_params.loss_functional is None:
		loss_functional = LossFunctional(regularizers=[XEntropyReg()])
	else:
		loss_functional = train_params.loss_functional
	loss_functional.attach_network(network)

	# Do the training loop
	for epoch_no in range(epoch_start_no, epoch_start_no + train_params.num_epochs + 1):
		if epoch_callback is not None:
			epoch_callback(network=network, epoch_no=epoch_no)
		
		for i, (examples, labels) in enumerate(train_params.trainset):
			if examples.dtype != network.dtype:
				examples = examples.type(network.dtype)
			if use_cuda:
				examples, labels = cudafy([examples, labels])

			optimizer.zero_grad()
			loss = loss_functional(examples, labels)
			loss.backward()
			optimizer.step()
		if (epoch_no) % test_after_epoch == 0:
			# If we run test accuracy this test
			test_acc = test_validation(network, train_params.valset, 
									   use_cuda=use_cuda)
			test_acc_str = '| Accuracy: %.02f' % (test_acc * 100)
			print(("Epoch %02d " % epoch_no) + test_acc_str)


	if use_cuda:
		network = network.cpu()
		train_params.cpu()


def test_validation(network, valset, loss_functional=None, use_cuda=False):
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
		if examples.dtype != network.dtype:
			examples = examples.type(network.dtype)
		if use_cuda:
			examples, labels = cudafy([examples, labels])
		total += labels.numel()
		if loss_functional is None:
			xout = network(examples).max(1)[1]
			xout == labels
			count_value += (network(examples).max(1)[1] == labels).sum().item()
		else:
			count_value += loss_functional(examples, labels).data.item()
	return float(count_value) / total


def best_of_k(base_network, train_params, k, use_cuda=False, **kwargs):
	""" Takes in a network and trains it K times, taking the best validation
		accuracy after all of them 
	ARGS: 
		base_network : ReLuNet instance - object that gets trained several times 
		train_params: TrainParameters instance - object that dictates how 
					  training goes 
		k: int - how many times we copy the training
		use_cuda : bool - if True we use CUDA to do things faster
		kwargs : any other keyword args to pass to training_loop
	RETURNS:
		None, but modifies the base_network parameters
	"""	
	best_val_acc = 0.0
	best_relunet = None
	if k == 1: # K =1 case
		return training_loop(base_network, train_params, use_cuda=use_cuda,
							 **kwargs)

	for i in range(k): #K > 1 case
		print("Starting training run %02d of %02d" % (i, k))
		this_net = base_network.clone()
		this_net.re_init_weights()
		training_loop(this_net, train_params, use_cuda=use_cuda, **kwargs)
		this_val_acc = test_validation(this_net, train_params.valset,
  								       use_cuda=use_cuda)
		print("...ending training run with acc %.02f" % (100 * this_val_acc))
		if this_val_acc > best_val_acc:
			best_val_acc = this_val_acc
			best_relunet = this_net

	base_network.fcs = best_relunet.fcs
	base_network.net = best_relunet.net
	return base_network



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
	def __init__(self, network=None, regularizers=None):
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


	def attach_network(self, network):
		self.network = network 
		for reg in self.regularizers:
			reg.attach_network(network)


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
		self.network = None
		pass

	def attach_network(self, network):
		self.network = network

	@abstractmethod
	def forward(self, examples, labels, outputs=None):
		pass


class XEntropyReg(Regularizer):
	def __init__(self, network=None, scalar=1.0):
		super(XEntropyReg, self).__init__(scalar)
		self.network = network
		self.requires_ff = True

	def __repr__(self):
		return 'XEntropy: (scalar: %.02e)' % self.scalar

	def forward(self, examples, labels, outputs=None):

		if outputs is None:
			outputs = self.network(examples)
		return self.scalar * nn.CrossEntropyLoss()(outputs, labels)

class MSEReg(Regularizer):
	def __init__(self, num_classes, network=None, scalar=1.0):
		super(MSEReg, self).__init__(scalar)
		self.num_classes = num_classes
		self.network = network 
		self.requires_ff = True 


	def __repr__(self):
		return 'MSE: (scalar: %.02e)' % self.scalar

	def forward(self, examples, labels, outputs=None):

		if outputs is None:
			outputs = self.network(examples)
		one_hot = F.one_hot(labels).type(outputs.dtype)
		return self.scalar * nn.MSELoss()(outputs, one_hot)



class LpWeightReg(Regularizer):
	def __init__(self, network=None, scalar=1.0, lp='l1'):
		super(LpWeightReg, self).__init__(scalar)

		self.network = network
		self.lp_str = lp
		self.p_norm = {'l1': 1, 'l2': 2, 'linf': float('inf')}[lp]
		self.requires_ff = False

	def __repr__(self):
		return 'L1Reg: (scalar: %.02e)' % self.scalar

	def forward(self, examples, labels, outputs=None):
		""" Just return the l1 norm of the weight matrices """
		weight_norm = lambda fc: fc.weight.norm(self.p_norm)
		sum_val = 0
		for layer in self.network.net:
			if hasattr(layer, 'weight'):
				sum_val += weight_norm(layer)
		return self.scalar * sum_val


class ReconstructionLoss(Regularizer):

	def __init__(self, network=None, scalar=1.0):
		super(ReconstructionLoss, self).__init__(scalar)
		self.network = network
		self.requires_ff = True

	def forward(self, examples, labels, outputs=None):
		if outputs is None:
			outputs = self.network(examples)

		num_examples = examples.shape[0]
		net_loss = self.scalar * (outputs - examples.view(num_examples, -1)).norm(p=2)
		return net_loss ** 2 / (2 * num_examples)



class ReluStability(Regularizer):

	def __init__(self, network=None, scalar=1.0, l_inf_radius=None,
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
	def __init__(self, network=None, scalar=1.0, tv_or_max='tv', lp_norm=1):
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
	def __init__(self, network=None, scalar=1.0, l_inf_radius=None,
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


class FGSM(Regularizer):
	def __init__(self, linf_bound, global_lo=0.0, global_hi=1.0, 
				 network=None, scalar=1.0):
		super(FGSM, self).__init__(scalar)
		self.linf_bound = linf_bound 
		self.global_lo = global_lo
		self.global_hi = global_hi
		self.network = network

	def forward(self, examples, labels, outputs=None):
		adv_examples = aa.fgsm(self.network, examples, labels, 
							   self.linf_bound, global_lo=self.global_lo,
							   global_hi=self.global_hi)
		adv_logits = self.network(adv_examples)
		return self.scalar * nn.CrossEntropyLoss()(adv_logits, labels)


class PGD(Regularizer):
	def __init__(self, linf_bound, num_iter=10, step_size=0.02, 
				 global_lo=0.0, global_hi=1.0, network=None, scalar=1.0):
		super(PGD, self).__init__(scalar)
		self.linf_bound = linf_bound 
		self.num_iter = num_iter
		self.step_size = step_size
		self.global_lo = global_lo
		self.global_hi = global_hi
		self.network = network

	def forward(self, examples, labels, outputs=None):
		adv_examples = aa.pgd(self.network, examples, labels, 
							  self.linf_bound, num_iter=self.num_iter, 
							  step_size=self.step_size, 
							  global_lo=self.global_lo,
							  global_hi=self.global_hi)
		adv_logits = self.network(adv_examples)
		return self.scalar * nn.CrossEntropyLoss()(adv_logits, labels)



class PGDEval(Regularizer):
	def __init__(self, linf_bound, num_iter=10, step_size=0.02, 
				 global_lo=0.0, global_hi=1.0, network=None, scalar=1.0,
				 top1=False):
		super(PGD, self).__init__(scalar)
		self.linf_bound = linf_bound 
		self.num_iter = num_iter
		self.step_size = step_size
		self.global_lo = global_lo
		self.global_hi = global_hi
		self.network = network

	def forward(self, examples, labels, outputs=None):
		adv_examples = aa.pgd(self.network, examples, labels, 
							  self.linf_bound, num_iter=self.num_iter, 
							  step_size=self.step_size, 
							  global_lo=self.global_lo,
							  global_hi=self.global_hi)
		adv_logits = self.network(adv_examples)

		if not self.top1:
			return self.scalar * nn.CrossEntropyLoss()(adv_logits, labels)

		else:
			count_correct = adv_logits.max(1)[1] == labels
			num_ex = labels.numel()
			return float(count_correct) / num_ex