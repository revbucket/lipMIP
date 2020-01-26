""" Very minimal adversarial attack techniques """

import torch 
import torch.nn as nn
import functools



# =============================================================
# =           ATTACK TECHNIQUES                               =
# =============================================================




def fgsm(network, examples, labels, linf_bound,
		 global_lo=0.0, global_hi=1.0):
	""" Takes in a neural net, and a minibatch of examples, labels
		and adds noise with linf_bound < linf_bound to each example 
		in order to MAXIMIZE loss.
	ARGS:
		network: ReLUNet instance - network we adversarially attack 
		examples: tensor - images to build adversarial examples for
		linf_bound: float - bound for how much to affect each image by 
		global_lo/hi : float - global lower/upper bounds for all coordinates
					   (if None, then no bounds)
	RETURNS:
		new_examples: modified examples
	"""
	new_ex = network.tensorfy_clone(examples, requires_grad=True)
	loss = nn.CrossEntropyLoss()(network(new_ex), labels)
	loss.backward()
	new_ex.data += torch.sign(new_ex.grad) * linf_bound
	if global_lo is not None:
		new_ex.data = torch.clamp(new_ex.data, min=global_lo)
	if global_hi is not None:
		new_ex.data = torch.clamp(new_ex.data, max=global_hi)		

	return new_ex


def pgd(network, examples, labels, linf_bound, num_iter=10,
		step_size=0.02, global_lo=0.0, global_hi=1.0):
	""" Runs PGD over an L_infty domain with fixed number of iterations, 
		step sizes. 
	"""
	new_ex = network.tensorfy_clone(examples, requires_grad=True)

	for i in range(num_iter):
		loss = nn.CrossEntropyLoss()(network(new_ex), labels)
		loss.backward() 
		delta = (new_ex.data - examples.data) + (torch.sign(new_ex.grad) * step_size)
		delta = torch.clamp(delta, -linf_bound, linf_bound)
		new_ex.data = examples.data + delta	
		if global_lo is not None:
			new_ex.data = torch.clamp(new_ex.data, min=global_lo)
		if global_hi is not None:
			new_ex.data = torch.clamp(new_ex.data, max=global_hi)

	return new_ex






# =============================================================
# =           ROBUSTNESS EVALUATORS                           =
# =============================================================

def build_attack_partial(fxn, **kwargs):
	return functools.partial(fxn, **kwargs)


def eval_dataset(network, dataset, attack_partial):
	""" Evaluates Dataset for adversarial robustness 
	ARGS:
		network: ReLUNet instance - network we evaluate 
		examples: dataset - iterable where each iteration is 
				  (examples, labels)
		attack_partial: partially applied function which takes in 
						only the (network, examples, labels) arguments
	RETURNS dict like:
		{num_correct: # of correct images, 
		 total:       # of total images,
		 percentage_correct: float in [0, 1]}
	"""
	results = []
	for ex, lab in dataset:
		adv_ex = attack_partial(network, ex, lab)
		results.append(eval_minibatch(network, adv_ex, lab))

	num_correct = sum(_['num_correct'] for _ in results)
	total = sum(_['total'] for _ in results)
	return {'num_correct': num_correct, 
			'total': total, 
			'percentage_correct': float(num_correct) / total}

def eval_minibatch(network, examples, labels):
	""" Evaluates the top1 accuracy of the provided minibatch
	ARGS:
		network: ReLUNet instance - network we evaluate 
		examples: tensor - images to input into the network 
		labels : LongTensor - correct labels of the images
	RETURNS:
		{num_correct: # of correct images
		 percentage_correct: percentage of correct images (float in [0, 1])}
	"""
	computed_labels = torch.max(network(examples), 1)[1]
	num_correct = (computed_labels == labels).sum().item()
	return {'num_correct': num_correct,
			'total': len(labels),
	  	    'percentage_correct': float(num_correct) / float(len(labels))}

