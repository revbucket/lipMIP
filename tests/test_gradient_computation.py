""" Tests for the computation of the gradient using MIP:

	Test strategy:
		- make random network 
		- make random C_vector 
		- make a bunch of random points 
		- compare gradient using pytorch + MIP 
"""

import utilities as utils
import lipMIP
from relu_nets import ReLUNet
from hyperbox import Hyperbox
import numpy as np
import torch
import random

# =====================================================================
# =           Main Helper Function                                    =
# =====================================================================

def test_network_hbox_grad(network, hbox, num_points, c_vector=None, 
						   rand_seed=None):
	""" Takes a given network, hyperbox, and computes random points
		and computes the gradient using LIP/pytorch.
	Comparison happens by recording each and their differences
	--- if c_vector is not specified, we choose one randomly	
	"""
	if rand_seed is None:
		rand_seed = random.randint(0, 10**6)
	print("--- random seed: ", rand_seed)
	torch.random.manual_seed(rand_seed)	
	# Make random points and c_vector	
	if c_vector is None:
		c_vector = 2 * torch.rand(network.layer_sizes[-1]) - 1
		c_vector.type(network.dtype)
	tens_points = hbox.random_point(num_points=num_points, 
									tensor_or_np='tensor')

	# Compute pytorch gradients	
	pytorch_grads = network.get_grad_at_point(tens_points, c_vector)


	# Now get lip gradients 
	np_points = utils.as_numpy(tens_points)
	c_vector = utils.as_numpy(c_vector)

	squire, model, preacts = lipMIP.build_gurobi_model(network, hbox, 
													   'l_inf', c_vector,
													   verbose=False)
	mip_grads = []
	for point in np_points:
		mip_grads.append(squire.get_grad_at_point(point))


	# Assert differences are bounded by infinity norm of 1e-6
	diffs = utils.as_numpy(pytorch_grads) - np.vstack(mip_grads)
	max_abs_diff = abs(diffs).max()
	try:
		assert max_abs_diff < 1e-7
	except AssertionError as e:
		print("MAX ERROR IS ", max_abs_diff)



# ======================================================================
# =           Test blocks                                              =
# ======================================================================


def simple_test(n=1000):
	""" Simple test over 2x2x2x2 neural network with unit hypercube domain """
	network = ReLUNet(layer_sizes=[2, 2, 2, 2], bias=True)
	hbox = Hyperbox.build_unit_hypercube(2)
	c_vector = np.array([1.0, -1.0])
	print("Starting simple test...")
	test_network_hbox_grad(network, hbox, n, c_vector)
	print("...simple test passed!\n\n")


def med_test(n=1000):
	""" Slightly larger test """
	network = ReLUNet(layer_sizes=[10, 20, 40, 5], bias=True)
	hbox = Hyperbox.build_unit_hypercube(10)
	print("Starting med test...")
	test_network_hbox_grad(network, hbox, n, c_vector=None)
	print("...med test passed!\n\n")

def mnist_test(n=1000):
	""" Test with mnist style network """
	network = ReLUNet(layer_sizes=[784, 20, 20, 20, 10], bias=True)
	hbox = Hyperbox.build_unit_hypercube(784)
	print("Starting MNIST test...")
	test_network_hbox_grad(network, hbox, n, c_vector=None)
	print("...MNIST test passed!\n\n")

if __name__ == '__main__':
	simple_test()
	med_test()
	mnist_test(n=100)

