""" Script to have a training series for an MNIST network. 
	Can set FREQUENCY to None to build jobs only at the end
"""

import matlab.engine
import numpy as np
import torch 
import sys 
sys.path.append('..')
from experiment import Experiment, MethodNest, Job
from hyperbox import Hyperbox 
from relu_nets import ReLUNet
from neural_nets import data_loaders as dl
from neural_nets import train
from lipMIP import LipMIP
from other_methods import CLEVER, FastLip, LipLP, LipSDP, NaiveUB, RandomLB, SeqLip
from other_methods import LOCAL_METHODS, GLOBAL_METHODS
from utilities import Factory, DoEvery
import os 


SCHEDULE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
					   'jobs', 'scheduled')

def main():
	NAME = None
	LAYER_SIZES = None
	C_VECTOR = None # list of digits or the string 'crossLipschitz'
	RANDOM_SEED = None
	RADIUS = None
	MNIST_DIGITS = None 	
	FREQUENCY = None
	EPOCHS = None
	assert all([_ is not None for _ in [NAME, LAYER_SIZES, C_VECTOR, 
										RANDOM_SEED, RADIUS, FREQUENCY, 
										EPOCHS]])

	exp_kwargs = {'c_vector': C_VECTOR,
				  'primal_norm': 'linf'}
	DIMENSION = 784
	GLOBAL_LO = np.zeros(DIMENSION)
	GLOBAL_HI = np.ones(DIMENSION)
	DOMAIN = Hyperbox.build_unit_hypercube(DIMENSION)
	BALL_FACTORY = Factory(Hyperbox.build_linf_ball, radius=RADIUS)
	NAMER = lambda epoch_no: '%s_EPOCH%04d' % (NAME, epoch_no)
	# ================================================================
	# =           Data Parameters Setup                              =
	# ================================================================
	# Make both the training/validation sets 

	train_set = dl.load_mnist_data('train', digits=MNIST_DIGITS)
	val_set = dl.load_mnist_data('val', digits=MNIST_DIGITS)

	# Make the data arg_bundle object
	loader_kwargs = {'batch_size': 100, 'digits': MNIST_DIGITS,
					 'shuffle': True}
	train_arg_bundle = {'data_type': 'MNIST',
		 			    'loader_kwargs': loader_kwargs,
		 			    'ball_factory': BALL_FACTORY,
		 			    'train_or_val': 'train'}
	val_arg_bundle = {'data_type': 'MNIST',
 	 			      'loader_kwargs': loader_kwargs,
		 			  'ball_factory': BALL_FACTORY,
		 			  'train_or_val': 'val'}

	# ================================================================
	# =           Training Parameter Setup                           =
	# ================================================================

	# Build the loss functional and set the optimizer 
	xentropy = train.XEntropyReg()
	l2_penalty = train.LpWeightReg(scalar=1e-2, lp='l2')
	loss_functional = train.LossFunctional(regularizers=[xentropy])
	train_params = train.TrainParameters(train_set, train_set, EPOCHS, 
										 loss_functional=loss_functional,
										 test_after_epoch=20)
	# Build the base network architecture
	network = ReLUNet(layer_sizes=LAYER_SIZES)


	# ================================================================
	# =           Build the Experiment objects                       =
	# ================================================================

	local_exp = Experiment([FastLip, LipLP, LipMIP], network=network,
						   **exp_kwargs)
	global_exp = Experiment(GLOBAL_METHODS, network=network, **exp_kwargs)

	# ================================================================
	# =           Build the methodNests                              =
	# ================================================================

	# --- randomly evaluated method nest
	random_nest = MethodNest(Experiment.do_random_evals, 
	   						  {'sample_domain': DOMAIN, 
							   'ball_factory': BALL_FACTORY,
							   'num_random_points': 20})

	# --- data-based method nest 
	train_nest = MethodNest(Experiment.do_data_evals, train_arg_bundle)
	val_nest = MethodNest(Experiment.do_data_evals, val_arg_bundle)


	# --- hypercube stuff 
	cube_nest = MethodNest(Experiment.do_unit_hypercube_eval)

	local_nests = [random_nest, train_nest, val_nest, cube_nest]
	global_nests = [cube_nest]


	def build_jobs(epoch_no, network=None):
		local_job_name = NAMER(epoch_no) + '_LOCAL'
		local_job = Job(local_job_name, local_exp, local_nests,
						save_loc=SCHEDULE_DIR)
		local_job.write()

		global_job_name = NAMER(epoch_no) + '_GLOBAL'
		global_job = Job(global_job_name, global_exp, global_nests,
						 save_loc=SCHEDULE_DIR)
		global_job.write()

	if FREQUENCY is None:
		job_do_every = None
	else:
		job_do_every = DoEvery(build_jobs, FREQUENCY)

	# ==============================================================
	# =           Train the network                                =
	# ==============================================================

	train.training_loop(network, train_params, epoch_callback=job_do_every)
	if FREQUENCY is None:
		build_jobs(EPOCHS)


if __name__ == '__main__':
	main()