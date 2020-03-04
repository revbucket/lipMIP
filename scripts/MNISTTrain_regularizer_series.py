""" Script to build a bunch of networks WITH THE SAME ARCHITECTURE,
	but different regularizers. Build jobs to evaluate each of these

	General parameter scheme:
	for training_param in training_params[]: (holds various regularizers)
		- train possibly k times, taking most accurate answer
		- build jobs at either {end} or {every m epochs}
	CONTROLS:
		- list of training parameters 
		- whether or not to train k times 
		- to build jobs at either {end} or {every m epochs}

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
	# ==========================================================
	# =           SETUP -- CHANGE THESE VARIABLES!!!           =
	# ==========================================================
	# --- COMMON BLOCK (holds names, objectives, etc)
	NAME = 'MNISTTRAIN' 			# Name for this experiment
	C_VECTOR = np.array([1.0, -1.0])
	PRIMAL_NORM = 'linf'
	MNIST_DIGITS = [1, 7]
	# DATA_PARAMS = {} 		# kwargs to RandomKParameters constructor
	LAYER_SIZES = [784, 20, 20, 2]  	# defines architecture of net to test

	# --- CONTROL BLOCK (holds training params, num_restarts, jobs at end?)
	REGULARIZER_SERIES = {'vanilla': [train.XEntropyReg()],
						  'l2Penalty': [train.XEntropyReg(), 
						  				 train.LpWeightReg(scalar=1e-2, lp='l2')],
						  'l1Penalty': [train.XEntropyReg(), 
						  				 train.LpWeightReg(scalar=1e-3, lp='l1')],
						  'fgsm': [train.FGSM(0.1)],
						  'l2LipReg': [train.XEntropyReg(),
						  			   train.LipschitzReg(scalar=0.1, lp_norm=2),
						  			   train.LipschitzReg(scalar=0.1, tv_or_max='max', lp_norm=2)],
						  'l1LipReg': [train.XEntropyReg(), 
						  			   train.LipschitzReg(scalar=0.1, lp_norm=1), 
						  			   train.LipschitzReg(scalar=0.1, lp_norm=1, tv_or_max='max')]}	
	NUM_RESTARTS = 1  	 	# how many times to restart training
	JOB_FREQUENCY = 5

	# --- EXPERIMENTAL PARAM BLOCK (holds ball_factory, which methods to eval)
	RADIUS = 0.2        # Ball factory radius for random/data evals
	RANDOM_SEED = 420    # Dataset random seed
	NUM_EXP_RANDOM = 100 # num of random points to test in experiments
	NUM_EXP_DATA   = 100 # num of data points to test in experiments
	LOCAL_METHODS = [FastLip, LipLP, LipMIP] #Methods to do random/data
	GLOBAL_METHODS = [LipMIP, FastLip, LipLP, SeqLip, LipSDP,
					  NaiveUB, RandomLB] # Methods to do unit hcube


	# -- COMPUTED HELPER BLOCK
	exp_kwargs = {'c_vector': C_VECTOR,
				  'primal_norm': PRIMAL_NORM}	
	DOMAIN = Hyperbox.build_unit_hypercube(784)
	BALL_FACTORY = Factory(Hyperbox.build_linf_ball, radius=RADIUS)
	# ================================================================
	# =           Data Parameters Setup                              =
	# ================================================================
	# Make both the training/validation sets 

	train_set = dl.load_mnist_data('train', digits=MNIST_DIGITS)
	val_set = dl.load_mnist_data('val', digits=MNIST_DIGITS)
	#data_params = dl.RandomKParameters(**DATA_PARAMS)
	#dataset = dl.RandomDataset(data_params, batch_size=128, 
	#						   random_seed=RANDOM_SEED)
	#train_set, _ = dataset.split_train_val(1.0)

	# Make the data arg_bundle object
	train_loader_kwargs = {'batch_size': NUM_EXP_DATA, 
		   			  	   'train_or_val': 'train'}
	val_loader_kwargs = {'batch_size': NUM_EXP_DATA, 
						 'train_or_val': 'val'}
	train_arg_bundle = {'data_type': 'synthetic', 
					   'loader_kwargs': train_loader_kwargs,
					   'ball_factory': BALL_FACTORY,
					   }
	val_arg_bundle = {'data_type': 'MNIST', 
					   'loader_kwargs': val_loader_kwargs,
					   'ball_factory': BALL_FACTORY}

	# ================================================================
	# =           Build the methodNests                              =
	# ================================================================

	# --- randomly evaluated method nest
	random_nest = MethodNest(Experiment.do_random_evals, 
	   						  {'sample_domain': DOMAIN, 
							   'ball_factory': BALL_FACTORY,
							   'num_random_points': NUM_EXP_DATA})

	# --- data-based method nest 
	train_nest = MethodNest(Experiment.do_data_evals, train_arg_bundle)
	val_nest = MethodNest(Experiment.do_data_evals, val_arg_bundle)

	# --- hypercube stuff 
	cube_nest = MethodNest(Experiment.do_unit_hypercube_eval)

	local_nests = [random_nest, train_nest, val_nest]
	global_nests = [cube_nest]


	def build_callback_and_final(reg_name):
		# Builds the epoch_callback, final call

		def build_jobs(epoch_no, network=None, NAME=NAME, reg_name=reg_name):
			prefix = '%s_REG|%s_EPOCH|%04d' % (NAME, reg_name, epoch_no)
			local_exp = Experiment(LOCAL_METHODS, network=network, **exp_kwargs)
			global_exp = Experiment(GLOBAL_METHODS, network=network, **exp_kwargs)

			local_job = Job('%s_LOCAL' % prefix, local_exp, local_nests,
							save_loc=SCHEDULE_DIR)
			global_job = Job('%s_GLOBAL' % prefix, global_exp, global_nests,
							 save_loc=SCHEDULE_DIR)

			local_job.write()
			global_job.write()


		if JOB_FREQUENCY is None:
			return None, build_jobs
		else:
			return DoEvery(build_jobs, JOB_FREQUENCY), build_jobs

	# ==============================================================
	# =           Train the networks                                =
	# ==============================================================
	for reg_name, regularizers in REGULARIZER_SERIES.items():
		print('-' * 30, 'TRAINING --', reg_name)
		# First build job builder:
		callback, final = build_callback_and_final(reg_name)

		# Then train function
		loss_functional = train.LossFunctional(regularizers=regularizers)
		train_params = train.TrainParameters(train_set, train_set, 50,
										 loss_functional=loss_functional,
										 test_after_epoch=5)
		network = ReLUNet(layer_sizes=LAYER_SIZES)

		train.best_of_k(network, train_params, NUM_RESTARTS,
						epoch_callback=callback)
		# Finally call the final fxn
		final(epoch_no=train_params.num_epochs, network=network)


if __name__ == '__main__':
	main()