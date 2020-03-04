""" Script to build a bunch of jobs that train a neural net and
	every X epochs builds several jobs to analyze this.
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
from other_methods import LOCAL_METHODS, GLOBAL_METHODS, OTHER_METHODS
from utilities import Factory, DoEvery
import os


SCHEDULE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
					   'jobs', 'scheduled')


def main():

	# --- CHANGE THESE
	NAME = None
	C_VECTOR = None
	DIMENSION = None
	RADIUS = None
	RANDOM_SEED = None
	NUM_EPOCHS = None
	FREQUENCY = None
	LAYER_SIZES = None
	LOCAL_METHODS = [FastLip, LipLP, LipMIP]
	GLOBAL_METHODS = OTHER_METHODS + [LipMIP]

	NUM_RANDOM = None # Number of points in exp.do_random_evals
	NUM_DATA = None   # Number of points in exp.do_data_evals

	assert all(_ is not None for _ in [NAME, C_VECTOR, DIMENSION, RADIUS,
									   RANDOM_SEED, NUM_EPOCHS, FREQUENCY,
									   LAYER_SIZES])

	# --- THESE CAN REMAIN AS IS
	exp_kwargs = {'c_vector': C_VECTOR, 'primal_norm': 'linf'}
	GLOBAL_LO = np.zeros(DIMENSION)
	GLOBAL_HI = np.ones(DIMENSION)
	DOMAIN = Hyperbox.build_unit_hypercube(DIMENSION)
	BALL_FACTORY = Factory(Hyperbox.build_linf_ball, radius=RADIUS)

	# ================================================================
	# =           Data Parameters Setup                              =
	# ================================================================
	# Make both the training/validation sets
	data_params = dl.RandomKParameters(num_points=300, k=10, radius=0.01,
									   dimension=DIMENSION)
	dataset = dl.RandomDataset(data_params, batch_size=128,
				  			   random_seed=RANDOM_SEED)
	train_set, _ = dataset.split_train_val(1.0)

	# Make the data arg_bundle object
	loader_kwargs = {'batch_size': NUM_DATA, 'random_seed': RANDOM_SEED}
	data_arg_bundle = {'data_type': 'synthetic',
					   'params': data_params,
					   'loader_kwargs': loader_kwargs,
					   'ball_factory': BALL_FACTORY}

	# ================================================================
	# =           Training Parameter Setup                           =
	# ================================================================
	# Build the loss functional and set the optimizer
	xentropy = train.XEntropyReg()
	l2_penalty = train.LpWeightReg(scalar=1e-2, lp='l2')
	loss_functional = train.LossFunctional(regularizers=[xentropy, l2_penalty])
	train_params = train.TrainParameters(train_set, train_set, NUM_EPOCHS,
										 loss_functional=loss_functional,
										 test_after_epoch=20)
	# Build the base network architecture
	network = ReLUNet(layer_sizes=LAYER_SIZES)


	# ================================================================
	# =           Build the Experiment objects                       =
	# ================================================================

	local_exp = Experiment(LOCAL_METHODS, network=network, **exp_kwargs)
	global_exp = Experiment(GLOBAL_METHODS, network=network, **exp_kwargs)

	# ================================================================
	# =           Build the methodNests                              =
	# ================================================================

	# --- randomly evaluated method nest
	random_nest = MethodNest(Experiment.do_random_evals,
	   						  {'sample_domain': DOMAIN,
							   'ball_factory': BALL_FACTORY,
							   'num_random_points': NUM_RANDOM})

	# --- data-based method nest
	data_nest = MethodNest(Experiment.do_data_evals, data_arg_bundle)


	# --- hypercube stuff
	cube_nest = MethodNest(Experiment.do_unit_hypercube_eval)

	local_nests = [random_nest, data_nest]
	global_nests = [cube_nest]


	def build_jobs(epoch_no, network=None):
		local_job_name = '%s_EPOCH%04d_LOCAL' % (NAME, epoch_no)
		local_job = Job(local_job_name, local_exp, local_nests,
						save_loc=SCHEDULE_DIR)
		local_job.write()


		global_job_name = '%s_EPOCH%04d_GLOBAL' % (NAME, epoch_no)
		global_job = Job(global_job_name, global_exp, global_nests,
						 save_loc=SCHEDULE_DIR)
		global_job.write()

	job_do_every = DoEvery(build_jobs, FREQUENCY)

	# ==============================================================
	# =           Train the network                                =
	# ==============================================================

	train.training_loop(network, train_params, epoch_callback=job_do_every)


if __name__ == '__main__':

	main()