""" Script to build a bunch of networks, train each of them, and then
	build jobs to evaluate each of these
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
	NAME = 'WideSeq'
	exp_kwargs = {'c_vector': np.array([1.0, -1.0]), 
				  'primal_norm': 'linf'}
	DIMENSION = 2
	RADIUS = 0.1
	GLOBAL_LO = np.zeros(DIMENSION)
	GLOBAL_HI = np.ones(DIMENSION)
	RANDOM_SEED = 420
	DOMAIN = Hyperbox.build_unit_hypercube(DIMENSION)		
	BALL_FACTORY = Factory(Hyperbox.build_linf_ball, radius=RADIUS)
	LAYER_SEQ = [([2, 20] + [20 * i] + [20, 2]) for i in range(1, 6)]
	def NAME_FXN(network):
		""" Returns a string based on the network """
		width = network.layer_sizes[2]
		return '%s_WIDTH%04d' % (NAME, width)
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
	loader_kwargs = {'batch_size': 100, 'random_seed': RANDOM_SEED}
	data_arg_bundle = {'data_type': 'synthetic', 
					   'params': data_params,
					   'loader_kwargs': loader_kwargs,
					   'ball_factory': BALL_FACTORY}

	# ================================================================
	# =           Training Parameter Setup                           =
	# ================================================================

	# Build the loss functional and set the optimizer 
	xentropy = train.XEntropyReg()
	l2_penalty = train.LpWeightReg(scalar=1e-4, lp='l2')
	loss_functional = train.LossFunctional(regularizers=[xentropy])
	train_params = train.TrainParameters(train_set, train_set, 500, 
										 loss_functional=loss_functional,
										 test_after_epoch=50)


	# ================================================================
	# =           Build the methodNests                              =
	# ================================================================

	# --- randomly evaluated method nest
	random_nest = MethodNest(Experiment.do_random_evals, 
	   						  {'sample_domain': DOMAIN, 
							   'ball_factory': BALL_FACTORY,
							   'num_random_points': 20})

	# --- data-based method nest 
	data_nest = MethodNest(Experiment.do_data_evals, data_arg_bundle)


	# --- hypercube stuff 
	cube_nest = MethodNest(Experiment.do_unit_hypercube_eval)

	local_nests = [random_nest, data_nest, cube_nest]
	global_nests = [cube_nest]


	def build_jobs(network, **exp_kwargs):
		local_exp = Experiment([FastLip, LipLP, LipMIP], network=network, 
							   **exp_kwargs)	
		global_exp = Experiment([LipMIP, FastLip, LipLP, SeqLip, LipSDP, NaiveUB], 
								network=network, **exp_kwargs)
		prefix = NAME_FXN(network)
		#prefix = '%s_RELUS%02d' % (NAME, network.num_relus)
		local_job_name = prefix + "_LOCAL"
		local_job = Job(local_job_name, local_exp, local_nests,
						save_loc=SCHEDULE_DIR)
		global_job_name = prefix + "_GLOBAL"
		global_job = Job(global_job_name, global_exp, global_nests,
						 save_loc=SCHEDULE_DIR)
		local_job.write()		
		global_job.write()



	# ==============================================================
	# =           Train the networks                                =
	# ==============================================================

	for layer_size in LAYER_SEQ:
		print("Starting training:", layer_size)
		network = ReLUNet(layer_sizes=layer_size)
		train.training_loop(network, train_params)
		build_jobs(network, **exp_kwargs)


if __name__ == '__main__':

	main()