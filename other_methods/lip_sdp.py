""" Porting of the SDP methods for lipschitz overestimation:
	Arxiv: https://arxiv.org/abs/1906.04893
	Github: https://github.com/arobey1/LipSDP
"""
import numpy as np
from .other_methods import OtherResult
import utilities as utils
import tempfile
from scipy.io import savemat 
import os
import matlab.engine
import secrets

class LipSDPCustom(OtherResult):
	LIPSDP_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
							  'LipSDP', 'LipSDP')
	DEFAULT_LIPSDP_KWARGS = {'formulation': 'neuron', 
						     'split': matlab.logical([[False]]), 
						     'parallel': matlab.logical([[False]]), 
						     'verbose': matlab.logical([[False]]), 
						     'split_size': matlab.double([[2.]]), 
						     'num_neurons': matlab.double([[100.]]), 
						     'num_workers': matlab.double([[10.]]), 
						     'num_dec_vars': matlab.double([[10.]])}
	MATLAB_PATHS = ['matlab_engine', 'matlab_engine/weight_utils', 
				    'matlab_engine/error_messages']

	@classmethod 
	def extract_weights(cls, relunet, c_vector):
		weight_list = []
		for fc in relunet.fcs:
			weight_list.append(utils.as_numpy(fc.weight).astype(np.double))
		final_weight = weight_list[-1]
		final_weight = utils.as_numpy(c_vector)\
							.dot(final_weight).reshape((1, -1))
		weight_list[-1] = final_weight

		return {'weights': np.array(weight_list, dtype=np.object)}


	def __init__(self, network, c_vector):
		""" Solves LipSDP for given network/c_vector """
		super(LipSDPCustom, self).__init__(network, c_vector, None, 'l2')


	def compute(self, attr_name=None):
		""" Takes ReLUNet and casts weights to a temp file so we can 
			run the Matlab/Mosek SDP solver on these. Kwargs to come 
		"""

		# Collect weights and put them in a temp file
		weights = self.extract_weights(self.network, self.c_vector)
		weight_file = secrets.token_hex(24) + '.mat'
		weight_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
  							       'saved_weights', weight_file)
		savemat(weight_path, weights)
		# Build matlab stuff
		eng = matlab.engine.start_matlab()
		for path in self.MATLAB_PATHS:
			eng.addpath(os.path.join(self.LIPSDP_DIR, path))
		eng.addpath(os.path.dirname(weight_path))

		network = {'alpha': matlab.double([[0.]]),
				   'beta': matlab.double([[1.]]),
				   'weight_path': [weight_path]}

		lip_params = self.DEFAULT_LIPSDP_KWARGS
		L = eng.solve_LipSDP(network, lip_params, nargout=1)

		if attr_name is None:
			attr_name = 'global_l2'
		setattr(self, attr_name, L)
		os.remove(weight_path)
		return L