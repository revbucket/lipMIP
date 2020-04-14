""" Lipschitz (over)Estimation using zonotopes
"""

import numpy as np
import math
from .other_methods import OtherResult 
import utilities as utils 
from pre_activation_bounds import PreactivationBounds
from interval_analysis import AbstractNN

class ZLip(OtherResult):
	def __init__(self, network, c_vector, domain, primal_norm):
		super(FastLip, self).__init__(network, c_vector, domain, primal_norm)

	def compute(self):
		# Fast lip is just interval bound propagation through backprop
		timer = utils.Timer()
		preacts = AbstractNN(self.network, self.domain, self.c_vector)
		preacts.compute_forward(technique='zonotope:deep')
		preacts.compute_backward(technique='zonotope:box')
		backprop_zono = preacts.gradient_range


		# Compute the maximum norm depending on the DUAL of the primal norm 
		if self.primal_norm == 'l_inf':
			value = backprop_zono.maximize_l1_norm_mip()
		elif self.primal_norm == 'l1':
			value = backprop_zono.maximize_linf_norm()

		self.value = value
		self.compute_time = timer.stop()
		return value
