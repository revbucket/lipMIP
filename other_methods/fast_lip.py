""" Lipschitz (over)Estimation from this paper/repo:
	Arxiv: https://arxiv.org/abs/1804.09699
	Github: https://github.com/huanzhang12/CertifiedReLURobustness
"""

import numpy as np
import math
from .other_methods import OtherResult 
import utilities as utils 
from pre_activation_bounds import PreactivationBounds
from interval_analysis import HBoxIA

class FastLip(OtherResult):

	def __init__(self, network, c_vector, domain, primal_norm):
		super(FastLip, self).__init__(network, c_vector, domain, primal_norm)

	def compute(self):
		# Fast lip is just interval bound propagation through backprop
		timer = utils.Timer()
		preacts = HBoxIA(self.network, self.domain, self.c_vector)
		preacts.compute_forward()
		preacts.compute_backward()

		backprop_box = preacts.gradient_range

		# Worst case vector is max([abs(lo), abs(hi)])
		self.worst_case_vec = np.maximum(abs(backprop_box.box_low),
										 abs(backprop_box.box_hi))
		# And take dual norm of this
		dual_norm = {'linf': 1, 'l1': np.inf, 'l2': 2}[self.primal_norm]
		value = np.linalg.norm(self.worst_case_vec, ord=dual_norm)

		self.value = value
		self.compute_time = timer.stop()
		return value
