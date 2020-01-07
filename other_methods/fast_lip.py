""" Lipschitz (over)Estimation from this paper/repo:
	Arxiv: https://arxiv.org/abs/1804.09699
	Github: https://github.com/huanzhang12/CertifiedReLURobustness
"""

import numpy as np
import math
from .other_methods import OtherResult 
import utilities as utils 
from pre_activation_bounds import PreactivationBounds

class FastLip(OtherResult):

	def __init__(self, network, c_vector, domain, lp):
		super(FastLip, self).__init__(network, c_vector, domain)
		assert lp in ['linf', 'l2', 'l1']
		self.lp = lp

	def compute(self, attr_name=None):
		# Fast lip is just interval bound propagation through backprop
		preacts = PreactivationBounds.naive_ia_from_hyperbox(self.network,
 														    self.domain)
		preacts.backprop_bounds(self.c_vector)

		backprop_lows = preacts.backprop_lows[0]
		backprop_highs = preacts.backprop_highs[0]

		# Worst case vector is max([abs(lo), abs(hi)])
		self.worst_case_vec = np.maximum(abs(backprop_lows), 
										 abs(backprop_highs))
		# And take dual norm of this
		value = np.linalg.norm(self.worst_case_vec, 
							   {'linf': 1, 'l1': np.inf, 'l2': 2}[self.lp])

		if attr_name is None:
			attr_name = 'global_%s' % self.lp
		setattr(self, attr_name, value)
		return value
