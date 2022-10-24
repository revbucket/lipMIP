""" Lipschitz (over)Estimation using zonotopes
"""

import numpy as np
import math
from .other_methods import OtherResult
import utilities as utils
from pre_activation_bounds import PreactivationBounds
from interval_analysis import AbstractNN
import bound_prop as bp


class ZLip(OtherResult):
	def __init__(self, network, c_vector, domain, primal_norm):
		super(ZLip, self).__init__(network, c_vector, domain, primal_norm)

	def compute(self):
		# Fast lip is just interval bound propagation through backprop
		timer = utils.Timer()
		ap = bp.AbstractParams.basic_zono()
		ann = bp.AbstractNN2(self.network)

		self.grad_range = ann.get_both_bounds(ap, self.domain, self.c_vector)[1].output_range

		if self.primal_norm == 'linf':
			value = self.grad_range.maximize_l1_norm_abs()
		else:
			value = torch.max(self.grad_range.lbs.abs(),
							  self.grad_range.ubs.abs()).max()

		value = value.item()
		self.value = value
		self.compute_time = timer.stop()
		return value
