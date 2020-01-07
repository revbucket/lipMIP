""" Naive methods for Lipschitz computation 
	For the L2 domain:
	Upper bound = Prod(||W_1||, ||W_2||, ..., ||W_l||)
	Lower bound = ||Prod(W_1, W_2, ..., W_l)||

	Need some work to show for L-inf/L-1
"""
import numpy as np 
from .other_methods import OtherResult
import utilities as utils

# ==========================================================================
# =                      UPPER BOUNDS                                      =
# ==========================================================================




class NaiveUB(OtherResult):

	# =====================================================================
	# =           Various Norm Functions                                  =
	# =====================================================================
	@classmethod
	def op_norm(cls, w):
		return np.linalg.norm(utils.as_numpy(w))

	@classmethod
	def linf_norm(cls, w):
		# BE CAREFUL HERE -- IS THIS THE BEST WE CAN DO????
		return np.linalg.norm(utils.as_numpy(w), np.inf) # IS THIS RIGHT???

	# ====================================================================
	# =           Main Function Block                                    =
	# ====================================================================
	def __init__(self, network, c_vector, primal_norm):
		super(NaiveUB, self).__init__(network, c_vector, None, primal_norm)


		if self.primal_norm == 'l2':
			self.norm_fxn = self.op_norm

		elif self.primal_norm == 'linf':
			self.norm_fxn = self.linf_norm

	def compute(self, attr_name=None):
		""" Computes the L2 global lipschitz constant for this network
			by multiplying the operator norm of each weight matrix
		"""
		c_vec_norm = self.norm_fxn(self.c_vector)
		operator_norms = []
		for fc in self.network.fcs:
			operator_norms.append(self.norm_fxn(fc.weight))

		running_norm = c_vec_norm
		for op_norm in operator_norms:
			running_norm *= op_norm

		if attr_name is None:
			attr_name = 'global_%s' % self.primal_norm
		setattr(self, attr_name, running_norm)
		return running_norm

# ==========================================================================
# =                      LOWER BOUNDS                                      =
# ==========================================================================

class RandomLB(OtherResult):
	""" Randomly select points and return the max gradient"""
	def __init__(self, network, c_vector, domain, primal_norm):
		super(RandomLB, self).__init__(network, c_vector, domain, primal_norm)
		self.max_norm = None 
		self.max_point = None 
		self.max_grad = None

	def compute(self, num_points=1000):
		""" Computes maximum of dual norm of gradients of random points. 
			Can be called multiple times and will only improve
		"""
		pnorm = {'l1':  np.inf, 'l2': 2, 'linf': 1}[self.primal_norm]

		random_output = self.network.random_max_grad(self.domain, self.c_vector, 
													 num_points, pnorm=pnorm)
		if (self.max_norm is None) or (random_output['norm'] > self.max_norm):
			self.max_norm  = random_output['norm']
			self.max_point = random_output['point']
			self.max_grad  = random_output['grad']

		return self.max_norm