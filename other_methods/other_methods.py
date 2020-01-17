""" Generic "Other Method" result object """
import inspect


class OtherResult:
	def __init__(self, network, c_vector, domain, primal_norm):
		""" Generic Abstract Class for holding results of other evaluations 
			of lipschitz constants 

		ARGS:
			network: relu_nets.ReLUNet object - which network we're evaluating
			c_vector: torch.Tensor - which c_vector we're multiplying the output 
					  by (recall that we require the 'f(x)' to be real valued)
			domain: if not None, hyperbox.Hyperbox object - 
					for local Lipschitz constants
			primal_norm: ['l1', 'l2', 'linf']: corresponds to the primal norm ||.||
					     for |f(x)-f(y)| <= L * ||x-y|| 
					     (so we'll often care about the dual norm of this)
		"""
		self.network = network
		self.c_vector = c_vector
		self.domain = domain
		assert primal_norm in ['l1', 'l2', 'linf']
		self.primal_norm = primal_norm
		self.compute_time = None
		self.value = None

	def attach_label(self, label):
		""" Cute way to attach a label to a result """
		self.label = label
