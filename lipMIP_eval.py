class EvaluationParameters(utils.ParameterObject):
	def __init__(self, hypercube_kwargs=None, radius_kwargs=None,
				 data_eval_kwargs=None, num_random_eval_kwargs=None,
				 max_lipschitz_kwargs=None):
		""" Stores parameters for evaluation objects """
		super(EvaluationParameters, self).__init__(**kwargs)

	def __call__(self, network, c_vector, data_iter=None):

		eval_obj = LipMIPEvaluation(network, c_vector)
		# Set up eval objects for each case of evaluation

		# Random evaluations
		if num_random_eval is not None:
			eval_obj.do_random_evals()

		# Hypercube evaluation


		# Large radius evaluation 


		# Data evaluation



		return 


class LipMIPEvaluation:
	""" Handy object to run evaluations of lipschitz constants of a 
		neural net -- with a fixed c_vector
	"""
	def __init__(self, network, c_vector):
		self.network = network
		self.c_vector
		self.unit_hypercube_eval = None
		self.large_radius_eval = None
		self.data_eval = []
		self.random_eval = []

	def do_random_evals(self, num_random_points, sample_domain, ball_factory, 
						max_lipschitz_kwargs=None):
		""" Generates several random points from the domain 
			and stores (space-minimized) in the random_evals attribute 
		ARGS:
			num_random_points: int - number of random points to try 
			sample_domain: Hyperbox object - domain to sample from 
			ball_factory: LinfBallFactory object - object to generate 
						  hyperboxes 
			max_lipschitz_kwargs: None or dict - contains kwargs to pass 
								  to the compute_max_lipschitz fxn.
		RETURNS:
			None, but appends the results to the 'random_eval' list
		"""
		max_lipschitz_kwargs = max_lipschitz_kwargs or {}
		random_points = sample_domain.random_point(num_random_points)
		for random_point in random_points:
			domain = ball_factory(random_point)
			result = compute_max_lipschitz(self.network, domain, 'linf', 
										   self.c_vector, **max_lipschitz_kwargs)
			self.random_eval.append(result)


	def do_unit_hypercube_eval(self, max_lipschitz_kwargs=None, 
							   force_recompute=False):
		""" Do evaluation over the entire [0,1] hyperbox """
		if self.unit_hypercube_eval is not None and force_recompute is False:
			return

		max_lipschitz_kwargs = max_lipschitz_kwargs or {}
		cube = Hyperbox.build_unit_hypercube(self.network.layer_size[0])
		result = compute_max_lipschitz(self.network, domain, 'linf', 
									   self.c_vector, **max_lipschitz_kwargs)
		self.unit_hypercube_eval = result


	def do_large_radius_eval(self, r, max_lipschitz_kwargs=None, 
							 force_recompute=False):
		""" Does evaluation of lipschitz constant of a super-large 
			radius 
		"""
		if force_recompute is False and self.large_radius_eval is not None:
			return 

		max_lipschitz_kwargs = max_lipschitz_kwargs or {}
		dim = self.network.layer_sizes[0]
		cube = Hyperbox.build_linf_ball(np.zeros(dim), r)
		result = compute_max_lipschitz(self.network, domain, 'linf', 
									   self.c_vector, **max_lipschitz_kwargs)
		self.large_radius_eval = result

	def do_data_evals(self, data_points, ball_factory, 
					  label=None, max_lipschitz_kwargs=None, 
					  force_unique=True):
		""" Given a bunch of data points, we build balls around them 
			and compute lipschitz constants for all of them
		ARGS:
			data_points: tensor or np.ndarray - data points to compute lip for 
						 (these are assumed to be unique)
			ball_factory: LinfBallFactory object - object to generate hyperboxes
			label: None or str - label to attach to each point to trust
			max_lipschitz_kwargs : None or dict - kwargs to pass to 
								   compute_max_lipschitz fxn
			force_unique : bool - if True we only compute lipschitz constants 
						   for elements that are not really really close to 
						   things we've already computed.
		RETURNS:
			None, but appends to self.data_eval list
		"""
		dim = ball_factory.dimension
		data_points = utils.as_numpy(data_points).reshape((-1, dim))

		if force_unique:
			TOLERANCE = 1e-6
			extant_points = [_.domain.center for _ in self.data_eval]
			unique = lambda p: not any([np.linalg.norm(p -_) < TOLERANCE
										for _ in extant_points])
			data_points = [p for p in data_points if unique(p)]

		for p in data_points:
			hbox = ball_factory(p)
			result = compute_max_lipschitz(self.network, hbox, 'linf', 
										   self.c_vector, **max_lipschitz_kwargs)
			if label is not None:
				result.attach_label(label)
			self.data_eval.append(result)
