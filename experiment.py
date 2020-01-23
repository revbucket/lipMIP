""" Evaluations and experimental helpers """
import other_methods as om
from lipMIP import LipProblem, LipResult
import utilities as utils
from hyperbox import Hyperbox
import numpy as np

class Experiment(utils.ParameterObject):
	""" Will set up factories for a bunch of methods """
	VALID_CLASSES = om.OTHER_METHODS + [LipProblem]
	def __init__(self, class_list, **kwargs):
		assert all(_ in self.VALID_CLASSES for _ in class_list)
		super(Experiment, self).__init__(class_list=class_list, **kwargs)
		self.factory_dict = {}
		self.constructor_kwargs = kwargs
		for el in class_list:
			self.factory_dict[el.__name__] = utils.Factory(el, **kwargs)

	def __call__(self, **kwargs):
		output_dict = {k : v(**kwargs) for k,v in self.factory_dict.items()}
		return InstanceGroup(output_dict, self.constructor_kwargs, kwargs)


	def _get_dimension(self, **kwargs):		
		if 'network' in kwargs:
			network = kwargs['network']
		else:
			network = self.network
		return network.layer_sizes[0]


	def compute(self, **kwargs):
		return self(**kwargs).compute()


	def do_random_evals(self, num_random_points, sample_domain, 
						ball_factory, **kwargs):
		""" Will pick a num_random_points in sample_domain and 
			use ball_factory to make a new domain over them and 
			will evaluate at each
		ARGS:
			num_random_points: int - number of random points to check 
			sample_domain: Domain - has method random_point(...) which 
						   returns many random points 
			ball_factory: functional that takes in a random point and 
						  outputs a Hyperbox
			kwargs: any other kwargs to send to the factories (no domain!)
		RETURNS:
			list of instance group output_dicts
		"""
		assert 'domain' not in kwargs
		outputs = []
		random_points = sample_domain.random_point(num_random_points)
		for random_point in random_points:
			domain = ball_factory(x=random_point)
			outputs.append(self(domain=domain, **kwargs).compute())
		return ResultList(outputs)


	def do_unit_hypercube_eval(self, **kwargs):
		""" Do evaluation over the entire [0,1] hyperbox """
		assert 'domain' not in kwargs
		dimension = self._get_dimension(**kwargs)
		cube = Hyperbox.build_unit_hypercube(dimension)
		return self(domain=cube, **kwargs).compute()


	def do_large_radius_eval(self, r, **kwargs):
		""" Does evaluation of lipschitz constant of a super-large 
			radius 
		"""
		assert 'domain' not in kwargs
		dimension =self._get_dimension(**kwargs)

		cube = Hyperbox.build_linf_ball(np.zeros(dimension), r)
		return self(domain=cube, **kwargs).compute()


	def do_data_evals(self, data_points, ball_factory, 
					  num_random=None, **kwargs):
		""" Given a bunch of data points, we build balls around them 
			and compute lipschitz constants for all of them
		ARGS:
			data_points: tensor or np.ndarray - data points to compute lip for 
						 (these are assumed to be unique)
			ball_factory: LinfBallFactory object - object to generate hyperboxes
			label: None or str - label to attach to each point to trust
			max_lipschitz_kwargs : None or dict - kwargs to pass to 
								   compute_max_lipschitz fxn
			num_random: if not None, is int - how many random points we 
						collect (randomly) from the data points
			force_unique : bool - if True we only compute lipschitz constants 
						   for elements that are not really really close to 
						   things we've already computed.
		RETURNS:
			None, but appends to self.data_eval list
		"""
		assert 'domain' not in kwargs
		dim = self._get_dimension()

		data_points = utils.as_numpy(data_points).reshape((-1, dim))
		if num_random is not None and num_random < data_points.shape[0]:
			idxs = np.random.choice(data_points.shape[0], num_random)
			data_points = data_points[idxs]

		outputs = []
		for p in data_points:
			outputs.append(self(domain=ball_factory(x=p), **kwargs).compute())
		return ResultList(outputs)


class InstanceGroup:
	""" Group of LipProblem, OtherResult that all share the same params
		Will evaluate all of them together and return the result in a nice
		dict
	"""
	def __init__(self, instance_dict, constructor_kwargs, call_kwargs):
		self.instance_dict = instance_dict
		self.total_kwargs = {k: v for k,v in constructor_kwargs.items()}
		for k, v in call_kwargs.items():
			self.total_kwargs[k] = v
		for k, v in self.total_kwargs.items():
			setattr(self, k, v)

	def compute(self, verbose=False):
		output_dict = {}
		for k, v in self.instance_dict.items():
			if verbose:
				print("Working on %s %s" % (k, suffix))
			if isinstance(v, LipProblem):
				result = v.compute_max_lipschitz()
			elif isinstance(v, om.OtherResult):
				v.compute()
				result = v
			output_dict[k] = v
		return Result(output_dict, total_kwargs=self.total_kwargs)

	def __repr__(self):
		return '<INSTANCE GROUP. ' + self.instance_dict.__repr__() + '>'


class Result:
	def __init__(self, input_dict, total_kwargs=None):
		self.input_dict = input_dict
		self.total_kwargs = total_kwargs
		for k, v in (total_kwargs or {}).items():
			setattr(self, k, v)

	def __getitem__(self, k):
		return self.input_dict[k]

	def get_subattr(self, attr, k=None):
		""" Gets subattr for the input_dict """
		if attr is not None:
			getter = lambda k: getattr(self.input_dict[k], attr)
		else:
			getter = lambda k: self.input_dict[k]
		if k is not None:
			return getter(k)
		else:
			return {k: getter(k) for k, v in self.input_dict.items()}

	def objects(self, k=None):
		return self.get_subattr(None, k=k)

	def values(self, k=None):
		return self.get_subattr('value', k=k)

	def compute_times(self, k=None):
		return self.get_subattr('compute_time', k=k)


class ResultList:
	def __init__(self, results):
		self.results = results 

	def average_stdevs(self, attr):
		""" Collects the average and standard deviations by keys in each 
			input dict
		ARGS: 
			attr: string - must be 'value' or 'time'
		RETURNS:
			dict like:
				{k: (mean for k, stdev for k, # k)} for each k in each 
				input dict
		"""
		getter = {'value': lambda r: r.values(), 
				  'time':  lambda r: r.compute_times()}[attr]
		# collect set of keys
		key_list = set()
		for result in self.results:
			for k in result.input_dict:
				key_list.add(k)

		# aggregate data for all keys
		data_lists = {k: [] for k in key_list}
		for result in self.results:
			for k, v in getter(result).items():
				data_lists[k].append(v)

		get_mean = lambda arr: np.array(arr).mean() 
		get_stdev = lambda arr: np.array(arr).std()
		get_count = lambda arr: len(arr)

		return {k: (get_mean(v), get_stdev(v), get_count(v)) 
				for k,v in data_lists.items()}
