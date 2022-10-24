""" Evaluations and experimental helpers """
import other_methods as om
from lipMIP import LipMIP, LipResult
import utilities as utils
from hyperbox import Hyperbox
import numpy as np
import pickle
import os
from neural_nets import data_loaders as dl
import math

class Experiment(utils.ParameterObject):
	""" Will set up factories for a bunch of methods """
	VALID_CLASSES = om.OTHER_METHODS + [LipMIP]
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

	def attach_kwargs(self, **kwargs):
		""" Attaches attributes to the experiment in the constructor_kwargs"""
		for k, v in kwargs.items():
			self.constructor_kwargs[k] = v


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
		if not isinstance(r, list):
			cube = Hyperbox.build_linf_ball(np.zeros(dimension), r)
			return self(domain=cube, **kwargs).compute()

		output = []
		for subr in r:
			cube = Hyperbox.build_linf_ball(np.zeros(dimension), subr)
			output.append(self(domain=cube, **kwargs).compute())
		return ResultList(output)


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
	""" Group of LipMIP, OtherResult that all share the same params
		Will evaluate all of them together and return the result in a nice
		dict
	"""
	def __init__(self, instance_dict, constructor_kwargs, call_kwargs,
				 ig_verbose=False):
		self.instance_dict = instance_dict
		self.total_kwargs = {k: v for k,v in constructor_kwargs.items()}
		self.ig_verbose = ig_verbose
		for k, v in call_kwargs.items():
			self.total_kwargs[k] = v
		for k, v in self.total_kwargs.items():
			setattr(self, k, v)

	def compute(self, verbose=False):
		output_dict = {}
		for k, v in self.instance_dict.items():
			if verbose or self.ig_verbose:
				print("Working on %s" % k)
			if isinstance(v, LipMIP):
				try: # This sometimes fails on random instances...
					result = v.compute_max_lipschitz().shrink()
				except:
					result = None
			if isinstance(v, om.LipLP):
				try: # This also sometimes fails =(
					v.compute()
					result = v
				except:
					result = None
			elif isinstance(v, om.OtherResult):
				v.compute()
				result = v
			if result is not None:
				output_dict[k] = result
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

	def get_rel_err(self, dim):
		""" Collects the relative error of each method and reports stats"""

		def dim_scale(k, val, dim):
			if k not in ['SeqLip', 'LipSDP']:
				return val
			else:
				return math.sqrt(dim) * val
		rel_errors = {}
		for result in self.results:
			val_dict = result.values()
			if 'LipMIP' not in val_dict:
				continue
			right_answer = val_dict['LipMIP']
			for k, v in val_dict.items():
				if k not in rel_errors:
					rel_errors[k] = []
				rel_errors[k].append(dim_scale(k, v, dim) / right_answer)
		return {k: (np.array(v).mean(), np.array(v).std(), len(v)) for
				k,v in rel_errors.items()}

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



# ==========================================================================
# =           OFFLINE EXPERIMENT SCRIPT HELPERS                            =
# ==========================================================================


class MethodNest:
	""" Think of this as a (method, set-of-arguments).
		We'll hand this object an Experiment Object and this will supply
		the
			-method of the Experiment object to run
		 	-arguments to call that method
	"""
	METHODS = {Experiment.do_random_evals, Experiment.do_unit_hypercube_eval,
				 Experiment.do_large_radius_eval, Experiment.do_data_evals}

	def __init__(self, method, arg_bundle=None):
		assert method in self.METHODS
		self.method = method
		self.arg_bundle = arg_bundle or {}


	def __call__(self, experiment, **kwargs):
		""" Runs the experiment object"""
		ARGMAPPER = {Experiment.do_random_evals: self.args_do_random_evals,
 					 Experiment.do_unit_hypercube_eval: self.args_do_unit_hypercube_eval,
					 Experiment.do_large_radius_eval: self.args_do_large_radius_eval,
					 Experiment.do_data_evals: self.args_do_data_evals}

		args = ARGMAPPER[self.method]()
		for k, v in kwargs.items():
			args[k] = v
		return self.method(experiment, **args)


	def args_do_random_evals(self):
		""" Handles arguments for random evals. Structure of arg_bundle
			looks like:
		{num_random_points: int for how many random points to take
		 sample_domain:     Hyperbox to draw random points from
		 ball_factory:      object that takes in point and makes a hyperbox
		}
		"""
		req_keys = ['num_random_points', 'sample_domain', 'ball_factory']
		return {k: self.arg_bundle[k] for k in req_keys}

	def args_do_unit_hypercube_eval(self):
		""" Needs no args! =)"""
		return {}


	def args_do_large_radius_eval(self):
		""" Needs a radius or list of radii for large-radius evals """
		return {'r': self.arg_bundle['r']}


	def args_do_data_evals(self):
		""" Complicated arg bundles!
		Must have a 'data_type', 'loader_kwargs', 'ball_factory' keys
		But if data_type is MNIST, 'loader_kwargs' corresponds to kwargs
		for dl.load_mnist_data.

		If data_type i= 'synthetic', then we need to have a dataset parameter
		object in arg_bundle['params']
		and then 'loader_kwargs' corresponds to kwargs for
		RandomDataset object
		"""

		# First consider the dataset:
		assert self.arg_bundle['data_type'] in ['MNIST', 'synthetic']

		# Do MNIST data generation
		if self.arg_bundle['data_type'] == 'MNIST':
			data_loader = dl.load_mnist_data(**self.arg_bundle['loader_kwargs'])
			data = next(iter(data_loader))[0]

		# Do synthetic data generation
		elif self.arg_bundle['data_type'] == 'synthetic':
			params = self.arg_bundle['params']
			dataset = dl.RandomDataset(params,
								  	   **self.arg_bundle['loader_kwargs'])
			dataset.split_train_val(1.0)
			data = dataset.train_data[0][0]
		else:
			pass

		return {'data_points': data,
				'ball_factory': self.arg_bundle['ball_factory'],
				'num_random': self.arg_bundle.get('num_random', None),
				}



class Job(utils.ParameterObject):
	""" Job is an object that represents a set of experiments to be run.
		It has the following properties:
		- ReLuNet
		- Which techniques to evaluate
		- Which methods to run for each
		- A 'name'
		And the following functions:
		- run(...) runs all the experiments, SAFELY, and returns the answer
		  in a pickleable object
		- write(...) writes this UNEXECUTED JOB to a file
		- @classmethod: load from file
	"""

	def __init__(self, name, experiment, method_nests,
				 save_loc=None, **extra_args):
		""" Builds an experiment object and stores instructions on how to
			run each method:
		ARGS:
			name : name of this job, helpful for writing files
			network : ReLUNet object
			class_list : list of lipschitz estimation classes
			method_nests: list of MethodNest objects
			exp_kwargs : any other kwargs to be used to build the
						 experiment object
		"""
		super(Job, self).__init__(name=name,
								  experiment=experiment,
								  method_nests=method_nests,
								  save_loc=save_loc,
								  **extra_args)


	@classmethod
	def from_file(cls, filename):
		# Loads file and unpickles from a job object
		with open(filename, 'rb') as f:
			return pickle.load(f)



	def _get_savefile(self, ext='.job'):
		assert ext[0] == '.'
		if self.save_loc is not None:
			write_file = os.path.join(self.save_loc, self.name)
		else:
			write_file = self.name
		return '%s%s' % (write_file, ext)


	def run(self, write_to_file=True, **kwargs):
		""" Safely runs every method as described by the method nests"""
		output_object = {}
		for method_nest in self.method_nests:
			output_object[method_nest.method.__name__] = \
					method_nest(self.experiment, **kwargs)


		output_object['Job'] = self

		if write_to_file:
			with open(self._get_savefile(ext='.result'), 'wb') as f:
				pickle.dump(output_object, f)
		return output_object


	def write(self):
		""" Pickles this object and writes it to a file """
		with open(self._get_savefile(), 'wb') as f:
			pickle.dump(self, f)