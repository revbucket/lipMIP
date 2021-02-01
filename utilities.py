""" General all-purpose utilities """
import sys
import torch
import torch.nn.functional as F
import numpy as np
import gurobipy as gb
import matplotlib.pyplot as plt
import io
import contextlib
import tempfile
import time 
import re
import pickle
import inspect 
import glob
import torch.nn as nn
import os

COMPLETED_JOB_DIR = os.path.join(os.path.dirname(__file__), 'jobs', 'completed')
# ===============================================================================
# =           Helpful all-purpose functions                                     =
# ===============================================================================

class ParameterObject:
	def __init__(self, **kwargs):
		self.attr_list = []
		assert 'attr_list' not in kwargs
		for k,v in kwargs.items():
			setattr(self, k, v)
			self.attr_list.append(k)

	def change_attrs(self, **kwargs):
		new_kwargs = {}
		for attr in self.attr_list:
			if attr in kwargs:
				new_kwargs[attr] = kwargs[attr]
			else:
				new_kwargs[attr] = getattr(self, attr)
		return self.__class__(**new_kwargs)

class Factory(ParameterObject):
	def __init__(self, constructor, **kwargs):
		self.constructor = constructor
		super(Factory, self).__init__(**kwargs)

	def __call__(self, **kwargs):
		cons_args = inspect.getfullargspec(self.constructor).args
		# Make default args from attributes
		args = {k: getattr(self, k) for k in self.attr_list if k in cons_args}

		# Update the default args
		for k,v in kwargs.items():
			if k in cons_args:
				args[k] = v

		# Build object
		return self.constructor(**args)

	def __repr__(self):
		return '<Factory: %s>' % self.constructor.__self__.__name__


class DoEvery:

	@classmethod 
	def dummy(cls, *args, **kwargs):
		pass

	def __init__(self, func, freq):
		""" Simple class that holds onto a function and it returns 
			this function every freq iterations
		ARGS:
			func: function object to be returned every freq iterations
			freq: int - how often to return the function 
		"""
		self.func = func 
		self.freq = freq 
		self.i = 0

	def __call__(self, *args, **kwargs):
		if self.i % self.freq == 0:
			returner = self.func 
		else:
			returner = self.dummy 
		self.i += 1
		return returner(*args, **kwargs)




class Timer:
	def __init__(self, start_on_init=True):
		if start_on_init:
			self.start()

	def start(self):
		self.start_time = time.time()

	def stop(self):
		self.stop_time = time.time()
		return self.stop_time - self.start_time

	def reset(self):
		self.start_time = self.stop_time = None

def cpufy(tensor_iter):
	""" Takes a list of tensors and safely pushes them back onto the cpu"""
	return [_.cpu() for _ in tensor_iter]

def cudafy(tensor_iter):
	""" Takes a list of tensors and safely converts all of them to cuda"""
	def safe_cuda(el):
		try:
			return el.cuda()
		except AssertionError:
			return el 
	return [safe_cuda(_) for _ in tensor_iter]

def prod(num_iter):
	""" returns product of all elements in this iterator *'ed together"""
	cumprod = 1
	for el in num_iter:
		cumprod *= el
	return cumprod


def partition(n, m):
	""" Given ints n > m, partitions n into an iterable where all 
		elements are m, except for the last one which is (n % m)
	"""
	count = 0
	while count < n:
		yield min([m, n - count])
		count += m

def flatten_list(lol):
	""" Given list of lists, flattens it into a single list. """

	output = []
	for el in lol:
		if not isinstance(el, list):
			output.append(el)
			continue
		output.extend(flatten_list(el))
	return output

def partition_by_suffix(iterable, func):
	""" Given an iterable and a boolean-valued function which takes in 
		elements of that iterable, outputs a list of lists, where each list 
		ends in an element for which the func returns true, (except for the 
		last one)		
		e.g. 
		iterable := [1, 2, 3, 4, 5,5, 5]
		func := lambda x: (x % 2) == 0
		returns [[1,2], [3,4], [5, 5, 5]]
	"""
	output = [] 
	sublist = [] 
	for el in iterable:
		sublist.append(el)
		if func(el):
			output.append(sublist)
			sublist = []

	if len(sublist) > 0:
		output.append(sublist)
	return output


def arraylike(obj):
	return isinstance(obj, (torch.Tensor, np.ndarray))


def as_numpy(tensor_or_array):
    """ If given a tensor or numpy array returns that object cast numpy array
    """
    if isinstance(tensor_or_array, torch.Tensor):
        tensor_or_array = tensor_or_array.cpu().detach().numpy()
    return tensor_or_array


def two_col(l, r):
	""" Takes two numpy arrays of size N and makes a numpy array of size Nx2
	"""
	return np.vstack([l, r]).T

def split_pos_neg(x):
	if isinstance(x, torch.Tensor):
		return split_tensor_pos_neg(x)
	else:
		return split_ndarray_pos_neg(x)


def split_tensor_pos_neg(x):
	""" Splits a tensor into positive and negative components """
	pos = F.relu(x)
	neg = -F.relu(-x)
	return pos, neg

def split_ndarray_pos_neg(x):
	""" Splits a numpy ndarray into positive and negative components """
	pos = x * (x >= 0)
	neg = x * (x <= 0)
	return pos, neg

def swap_axes(x, source, dest):
	""" Swaps the dimensions of source <-> dest for torch/numpy 
	ARGS:
		x : numpy array or tensor 
		source : int index
		dest : int index
	RETURNS
		x' - object with same data as x, but with axes swapped 
	"""
	if isinstance(x, torch.Tensor):
		return x.transpose(source, dest)
	else:
		return np.moveaxis(x, source, dest)


def build_var_namer(k):
	return lambda d: '%s[%s]' % (k, d)

@contextlib.contextmanager
def silent():
	save_stdout = sys.stdout 
	temp = tempfile.TemporaryFile(mode='w')
	sys.stdout = temp
	yield 	
	sys.stdout = save_stdout
	temp.close()

def ia_mm(matrix, intervals, lohi_dim, matrix_or_vec='matrix'):
	""" Interval analysis matrix(-vec) multiplication for torch/np intervals
		
	ARGS:
		matrix : tensor or numpy array of shape (m,n) - 
		intervals : tensor or numpy array with shape (n1, ..., 2, n_i, ...) - 
				    "vector" of intervals to be multiplied by a matrix 
				    one such n_i must be equal to n (from matrix shape)
		lohi_dim : int - which dimension (index) of intervals corresponds
				   		  to the lo/hi split 
		matrix_or_vec : string - must be matrix or vec, corresponds to whether
						intervals is to be treated as a matrix or a vector. 
						If a v
	RETURNS:
		object of same type as intervals, but with the shape slightly 
		different: len(output[-1/-2]) == m
	"""


	# asserts for shapes and things 
	assert isinstance(matrix, torch.Tensor) # TENSOR ONLY FOR NOW 
	assert isinstance(intervals, torch.Tensor)
	m, n = matrix.shape 
	assert intervals.shape[lohi_dim] == 2	
	assert matrix_or_vec in ['matrix', 'vec']

	if matrix_or_vec == 'vec':
		intervals = intervals.unsqueeze(-1)

	assert lohi_dim != intervals.dim() - 2 
	assert intervals[dim][-2] == n


	# define operators based on tensor/numpy case 
	matmul = lambda m, x: m.matmul(x)
	stack = lambda a, b: torch.stack([a, b])

	# now do IA stuff
	intervals = swap_axes(intervals, 0, lohi_dim)
	matrix_pos, matrix_neg = split_pos_neg(matrix)
	los, his = intervals

	new_los = matmul(matrix_pos, los) + matmul(matrix_neg, his)
	new_his = matmul(matrix_pos, his) + matmul(matrix_neg, los)

	intervals = swap_axes(stack(new_los, new_his), 0, lohi_dim)
	if matrix_or_vec == 'vec':
		intervals = interval.squeeze(-1)
	return intervals


def random_ortho2(input_dim):
	# Get 2 random orthogonal vectors in input_dim 

	dirs = torch.randn(2, input_dim)
	dir1 = dirs[0] / torch.norm(dirs[0])
	dir2_unnorm = dirs[1] - (dir1 @ dirs[1]) * dir1 
	dir2 = dir2_unnorm / torch.norm(dir2_unnorm)
	return torch.stack([dir1, dir2])

def monotone_down_zeros(f, lb, ub, num_steps=100, tol=1e-8):
    # Finds the zeros of a monotone decreasing function (along the interval [lb, ub])
    for step in range(num_steps): 
        if f((lb + ub) / 2.0) > 0:
            lb = (lb + ub) / 2.0 
        else:
            ub = (lb + ub) / 2.0 
        if ub - lb < tol:
        	return (lb + ub) / 2.0

    return (lb + ub) / 2.0



# =============================================================================
# =           Image display functions                                         =
# =============================================================================

def display_images(image_rows, figsize=(8, 8)):
	""" Given either a tensor/np.array (or list of same), will display each 
		element in the row or tensor
	ARGS:
		image_rows: tensor or np.array or tensor[], np.array[] - 
				    image or list of images to display 
	RETURNS: None, but displays images 
	"""

	if not isinstance(image_rows, list):
		image_rows = [image_rows]

	np_rows = [as_numpy(row) for row in image_rows] 

	# Transpose channel to last dimension and stack to make rows
	np_rows = [np.concatenate(_.transpose([0, 2, 3, 1]), axis=1) 
			   for _ in np_rows]

	# Now stack rows
	full_image = np.concatenate(np_rows, axis=0)

	# And then show image 
	imshow_kwargs = {}
	if full_image.shape[-1] == 1:
		full_image = full_image.squeeze() 
		imshow_kwargs['cmap'] = 'gray'

	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot()
	ax.axis('off')
	ax.imshow(full_image, **imshow_kwargs)	
	plt.show()



# ======================================================
# =           Pytorch helpers                          =
# ======================================================

class NNAbs(nn.Module):
	def forward(self, x): 
		return torch.abs(x) 


def tensorfy(x, dtype=torch.float32):
	if isinstance(x, torch.Tensor):
		return x
	else:
		return torch.from_numpy(x).type(dtype)


def one_hot(labels, num_classes):
	""" Take a minibatch of labels and makes them 1-hot encoded """
	labels = labels.view(-1, 1)
	one_hot_vecs = torch.zeros(labels.numel(), num_classes)
	one_hot_vecs.scatter_(1, labels, 1)
	labels = labels.view(-1)
	return one_hot_vecs

def one_hot_training_data(trainset, num_classes):
	output = []
	for data, labels in trainset:
		output.append((data, one_hot(labels, num_classes)))
	return output 

def seq_append(seq, module):
	""" Takes a nn.sequential and a nn.module and creates a nn.sequential
		with the module appended to it
	ARGS:
		seq: nn.Sequntial object 
		module: <inherits nn.Module>
	RETURNS:
		nn.Sequential object 
	"""
	seq_modules = [seq[_] for _ in range(len(seq))] + [module]
	return nn.Sequential(*seq_modules)

def cpufy(tensor_iter):
	""" Takes a list of tensors and safely pushes them back onto the cpu"""
	output = []
	for el in tensor_iter:
		if isinstance(el, tuple):
			output.append(tuple(_.cpu() for _ in el))
		else:
			output.append(el.cpu())
	return output


def cudafy(tensor_iter):
	""" Takes a list of tensors and safely converts all of them to cuda"""
	def safe_cuda(el):
		try:
			if isinstance(el, tuple):
				return tuple(_.cuda() for _ in el)
			else:
				return el.cuda()
		except AssertionError:
			return el 
	return [safe_cuda(_) for _ in tensor_iter]


def conv2d_counter(x_size, conv2d):
	""" Returns the size of the output of a convolution operator 
	ARGS:
		x_size : tuple(int) - tuple of input sizes (c_in x H_in x W_in)
		conv2d : nn.Conv2D object 
	RETURNS:
		the shape of the output 
	"""
	c_in, h_in, w_in = x_size
	c_out = conv2d.out_channels 
	k0, k1 = conv2d.kernel_size 
	p0, p1 = conv2d.padding 
	s0, s1 = conv2d.stride 

	h_out = (h_in + 2 * p0 - k0) // s0 + 1 # round down apparently
	w_out = (w_in + 2 * p1 - k1) // s1 + 1
	return (c_out, h_out, w_out)


def conv2d_mod(x, conv2d, bias=True, abs_kernel=False):
	""" Helper method to do convolution suboperations:
	ARGS:
		x : tensor - input to convolutional layer
		conv2d : nn.Conv2d - convolutional operator we 'modify'
		bias: bool - true if we want to include bias, false o.w. 
		abs_kernel : bool - true if we use the absolute value of the kernel
	RETURNS:
		tensor output 
	"""
	if bias:
		bias = conv2d.bias 
	else:
		bias = None
	if abs_kernel: 
		weight = conv2d.weight.abs()
	else:
		weight = conv2d.weight

	if x.dim() == 3:
		x = x.unsqueeze(0)
	return F.conv2d(x, weight=weight, bias=bias, stride=conv2d.stride,
				    padding=conv2d.padding, dilation=conv2d.dilation,
				    groups=conv2d.groups)

def conv_transpose_2d_mod(x, layer, bias=True, abs_kernel=False):
	""" Helper method to do convolution suboperations:
	ARGS:
		x : tensor - input to convolutional layer
		layer : nn.Conv2d - convolutional operator we 'modify'
		bias: bool - true if we want to include bias, false o.w. 
		abs_kernel : bool - true if we use the absolute value of the kernel
	RETURNS:
		tensor output 
	"""
	if bias:
		bias = layer.bias 
	else:
		bias = None
	if abs_kernel: 
		weight = layer.weight.abs()
	else:
		weight = layer.weight

	if x.dim() == 3:
		x = x.unsqueeze(0)
	return F.conv_transpose2d(x, weight=weight, bias=bias, stride=layer.stride,
		  				    padding=layer.padding, dilation=layer.dilation,
						    groups=layer.groups)



# =======================================
# =           Polytope class            =
# =======================================

class Polytope:
	INPUT_KEY = 'input'
	SLACK_KEY = 'slack'
	def __init__(self, A, b):
		""" Represents a polytope of the form {x | AX <= b}
		    (where everything is a numpy array)
		"""
		self.A = A  
		self.b = b 

	def _input_from_model(self, model):
		var_namer = build_var_namer(self.INPUT_KEY)
		return np.array([model.getVarByName(var_namer(i)).X 
 						 for i in range(self.A.shape[1])])


	def _build_model(self, slack=False):
		""" Builds a gurobi model of this object """
		with silent():
			model = gb.Model() 

		input_namer = build_var_namer(self.INPUT_KEY)
		input_vars = [model.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, 
								   name=input_namer(i))
		 		      for i in range(self.A.shape[1])]
		if slack == True:		 		      
			slack_var = model.addVar(lb=0, ub=1.0, name=self.SLACK_KEY)
		else: 
			slack_var = 0

		for i, row in enumerate(self.A):
			model.addConstr(gb.LinExpr(row, input_vars) + slack_var <= self.b[i])
		model.update()
		return model

	def contains(self, x, tolerance=1e-6):
		return all(self.A @ x <= self.b + tolerance)


	def interior_point(self):
		model = self._build_model(slack=True)
		slack_var = model.getVarByName(self.SLACK_KEY)
		model.setObjective(slack_var, gb.GRB.MAXIMIZE)
		model.update()
		model.optimize()

		assert model.Status == 2
		return self._input_from_model(model)


	def intersects_hbox(self, hbox):
		""" If this intersects a given hyperbox, returns a 
			point contained in both
		"""

		model = self._build_model(slack=True)
		input_namer = build_var_namer(self.INPUT_KEY)

		for i, (lb, ub) in enumerate(hbox):
			var = model.getVarByName(input_namer(i))
			model.addConstr(lb <= var <= ub)

		slack_var = model.getVarByName(self.SLACK_KEY)
		model.setObjective(slack_var, gb.GRB.MAXIMIZE)
		model.update()		
		model.optimize()

		assert model.Status == 2
		return self._input_from_model(model)


# =========================================================
# =           experiment.Result object helpers            =
# =========================================================

def filename_to_epoch(filename):
	return int(re.search(r'_EPOCH\d{4}_', filename).group()[-5:-1])

def read_result_files(result_files):
	output = []
	for result_file in result_files:
		try:
			with open(result_file, 'rb') as f:
				output.append((result_file, pickle.load(f)))
		except Exception as err:
			print("Failed on file: ", result_file, err)
	return output

def job_out_series(job_outs, eval_style, method, 
				   value_or_time='value', avg_stdev='avg'):
	""" Takes in some result or resultList objects and 
		a 'method', and desired object, and returns these objects
		in a list
	ARGS:
		results: Result[] or ResultList[], results to consider
		eval_style: str - which method of Experiment we look at
		method: str - which Lipschitz-estimation technique to consider
		value_or_time: 'value' or 'time' - which number to return 
		avg_stdev: 'avg' or 'stdev' - for ResultList[], we can 
				   get average or stdev values 
	RETURNS:
		list of floats
	"""
	# check everything is the same type
	assert value_or_time in ['value', 'time']
	assert avg_stdev in ['avg', 'stdev']
	assert eval_style in ['do_random_evals', 'do_unit_hypercube_eval',
						      'do_data_evals', 'do_large_radius_evals']

	results = [job_out[eval_style] for job_out in job_outs]
	output = []
	for result in results:
		try: #Result object case
			if value_or_time == 'value':
				output.append(result.values(method))
			else:
				output.append(result.compute_times(method))
		except:
			triple = result.average_stdevs(value_or_time)[method]
			if avg_stdev == 'avg':
				output.append(triple[0])
			else:
				output.append(triple[1])
	return output


def collect_result_outs(filematch):
	""" Uses glob to collect and load result objects matching a series
	ARGS:
		filematch: string with *'s associated with it
				   e.g. 'NAME*SUBNAME*GLOBAL.result'
	RESULTS:
		list of (filename, experiment.Result) objects
	"""
	search_str = os.path.join(COMPLETED_JOB_DIR, filematch)
	sorted_filenames = sorted(glob.glob(search_str))
	return read_result_files(sorted_filenames)


def collect_epochs(filename_list):
	""" Given a list of (filename) objects, converts
		the filenames into integers, pulling the EPOCH attribute from 
		the filename 
	str[] -> int[]
	"""
	def epoch_gleamer(filename):
		basename = os.path.basename(filename)
		return int(re.search('_EPOCH\d+_', filename).group()[6:-1])
	return [epoch_gleamer(_) for _ in filename_list]


def data_from_results(result_iter, method, lip_estimator, time_or_value='value',
					  avg_or_stdev='avg'):
	""" Given a list of experiment.Result or experiment.ResultList objects
		will return the time/value for the lip_estimator of the method 
		for result (or avg/stdev if resultList objects)
		e.g., data_from_results('do_unit_hypercube_eval', 'LipMIP',
								 'value') gets a list of values of the 
								 LipMIP over the unitHypercube domain
	ARGS:
		method: str - name of one of the experimental methods 
		lip_estimator : str - name of the class of lipschitz estimator to use
		time_or_value : 'time' or 'value' - returning the time or value here
		avg_or_stdev  : 'avg' or 'stdev' - returning either avg or stdev of 
						results from ResultListObjects
	"""
	assert method in ['do_random_evals', 'do_data_evals',
					  'do_unit_hypercube_eval']
	assert lip_estimator in ['LipMIP', 'FastLip', 'LipLP', 'CLEVER', 
							 'LipSDP', 'NaiveUB', 'RandomLB', 'SeqLip']
	assert time_or_value in ['time', 'value']
	assert avg_or_stdev in ['avg', 'stdev']

	def datum_getter(result_obj):
		if not hasattr(result_obj, 'average_stdevs'):
			if time_or_value == 'value':
				return result_obj[method].values(lip_estimator)
			else:
				return result_obj[method].compute_times(lip_estimator)
		else:
			triple = result_obj.average_stdevs(time_or_value)
			if avg_or_stdev == 'avg':
				return triple[0]
			else:
				return triple[1]

	return [datum_getter(_) for _ in result_iter]

