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
import inspect 
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
