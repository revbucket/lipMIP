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

# ===============================================================================
# =           Helpful all-purpose functions                                     =
# ===============================================================================



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
