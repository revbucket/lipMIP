""" General all-purpose utilities """
import torch
import torch.nn.functional as F
import numpy as np


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


def split_tensor_pos_neg(x):
	""" Splits a tensor into positive and negative components """
	pos = F.relu(x)
	neg = -F.relu(-x)
	return pos, neg

def build_var_namer(k):
	return lambda d: '%s[%s]' % (k, d)