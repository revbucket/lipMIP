import numpy
import torch
import torch.nn as nn
import copy
import numpy as np
import numbers
import utilities as utils


class LinfBallFactory(object):
    """ Class to easily generate l_inf balls with fixed radius and global bounds
        (but no center)
    """
    def __init__(self, dimension, radius, global_lo=None, global_hi=None):
        self.dimension = dimension
        self.radius = radius
        self.global_lo = global_lo
        self.global_hi = global_hi

    def make_linf_ball(center):
        return Hyperbox.build_linf_ball(center, self.radius, 
                                        global_lo=self.global_lo,
                                        global_hi=self.global_hi)



class Hyperbox(object):
    def __init__(self, dimension):
        self.dimension = dimension

        # in the l_inf ball case
        self.center = None
        self.radius = None


        self.box_low = None # ARRAY!
        self.box_hi = None # ARRAY!

    def as_dict(self):
        return {'dimension':                self.dimension,
                'center':                   self.center,
                'radius':                   self.radius,
                'box_low':                  self.box_low,
                'box_hi':                   self.box_hi}



    @classmethod
    def from_dict(cls, saved_dict):
        domain = cls(saved_dict['dimension'])
        for s in ['center', 'radius', 'box_low', 'box_hi']:
            setattr(domain, s, saved_dict[s])
        return domain


    # ==============================================================
    # =           Forward facing methods                           =
    # ==============================================================

    @classmethod
    def build_unit_hypercube(cls, dim):
        return cls.build_linf_ball(np.ones(dim) * 0.5, 0.5)

    @classmethod
    def build_linf_ball(cls, x, radius, global_lo=None, global_hi=None):
        """ Case we mostly care about -- builds an L_infinity ball centered
            at x with specified radius and also intersects with hyperbox
            with specified global lo and hi bounds
        ARGS:
            x: np.Array or torch.Tensor - center of the linf ball
            radius: float - size of L_inf ball
            global_lo: float or np.Array/torch.Tensor (like x) -
                       lower bounds if there's a domain we care about too
            global_hi : float or np.Array/torch.Tensor (like x) - upper bounds
                        if there's a domain we care about too
        RETURNS:
            Domain object
        """
        x_np = utils.as_numpy(x).reshape(-1)
        domain = cls(x_np.size)
        domain.center = x_np
        domain.radius = radius

        domain._add_box_bound(x_np - radius, lo_or_hi='lo')
        domain._add_box_bound(x_np + radius, lo_or_hi='hi')
        if global_lo is not None:
            domain._add_box_bound(global_lo, lo_or_hi='lo')
        if global_hi is not None:
            domain._add_box_bound(global_hi, lo_or_hi='hi')

        return domain



    def random_point(self, num_points=1, tensor_or_np='np'):
    	""" Returns a uniformly random point in this hyperbox
    	ARGS:
    		num_points: int - number of output points to return
    		tensor_or_np: string ['np' or 'tensor'] - decides whether to
    					  return a torch.Tensor or a numpy array
    	RETURNS:
    		(numpy array | tensor) of w/ shape (num_points, self.x.shape[0])
    	"""
    	assert tensor_or_np in ['np', 'tensor']
    	diameter = (self.box_hi - self.box_low)
    	random_points = np.random.random((num_points, self.center.size))
    	random_points = random_points * diameter + self.box_low

    	if tensor_or_np == 'tensor':
    		return torch.Tensor(random_points)
    	else:
    		return random_points


    # ==========================================================================
    # =           Helper methods                                               =
    # ==========================================================================


    def _add_box_bound(self, val, lo_or_hi='lo'):
        """ Adds lower bound box constraints
        ARGS:
            val: float or np.array(self.dimension) -- defines the coordinatewise
                 bounds bounds
            lo_or_hi: string ('lo' or 'hi') -- defines if these are lower or upper
                      bounds
        RETURNS:
            None
        """
        if isinstance(val, numbers.Real):
            val = self._number_to_arr(val)

        attr, comp = {'lo': ('box_low', np.maximum),
                      'hi': ('box_hi', np.minimum)}[lo_or_hi]

        if getattr(self, attr) is None:
            setattr(self, attr, val)
        else:
            setattr(self, attr, comp(getattr(self, attr), val))
        return None


    def _number_to_arr(self, number_val):
        """ Converts float to array of dimension self.dimension """
        assert isinstance(number_val, numbers.Real)
        return np.ones(self.dimension) * number_val