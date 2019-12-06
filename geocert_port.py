""" File to handle running of GeoCert --
	I rewrote some common classes for the lipMIP directory, 
	but I'll want to compare against what I get from geoCert for 
	verification purposes. This file handles conversion between 
	similar classes 

"""

import torch

# lipMIP objects
from hyperbox import Hyperbox
from relu_nets import ReLUNet


# Geocert objects
import sys
sys.path.append('./geometric_certificates/')
import geometric_certificates as geo

# ====================================================
# =           Object conversion functions            =
# ===========================================.=========

def hyperbox_to_geodomain(hyperbox):
	""" Converts a lipMIP.hyperbox.Hyperbox object to a 
		<geocert>.domains.Domain object
	"""
	hyperbox_dict = hyperbox.as_dict() 
	hyperbox_dict['dimension'] = hyperbox_dict['center'].size
	hyperbox_dict['l2_radius'] = (hyperbox_dict['dimension'] ** 0.5 * 
								  hyperbox_dict['radius'])
	pairmaps = [('box_high', 'box_hi'), ('linf_radius', 'radius'), 
				('x', 'center'), ('original_box_low', 'box_low'), 
				('original_box_high', 'box_hi'), 
				('unmodified_bounds_low', 'box_low'),
				('unmodified_bounds_high', 'box_hi')]

	for target, source in pairmaps:
		hyperbox_dict[target] = hyperbox_dict[source]
	return geo.Domain.from_dict(hyperbox_dict)

def relunet_to_geoplnn(network):
	""" Converts a lipMIP.relu_nets.ReLUNet object to a <geocert>.plnn.PLNN object """

	geo_plnn = geo.PLNN(layer_sizes=network.layer_sizes, 
						bias=network.bias, 
						dtype=network.dtype)

	geo_plnn.net = network.net 
	geo_plnn.fcs = network.fcs
	return geo_plnn

# ====================================================
# =           GeoCert Block                          =
# ====================================================

def geocert_max_lipschitz(network, hyperbox, l_p, c_vector, 
						  preact_method='ia', verbose=False, 
						  timeout=None):
	""" Computes the max lipschitz constant over the domain using geocert 
		-- has the same signature as 'lipMIP.compute_max_lipschitz(...) --
	ARGS:
		network: lipMip.relunets.ReLUNet object 
		hyperbox: lipmip.hyperbox.Hyperbox object 
		c_vector: torch.Tensor or np.array 
	RETURNS:
		output objects from the geocert instance
		{'return_obj': GeoCertReturn object, 
		'linreg_coll': LinearRegionCollection object,
		 'max_lipschitz': maximum lipschitz constant}
	"""

	# First convert network and domain to <geocert>-style objects 
	geodomain = hyperbox_to_geodomain(hyperbox)
	geoplnn = relunet_to_geoplnn(network)

	# And then make the geocert object and run it over the domain 
	bound_fxn = {'ia': 'ia'}[preact_method]
	hbox_bounds = geodomain.box_low, geodomain.box_high
	geocert_obj = geo.GeoCert(geoplnn, hyperbox_bounds=hbox_bounds, 
							  verbose=verbose, neuron_bounds=bound_fxn)
	center_tensor = torch.tensor(hyperbox.center, dtype=geoplnn.dtype)
	geocert_return = geocert_obj.run(center_tensor, lp_norm=l_p, 
									 problem_type='count_regions', 
									 collect_graph=True, max_runtime=timeout,
									 decision_radius=hyperbox.radius)
	# And then finally examine the output object and compute max-lipschitz 
	linear_regions = geo.LinearRegionCollection(geoplnn, geocert_return,
	 											objective_vec=c_vector,
 												do_setup=True)
	return {'return_obj': geocert_return, 
			'linreg_coll': linear_regions,
			'max_lipschitz': linear_regions.get_maximum_lipschitz_constant()}



if __name__ == '__main__':
	pass