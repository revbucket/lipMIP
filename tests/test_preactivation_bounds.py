""" Tests for the preactivation bound files. This is a little tricky
    so we test to make sure this works well. These tests are not conclusive, 
    but will hopefully fail if there's a bug involved 
"""

import plnn
import pre_activation_bounds as pab
from hyperbox import Hyperbox
import numpy as np 
import torch 
import utilities as utils

# ========================================================================
# =           Test Naive Interval Analysis                               =
# ========================================================================

# HELPER FUNCTION
def verify_ia(x, bound_obj):
    """ Verifies that point is indeed within the preactivation bounds
        x is a torch.tensor of shape [N, #in_features]
        bound_obj is a PreactivationBounds object
    """
    preact_values = bound_obj.network.forward(x, return_preacts=True)
    sats = []
    for i, layer_val in enumerate(preact_values):
        val_np = utils.as_numpy(layer_val)
        low_sat = (val_np >= bound_obj.low_dict[i]).all()
        high_sat = (val_np <= bound_obj.high_dict[i]).all()
        sats.append(low_sat and high_sat)
    assert all(sats)

# Simple test function
def test_2layer_naive_ia_bounds(num_test_points=1000):
    network = plnn.PLNN(layer_sizes=[10, 100, 10], bias=True)
    center = np.zeros(10)
    radius = 1.0
    global_lo = -1.0
    global_hi = 1.0

    simple_box = Hyperbox.build_linf_ball(center, radius, global_lo, global_hi)
    bound_obj = pab.PreactivationBounds.naive_ia_from_hyperbox(network, simple_box)
    x = torch.rand(num_test_points, 10) * 2 - 1
    verify_ia(x, bound_obj)

# More complicated test
def test_5layer_naive_ia_bounds(num_test_points=1000):
    network = plnn.PLNN(layer_sizes=[128, 256, 512, 128, 64, 32, 16], bias=True)
    center = np.ones(128) * 0.5
    radius = 1.0
    global_lo = 0.2
    global_hi = 1.0
    simple_box = Hyperbox.build_linf_ball(center, radius, global_lo, global_hi)
    bound_obj = pab.PreactivationBounds.naive_ia_from_hyperbox(network, simple_box)
    x = torch.rand(num_test_points, 128).clamp(0.2, 1.0)
    verify_ia(x, bound_obj)





# ========================================================================
# =           Test FastLip Values                                        =
# ========================================================================
def verify_gradient_bounds(num_test_points, bound_obj, c_tens):
    random_points = bound_obj.hyperbox.random_point(num_test_points, 
                                                    tensor_or_np='np')
    random_tens = torch.tensor(random_points, requires_grad=True, 
                               dtype=torch.float32)
    output = torch.sum(bound_obj.network(random_tens).mv(c_tens))
    output.backward()
    np_grads = utils.as_numpy(random_tens.grad)
    lo, hi = bound_obj.get_ith_layer_backprop_bounds(0)
    lo = lo.reshape(1, -1).repeat(num_test_points, axis=0)
    hi = hi.reshape(1, -1).repeat(num_test_points, axis=0)
    assert (lo <= np_grads).all() and (np_grads <= hi).all()


def test_2layer_backprop_bounds(num_test_points=1000):
    torch.random.manual_seed(42069)
    network = plnn.PLNN(layer_sizes=[2, 10, 2], bias=True)
    center = np.zeros(2)
    radius = 1.0
    global_lo = -1.0
    global_hi = 1.0
    simple_box = Hyperbox.build_linf_ball(center, radius, global_lo, global_hi)

    preact_bounds = pab.PreactivationBounds.naive_ia_from_hyperbox(network, simple_box)
    c_vec = np.array([1.0, -1.0])
    c_tens = torch.Tensor(c_vec)
    preact_bounds.backprop_bounds(c_vec)

    verify_gradient_bounds(num_test_points, preact_bounds, c_tens)


def test_5layer_backprop_bounds(num_test_points=1000):
    torch.random.manual_seed(42069)    
    network = plnn.PLNN(layer_sizes=[128, 256, 512, 128, 64, 32, 4], bias=True)
    center = np.ones(128) * 0.5
    radius = 1.0
    global_lo = 0.2
    global_hi = 1.0
    simple_box = Hyperbox.build_linf_ball(center, radius, global_lo, global_hi)
    preact_bounds = pab.PreactivationBounds.naive_ia_from_hyperbox(network, simple_box)
    x = torch.rand(num_test_points, 128).clamp(0.2, 1.0)
    c_vec = np.array([3.0, -1.0, 2.0, -4.0])
    c_tens = torch.Tensor(c_vec)
    preact_bounds.backprop_bounds(c_vec)

    verify_gradient_bounds(num_test_points, preact_bounds, c_tens)

if __name__ == '__main__':
    test_2layer_naive_ia_bounds()
    test_5layer_naive_ia_bounds()
    test_2layer_backprop_bounds()
    test_5layer_backprop_bounds()



