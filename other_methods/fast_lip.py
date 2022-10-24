""" Lipschitz (over)Estimation from this paper/repo:
    Arxiv: https://arxiv.org/abs/1804.09699
    Github: https://github.com/huanzhang12/CertifiedReLURobustness
"""

import numpy as np
import math
from .other_methods import OtherResult 
import utilities as utils 

import bound_prop as bp 

class FastLip(OtherResult):

    def __init__(self, network, c_vector, domain, primal_norm):
        super(FastLip, self).__init__(network, c_vector, domain, primal_norm)

    def compute(self):
        # Fast lip is just interval bound propagation through backprop
        timer = utils.Timer()
        preacts = AbstractNN(self.network, self.domain, self.c_vector)
        preacts.compute_forward(technique='naive_ia')
        preacts.compute_backward(technique='naive_ia')

        backprop_box = preacts.gradient_range

        # Worst case vector is max([abs(lo), abs(hi)])
        self.worst_case_vec = np.maximum(abs(backprop_box.box_low.detach()),
                                         abs(backprop_box.box_hi.detach()))
        # And take dual norm of this
        dual_norm = {'linf': 1, 'l1': np.inf, 'l2': 2}[self.primal_norm]
        value = np.linalg.norm(self.worst_case_vec, ord=dual_norm)

        self.value = value
        self.compute_time = timer.stop()
        return value


class FastLip2(OtherResult):
    def __init__(self, network, c_vector, domain, primal_norm):
        super(FastLip2, self).__init__(network, c_vector, domain, primal_norm)

    def compute(self):
        timer = utils.Timer() 
        ap = bp.AbstractParams.hyperbox_params() 
        ann = bp.AbstractNN2(self.network)
        grad_range = ann.get_both_bounds(ap, self.domain, self.c_vector)[1].output_range

        max_coords = grad_range.as_twocol().abs().max(dim=1)[0]
        dual_norm = {'linf': 1, 'l1': np.inf, 'l2': 2}[self.primal_norm]        
        self.value = max_coords.norm(p=dual_norm).item() 
        self.compute_time = timer.stop() 
        return self.value 
