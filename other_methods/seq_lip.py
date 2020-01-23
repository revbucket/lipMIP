""" Lipschitz (over)Estimation from this paper/repo: 
    Arxiv: https://arxiv.org/abs/1805.10965
    Github: https://github.com/avirmaux/lipEstimation
"""

from .other_methods import OtherResult
import other_methods.lipEstimation.seqlip as sl
import utilities as utils
import torch
import numpy as np

class SeqLip(OtherResult):

    def __init__(self, network, c_vector, domain, primal_norm):
        super(SeqLip, self).__init__(network, c_vector, domain, primal_norm)

    def compute(self):
        """ Estimate (without any guarantee if upper or lower bound) by 
            searching over ReLU signs of each layer. 
        """
        timer = utils.Timer()
        # Step 1: split up the network
        # f(x) matches r/(LR)+Lx/ 
        # J(x) matches r/(L Lambda)+L/ 

        # We do SVD's on each linear layer
        svds = []
        for linear_layer in self.network.fcs:
            if (linear_layer == self.network.fcs[-1] and 
                self.c_vector is not None):
                c_vec = torch.tensor(self.c_vector).view(1, -1)
                c_vec = c_vec.type(self.network.dtype)
                weight = c_vec @ linear_layer.weight
            else:
                weight = linear_layer.weight
            svds.append(torch.svd(weight))

        num_relus = len(self.network.fcs) - 1

        # Then set up each of the (num_relus) subproblems:
        subproblems = [] # [(Left, Right), ....]
        for relu_num in range(num_relus):
            left_svd = svds[relu_num + 1]
            right_svd = svds[relu_num]
            _, sigma_ip1, v_ip1 = svds[relu_num + 1]
            u_i, sigma_i, _ = svds[relu_num]

            if relu_num != num_relus - 1:
                sigma_ip1 = torch.sqrt(sigma_ip1)
            if relu_num != 0:
                sigma_i = torch.sqrt(sigma_i)
            sigma_i = torch.diag(sigma_i)
            sigma_ip1 = torch.diag(sigma_ip1)
            subproblems.append(((sigma_ip1 @ v_ip1.t()).data,
                                (u_i @ sigma_i).data))
        # And solve each of the subproblems:
        dual_norm = {'linf': 1, 'l2': 2, 'l1': np.inf}[self.primal_norm]

        lips = [sl.optim_nn_pca_greedy(*_, verbose=False, use_tqdm=False,
                                       norm_ord=2)[0]
                for _ in subproblems]
        self.value = utils.prod(lips)
        self.compute_time = timer.stop()
        return self.value