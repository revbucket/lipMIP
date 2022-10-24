""" Lipschitz estimation using extreme value methods 
    Arxiv: https://arxiv.org/pdf/1801.10578.pdf
    Github: https://github.com/IBM/CLEVER-Robustness-Score
"""


from functools import partial
import scipy
import scipy.io as sio
from scipy.stats import weibull_min
import scipy.optimize
import numpy as np 
from .other_methods import OtherResult 
import utilities as utils

class CLEVER(OtherResult):
    MAX_BACKPROP_SIZE = 128 # So we don't backprop on batches of size 1024

    def __init__(self, network, c_vector, domain, primal_norm):
        super(CLEVER, self).__init__(network, c_vector, domain, primal_norm)


    def compute(self, num_batches=500, batch_size=1024, 
                weibull_fit_kwargs=None, nofit=False):
        """ Uses CLEVER to compute an estimate of maximum gradient.
            c.f. Algorithm 1 in the linked arxiv paper
            - takes num_batches groups of batch_size random points,
              and evaluates the dual norm of the gradient of each.
              Collects the max dual norm of each batch and then 
              uses extreme value theory to estimate the maximum
        ARGS:
            num_batches: int - how many batches we use 
            batch_size : int - size of each batch
            weibull_fit_kwargs : None or dict with keys 
                                {use_reg: bool, 
                                 shape_reg: float,
                                 c_init: float[]}
                                tbh, I don't know what this does: 
                                ask the IBM folks
        RETURNS:
        """
        weibull_fit_kwargs = weibull_fit_kwargs or {}
        timer = utils.Timer()
        dual = {'l1':  np.inf, 'l2': 2, 'linf': 1}[self.primal_norm]        
        batch_maxes = []
        for batch in range(num_batches):
            batch_max = None
            for subbatch_size in utils.partition(batch_size, 
                                                 self.MAX_BACKPROP_SIZE):
                rand_points = self.domain.random_point(num_points=subbatch_size)
                grads = self.network.get_grad_at_point(rand_points, self.c_vector)
                grad_norms = grads.norm(p=dual, dim=1)
                max_grad_norm = grad_norms.max().item()
                if batch_max is None or batch_max < max_grad_norm:
                    batch_max = max_grad_norm

            batch_maxes.append(batch_max)
        with utils.silent():
            batch_maxes = np.array(batch_maxes)
            weibull_fit_results = get_best_weibull_fit(batch_maxes,
                                                       **weibull_fit_kwargs)

        self.value =  -1 * weibull_fit_results[2]
        self.compute_time = timer.stop()
        return self.value

class CLEVER2(OtherResult):
    MAX_BACKPROP_SIZE = 128 
    def __init__(self, network, c_vector, domain, primal_norm):
        self(CLEVER2, self).__init__(network, c_vector, domain, primal_norm) 

    def compute(self, num_batches=500, batch_size=1024, use_cuda=False, 
                weibull_fit_kwargs=None):

        weibull_fit_kwargs = weibull_fit_kwargs or {}
        if use_cuda:
            self.network.cuda() 
        timer = utils.Timer()
        dual = {'l1':  np.inf, 'l2': 2, 'linf': 1}[self.primal_norm]        
        batch_maxes = []
        for batch in range(num_batches):
            batch_max = None
            for subbatch_size in utils.partition(batch_size, 
                                                 self.MAX_BACKPROP_SIZE):
                rand_points = self.domain.random_point(num_points=subbatch_size)
                if use_cuda:
                    rand_points = rand_points.cuda() 
                rand_points.requires_grad_(True)
                output = (self.network(rand_points) @ self.c_vector).sum() 
                grads = torch.autograd.grad(output, rand_points)[0].view(subbatch_size, -1)
                grad_norms = grads.norm(p=dual, dim=1)
                max_grad_norm = grad_norms.max().cpu().item()
                if batch_max is None or batch_max < max_grad_norm:
                    batch_max = max_grad_norm
            batch_maxes.append(batch_max)
        with utils.silent():
            batch_maxes = np.array(batch_maxes)
            weibull_fit_results = get_best_weibull_fit(batch_maxes,
                                                       **weibull_fit_kwargs)

        self.value =  -1 * weibull_fit_results[2]
        self.compute_time = timer.stop()
        self.network.cpu()
        return self.value


""" Copied Weibull fitting technique from CLEVER repo here because I don't 
    want to use multiprocessing with a Pool coming from the __main__ method.

    Code from IBM folks incoming....
"""


# We observe that the scipy.optimize.fmin optimizer (using Nelder–Mead method)
# sometimes diverges to very large parameters a, b and c. Thus, we add a very
# small regularization to the MLE optimization process to avoid this divergence
def fmin_with_reg(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None, 
                  maxfun=None, full_output=0, disp=1, retall=0, callback=None, 
                  initial_simplex=None, shape_reg = 0.01):
    # print('my optimier with shape regularizer = {}'.format(shape_reg))
    def func_with_reg(theta, x):
        shape = theta[2]
        log_likelyhood = func(theta, x)
        reg = shape_reg * shape * shape
        # penalize the shape parameter
        return log_likelyhood + reg
    return scipy.optimize.fmin(func_with_reg, x0, args, xtol, ftol, maxiter, maxfun,
         full_output, disp, retall, callback, initial_simplex)

# fit using weibull_min.fit and run a K-S test
def fit_and_test(rescaled_sample, sample, loc_shift, 
                 shape_rescale, optimizer, c_i):
    [c, loc, scale] = weibull_min.fit(-rescaled_sample, c_i, optimizer=optimizer)
    loc = - loc_shift + loc * shape_rescale
    scale *= shape_rescale
    ks, pVal = scipy.stats.kstest(-sample, 'weibull_min', args = (c, loc, scale))
    return c, loc, scale, ks, pVal




def get_best_weibull_fit(sample, use_reg = False, shape_reg = 0.01,
                         c_init=None):
    if c_init is None:
        c_init = [0.1,1,5,10,20,50,100]     
    # initialize dictionary to save the fitting results
    fitted_paras = {"c":[], "loc":[], "scale": [], "ks": [], "pVal": []}
    # reshape the data into a better range 
    # this helps the MLE solver find the solution easier
    loc_shift = np.amax(sample)
    dist_range = np.amax(sample) - np.amin(sample)
    # if dist_range > 2.5:
    shape_rescale = dist_range
    if shape_rescale < 1e-9:
        shape_rescale = 1.0
    # else:
    #     shape_rescale = 1.0
    print("shape rescale = {}".format(shape_rescale))
    rescaled_sample = np.copy(sample)
    rescaled_sample -= loc_shift
    rescaled_sample /= shape_rescale

    print("loc_shift = {}".format(loc_shift))
    ##print("rescaled_sample = {}".format(rescaled_sample))

    # fit weibull distn: sample follows reverse weibull dist, so -sample follows weibull distribution
    if use_reg:
        optimizer = partial(fmin_with_reg, shape_reg = shape_reg)
    else:
        optimizer = scipy.optimize.fmin


    results = [fit_and_test(rescaled_sample, sample, loc_shift, 
                            shape_rescale, optimizer, c_i) for c_i in c_init]

    for res, c_i in zip(results, c_init):
        c = res[0]
        loc = res[1]
        scale = res[2]
        ks = res[3]
        pVal = res[4]
        print("[DEBUG][L2] c_init = {:5.5g}, fitted c = {:6.2f}, loc = {:7.2f}, scale = {:7.2f}, ks = {:4.2f}, pVal = {:4.2f}, max = {:7.2f}".format(c_i,c,loc,scale,ks,pVal,loc_shift))
        
        ## plot every fitted result
        #plot_weibull(sample,c,loc,scale,ks,pVal,p)
        
        fitted_paras['c'].append(c)
        fitted_paras['loc'].append(loc)
        fitted_paras['scale'].append(scale)
        fitted_paras['ks'].append(ks)
        fitted_paras['pVal'].append(pVal)
    
    
    # get the paras of best pVal among c_init
    max_pVal = np.nanmax(fitted_paras['pVal'])
    if np.isnan(max_pVal) or max_pVal < 0.001:
        print("ill-conditioned samples. Using maximum sample value.")
        # handle the ill conditioned case
        return -1, -1, -max(sample), -1, -1, -1

    max_pVal_idx = fitted_paras['pVal'].index(max_pVal)
    
    c_init_best = c_init[max_pVal_idx]
    c_best = fitted_paras['c'][max_pVal_idx]
    loc_best = fitted_paras['loc'][max_pVal_idx]
    scale_best = fitted_paras['scale'][max_pVal_idx]
    ks_best = fitted_paras['ks'][max_pVal_idx]
    pVal_best = fitted_paras['pVal'][max_pVal_idx]
    
    return c_init_best, c_best, loc_best, scale_best, ks_best, pVal_best
    