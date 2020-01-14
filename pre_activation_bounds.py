""" Techniques to compute preactivation bounds for piecewise linear nets"""

from relu_nets import ReLUNet
from hyperbox import Hyperbox
import utilities as utils
import torch
import numpy as np
import torch.nn as nn


# ====================================================================
# =           Class to store preactivation bound results             =
# ====================================================================
#  REFACTOR THIS -- SOOOOO UGLY!!!

class PreactivationBounds(object):

    @classmethod
    def naive_ia(cls, network, hyperbox):
        return naive_interval_analysis(network, hyperbox)

    @classmethod
    def preact_constructor(cls, preact_input, network, hyperbox):
        if preact_input == 'ia':
            return cls.naive_ia_from_hyperbox(network, hyperbox)
        elif isinstance(preact_input, PreactivationBounds):
            return preact_input


    def __init__(self, network, hyperbox):
        self.network = network
        self.hyperbox = hyperbox
        self.low_dict = {}
        self.high_dict = {}
        self.backprop_lows = {}
        self.backprop_highs = {}
        self.backprop_vector = None

    def _forward_computed(self):
        return sum(len(_) for _ in [self.low_dict, self.high_dict]) > 0

    def _backward_computed(self):
        return sum(len(_) for _ in
                   [self.backprop_lows, self.backprop_highs]) > 0


    def add_ith_layer_bounds(self, i, lows, highs):
        """ Adds the bounds as a [n_neurons, 2] numpy array """
        self.low_dict[i] = lows
        self.high_dict[i] = highs

    def get_ith_layer_bounds(self, i, two_col=False):
        output = self.low_dict[i], self.high_dict[i]
        if two_col:
            return utils.two_col(*output)
        return output

    def bound_iter(self, two_col=False):
        for i in range(len(self.low_dict)):
            yield self.get_ith_layer_bounds(i, two_col=two_col)

    def get_ith_layer_backprop_bounds(self, i, two_col=False):
        output = self.backprop_lows[i], self.backprop_highs[i]
        if two_col:
            return utils.two_col(*output)
        return output

    def get_ith_layer_on_off_set(self, i):
        """ For the i^th layer neurons of a neural_net,
            will return a dict like
            {'on': {on_indices in a set}
             'off': {off_indices in a set}
             'uncertain': {indices that may be on or off}}
        """
        output = {'on': set(),
                  'off': set(),
                  'uncertain': set()}
        ith_low, ith_high = self.get_ith_layer_bounds(i)

        for j, (lo, hi) in enumerate(zip(ith_low, ith_high)):
            if hi < 0:
                output['off'].add(j)
            elif lo > 0:
                output['on'].add(j)
            else:
                output['uncertain'].add(j)
        return output

    def backprop_bounds(self, c_vector):
        """ For a given vector, vec, computes upper and lower bounds
            on the partial derivative d<vec, network(x)>/dNeuron
            for each neuron.

            This is related, but not identical to fast-lip
        ARGS:
            c_vector: torch.Tensor, np.ndarray, or 
            
            c_vector: torch.Tensor or np.ndarray - tensor or array
                      that is the output of self.network gets multiplied by
        RETURNS:
            None, but sets self.backprop_vector, self.backprop_low,
                           self.backprop_high
        """
        if self._backward_computed():
            return 
            
        preswitch_lows = {}
        preswitch_highs = {}
        postswitch_lows = {}
        postswitch_highs = {}

        # Do backprop bounds by iterating backwards
        # General rules -- only two layers in the backprop
        # Push intervals backwards through each layer
        # Initial interval starts out as [d_i, d_i] for each neuron i
        # where d = W^l * c_vector

        # EVERYTHING IS A NUMPY ARRAY!!!
        c_vector = utils.as_numpy(c_vector)
        for layer_no in range(len(self.network.fcs) - 1, -1, -1):
            fc = self.network.fcs[layer_no]
            # First case is special
            if layer_no == len(self.network.fcs) - 1:
                layer_lows = layer_highs = (c_vector.dot(utils.as_numpy(fc.weight)))
                preswitch_lows[layer_no] = layer_lows
                preswitch_highs[layer_no] = layer_highs
                continue
            # other cases are pretty usual of (switch -> linear layer)
            prev_lo = preswitch_lows[layer_no + 1]
            prev_hi = preswitch_highs[layer_no + 1]
            preact_lo, preact_hi = self.low_dict[layer_no], self.high_dict[layer_no]
            postswitch_lo, postswitch_hi = \
                PreactivationBounds._backprop_switch_layer(prev_lo, prev_hi,
                                                           preact_lo, preact_hi)
            next_lo, next_hi =\
                    PreactivationBounds._backprop_linear_layer(fc, postswitch_lo,
                                                               postswitch_hi)

            preswitch_lows[layer_no] = next_lo
            preswitch_highs[layer_no] = next_hi

        # Set values and return

        self.backprop_vector = c_vector
        self.backprop_lows = preswitch_lows
        self.backprop_highs = preswitch_highs


    @classmethod
    def _backprop_linear_layer(self, fc_layer, input_lows, input_highs):
        """ Subroutine to handle the backprop of a linear layer:
            i.e., given a function defined as y=Wx + b
            and some interval on the value of df/dy, want intervals for df/dx
        ARGS:
            layer_no: nn.Linear object - object we are backpropping through
            input_lows: np.Array - array for the lower bounds of the
                                       input gradient
            input_highs: np.Array - array for the upper bounds of the
                                        input gradient
        RETURNS:
            output_lows : np.Array - array for the lower bounds on
                                     the output gradients
            output_highs : np.Array - tensor for the high bounds on
                                      the output gradients
        """
        weight_t = utils.as_numpy(fc_layer.weight.t())
        midpoint = (input_lows + input_highs) / 2.0
        radius = (input_highs - input_lows) / 2.0
        new_midpoint = weight_t.dot(midpoint)
        new_radius = np.abs(weight_t).dot(radius)
        output_lows = new_midpoint - new_radius
        output_highs = new_midpoint + new_radius
        return output_lows, output_highs

    @classmethod
    def _backprop_switch_layer(self, input_lows, input_highs, preact_lows,
                               preact_highs):
        """ Does interval bound propagation through a switch layer. Follows
            the following rules: (elementwise)
            switch([lo, hi], a) :=
                --- [lo        , hi        ] if a is guaranteed to be 1
                --- [0         , 0         ] if a is guaranteed to be 0
                --- [min(lo, 0), max(0, hi)] if a is uncertain
        ARGS:
            input_lows: np.Array - array for lower bounds on the input
            input_highs: np.Array - array for upper bounds on the input
            preact_lows: np.Array - array for lower bounds on the relu
                                        preactivation (useful for computing
                                        which neurons are on/off/uncertain)
            preact_highs: np.Array - array for upper bounds on the relu
                                         preactivation (useful for computing
                                         which neurons are on/off/uncertain)
        RETURNS:
            output_lows : np.Array - array for the lower bounds on
                                         the output gradients
            output_highs : np.Array - array for the high bounds on
                                          the output gradients
        """

        # On case by default
        new_lows = np.copy(input_lows)
        new_highs = np.copy(input_highs)

        on_neurons = preact_lows > 0
        off_neurons = preact_highs < 0


        uncertain_neurons = (1 - (on_neurons + off_neurons))

        # Handle the off case
        new_lows[off_neurons] = 0.0
        new_highs[off_neurons] = 0.0


        # Handle the uncertain case
        new_lows[np.logical_and(uncertain_neurons, (input_lows > 0))] = 0.0
        new_highs[np.logical_and(uncertain_neurons, (input_highs < 0))] = 0.0



        return new_lows, new_highs




# ===============================================================
# =           Preactivation Bound Compute Techniques            =
# ===============================================================


def naive_interval_analysis(network, domain):
    """ Most naive form of interval bound propagation --
        implemented using equation (6) from
        https://arxiv.org/pdf/1810.12715.pdf
    ARGS:
        network : ReLUNet object - network we're building bounds for
        domain: Hyperbox object - bounds on the input we allow
    RETURNS:
        PreactivationBounds object which holds the values we care
        about
    """

    preact_object = PreactivationBounds(network, domain)
    prev_lows, prev_highs = domain.box_low, domain.box_hi
    relu_num = 0
    for layer_num, layer in enumerate(network.net):
        if isinstance(layer, nn.ReLU):
            preact_object.add_ith_layer_bounds(relu_num, prev_lows, prev_highs)
            relu_num += 1
            prev_lows = np.maximum(prev_lows, 0)
            prev_highs = np.maximum(prev_highs, 0)
        elif isinstance(layer, nn.Linear):
            midpoint = (prev_lows + prev_highs) / 2.0
            radius = (prev_highs - prev_lows) / 2.0
            new_midpoint = utils.as_numpy(layer(torch.Tensor(midpoint)))
            new_radius = utils.as_numpy(torch.abs(layer.weight)).dot(radius)
            prev_lows = new_midpoint - new_radius
            prev_highs = new_midpoint + new_radius


    if isinstance(network.net[-1], nn.Linear):
        preact_object.add_ith_layer_bounds(relu_num, prev_lows, prev_highs)

    return preact_object


def improved_interval_analysis(network, domain):
    pass # Do later

def linear_programming_relaxation(network, domain):
    pass # Do later



