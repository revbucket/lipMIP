""" Doing the convex relaxation of LipMIP only"""

from .other_methods import OtherResult
import lipMIP as lm
import utilities as utils
import gurobipy as gb

class LipLP(OtherResult):

    def __init__(self, network, c_vector, domain, primal_norm):
        assert primal_norm != 'l2'
        super(LipLP, self).__init__(network, c_vector, domain, primal_norm)

    def compute(self, preact_method='ia', tighter_relu=False):
        timer = utils.Timer()

        # Use the GurobiSquire / model constuctor in lipMIP file
        squire, _, _ = lm.build_gurobi_model(self.network, self.domain,
                                             self.primal_norm, self.c_vector,
                                             preact_method=preact_method)

        # And then we'll just change any binary variables to continous ones:
        model = squire.lp_ify_model(tighter_relu=tighter_relu)

        # And optimize
        model.optimize()
        if model.Status == 3:
            print('INFEASIBLE')
        self.value = model.getObjective().getValue()
        self.compute_time = timer.stop()

        return self.value
