from cvxpy.expressions.variables import Variable, NonNegative
from cvxpy.expressions.types import add_expr, neg_expr
from cvxpy.atoms import max_elemwise, square, inv_pos
from cvxpy.lin_ops import lin_utils
from cvxpy.constraints import *

from expectation import expectation
from expectation import clamp_or_sample_rvs
import strings

import math
import copy

class Phi:
    NONE = 1
    HINGE = 2

class prob:

    """
    Notes:
    - constr must be in the form Prob( f(x) >= g(x) ), where f is convex and g is concave (g is often just 0)
    - if phi == Phi.HINGE, then we expect Prob( f(x) >= g(x) ) to be <= beta, where beta is usually small (e.g., beta = 0.05)
    """
    def __init__(self, constr, num_samples, phi=Phi.HINGE):
        self.constr = constr
        self.num_samples = num_samples
        self.phi = phi

        if self.phi == Phi.HINGE:
            self.alpha = NonNegative(1)

    def __le__(self, beta):
        constr_rearranged = self.constr.args[1] - self.constr.args[0] # Note: constr.rh_exp <==> f(x) and constr.lh_exp <==> g(x) in the "Note" above
        self.conservative = expectation(max_elemwise(0.0, constr_rearranged + self.alpha), self.num_samples)

        if self.phi == Phi.HINGE:
            return self.conservative - self.alpha*beta <= 0
        else:
            raise Exception(strings.UNSUPPORTED_BOUND_TYPE)

    def get_prob(self):
        if self.phi == Phi.HINGE:
            return self.conservative.value / self.alpha.value # Note: the returned value here is an *upper bound* on Prob( f(x) >= 0 ), and it should be  <= beta
        else:
            constr_rearranged = self.constr.lh_exp - self.constr.rh_exp

            true_cnt = 0.0
            rvs2samples, ignore = clamp_or_sample_rvs(constr_rearranged, rvs2samples={}, want_de=False, want_mf=False, num_samples=self.num_samples, num_burnin_samples=0)
            for s in range(self.num_samples):
                constr_rearranged_copy = copy.deepcopy(constr_rearranged)
                constr_rearranged_copy_rlzd = clamp_or_sample_rvs(constr_rearranged_copy, rvs2samples, want_de=False, want_mf=False, sample_idx=s)
                if constr_rearranged_copy_rlzd.value <= 0:
                    true_cnt += 1.0

            return true_cnt / self.num_samples