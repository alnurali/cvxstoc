import copy, Queue
import numpy

import cvxpy.expressions.types
from cvxpy.transforms.partial_optimize import PartialProblem

from random_variable import RandomVariable

def expectation(expr, num_samples=0, want_de=False, want_mf=False, num_burnin_samples=0):
    if want_de:
        return construct_det_equiv(expr)
    elif want_mf:
        return construct_mean_field(expr)
    else:
        return construct_saa(expr, num_samples, num_burnin_samples)

def get_all_poss_combs_of_rv_realizations(rvs):

    num_rvs = len(rvs)
    num_combs = 1
    for rv in rvs:
        num_combs *= len(rv._metadata["vals"])

    idx_combs = numpy.zeros((num_combs, num_rvs))


    for i, rv in enumerate(rvs):

        num_reps = 1
        if i < (num_rvs-1):
            for j in range(i+1, num_rvs):
                num_reps *= len(rvs[j]._metadata["vals"])

        num_tiles = 1
        if i > 0:
            for j in range(i):
                num_tiles *= len(rvs[j]._metadata["vals"])


        idxes = range(len(rv._metadata["vals"]))
        col = numpy.tile(numpy.repeat(idxes, num_reps), num_tiles)

        idx_combs[:,i] = col[:]


    return idx_combs

def construct_mean_field(expr):
    rvs2samples, rvs = clamp_or_sample_rvs(expr, rvs2samples={}, want_de=False, want_mf=True, num_samples=0, num_burnin_samples=0)
    return clamp_or_sample_rvs(copy.deepcopy(expr), rvs2samples, want_de=False, want_mf=True, num_samples=0, num_burnin_samples=0)

def construct_det_equiv(expr):

    rvs2samples, rvs = clamp_or_sample_rvs(expr, rvs2samples={}, want_de=True, want_mf=False, num_samples=0, num_burnin_samples=0)

    if not rvs2samples.keys():
        return expr

    else:

        combs = get_all_poss_combs_of_rv_realizations(rvs)
        num_combs = combs.shape[0]


        mul_exprs = []
        for s in range(num_combs):

            expr_copy = copy.deepcopy(expr)
            expr_copy_realized = clamp_or_sample_rvs(expr_copy, rvs2samples, want_de=True, want_mf=False, sample_idxes=combs[s,:])


            prob = 1
            for i, rv in enumerate(rvs):
                idx  = int(combs[s,i])
                prob *= rv._metadata["probs"][idx]

            prob = cvxpy.expressions.types.constant()(prob)


            expr_copy_realized_scaled = cvxpy.expressions.types.mul_expr()(prob, expr_copy_realized)
            mul_exprs.append(expr_copy_realized_scaled)


        return cvxpy.expressions.types.add_expr()(mul_exprs)

def construct_saa(expr, num_samples, num_burnin_samples):

    # Get samples of each RandomVariable in expr
    rvs2samples, ignore = clamp_or_sample_rvs(expr, rvs2samples={}, want_de=False, want_mf=False, num_samples=num_samples, num_burnin_samples=num_burnin_samples)

    # If expr contained no RandomVariables, we're done
    if not rvs2samples.keys():
        return expr

    # Else expr contained RandomVariables
    else:

        mul_exprs = []
        for s in range(num_samples-num_burnin_samples):

            expr_copy = copy.deepcopy(expr) # Deep copy expr (num_samples times)
            expr_copy_realized = clamp_or_sample_rvs(expr_copy, rvs2samples, want_de=False, want_mf=False, sample_idx=s) # Plug in a realization of each RandomVariable

            prob = cvxpy.expressions.types.constant()(1.0/(num_samples-num_burnin_samples))

            expr_copy_realized_scaled = cvxpy.expressions.types.mul_expr()(prob, expr_copy_realized)
            mul_exprs.append(expr_copy_realized_scaled)


        # Return an AddExpression using all the realized deep copies to the caller
        return cvxpy.expressions.types.add_expr()(mul_exprs)

def clamp_or_sample_rvs(expr, rvs2samples={}, want_de=None, want_mf=None, num_samples=None, num_burnin_samples=None, sample_idx=None, sample_idxes=None):

    # Walk expr and "process" each RandomVariable

    if not rvs2samples.keys():
        draw_samples = True
    else:
        draw_samples = False


    my_queue = Queue.Queue()
    my_queue.put(ExpectationQueueItem(expr))

    rvs = []

    rv_ctr = 0

    while True:

        queue_item = my_queue.get()
        cur_expr = queue_item._item
        if isinstance(cur_expr, RandomVariable):

            if draw_samples:
                rvs += [cur_expr]

                if cur_expr not in rvs2samples:
                    samples = cur_expr.sample(num_samples, num_burnin_samples)
                    rvs2samples[cur_expr] = samples

            else:

                if want_de:
                    idx = int(sample_idxes[rv_ctr])
                    cur_expr.value = cur_expr._metadata["vals"][idx]

                    rv_ctr += 1
                elif want_mf:
                    cur_expr.value = cur_expr.mean
                else:
                    cur_expr.value = rvs2samples[cur_expr][sample_idx]

        else:

            if isinstance(cur_expr, PartialProblem):
                my_queue.put(ExpectationQueueItem(cur_expr.args[0].objective.args[0]))

                for constr in cur_expr.args[0].constraints:
                    my_queue.put(ExpectationQueueItem(constr.args[0]))
                    my_queue.put(ExpectationQueueItem(constr.args[1]))

            else:
                for arg in cur_expr.args:
                    my_queue.put(ExpectationQueueItem(arg))

        if my_queue.empty():
            break


    if draw_samples:
        return rvs2samples, rvs
    else:
        return expr

class ExpectationQueueItem:
    def __init__(self, item):
        self._item = item
