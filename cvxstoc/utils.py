from cvxstoc.random_variable import RandomVariable
from cvxpy.transforms.partial_optimize import PartialProblem
from cvxpy import Problem
import copy


def replace_rand_vars(obj):
    """Replaces all random variables with copies.

    Parameters
    ----------
    obj : Object
        The object to replace random variables in.

    Returns
    -------
    Object
        An object identical to obj, but with the random variables replaced.
    """
    if isinstance(obj, RandomVariable):
        return copy.deepcopy(obj)
    # Leaves other than random variables are preserved.
    elif len(obj.args) == 0:
        return obj
    elif isinstance(obj, PartialProblem):
        prob = obj.args[0]
        new_obj = replace_rand_vars(prob.objective)
        new_constr = []
        for constr in prob.constraints:
            new_constr.append(replace_rand_vars(constr))
        new_args = [Problem(new_obj, new_constr)]
        return obj.copy(new_args)
    # Parent nodes are copied.
    else:
        new_args = []
        for arg in obj.args:
            new_args.append(replace_rand_vars(arg))
        return obj.copy(new_args)
