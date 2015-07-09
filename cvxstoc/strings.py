# random_variable.py
DIDNT_PASS_EITHER_RV_OR_MODEL = "You didn't pass either a pymc random variable or pymc.Model object into the RandomVariable constructor."
DIDNT_PASS_PYMC_MODEL_OBJ = "You didn't pass a pymc.Model object into the RandomVariable constructor."
CANT_FIND_PYMC_RV_IN_PYMC_MODEL_OBJ = "The pymc random variable name that you passed into the RandomVariable constructor didn't match the name of any of the pymc variables attached to the pymc.Model that you also passed into the RandomVariable constructor."
BAD_RV_DIMS = "The pymc random variable that is attached to the pymc.Model object that you passed into the RandomVariable constructor had more than two dimensions (i.e. you wanted to create a RandomVariable that is neither a scalar nor a matrix)."
UNSUPPORTED_PYMC_RV = "It appears that pymc.Model object that you passed into the RandomVariable constructor contains a pymc random variable object that is unsupported."

UNSUPPORTED_RV_TYPE = "You asked RandomVariableFactory to create a RandomVariable that it doesn't know how to create."
BAD_VAL_MAP = "You passed into RandomVariableFactory a vals list that contains items that are neither int, float, nor numpy.ndarray."
UNSUPPORTED_CONSTR = "You created a Problem object that you passed into partial_optimize that contained a constraint that is unsupported right now."

# partial_optimize.py
CANT_DETERMINE_CURVATURE = "You called partial_optimize with a Problem object that contains neither a Minimize nor a Maximize statement; this is not supported."

# prob.py
UNSUPPORTED_BOUND_TYPE = "You called prob with an upper bound type that is not supported."
