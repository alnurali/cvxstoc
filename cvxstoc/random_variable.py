import copy

import pymc.Model, pymc.MCMC

import numpy
import scipy

import strings
from cvxpy import utilities, interface
import cvxpy.expressions.constants.constant, cvxpy.lin_ops.lin_utils


class RandomVariableFactory:
    _id = 0

    def __init__(self):
        pass

    def create_dirichlet_rv(self, alpha):                   # Note: alpha here is in R_+^K <==> x is in R^K ~ Dir(alpha)
        rv_inner_name = self.get_rv_name()
        rv_outer_name = self.get_rv_name()

        rv_pymc = pymc.CompletedDirichlet(name=rv_outer_name, D=pymc.Dirichlet(name=rv_inner_name, theta=alpha))

        metadata = {}
        metadata["mu"] = [ alpha[i]/numpy.sum(alpha) for i in range(len(alpha)) ]

        metadata["alpha"] = alpha

        return RandomVariable(rv=rv_pymc, metadata=metadata)

    def create_unif_rv(self, lower=0, upper=1, cont=True, shape=1):
        rv_name = self.get_rv_name()

        rv_pymc = None
        if cont:
            rv_pymc = pymc.Uniform(name=rv_name, lower=lower, upper=upper, size=shape)
        else:
            rv_pymc = pymc.DiscreteUniform(name=rv_name, lower=lower, upper=upper, size=shape)

        metadata = {}
        metadata["mu"] = (1.0/2.0) * (upper+lower)

        metadata["lower"] = lower
        metadata["upper"] = upper

        return RandomVariable(rv=rv_pymc, metadata=metadata)

    def create_categorical_rv(self, vals=None, probs=None, shape=1):
        rv_name = self.get_rv_name()

        rv_pymc = pymc.Categorical(name=rv_name, p=list(probs), size=shape)

        metadata = {}
        metadata["mu"] = len(probs) * numpy.asarray(probs)

        val_map = {}
        if vals:                                            # True when vals != None && vals != []
            for val in range(0, len(probs)):
                val_map[val] = vals[val]
        metadata["vals"] = vals
        metadata["probs"] = probs

        return RandomVariable(rv=rv_pymc, val_map=val_map, metadata=metadata)

    def create_normal_rv(self, mu, cov, shape=1):
        rv_name = self.get_rv_name()

        rv_pymc = None
        metadata = {}
        if isinstance(mu, numpy.ndarray):
            rv_pymc = pymc.MvNormal(name=rv_name, mu=mu, tau=scipy.linalg.inv(cov))

            metadata["mu"] = mu
            metadata["cov"] = cov
        else:
            rv_pymc = pymc.Normal(name=rv_name, mu=mu, tau=1.0/cov, size=shape)

            if shape == 1:
                metadata["mu"] = mu
                metadata["cov"] = cov
            else:
                metadata["mu"] = numpy.tile(mu,shape)
                metadata["cov"] = cov

        return RandomVariable(rv=rv_pymc, metadata=metadata)

    def get_rv_name(self):
        return "rv" + str(RandomVariableFactory.get_next_avail_id())

    @staticmethod
    def get_next_avail_id():
        RandomVariableFactory._id += 1
        return RandomVariableFactory._id

def NormalRandomVariable(mean, cov, shape=1):
    return RandomVariableFactory().create_normal_rv(mu=mean, cov=cov, shape=shape)

def CategoricalRandomVariable(vals, probs, shape=1):
    return RandomVariableFactory().create_categorical_rv(vals=vals, probs=probs, shape=shape)

def UniformRandomVariable(lower=0, upper=1, cont=True, shape=1):
    return RandomVariableFactory().create_unif_rv(lower=lower, upper=upper, cont=cont, shape=shape)

class RandomVariable(cvxpy.expressions.constants.parameter.Parameter):
    def __init__(self, rv=None, model=None, name=None, val_map=None, metadata=None):        # model == pymc.Model object
        if name is not None:
            self._name = name

        self._metadata = metadata

        self._val_map = val_map

        self.set_rv_model_and_maybe_name(rv, model)

        self.set_shape()

        rows, cols = self._shape.size
        super(RandomVariable, self).__init__(rows, cols, self._name)

    @property
    def mean(self):
        return self._metadata["mu"]

    def set_rv_model_and_maybe_name(self, rv, model):
        if rv is not None and model is None:
            self._rv = rv
            self._model = pymc.Model([self._rv])

            self._name = self._rv.__name__

        elif rv is not None and model is not None:
            self._rv = rv
            self._model = model

            self._name = self._rv.__name__

        elif rv is None and model is not None:

            self._model = model

            self._rv = None
            for pymc_variable in self._model.variables:
                if pymc_variable.__name__ == self._name:
                    if isinstance(pymc_variable, pymc.CompletedDirichlet):
                        if pymc_variable.parents["D"].observed:
                            continue
                        # Success.
                    elif hasattr(pymc_variable, "observed"):
                        if pymc_variable.observed:
                            continue
                        # Success.
                     # Failure.
                    else:
                        raise Exception(strings.UNSUPPORTED_PYMC_RV)

                    self._rv = pymc_variable
                    break

            if self._rv is None:
                raise Exception(strings.CANT_FIND_PYMC_RV_IN_PYMC_MODEL_OBJ)

        else:
            raise Exception(strings.DIDNT_PASS_EITHER_RV_OR_MODEL)

    def set_shape(self):
        shape = ()
        if self.has_val_map():
            val = self._val_map.values()[0]

            if isinstance(val, int) or isinstance(val, float):
                shape = (1,1)

            elif isinstance(val, numpy.ndarray):

                numpy_shape = val.shape
                if len(numpy_shape) == 1:
                    shape = (numpy_shape[0], 1)
                elif len(numpy_shape) == 2:
                    shape = (numpy_shape[0], numpy_shape[1])
                else:
                    raise Exception(strings.BAD_RV_DIMS)

            else:
                raise Exception(strings.BAD_VAL_MAP)

        else:
            pymc_shape = ()
            if isinstance(self._rv, pymc.CompletedDirichlet):
                pymc_shape = self._rv.parents["D"].shape
            else:
                pymc_shape = self._rv.shape

            if len(pymc_shape) == 0:
                shape = (1,1)
            elif len(pymc_shape) == 1:
                shape = (pymc_shape[0], 1)
            elif len(pymc_shape) == 2:
                shape = (pymc_shape[0], pymc_shape[1])
            else:
                raise Exception(strings.BAD_RV_DIMS)

        self._shape = utilities.Shape(*shape)

    def name(self):
        # Override.
        if self.value is None:
            return self._name
        else:
            return str(self.value)

    def __repr__(self):
        # Override.
        return "RandomVariable(%s, %s, %s)" % (self.curvature, self.sign, self.size)

    def __eq__(self, rv):
        # Override.
        return self._name == rv._name

    def __hash__(self):
        # Override.
        return hash(self._name)

    def __deepcopy__(self, memo):
        # Override.
        return self.__class__(rv=self._rv,
                              model=self._model,
                              name=self.name(),
                              val_map=self._val_map,
                              metadata=self._metadata)

    def sample(self, num_samples, num_burnin_samples=0):
        if num_samples == 0:
            return [None]

        mcmc = pymc.MCMC(self._model)
        mcmc.sample(num_samples, num_burnin_samples, progress_bar=False)
        samples = mcmc.trace(self._name)[:]

        if not self.has_val_map():
            return samples
        else:
            samples_mapped = [self._val_map[sample[0]] for sample in samples]
            return samples_mapped

    def has_val_map(self):
        if self._val_map is not None and len(self._val_map.values()) > 0:
            return True
        else:
            return False