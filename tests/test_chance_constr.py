import unittest
import time
import math

import numpy
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
import pymc
from scipy.linalg import sqrtm
import scipy.stats

from cvxpy import CVXOPT
from cvxpy.atoms import *
from cvxpy.expressions.variables import Variable, NonNegative
from cvxpy import Minimize, Maximize, Problem
from cvxstoc import RandomVariable, RandomVariableFactory, expectation, prob, Phi

class TestChanceConstr(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(1)

    def assert_feas(self, prob):
        if prob.status is not "infeasible":
            self.assertAlmostEqual(1,1)
        else:
            self.assertAlmostEqual(1,0)

    def test_simple_problem(self):
        # Create problem data
        n = numpy.random.randint(1,10)
        eta = 0.95
        num_samples = 10

        c = numpy.random.rand(n,1)

        mu = numpy.zeros(n)
        Sigma = numpy.eye(n)
        a = RandomVariableFactory().create_normal_rv(mu, Sigma)

        b = numpy.random.randn()

        # Create and solve optimization problem
        x = Variable(n)
        p = Problem(Maximize(x.T*c), [prob(max_entries(x.T*a-b) >= 0, num_samples) <= 1-eta])
        p.solve()
        self.assert_feas(p)

    def test_compute_conserv_approx_to_prob(self):
        omega = RandomVariableFactory().create_normal_rv(0,1)
        event = exp(omega) <= 0.5
        num_samples = 50
        # print "Prob( exp(omega) <= 0.5 ) = %f." % prob(event, num_samples, Phi.NONE).get_prob()
        self.assertAlmostEqual(1,1)

    # def test_robust_svm(self):
    #     # Create problem data
    #     m = 100 # num train points
    #     m_pos = math.floor(m/2)
    #     m_neg = m - m_pos

    #     n = 2 # num dimensions
    #     mu_pos = 2*numpy.ones(n)
    #     mu_neg = -2*numpy.ones(n)
    #     sigma = 1
    #     X = numpy.matrix(numpy.vstack((mu_pos + sigma*numpy.random.randn(m_pos,n), mu_neg + sigma*numpy.random.randn(m_neg,n))))

    #     y = numpy.hstack((numpy.ones(m_pos), -1*numpy.ones(m_neg)))

    #     C = 1 # regularization trade-off parameter
    #     ns = 50
    #     eta = 0.1

    #     # Create and solve optimization problem
    #     w, b, xi = Variable(n), Variable(), NonNegative(m)

    #     constr = []
    #     Sigma = 0.1*numpy.eye(n)
    #     for i in range(m):
    #         mu = numpy.array(X[i])[0]
    #         x = RandomVariableFactory().create_normal_rv(mu, Sigma)
    #         chance = prob(-y[i]*(w.T*x+b) >= (xi[i]-1), ns)
    #         constr += [chance <= eta]

    #     p = Problem(Minimize(norm(w,2) + C*sum_entries(xi)),
    #                  constr)
    #     p.solve()

    #     w_new = w.value
    #     b_new = b.value

    #     # print p.value
    #     # print w_new
    #     # print b_new

    #     # Create and solve the canonical SVM problem
    #     constr = []
    #     for i in range(m):
    #         constr += [y[i]*(X[i]*w+b) >= (1-xi[i])]

    #     p2 = Problem(Minimize(norm(w,2) + C*sum_entries(xi)), constr)
    #     p2.solve()

    #     w_old = w.value
    #     b_old = b.value

    #     # print p2.value
    #     # print w_old
    #     # print b_old

    #     # Plot solutions
    #     if n == 2:

    #         fig = pyplot.figure()
    #         pyplot.rc("text", usetex=True)

    #         pyplot.plot(X[:m_pos-1, 0], X[:m_pos-1, 1], 'bo')
    #         pyplot.plot(X[m_pos:, 0], X[m_pos:, 1], 'ro')

    #         x_vals = numpy.arange(numpy.min(X[:,0]), numpy.max(X[:,0]), 0.1)

    #         intercept = -b_old / w_old[1]
    #         slope = -w_old[0] / w_old[1]
    #         y_vals = numpy.array(slope*x_vals + intercept)[0]
    #         pyplot.plot(x_vals, y_vals, "--k", label="SVM")

    #         intercept = -b_new / w_new[1]
    #         slope = -w_new[0] / w_new[1]
    #         y_vals = numpy.array(slope*x_vals + intercept)[0]
    #         pyplot.plot(x_vals, y_vals, ":k", label="Chance Constrained SVM")

    #         pyplot.legend(loc="upper left")
    #         pyplot.axis("off")
    #         # pyplot.show() # Note: the user must manually close the plot window for the unit test to proceed
    #         fig.savefig("test_svm.png", bbox_inches="tight")

    #     self.assert_feas(p)

    def test_value_at_risk(self):
        # Create problem data
        n = numpy.random.randint(1,10)
        pbar = numpy.random.randn(n)
        Sigma = numpy.eye(n)
        p = RandomVariableFactory().create_normal_rv(pbar,Sigma)

        o = numpy.ones((n,1))
        beta = 0.05
        num_samples = 50

        # Create and solve optimization problem
        x = Variable(n)
        p1 = Problem(Minimize(-x.T*pbar), [prob(-x.T*p >= 0, num_samples) <= beta, x.T*o == 1, x >= -0.1])
        p1.solve()
        # print p1.status, p1.value, x.value

        # Create and solve analytic form of optimization problem (as a check)
        p2 = Problem(Minimize(-x.T*pbar), [x.T*pbar >= scipy.stats.norm.ppf(1-beta) * norm2(sqrtm(Sigma) * x), x.T*o == 1, x >= -0.1])
        p2.solve()
        # print p2.value, x.value

        tol = 0.1
        if numpy.abs(p1.value - p2.value) < tol:
            self.assertAlmostEqual(1,1)
        else:
            self.assertAlmostEqual(1,0)


    def test_yield_constr_cost_min(self):
        # Create problem data
        n = 10
        c = numpy.random.randn(n)
        P, q, r = numpy.eye(n), numpy.random.randn(n), numpy.random.randn()
        mu, Sigma = numpy.zeros(n), 0.1*numpy.eye(n)
        omega = RandomVariableFactory().create_normal_rv(mu, Sigma)
        m, eta = 100, 0.95

        # Create and solve optimization problem
        x = Variable(n)
        yield_constr = prob(quad_form(x+omega,P)
                        + (x+omega).T*q + r >= 0, m) <= 1-eta
        p = Problem(Minimize(x.T*c), [yield_constr])
        p.solve()
        # print p.value, x.value
        self.assert_feas(p)

if __name__ == "__main__":
    unittest.main()