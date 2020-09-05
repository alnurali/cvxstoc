import unittest
import time
import math

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
import pymc
from scipy.linalg import sqrtm
import scipy.stats

from cvxpy import CVXOPT
from cvxstoc import NormalRandomVariable, prob
import cvxpy as cp
import numpy


class TestChanceConstr(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(1)

    def assert_feas(self, prob):
        if prob.status is not "infeasible":
            self.assertAlmostEqual(1, 1)
        else:
            self.assertAlmostEqual(1, 0)

    def test_simple_problem(self):
        # Create problem data.
        n = numpy.random.randint(1, 10)
        eta = 0.95
        num_samples = 10

        c = numpy.random.rand(n, 1)

        mu = numpy.zeros(n)
        Sigma = numpy.eye(n)
        a = NormalRandomVariable(mu, Sigma)

        b = numpy.random.randn()

        # Create and solve optimization problem.
        x = cp.Variable(n)
        p = cp.Problem(
            cp.Maximize(x.T * c),
            [prob(cp.max(x.T * a - b) >= 0, num_samples) <= 1 - eta],
        )
        p.solve()
        self.assert_feas(p)

    def test_compute_conserv_approx_to_prob(self):
        omega = NormalRandomVariable(0, 1)
        event = cp.exp(omega) <= 0.5
        num_samples = 50
        self.assertAlmostEqual(1, 1)

    def test_robust_svm(self):
        # Create problem data.
        m = 100  # num train points
        m_pos = math.floor(m / 2)
        m_neg = m - m_pos

        n = 2  # num dimensions
        mu_pos = 2 * numpy.ones(n)
        mu_neg = -2 * numpy.ones(n)
        sigma = 1
        X = numpy.matrix(
            numpy.vstack(
                (
                    mu_pos + sigma * numpy.random.randn(m_pos, n),
                    mu_neg + sigma * numpy.random.randn(m_neg, n),
                )
            )
        )

        y = numpy.hstack((numpy.ones(m_pos), -1 * numpy.ones(m_neg)))

        C = 1  # regularization trade-off parameter
        ns = 50
        eta = 0.1

        # Create and solve optimization problem.
        w, b, xi = cp.Variable(n), cp.Variable(), cp.Variable(m, nonneg=True)

        constr = []
        Sigma = 0.1 * numpy.eye(n)
        for i in range(m):
            mu = numpy.array(X[i])[0]
            x = NormalRandomVariable(mu, Sigma)
            chance = prob(-y[i] * (w.T * x + b) >= (xi[i] - 1), ns)
            constr += [chance <= eta]

        p = cp.Problem(cp.Minimize(cp.norm(w, 2) + C * cp.sum(xi)), constr)
        p.solve(verbose=True)

        w_new = w.value
        b_new = b.value

        # Create and solve the canonical SVM problem.
        constr = []
        for i in range(m):
            constr += [y[i] * (X[i] * w + b) >= (1 - xi[i])]

        p2 = cp.Problem(cp.Minimize(cp.norm(w, 2) + C * cp.sum(xi)), constr)
        p2.solve()

        w_old = w.value
        b_old = b.value

        self.assert_feas(p)

    def test_value_at_risk(self):
        # Create problem data.
        n = numpy.random.randint(1, 10)
        pbar = numpy.random.randn(n)
        Sigma = numpy.eye(n)
        p = NormalRandomVariable(pbar, Sigma)

        o = numpy.ones((n, 1))
        beta = 0.05
        num_samples = 50

        # Create and solve optimization problem.
        x = cp.Variable(n)
        p1 = cp.Problem(
            cp.Minimize(-x.T * pbar),
            [prob(-x.T * p >= 0, num_samples) <= beta, x.T * o == 1, x >= -0.1],
        )
        p1.solve(solver=cp.ECOS)

        # Create and solve analytic form of optimization problem (as a check).
        p2 = cp.Problem(
            cp.Minimize(-x.T * pbar),
            [
                x.T * pbar
                >= scipy.stats.norm.ppf(1 - beta) * cp.norm2(sqrtm(Sigma) * x),
                x.T * o == 1,
                x >= -0.1,
            ],
        )
        p2.solve()

        tol = 0.1
        if numpy.abs(p1.value - p2.value) < tol:
            self.assertAlmostEqual(1, 1)
        else:
            self.assertAlmostEqual(1, 0)

    def test_yield_constr_cost_min(self):
        # Create problem data.
        n = 10
        c = numpy.random.randn(n)
        P, q, r = numpy.eye(n), numpy.random.randn(n), numpy.random.randn()
        mu, Sigma = numpy.zeros(n), 0.1 * numpy.eye(n)
        omega = NormalRandomVariable(mu, Sigma)
        m, eta = 100, 0.95

        # Create and solve optimization problem.
        x = cp.Variable(n)
        yield_constr = (
            prob(cp.quad_form(x + omega, P) + (x + omega).T * q + r >= 0, m) <= 1 - eta
        )
        p = cp.Problem(cp.Minimize(x.T * c), [yield_constr])
        p.solve(solver=cp.ECOS)
        self.assert_feas(p)


if __name__ == "__main__":
    unittest.main()
