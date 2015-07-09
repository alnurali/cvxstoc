import unittest

from matplotlib import pyplot
import math
import pymc
import numpy
import scipy.io
import time

from cvxpy import CVXOPT
from cvxpy.atoms import *
from cvxpy.expressions.variables import Variable, NonNegative
from cvxpy import Minimize, Maximize, Problem, utilities
from cvxstoc import RandomVariable, RandomVariableFactory, expectation
import cvxpy

class WordCountVecRV(RandomVariable):
    
    def sample(self, num_samples, num_burnin_samples=0): # Override
        
        mcmc = pymc.MCMC(self._model)        
        mcmc.sample(num_samples, num_burnin_samples, progress_bar=False)

        doc_idx = str(self._metadata["doc_idx"])
        
        num_unique_words = self._metadata["num_unique_words"]
        samples = numpy.zeros((num_samples, num_unique_words))
        for j in range(num_unique_words):

            rv_name = "w_" + doc_idx + "_" + str(j)
            sample = mcmc.trace(rv_name)[:]

            for i, val in enumerate(sample):
                samples[i,j] = val        
        return samples

    def set_shape(self): # Override
        
        num_unique_words = self._metadata["num_unique_words"]
        self._shape = utilities.Shape(num_unique_words,1)

        doc_idx = str(self._metadata["doc_idx"])
        self._name = "w_" + doc_idx


class TestStochastic(unittest.TestCase):

    def setUp(self):        
        numpy.random.seed(1)        
        self.run_test_web = False

    def assert_feas(self, prob):        
        if prob.status is not "infeasible":
            self.assertAlmostEqual(1,1)
        else:
            self.assertAlmostEqual(1,0)

    def test_news_vendor(self):
        """
        Here is a another (completely automated) way to generate a probability distribution over demand (d):
        num_scenarios = 3
        
        alpha = numpy.ones(num_scenarios)
        d_probs = list(numpy.random.dirichlet(alpha))
        
        d_min = 50
        d_max = 150        
        d_vals = list(numpy.random.randint(d_min, d_max, num_scenarios))        
        d = RandomVariableFactory().create_categorical_rv(d_vals, d_probs)
        
        u = d_max
        """
        
        start = time.time()        
        
        # Create problem data
        b, s, r, u = 10, 25, 5, 150        
        d_probs = [0.3, 0.6, 0.1]
        d_vals = [55, 139, 141]
        d = RandomVariableFactory().create_categorical_rv(d_vals, d_probs)
    
        # Create optimization variables
        x = NonNegative()
        y1, y2 = NonNegative(), NonNegative()

        # Create second stage problem
        obj = -s*y1 - r*y2
        constrs = [y1+y2<=x, y1<=d]
        p2 = Problem(Minimize(obj), constrs)
        Q = partial_optimize(p2, [y1, y2], [x])

        # Create and solve first stage problem
        p1 = Problem(Minimize(b*x + expectation(Q(x), want_de=True)), [x<=u])
        p1.solve()
        
        end = time.time()
        # print p1.value, x.value, y1.value, y2.value
        # print "time = [" + str(end-start) + "]"

        # Solve problem the old way (i.e., deterministic equivalent) as a check
        y = Variable(2*len(d_vals))
        num_y = y.shape.rows
        
        ev = 0
        constrs = []
        i = 0
        while (i < num_y):
            d_val = d_vals[i/2]
            d_prob = d_probs[i/2]
            
            ev += d_prob * (-s*y[i] - r*y[i+1])            
            constrs += [y[i] + y[i+1] <= x, 0 <= y[i], y[i] <= d_val, y[i+1] >= 0]
            
            i += 2
            
        p3 = Problem(Minimize(b*x + ev), [0 <= x, x <= u] + constrs)
        p3.solve()
        
        # print p3.value, x.value
        # for i in range(num_y):
        #     print y[i].value

        self.assertAlmostEqual(p1.value, p3.value, 5)

    def test_cvar(self):

        # Create problem data
        n = 10
        pbar, Sigma = numpy.random.randn(n), numpy.eye(n)
        p = RandomVariableFactory().create_normal_rv(pbar, Sigma)
        u, eta, m = numpy.random.rand(), 0.95, 100

        # Create optimization variables
        x, beta = NonNegative(n), Variable()

        # Create and solve optimization problem
        cvar = expectation(pos(-x.T*p - beta), m)
        cvar = beta + 1/(1-eta)*cvar
        prob = Problem(Minimize(expectation(-x.T*p,m)),
                       [x.T*numpy.ones((n,1)) == 1, cvar<=u])
        prob.solve()        
        # print prob.value, x.value, beta.value
        self.assert_feas(prob)

    def test_airline(self):

        # Create problem data
        E = 6
        C = 2
        P = 6

        u = numpy.random.randint(10,20,E)
        p = numpy.random.rand(P)

        d_mu = 100*numpy.ones(P) # Not used
        d_Sigma = 0.01*numpy.eye(P) # Not used

        m = 10 # Num samples        

        A = [self.get_A(i,E,P) for i in range(C)]
        
        # Create optimization variables
        x = [NonNegative(E) for i in range(C)]
        y = NonNegative(P)

        # Create second stage problem
        capacity = [A[i]*y<=x[i] for i in range(C)]            
        d = RandomVariable(pymc.Lognormal(name="d", mu=0,
                                          tau=1, size=P))
        p2 = Problem(Minimize(-y.T*p), [y<=d] + capacity)
        Q = partial_optimize(p2, [y], [x[0], x[1]]) 

        # Create and solve first stage problem
        p1 = Problem(Minimize(expectation(Q(*x), m)),
                     [sum(x) <= u])
        p1.solve()
        
        # print p1.value
        #
        # for i in range(C):
        #     print x[i].value
        #
        # for i in range(P):
        #     print y[i].value

        self.assert_feas(p1)

    def get_A(self, i, E, P):
        """
        Some notes:
        - A_jk == 1 <==> path k contains edge j.
        """
        A = numpy.matrix("1 1 1 0 0 0;" + 
                         "1 1 1 1 1 1;" +
                         "1 0 0 1 0 0;" +
                         "0 1 0 0 1 0;" +
                         "0 0 1 0 0 1;" +
                         "0 0 0 1 1 1")

        return A

    def test_dc_power(self):

        # Load problem data
        fp = "/Users/alnurali/shared_code/cvxpy/cvxpy/cvxpy/stochastic/pf_dc/pf_dc.mat"
        data = scipy.io.loadmat(fp)        

        A = data.get("A")
        n,E = A.shape

        gen_idxes = data.get("gen_idxes")
        wind_idxes = data.get("wind_idxes")
        load_idxes = data.get("load_idxes")

        num_gens = gen_idxes.size
        num_winds = wind_idxes.size
        num_loads = load_idxes.size
        
        gen_idxes = gen_idxes.reshape(num_gens) - 1 # "-1" to switch from Matlab to Python indexing
        wind_idxes = wind_idxes.reshape(num_winds) - 1    
        load_idxes = load_idxes.reshape(num_loads) - 1

        c_g = data.get("c_g")
        temp = c_g[0]
        c_g[0] = c_g[1]
        c_g[1] = temp
        
        c_w = data.get("c_w")

        p = data.get("p").reshape(n)
        p_orig = p
        
        u_lines = 1 # Note: depending on the choice of this (and the realizations of p_w below, the problem may or may not be feasible
        u_gens = 1

        m = 100 # Num samples

        # Create optimization variables
        p_g1, p_g2 = NonNegative(), NonNegative()
        z = NonNegative(num_winds)
        p_lines = Variable(E)
        p_w = RandomVariable(pymc.Lognormal(name="p_w", mu=1,
                                    tau=1, size=num_winds))

        # Create second stage problem
        p_g = vstack(p_g1, p_g2)        
        p = vstack(p_g1,
                   p_g2,
                   p[load_idxes[:-1]],
                   p_w-z,
                   p[load_idxes[-1]])
        p2 = Problem(Minimize(p_g.T*c_g + z.T*c_w),
                     [A*p_lines == p, p_g<=u_gens, z<=p_w,
                      abs(p_lines)<=u_lines])
        Q = partial_optimize(p2, [p_g2, z, p_lines], [p_g1])
        
        # Create and solve first stage problem
        p1 = Problem(Minimize(expectation(Q(p_g1), m)))
        p1.solve()

        ### Plot results ###
        B_binary = data.get("B_binary")
        
        coords = data.get("coords")
        coords[0,0] += 0.1 # Drawing was getting clipped in Python...

        # Draw edges between vertices
        fig = pyplot.figure()
        
        for i in range(n-1):
            for j in range(i+1,n):
                if B_binary[i,j] == 1:
                    pyplot.plot((coords[i,0], coords[j,0]), (coords[i,1], coords[j,1]), '-k')

        # Draw symbols and power generation/consumption for each vertex
        lognorm_mean = math.exp(1+pow(1,2)/2.0)

        fs = 16
        shift_x = 0
        shift_y = 0.125
        for i in range(n):
            
            if i in gen_idxes:
                pyplot.plot(coords[i,0], coords[i,1], color="crimson", marker="s", markersize=12)
                
                if i == 0:
                    pyplot.text(coords[i,0]+shift_x, coords[i,1]+shift_y, "{0:.2f}".format(p_g1.value), fontsize=fs)
                else:
                    pyplot.text(coords[i,0]+shift_x, coords[i,1]+shift_y, "sec. stg.", fontsize=fs)
                
            elif i in wind_idxes:
                pyplot.plot(coords[i,0], coords[i,1], color="blue", marker="s", markersize=12)
                pyplot.text(coords[i,0]+shift_x, coords[i,1]+shift_y, "{0:.2f}".format(lognorm_mean), fontsize=fs)
            
            else:
                pyplot.plot(coords[i,0], coords[i,1], 'ko')
                pyplot.text(coords[i,0]+shift_x, coords[i,1]+shift_y, "{0:.2f}".format(p_orig[i]), fontsize=fs)

        #pyplot.axis([0, 4, 0, 3])                
        pyplot.axis("off")
        # pyplot.show()
        fig.savefig("grid.png", bbox_inches="tight")
        
        # Check results
        # print p1.value
        # print p_g1.value
        self.assert_feas(p1)

    def test_web(self):

        if self.run_test_web is False:
            self.assertAlmostEqual(1,1)
            return
        
        # Create problem data
        n = 3 # Num dimensions of x        
        m = 20 # Num (x,y) train points
        m_pos = math.floor(m/2)
        m_neg = m - m_pos        
        
        q = 4 # Num dimensions of z        
        p = 30 # Num (z,w) train points
        p_pos = math.floor(p/2)
        p_neg = p - p_pos

        l = math.floor((n+q)/3)
        u = math.floor(2*(n+q)/3)

        C = 1 # L2 regularization trade-off parameter
        ns = 10 # Num samples

        # Create (x,y) data
        mu_pos = 0.5*numpy.ones(n)
        mu_neg = -0.5*numpy.ones(n)
        Sigma = numpy.eye(n)
        x = numpy.matrix(numpy.vstack((numpy.random.randn(m_pos,n)+mu_pos, numpy.random.randn(m_neg,n)+mu_neg)))        

        y = numpy.hstack((numpy.ones(m_pos), -1*numpy.ones(m_neg)))

        # Set up probabilistic model for (z,w) data
        z = self.get_z_data(p, p_pos, q)        
        w = numpy.hstack((numpy.ones(p_pos), -1*numpy.ones(p_neg)))        

        # Create optimization variables
        a, b = Variable(n), Variable()
        c, d = Variable(q), Variable()

        # Create second stage problem
        obj2 = [log_sum_exp(vstack(0, -w[i]*(c.T*z[i]+d)))
                for i in range(p)]
        budget = norm1(a) + norm1(c)
        p2 = Problem(Minimize(sum(obj2) + C*norm(c,2)),
                     [budget<=u])
        Q = partial_optimize(p2, [c,d], [a,b])

        # Create and solve first stage problem
        obj1 = [log_sum_exp(vstack(0, -y[i]*(x[i]*a+b)))
                for i in range(m)]
        p1 = Problem(Minimize(sum(obj1) + C*norm(a,2) +
                              expectation(Q(a,b), ns)), [])
        p1.solve()
        
        # print p1.value
        # print a.value
        # print b.value
        # print c.value
        # print d.value

        self.assert_feas(p1)

    """
    Alternatively, here is a simpler (Gaussian) model for the z variables:
    
    def get_z_data(self, p, p_pos, q):
    
        z = []
        for m in range(M):
        
            mu = None
            if m < p_pos:
                mu = 0.5*numpy.ones(q)                
            else:
                mu = -0.5*numpy.ones(q)
            
            Sigma = numpy.eye(q)
                
            rv = RandomVariableFactory().create_normal_rv(mu, Sigma)
            z += [rv]

        return z
    """
    def get_z_data(self, p, p_pos, q):

        K = 2 # Num topics        
        M = p # Num documents
        N = q # Total num of unique words across all documents
        
        alpha = 1.0 # Concentration parameter for distribution over distributions over words (one for each topic)
        beta = 1.0 # Concentration parameter for distribution over distributions over topics (one for each document)

        phi = pymc.Container([pymc.CompletedDirichlet(name="phi_" + str(k), D=pymc.Dirichlet(name="phi_temp_" + str(k), theta=beta*numpy.ones(N))) for k in range(K)])
        theta = pymc.Container([pymc.CompletedDirichlet(name="theta_" + str(m), D=pymc.Dirichlet(name="theta_temp_" + str(m), theta=alpha*numpy.ones(K))) for m in range(M)])
        z = pymc.Container([pymc.Categorical(name="z_" + str(m), p=theta[m], size=N) for m in range(M)])
        w = pymc.Container([pymc.Categorical(name="w_" + str(m) + "_" + str(n), p=pymc.Lambda("phi_z_" + str(m) + str(n), lambda z_in=z[m][n], phi_in=phi : phi_in[z_in])) for m in range(M) for n in range(N)])
        lda = pymc.Model([w, z, theta, phi])

        z_rvs = []
        for m in range(M):            
            metadata = {"doc_idx": m, "num_unique_words": N}            
            rv = WordCountVecRV(model=lda, name="w_0_0", metadata=metadata) # Note: w_0_0 is just a dummy argument that must be present in the pymc.Model            
            z_rvs += [rv]        
        return z_rvs        
    
    def test_multistage(self):
        
        # Create problem data
        b = 10
        s = 25
        r = 5

        d_probs = [0.3, 0.7]
        d_vals = [55, 141]
        d = RandomVariableFactory().create_categorical_rv(d_vals, d_probs)
        
        u = 150
    
        # Create optimization variables
        x = Variable()
        y1, y2 = Variable(), Variable()
        y3, y4 = Variable(), Variable()

        # Create third stage problem
        p3 = Problem(Minimize(-s*y3 - r*y4), [y3+y4<=x, 0<=y3, y3<=d, y4>=0])
        Q2 = partial_optimize(p3, [y3, y4], [x, y1, y2])

        # Create second stage problem
        p2 = Problem(Minimize(-s*y1 - r*y2 + expectation(Q2(x, y1, y2), want_de=True)), [y1+y2<=x, 0<=y1, y1<=d, y2>=0])
        Q1 = partial_optimize(p2, [y1, y2], [x])        

        # Create and solve first stage problem
        p1 = Problem(Minimize(b*x + expectation(Q1(x), want_de=True)), [0<=x, x<=u])
        p1.solve()
        # print p1.value, x.value, y1.value, y2.value, y3.value, y4.value

        self.assert_feas(p1)

    def test_scale(self):

        num_samples = 10 # Note: everything works fine with num_samples <= 100,000 and n <= 10
        n = 1
        
        mu = numpy.zeros(n)
        Sigma = numpy.eye(n)
        c = RandomVariableFactory().create_normal_rv(mu, Sigma)

        x = Variable(n)

        p = Problem(Minimize(expectation(x.T*c, num_samples)), [x >= -1, x <= 1])
        p.solve()
        # print p.status, p.value, x.value

        self.assert_feas(p)

if __name__ == "__main__":
    unittest.main()
