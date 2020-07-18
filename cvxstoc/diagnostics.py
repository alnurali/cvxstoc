import sys  # TODO: temp

sys.path.insert(1, "/Users/alnurali/shared_code/cvxpy/cvxpy")  # TODO: temp
import cvxpy as cvx
import cvxpy.stochastic
from cvxpy import CVXOPT

import numpy as np
import math
import scipy.stats

import matplotlib.pyplot as plt

# from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class HowEstUnconstrOptVal:
    AVERAGE_SAAS = 1
    MEANFIELD = 2


class HowEstConstrOptVal:
    LAGRANGIAN = 1


class ConfInt:
    T = 1
    BOOT_TMP = 2


class HowGetCritVal:
    T = 1
    NORMAL = 2


class ConstrDiagnostics:
    def est_optval(
        self, how, obj, constrs
    ):  # Note: obj and constrs are expected to be *deterministic* functions

        if how == HowEstConstrOptVal.LAGRANGIAN:
            lagrangian = obj
            for i in range(len(constrs)):
                stoch_constr = constrs[i].lh_exp - constrs[i].rh_exp
                (m, n) = stoch_constr.size
                dual_var = 0.1 * np.ones(m)  # 0.1+np.random.rand(m)
                lagrangian += stoch_constr.T * dual_var
            prob = cvx.Problem(cvx.Minimize(lagrangian))
            prob.solve()
            return prob.value


class UnconstrDiagnostics:
    def est_optgap(self, how, stoch_objf, num_replications, num_samples, alpha):
        (ignore, fhatmean, fhatvar, ignore, ignore) = self.est_cost(
            how, stoch_objf, num_replications, alpha
        )
        (pstarhatmean, pstarhatvar, ignore, ignore) = self.est_optval(
            how, stoch_objf, num_replications, num_samples, alpha
        )
        return self.est_optgap_core(fhatmean, fhatvar, pstarhatmean, pstarhatvar, alpha)

    def est_optgap_core(self, fhatmean, fhatvar, pstarhatmean, pstarhatvar, alpha):
        gaphat = fhatmean - pstarhatmean
        pm = scipy.stats.norm.ppf(1.0 - (1.0 - alpha) / 2.0) * np.sqrt(
            fhatvar + pstarhatvar
        )
        return (gaphat, gaphat + pm, gaphat - pm)

    def est_cost(self, stoch_objf, num_replications, alpha):
        # Evaluate the SAA objective many times; compute the mean and variance
        fhats = [
            cvx.stochastic.expectation(stoch_objf, 1).value
            for i in range(num_replications)
        ]
        fhatmean = np.mean(fhats)

        # Compute confidence intervals
        fhatvar = np.var(fhats, ddof=1)

        fhatstderr = np.sqrt(1.0 / num_replications * fhatvar)
        fhatucb = fhatmean + UnconstrDiagnostics.get_critval(alpha) * fhatstderr
        fhatlcb = fhatmean - UnconstrDiagnostics.get_critval(alpha) * fhatstderr

        # Return
        return (fhatmean, fhatvar, fhatucb, fhatlcb)

    def est_optval(
        self,
        how_est_optval,
        stoch_objf,
        num_replications,
        num_samples,
        how_conf_int,
        alpha,
        A=None,
        b=None,
    ):

        if how_est_optval == HowEstUnconstrOptVal.AVERAGE_SAAS:

            if how_conf_int == ConfInt.T:

                pstarhats = []
                critval = UnconstrDiagnostics.get_critval(
                    alpha, HowGetCritVal.T, num_replications - 1
                )
                for i in range(num_replications):
                    prob = cvx.Problem(
                        cvx.Minimize(
                            cvx.stochastic.expectation(stoch_objf, num_samples)
                        )
                    )
                    prob.solve()
                    pstarhats.append(prob.value)

                pstarhatmean = np.mean(pstarhats)

                pstarhatvar = np.var(pstarhats, ddof=1)

                pstarhatstderr = np.sqrt(1.0 / num_replications * pstarhatvar)
                pstarhatucb = pstarhatmean + critval * pstarhatstderr
                pstarhatlcb = pstarhatmean - critval * pstarhatstderr

            else:  # Use bootstrap

                dims = A.shape
                (m, n) = dims.rows, dims.cols
                x = cvx.Variable(n)

                A_samples = A.sample(num_samples)
                b_samples = b.sample(num_samples)

                objf = 0
                for idx in range(num_samples):
                    objf += cvx.sum_squares(
                        A_samples[idx, :, :] * x - b_samples[idx, :]
                    )
                prob = cvx.Problem(cvx.Minimize((1.0 / num_samples) * objf))
                prob.solve()
                pstarhatmean = prob.value

                pstarhats = []
                for i in range(
                    num_replications
                ):  # Each iteration here is a bootstrap replication
                    idxes = np.random.randint(0, num_samples, num_samples)

                    objf = 0
                    for idx in idxes:
                        objf += cvx.sum_squares(
                            A_samples[idx, :, :] * x - b_samples[idx, :]
                        )

                    prob = cvx.Problem(cvx.Minimize((1.0 / num_samples) * objf))
                    prob.solve()

                    pstarhats.append(prob.value)

                pstarhatvar = np.var(pstarhats)

                pstarhatucb = np.percentile(pstarhats, 100.0 * (1.0 - alpha / 2.0))
                pstarhatlcb = np.percentile(pstarhats, 100.0 * (alpha / 2.0))

        else:  # Use mean field approximation (no conf ints)
            prob = cvx.Problem(
                cvx.Minimize(cvx.stochastic.expectation(stoch_objf, want_mf=True))
            )
            prob.solve()
            pstarhatmean = prob.value

            pstarhatvar, pstarhatucb, pstarhatlcb = None, None, None

        # Return
        return (pstarhatmean, pstarhatvar, pstarhatucb, pstarhatlcb)

    @staticmethod
    def get_critval(
        alpha, how=HowGetCritVal.NORMAL, df=None
    ):  # Note: alpha is (expected to be) small here
        if how == HowGetCritVal.T:
            return scipy.stats.t.ppf(1.0 - alpha / 2.0, df)
        else:  # Use inverse normal cdf
            return scipy.stats.norm.ppf(1.0 - alpha / 2.0)


class TestDiagnostics:
    def __init__(self):
        np.random.seed(1)

        self.unconstr_diag = UnconstrDiagnostics()
        self.constr_diag = ConstrDiagnostics()

        self.fontsize = 24  # 16 For plots
        self.fontsize_axis = 16
        self.linewidth = 3  # 2

    def test_least_squares_plot_saa_quality(
        self, A, b, P, q, r, num_samples_list, fn, num_replications, alpha
    ):
        dims = A.shape
        (m, n) = dims.rows, dims.cols

        # Solve the SAA
        pstarhat_avgs, pstarhat_ucbs, pstarhat_lcbs = [], [], []
        critval = UnconstrDiagnostics.get_critval(
            alpha, HowGetCritVal.T, num_replications - 1
        )
        for num_samples in num_samples_list:
            print "Solving a SAA to the stochastic least squares problem with [%d] Monte Carlo samples." % num_samples
            pstarhats = []
            for i in range(num_replications):
                x = cvx.Variable(n)
                prob = cvx.Problem(
                    cvx.Minimize(
                        cvx.stochastic.expectation(
                            cvx.sum_squares(A * x - b), num_samples
                        )
                    )
                )
                prob.solve()
                # print "Got objective value [%f]." % prob.value
                pstarhats.append(prob.value)

            pstarhat_avg = np.mean(pstarhats)
            pstarhat_avgs.append(pstarhat_avg)

            pstarhat_se = scipy.stats.sem(pstarhats)
            pstarhat_ucbs.append(pstarhat_avg + critval * pstarhat_se)
            pstarhat_lcbs.append(pstarhat_avg - critval * pstarhat_se)

        # Solve the true problem
        prob = cvx.Problem(cvx.Minimize(cvx.quad_form(x, P) - 2 * x.T * q + r))
        print "Solving the true stochastic least squares problem."
        prob.solve()
        # print "Got objective value [%f]." % prob.value
        pstars = np.tile(prob.value, len(num_samples_list))

        # Make plots
        print "Plotting results to [" + fn + "] (on a linear/log scale)."
        fig, ax = plt.subplots()
        plt.rc("text", usetex=True)
        # plt.rc('font', family='serif')
        ax.plot(
            num_samples_list,
            pstarhat_avgs,
            "-",
            label=r"$\hat{p}^*_N$",
            color="blue",
            lw=self.linewidth,
        )
        ax.fill_between(
            num_samples_list,
            pstarhat_ucbs,
            pstarhat_lcbs,
            facecolor="lightskyblue",
            alpha=0.5,
        )
        ax.plot(
            num_samples_list,
            pstars,
            "-",
            label=r"$p^*$",
            color="crimson",
            lw=self.linewidth,
        )

        ax.legend(loc="lower right", fontsize=self.fontsize)
        ax.set_xlabel(
            r"\# Monte Carlo samples $N$", fontsize=self.fontsize
        )  # number of Monte Carlo samples $N$
        # ax.xticks(num_samples_list)
        # plt.savefig(fn, bbox_inches="tight")
        ax.set_xscale("log")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(self.fontsize_axis)
        fig.savefig(fn, bbox_inches="tight")
        print "All done."

    def test_least_squares_plot_cost_ests(
        self, A, b, P, q, r, x, num_replications_list, alpha, fn
    ):
        # Estimate the true objective value
        stoch_objf = cvx.sum_squares(A * x - b)
        fhatmeans, fhatvars, fhatucbs, fhatlcbs = [], [], [], []
        for num_replications in num_replications_list:
            print "Averaging [%d] times the obj. vals. at a candidate point of SAAs with 1 Monte Carlo sample to the stochastic least squares objective." % num_replications
            (fhatmean, fhatvar, fhatucb, fhatlcb) = self.unconstr_diag.est_cost(
                stoch_objf, num_replications, alpha
            )
            fhatmeans.append(fhatmean)
            fhatvars.append(fhatvar)
            fhatucbs.append(fhatucb)
            fhatlcbs.append(fhatlcb)

        # Compute the true objective value
        print "Evaluating the true stochastic least squares objective value at a candidate point."
        fstar = (cvx.quad_form(x, P) - 2 * x.T * q + r).value
        fstars = np.tile(fstar, len(num_replications_list))

        # Make plots
        print "Plotting results to [" + fn + "] (on a linear/log scale)."
        fig, ax = plt.subplots()
        plt.rc("text", usetex=True)
        ax.plot(
            num_replications_list,
            fhatmeans,
            "-",
            label=r"$\hat{f}_{0,N}(x)$",
            color="blue",
            lw=self.linewidth,
        )
        ax.fill_between(
            num_replications_list,
            fhatucbs,
            fhatlcbs,
            facecolor="lightskyblue",
            alpha=0.5,
        )
        ax.plot(
            num_replications_list,
            fstars,
            "-",
            label=r"$f_0(x)$",
            color="crimson",
            lw=self.linewidth,
        )

        ax.legend(loc="upper right", fontsize=self.fontsize)
        ax.set_xlabel(r"\# samples $N$", fontsize=self.fontsize)
        ax.set_xscale("log")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(self.fontsize_axis)
        fig.savefig(fn, bbox_inches="tight")
        print "All done."

        # Return
        return (fhatmeans, fhatvars, fstars)

    def test_least_squares_plot_optval_ests(
        self,
        A,
        b,
        P,
        q,
        r,
        num_replications_list,
        num_samples_list,
        alpha,
        fn1,
        fn2,
        fn3,
        fn4,
        fn5,
        fhatmeans,
        fstars,
    ):
        # Compute the true optimal value
        dims = A.shape
        m, n = dims.rows, dims.cols

        x = cvx.Variable(n)
        prob = cvx.Problem(cvx.Minimize(cvx.quad_form(x, P) - 2 * x.T * q + r))
        print "Solving the true stochastic least squares problem."
        prob.solve()
        pstar = prob.value

        # Compute the true optimality gap
        gap = fstars[0] - pstar

        # Estimate the optimality gap (using Monte Carlo)
        X = len(num_replications_list)
        Y = len(num_samples_list)

        pstarhats = np.zeros((X, Y))
        pstarhat_ucbs = np.zeros((X, Y))
        pstarhat_lcbs = np.zeros((X, Y))

        gaphats = np.zeros((X, Y))

        errors = np.zeros((X, Y))
        for i in range(X):  # range(X-1,-1,-1):
            num_replications = num_replications_list[i]

            for j in range(Y):  # range(Y-1,-1,-1):
                num_samples = num_samples_list[j]

                # Estimate the true optimal value
                stoch_objf = cvx.sum_squares(A * x - b)
                print "Averaging the opt. vals. of [%d] SAAs with [%d] Monte Carlo samples to the stoch. least squares problem; requesting t-based conf. ints." % (
                    num_replications,
                    num_samples,
                )
                (
                    pstarhatmean,
                    pstarhatvar_ignore,
                    pstarhatucb,
                    pstarhatlcb,
                ) = self.unconstr_diag.est_optval(
                    HowEstUnconstrOptVal.AVERAGE_SAAS,
                    stoch_objf,
                    num_replications,
                    num_samples,
                    ConfInt.T,
                    alpha,
                )
                pstarhats[i, j] = pstarhatmean
                pstarhat_ucbs[i, j] = pstarhatucb
                pstarhat_lcbs[i, j] = pstarhatlcb

                gaphat = fhatmeans[i] - pstarhatmean
                gaphats[i, j] = gaphat

                error = gaphat - gap
                print "Error between a gap estimator with [%d] reps. and [%d] samples and the true gap is [%f]." % (
                    num_replications,
                    num_samples,
                    error,
                )
                errors[i, j] = error  # np.square( )

        # Estimate the true optimal value (using the mean field approx.)
        print "Computing the mean field approximation."
        (pstarhat_mf, ignore, ignore, ignore) = self.unconstr_diag.est_optval(
            HowEstUnconstrOptVal.MEANFIELD, stoch_objf, None, None, None, None
        )

        # Plot num. Monte Carlo samples vs. num. replications vs. gap error (i.e., gaphat - gap)
        print "Plotting results to [" + fn1 + "] (on a linear/log scale)."
        print "FYI, the errors are:"
        print errors

        fig = plt.figure()
        plt.rc("text", usetex=True)
        ax = fig.gca(projection="3d")
        (num_reps_list_meshed, num_samps_list_meshed) = np.meshgrid(
            num_replications_list, num_samples_list
        )
        surf = ax.plot_surface(
            num_reps_list_meshed,
            num_samps_list_meshed,
            errors,
            rstride=1,
            cstride=1,
            cmap=cm.autumn,
            linewidth=0,
            antialiased=False,
            alpha=0.75,
        )
        # cont = plt.contour(num_reps_list_meshed, num_samps_list_meshed, errors)

        ax.set_xlabel(r"\# reps.")
        ax.set_ylabel(r"\# samples")
        ax.set_zlabel(r"error")  # $\hat{\Delta}_{M,N} - \Delta$
        ax.xaxis.set_scale("log")
        ax.yaxis.set_scale("log")
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.savefig(fn1, bbox_inches="tight")
        print ""

        # Plot gaphat and gap vs. num. samples
        print "Plotting results to [" + fn2 + "] (on a linear/log scale)."
        print "FYI, the gaphats are:"
        print gaphats

        num_reps_fixed = 10
        num_samps_addl = 1000
        idx = 0

        stoch_objf = cvx.sum_squares(A * x - b)
        print "Averaging the opt. vals. of [%d] SAAs with [%d] Monte Carlo samples to the stoch. least squares problem; requesting t-based conf. ints." % (
            num_reps_fixed,
            num_samps_addl,
        )
        (
            pstarhatmean,
            pstarhatvar_ignore,
            pstarhatucb,
            pstarhatlcb,
        ) = self.unconstr_diag.est_optval(
            HowEstUnconstrOptVal.AVERAGE_SAAS,
            stoch_objf,
            num_reps_fixed,
            num_samps_addl,
            ConfInt.T,
            alpha,
        )

        gaphat_addl = fhatmeans[idx] - pstarhatmean
        print "Error between a gap estimator with [%d] reps. and [%d] samples and the true gap is [%f]." % (
            num_reps_fixed,
            num_samps_addl,
            gaphat_addl - gap,
        )

        gaphats_plot = np.append(gaphats[idx, :], gaphat_addl)
        num_samples_list.append(num_samps_addl)
        gaps = np.tile(gap, len(num_samples_list))

        fig, ax = plt.subplots()
        ax.plot(
            num_samples_list,
            gaphats_plot,
            "-",
            label=r"$\hat{\Delta}_{M,N}$",
            color="blue",
            lw=self.linewidth,
        )
        ax.plot(
            num_samples_list,
            gaps,
            "-",
            label=r"$\Delta$",
            color="crimson",
            lw=self.linewidth,
        )

        ax.legend(loc="upper right", fontsize=self.fontsize)
        ax.set_xlabel(
            r"\# samples $N$ (\# replications $M$ = %d fixed)" % num_reps_fixed,
            fontsize=self.fontsize,
        )
        ax.set_xscale("log")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(self.fontsize_axis)
        fig.savefig(fn2, bbox_inches="tight")
        print ""

        # Plot phatstar vs. num. samples
        print "Plotting results to [" + fn3 + "] (on a linear/log scale)."

        pstarhats_plot = np.append(pstarhats[idx, :], pstarhatmean)
        pstarhat_ucbs_plot = np.append(pstarhat_ucbs[idx, :], pstarhatucb)
        pstarhat_lcbs_plot = np.append(pstarhat_lcbs[idx, :], pstarhatlcb)

        pstars = np.tile(pstar, len(num_samples_list))

        pstarhats_mf = np.tile(pstarhat_mf, len(num_samples_list))

        fig, ax = plt.subplots()
        plt.rc("text", usetex=True)
        ax.plot(
            num_samples_list,
            pstarhats_plot,
            "-",
            label=r"$\hat{p}^*_{M,N}$",
            color="blue",
            lw=self.linewidth,
        )
        ax.fill_between(
            num_samples_list,
            pstarhat_ucbs_plot,
            pstarhat_lcbs_plot,
            facecolor="lightskyblue",
            alpha=0.5,
        )
        ax.plot(
            num_samples_list,
            pstarhats_mf,
            "-",
            label=r"$\hat{p}^*_{\mathrm{mf}}$",
            color="forestgreen",
            lw=self.linewidth,
        )
        ax.plot(
            num_samples_list,
            pstars,
            "-",
            label=r"$p^*$",
            color="crimson",
            lw=self.linewidth,
        )

        ax.legend(loc="lower left", fontsize=self.fontsize)
        ax.set_xlabel(
            r"\# samples $N$ (\# replications $M$ = %d fixed)" % num_reps_fixed,
            fontsize=self.fontsize,
        )
        ax.set_xscale("log")
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - 1, ymax)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(self.fontsize_axis)
        fig.savefig(fn3, bbox_inches="tight")
        print ""

        # Plot pstarhat vs. num. replications
        print "Plotting results to [" + fn4 + "] (on a linear/log scale)."

        num_reps_addl = 1000
        num_samps_fixed = 100
        idx = 1

        stoch_objf = cvx.sum_squares(A * x - b)
        print "Averaging the opt. vals. of [%d] SAAs with [%d] Monte Carlo samples to the stoch. least squares problem; requesting t-based conf. ints." % (
            num_reps_addl,
            num_samps_fixed,
        )
        (
            pstarhatmean,
            pstarhatvar_ignore,
            pstarhatucb,
            pstarhatlcb,
        ) = self.unconstr_diag.est_optval(
            HowEstUnconstrOptVal.AVERAGE_SAAS,
            stoch_objf,
            num_reps_addl,
            num_samps_fixed,
            ConfInt.T,
            alpha,
        )

        pstarhats_plot = np.append(pstarhats[:, idx], pstarhatmean)
        pstarhat_ucbs_plot = np.append(pstarhat_ucbs[:, idx], pstarhatucb)
        pstarhat_lcbs_plot = np.append(pstarhat_lcbs[:, idx], pstarhatlcb)

        num_replications_list.append(num_reps_addl)

        fig, ax = plt.subplots()
        plt.rc("text", usetex=True)

        ax.plot(
            num_replications_list,
            pstarhats_plot,
            "-",
            label=r"$\hat{p}^*_{M,N}$",
            color="blue",
            lw=self.linewidth,
        )
        ax.fill_between(
            num_replications_list,
            pstarhat_ucbs_plot,
            pstarhat_lcbs_plot,
            facecolor="lightskyblue",
            alpha=0.5,
        )

        # ax.plot(num_replications_list, pstarhatmeans_boot, "-", label=r"$\hat{p}^*_{M,N,\textrm{boot}}$", color="darkorchid", lw=self.linewidth)
        # ax.fill_between(num_replications_list, pstarhatucbs_boot, pstarhatlcbs_boot, facecolor="plum", alpha=0.25)

        ax.plot(
            num_replications_list,
            pstarhats_mf,
            "-",
            label=r"$\hat{p}^*_{\mathrm{mf}}$",
            color="forestgreen",
            lw=self.linewidth,
        )

        ax.plot(
            num_replications_list,
            pstars,
            "-",
            label=r"$p^*$",
            color="crimson",
            lw=self.linewidth,
        )

        ax.legend(loc="lower left", fontsize=self.fontsize)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - 1, ymax)
        ax.set_xlabel(
            r"\# replications $M$ (\# samples $N$ = %d fixed)" % num_samps_fixed,
            fontsize=self.fontsize,
        )
        ax.set_xscale("log")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(self.fontsize_axis)
        fig.savefig(fn4, bbox_inches="tight")
        print ""

        # Plot gaphat and gap vs. num. replications
        print "Plotting results to [" + fn5 + "] (on a linear/log scale)."

        num_samps_fixed = 100
        idx = 1
        gaphat_addl = fhatmeans[2] - pstarhatmean
        gaphats_plot = np.append(gaphats[:, idx], gaphat_addl)

        fig, ax = plt.subplots()
        ax.plot(
            num_replications_list,
            gaphats_plot,
            "-",
            label=r"$\hat{\Delta}_{M,N}$",
            color="blue",
            lw=self.linewidth,
        )
        ax.plot(
            num_replications_list,
            gaps,
            "-",
            label=r"$\Delta$",
            color="crimson",
            lw=self.linewidth,
        )

        ax.legend(loc="upper right", fontsize=self.fontsize)
        ax.set_xlabel(
            r"\# replications $M$ (\# samples $N$ = %d fixed)" % num_samps_fixed,
            fontsize=self.fontsize,
        )
        ax.set_xscale("log")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(self.fontsize_axis)
        fig.savefig(fn5, bbox_inches="tight")
        print "All done."

    def test_least_squares(self, m, n, want_plot_saa_quality=False):
        # Create problem data
        Aij_mu, Aij_cov = 1, 1
        bi_mu, bi_cov = -1, 1
        A = cvx.stochastic.RandomVariableFactory().create_normal_rv(
            Aij_mu, Aij_cov, (m, n)
        )
        b = cvx.stochastic.RandomVariableFactory().create_normal_rv(
            bi_mu, bi_cov, (m, 1)
        )

        P = np.tile(m * Aij_mu * Aij_mu, (n, n))
        np.fill_diagonal(P, m * (math.pow(Aij_mu, 2) + Aij_cov))
        q = np.tile(m * Aij_mu * bi_mu, n)
        r = m * (math.pow(bi_mu, 2) + bi_cov)

        alpha = 0.01

        # Plot the optimal value of the sample average approximation vs. num. samples used to construct the SAA
        if want_plot_saa_quality:
            num_samples_list = [
                1,
                10,
                100,
                1000,
            ]  # Cut (too costly): ,10000].  Not used for any other tests.
            num_replications = 10  # Not used for any other tests
            alpha = 0.05  # Not used for any other tests
            fn = "fig1.png"

            self.test_least_squares_plot_saa_quality(
                A, b, P, q, r, num_samples_list, fn, num_replications, alpha
            )
            print ""

            return

        # Plot an estimate of the true objective value at a point vs. num. replications used to construct the estimate
        num_replications_list = [10, 100, 1000]  # Cut (too costly): ,10000]
        fn = "fig2.png"

        x = cvx.Variable(n)  # Get a candidate solution x first
        num_samples = 1
        prob = cvx.Problem(
            cvx.Minimize(
                cvx.stochastic.expectation(cvx.sum_squares(A * x - b), num_samples)
            )
        )
        prob.solve()

        (fhatmeans, fhatvars, fstars) = self.test_least_squares_plot_cost_ests(
            A, b, P, q, r, x, num_replications_list, alpha, fn
        )
        print ""

        # Make lots of other plots
        num_replications_list = [10, 100]
        num_samples_list = [10, 100]
        fn1, fn2, fn3, fn4, fn5 = (
            "fig3.png",
            "fig4.png",
            "fig5.png",
            "fig6.png",
            "fig12.png",
        )

        self.test_least_squares_plot_optval_ests(
            A,
            b,
            P,
            q,
            r,
            num_replications_list,
            num_samples_list,
            alpha,
            fn1,
            fn2,
            fn3,
            fn4,
            fn5,
            fhatmeans,
            fstars,
        )  # Cut (no longer used): (pstarhatmeans, pstarhatvars, pstars) =
        print ""

    def test_QCQP_compute_fhat(self, q0, r0, x, num_replications, alpha):
        n = q0.size

        fhats = []
        for i in range(num_replications):
            A0 = np.random.randn(n, n)
            P0 = A0.T.dot(A0)
            fhat = (0.5 * cvx.quad_form(x, P0) + x.T.dot(q0) + r0).value
            fhats.append(fhat)

        fhat_avg = np.mean(fhats)

        fhat_se = scipy.stats.sem(fhats)
        critval = UnconstrDiagnostics.get_critval(alpha)
        fhat_ucb = fhat_avg + critval * fhat_se
        fhat_lcb = fhat_avg - critval * fhat_se

        return (fhat_avg, fhat_ucb, fhat_lcb)

    def test_QCQP_compute_pstarhat(
        self, q0, r0, q1, r1, x, num_replications, num_samples, alpha
    ):
        n = q0.size

        pstarhats = []
        for k in range(num_replications):
            obj, constr = 0, 0
            for l in range(num_samples):
                A0 = np.random.randn(n, n)
                P0 = A0.T.dot(A0)
                obj += (1.0 / num_samples) * (
                    0.5 * cvx.quad_form(x, P0) + x.T * q0 + r0
                )

                A1 = np.random.randn(n, n)
                P1 = A1.T.dot(A1)
                constr += (1.0 / num_samples) * (
                    0.5 * cvx.quad_form(x, P1) + x.T * q1 + r1
                )

            pstarhat = self.constr_diag.est_optval(
                HowEstConstrOptVal.LAGRANGIAN, obj, [constr <= 0]
            )
            pstarhats.append(pstarhat)

        pstarhat_avg = np.mean(pstarhats)

        pstarhat_se = scipy.stats.sem(pstarhats)
        critval = UnconstrDiagnostics.get_critval(
            alpha, HowGetCritVal.T, num_replications - 1
        )
        pstarhat_ucb = pstarhat_avg + critval * pstarhat_se
        pstarhat_lcb = pstarhat_avg - critval * pstarhat_se

        return (pstarhat_avg, pstarhat_ucb, pstarhat_lcb)

    def test_QCQP(
        self,
        n=5,
        num_samples_list=[10, 100],
        num_replications_list=[10, 100],
        alpha=0.05,
        fn1="fig9.png",
        fn2="fig10.png",
        fn3="fig11.png",
    ):  # Cut (too costly): ,1000
        # Create problem data
        np.random.seed(
            4
        )  # I had to play with the seed to get the QCQP constraint in the true problem below to be feasible (similar to http://web.cvxr.com/cvx/examples/cvxbook/Ch05_duality/html/qcqp.html)

        Aij_mu, Aij_cov = 0, 1
        q0 = np.random.randn(n)
        q1 = np.random.randn(n)
        r0 = np.random.randn()
        r1 = np.random.randn()

        # Compute the true optimal value
        P = np.tile(n * Aij_mu * Aij_mu, (n, n))
        np.fill_diagonal(P, n * (math.pow(Aij_mu, 2) + Aij_cov))

        x = cvx.Variable(n)
        prob = cvx.Problem(
            cvx.Minimize(0.5 * cvx.quad_form(x, P) + x.T * q0 + r0),
            [0.5 * cvx.quad_form(x, P) + x.T * q1 + r1 <= 0],
        )
        prob.solve()
        pstar = prob.value

        # Compute the true cost
        xtilde = cvx.Variable(n)
        prob = cvx.Problem(
            cvx.Minimize(0), [0.5 * cvx.quad_form(xtilde, P) + x.T * q1 + r1 <= 0]
        )  # Just get a feasible point
        prob.solve()

        xtilde = xtilde.value
        fval = (0.5 * cvx.quad_form(xtilde, P) + xtilde.T.dot(q0) + r0).value

        # Compute the true optimality gap
        gap = fval - pstar

        # Estimate the optimality gap
        X = len(num_replications_list)
        Y = len(num_samples_list)

        pstarhats = np.zeros((X, Y))
        pstarhat_ucbs = np.zeros((X, Y))
        pstarhat_lcbs = np.zeros((X, Y))

        fhats = np.zeros(X)

        gaphats = np.zeros((X, Y))

        errors = np.zeros((X, Y))
        for i in range(X):  # range(X-1,-1,-1):
            num_replications = num_replications_list[i]

            # Estimate the true cost
            (fhat, fhat_ucb, fhat_lcb) = self.test_QCQP_compute_fhat(
                q0, r0, xtilde, num_replications, alpha
            )
            fhats[i] = fhat

            for j in range(Y):  # range(Y-1,-1,-1):
                num_samples = num_samples_list[j]

                # Estimate the true optimal value
                print "Minimizing Lagrangian [%d] times with [%d] Monte Carlo samples." % (
                    num_replications,
                    num_samples,
                )
                (
                    pstarhat,
                    pstarhat_ucb,
                    pstarhat_lcb,
                ) = self.test_QCQP_compute_pstarhat(
                    q0, r0, q1, r1, x, num_replications, num_samples, alpha
                )
                pstarhats[i, j] = pstarhat
                pstarhat_ucbs[i, j] = pstarhat_ucb
                pstarhat_lcbs[i, j] = pstarhat_lcb

                gaphat = fhat - pstarhat
                gaphats[i, j] = gaphat

                error = gaphat - gap
                print "Error between a gap estimator with [%d] reps. and [%d] samples and the true gap is [%f]." % (
                    num_replications,
                    num_samples,
                    error,
                )
                errors[i, j] = error  # np.square( )

        # Plot num. Monte Carlo samples vs. num. replications vs. gap error (i.e., gaphat - gap)
        print "Plotting results to [" + fn1 + "] (on a linear/log scale)."
        print "FYI, the errors are:"
        print errors

        fig = plt.figure()
        plt.rc("text", usetex=True)
        ax = fig.gca(projection="3d")
        (num_reps_list_meshed, num_samps_list_meshed) = np.meshgrid(
            num_replications_list, num_samples_list
        )
        surf = ax.plot_surface(
            num_reps_list_meshed,
            num_samps_list_meshed,
            errors,
            rstride=1,
            cstride=1,
            cmap=cm.autumn,
            linewidth=0,
            antialiased=False,
            alpha=0.75,
        )
        # cont = plt.contour(num_reps_list_meshed, num_samps_list_meshed, errors)

        ax.set_xlabel(r"\# reps.")
        ax.set_ylabel(r"\# samples")
        ax.set_zlabel(r"error")  # $\hat{\Delta}_{M,N} - \Delta$
        ax.xaxis.set_scale("log")
        ax.yaxis.set_scale("log")
        # plt.gca().invert_yaxis()
        # plt.gca().invert_xaxis()
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.savefig(fn1, bbox_inches="tight")

        # Plot gaphat and gap vs. num. samples
        print "Plotting results to [" + fn2 + "] (on a linear/log scale)."
        print "FYI, the gaphats are:"
        print gaphats

        num_reps_fixed = 10  # Maybe unnecessary and too costly: 100
        num_samps_addl = 1000
        idx = 0

        print "Minimizing Lagrangian [%d] times with [%d] Monte Carlo samples." % (
            num_reps_fixed,
            num_samps_addl,
        )
        (
            pstarhat_addl,
            pstarhat_addl_ucb,
            pstarhat_addl_lcb,
        ) = self.test_QCQP_compute_pstarhat(
            q0, r0, q1, r1, x, num_reps_fixed, num_samps_addl, alpha
        )

        gaphat_addl = fhats[idx] - pstarhat_addl
        print "Error between a gap estimator with [%d] reps. and [%d] samples and the true gap is [%f]." % (
            num_reps_fixed,
            num_samps_addl,
            gaphat_addl - gap,
        )

        gaphats_plot = np.append(gaphats[idx, :], gaphat_addl)
        num_samples_list.append(num_samps_addl)
        gaps = np.tile(gap, len(num_samples_list))

        fig, ax = plt.subplots()
        ax.plot(
            num_samples_list,
            gaphats_plot,
            "-",
            label=r"$\hat{\Delta}_{M,N}$",
            color="blue",
            lw=self.linewidth,
        )
        ax.plot(
            num_samples_list,
            gaps,
            "-",
            label=r"$\Delta$",
            color="crimson",
            lw=self.linewidth,
        )

        ax.legend(loc="upper right", fontsize=self.fontsize)
        ax.set_xlabel(
            r"\# samples $N$ (\# replications $M$ = %d fixed)" % num_reps_fixed,
            fontsize=self.fontsize,
        )
        ax.set_xscale("log")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(self.fontsize_axis)
        fig.savefig(fn2, bbox_inches="tight")

        # Plot phatstar vs. num. samples
        print "Plotting results to [" + fn3 + "] (on a linear/log scale)."

        pstarhats_plot = np.append(pstarhats[idx, :], pstarhat_addl)
        pstarhat_ucbs_plot = np.append(pstarhat_ucbs[idx, :], pstarhat_addl_ucb)
        pstarhat_lcbs_plot = np.append(pstarhat_lcbs[idx, :], pstarhat_addl_lcb)

        pstars = np.tile(pstar, len(num_samples_list))

        fig, ax = plt.subplots()
        plt.rc("text", usetex=True)
        ax.plot(
            num_samples_list,
            pstarhats_plot,
            "-",
            label=r"$\hat{p}^*_{M,N}$",
            color="blue",
            lw=self.linewidth,
        )
        ax.fill_between(
            num_samples_list,
            pstarhat_ucbs_plot,
            pstarhat_lcbs_plot,
            facecolor="lightskyblue",
            alpha=0.5,
        )
        ax.plot(
            num_samples_list,
            pstars,
            "-",
            label=r"$p^*$",
            color="crimson",
            lw=self.linewidth,
        )

        ax.legend(loc="lower right", fontsize=self.fontsize)
        ax.set_xlabel(
            r"\# samples $N$ (\# replications $M$ = %d fixed)" % num_reps_fixed,
            fontsize=self.fontsize,
        )
        ax.set_xscale("log")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(self.fontsize_axis)
        fig.savefig(fn3, bbox_inches="tight")
        print "All done."

    def test_port_opt_plot_saa_quality(
        self,
        n=50,
        num_samples_list=[1, 10, 100, 1000],
        fn_objvals="fig7.png",
        fn_probs="fig8.png",
        num_replications=10,
        alpha=0.05,
    ):  # Cut (too costly): ,10000
        # Create problem data
        pbar = 10 + np.random.randn(n)
        A = np.random.rand(n, n)
        Sigma = 5 * A.T.dot(A)
        beta = 0.05  # Probability of portfolio loss (should be small)
        o = np.ones((n, 1))

        # Solve the SAA
        pstarhat_avgs, pstarhat_ucbs, pstarhat_lcbs = [], [], []
        prob_avgs, prob_ucbs, prob_lcbs = [], [], []
        critval = UnconstrDiagnostics.get_critval(
            alpha, HowGetCritVal.T, num_replications - 1
        )
        for num_samples in num_samples_list:
            print "Solving [%d] times a SAA to the stochastic port. opt. problem with [%d] Monte Carlo samples." % (
                num_replications,
                num_samples,
            )
            pstarhats = []
            probs = []
            for i in range(num_replications):
                x = cvx.Variable(n)
                p = cvx.stochastic.RandomVariableFactory().create_normal_rv(pbar, Sigma)
                cc = cvx.stochastic.prob(-x.T * p >= 0, num_samples)
                prob = cvx.Problem(
                    cvx.Maximize(x.T * pbar), [cc <= beta, x.T * o == 1, x >= -0.1]
                )
                prob.solve()  # solver=CVXOPT
                pstarhats.append(prob.value)
                probs.append(cc.get_prob())

            pstarhat_avg = np.mean(pstarhats)
            pstarhat_avgs.append(pstarhat_avg)

            pstarhat_se = scipy.stats.sem(pstarhats)
            pstarhat_ucbs.append(pstarhat_avg + critval * pstarhat_se)
            pstarhat_lcbs.append(pstarhat_avg - critval * pstarhat_se)

            prob_avg = np.mean(probs)
            prob_avgs.append(prob_avg)

            prob_se = scipy.stats.sem(probs)
            prob_ucbs.append(prob_avg + critval * prob_se)
            prob_lcbs.append(prob_avg - critval * prob_se)

        # Solve the true problem
        var = x.T * pbar >= scipy.stats.norm.ppf(1 - beta) * cvx.norm2(
            scipy.linalg.sqrtm(Sigma) * x
        )
        prob = cvx.Problem(cvx.Maximize(x.T * pbar), [var, x.T * o == 1, x >= -0.1])
        print "Solving the true stochastic port. opt. problem."
        prob.solve()
        pstars_var = np.tile(prob.value, len(num_samples_list))

        # Solve the true problem (with an exact CVaR constraint)
        cvar = x.T * pbar >= np.exp(
            -np.power(scipy.stats.norm.ppf(1 - beta), 2) / 2
        ) / (np.sqrt(2 * np.pi) * beta) * cvx.norm2(scipy.linalg.sqrtm(Sigma) * x)
        prob = cvx.Problem(cvx.Maximize(x.T * pbar), [cvar, x.T * o == 1, x >= -0.1])
        print "Solving the true stochastic port. opt. problem (with an exact CVaR constraint)."
        prob.solve()
        pstars_cvar = np.tile(prob.value, len(num_samples_list))

        # Make plots
        print "Plotting results to [" + fn_objvals + "] (on a linear/log scale)."
        fig, ax1 = plt.subplots()
        plt.rc("text", usetex=True)
        ax1.plot(
            num_samples_list,
            pstars_var,
            "-",
            label=r"$p^*$ (VaR)",
            color="forestgreen",
            lw=self.linewidth,
        )

        ax1.plot(
            num_samples_list,
            pstars_cvar,
            "-",
            label=r"$p^*$ (CVaR)",
            color="crimson",
            lw=self.linewidth,
        )

        ax1.plot(
            num_samples_list,
            pstarhat_avgs,
            "-",
            label=r"$\hat{p}^*_N$ (CVaR)",
            color="blue",
            lw=self.linewidth,
        )
        ax1.fill_between(
            num_samples_list,
            pstarhat_ucbs,
            pstarhat_lcbs,
            facecolor="lightskyblue",
            alpha=0.5,
        )
        # ax1.set_ylabel("objective value", fontsize=self.fontsize)

        ax1.legend(loc="lower left", fontsize=self.fontsize)

        ax1.set_xlabel(r"\# Monte Carlo samples $N$", fontsize=self.fontsize)
        ax1.set_xscale("log")

        # ax2 = ax1.twinx()

        for label in (
            ax1.get_xticklabels() + ax1.get_yticklabels()
        ):  # See http://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
            label.set_fontsize(self.fontsize_axis)
        fig.savefig(fn_objvals, bbox_inches="tight")

        print "Plotting results to [" + fn_probs + "] (on a linear/log scale)."
        fig, ax2 = plt.subplots()
        plt.rc("text", usetex=True)
        ax2.plot(
            num_samples_list,
            prob_avgs,
            "-",
            label=r"$\mathop{\bf E{}} (-x^T \bar{p} / \alpha + 1)_+$",
            color="blue",
            lw=self.linewidth,
        )
        ax2.fill_between(
            num_samples_list, prob_ucbs, prob_lcbs, facecolor="lightskyblue", alpha=0.5
        )
        ax2.plot(
            num_samples_list,
            np.tile(beta, len(num_samples_list)),
            "-",
            label=r"$\beta$",
            color="crimson",
            lw=self.linewidth,
        )

        # ax2.set_ylabel("probability", fontsize=self.fontsize)
        # ax2.set_ylim((0,beta+0.01))
        ax2.legend(loc="lower right", fontsize=self.fontsize)
        ax2.set_xlabel(r"\# Monte Carlo samples $N$", fontsize=self.fontsize)
        ax2.set_xscale("log")
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontsize(self.fontsize_axis)
        fig.savefig(fn_probs, bbox_inches="tight")
        print "All done."


if __name__ == "__main__":

    test_diag = TestDiagnostics()

    # Note: all the methods below work, I just commented some of them out so I could efficiently test them 1-by-1
    # test_diag.test_least_squares(m=100, n=50, want_plot_saa_quality=True)
    test_diag.test_least_squares(m=10, n=5)

    # test_diag.test_port_opt_plot_saa_quality()

    # test_diag.test_QCQP()
