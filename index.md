---
layout: page
title: cvxstoc
---
<div class="nav">
<ul>
<li>&bull; <a href="#what-is-cvxstoc">Overview</a></li>
<li>&bull; <a href="#how-do-i-install-cvxstoc">Install</a></li>
<li>&bull; <a href="#what-can-i-do-with-cvxstoc">Tutorial</a></li>
<li>&bull; <a href="#where-can-i-learn-more">Questions?</a></li>
</ul>
</div>

# What is cvxstoc? <!-- <span style="font-family:Courier">cvxstoc</span> -->

cvxstoc is a Python package that makes it easy to code and solve [convex optimization problems](http://stanford.edu/class/ee364a/lectures/intro.pdf) that include random variables.

Here, we'll go over:

* how you can [install](#how-do-i-install-cvxstoc) cvxstoc
* what you can [do](#what-can-i-do-with-cvxstoc) with cvxstoc (including examples)
* where you can [learn](#where-can-i-learn-more) more

Prerequisites:

* you've taken a course in convex optimization
* you've seen some probability
* you've seen some [cvxpy](http://cvxpy.readthedocs.org/en/latest/) code

# How do I install cvxstoc?

On a Mac:

1. TODO
2. ...

# What can I do with cvxstoc?

Here, we walk through some basic scenarios that hint at what you can accomplish with cvxstoc; more advanced scenarios are detailed in the cvxstoc [paper](TODO).

## Random variables, expectations, and events
Suppose we're interested in a random variable $$\omega\sim \mathrm{Normal}(0, 1)$$ (i.e., $$\omega$$ is a standard normal random variable).  In cvxstoc, we can declare this random variable like so:

{% highlight python %}
from cvxstoc import NormalRandomVariable
omega = NormalRandomVariable(0, 1)
{% endhighlight %}

Now, intuitively, we might expect that as we repeatedly draw samples of this random variable and average them (i.e., we compute $$\omega$$'s *sample mean*), the sample mean will converge to 0.  In cvxstoc, we can compute the sample mean like this (repeating some of the previous code):

{% highlight python %}
from cvxstoc import NormalRandomVariable, expectation
omega = NormalRandomVariable(0, 1)
sample_mean = expectation(omega, 100)
{% endhighlight %}

Here, "100" specifies the number of samples to use when computing the sample mean --- we could have specified any (natural) number instead.  As might be expected, `sample_mean` stores the (scalar) sample mean.

Indeed, if we execute this code for various values of the number of samples to use and plot the resulting `sample_mean` values, we get (as expected):

![TODO](../TODO/TODO.png?raw=true) <!-- See http://webapps.stackexchange.com/questions/29602/markdown-to-insert-and-display-an-image-on-github-repo -->

Notice that the sample mean appears to lie above TODO once we use TODO samples --- thus, we also may be interested in computing the certainty (probability) that the sample mean lies above TODO, i.e.,

\begin{equation}
{\bf Prob}( TODO \leq \textrm{sample\_mean} ). \label{eq:event}
\end{equation}

In cvxstoc, we can accomplish this like so (by appealing to the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem), and also repeating some of the previous code):

{% highlight python %}
from cvxstoc import NormalRandomVariable, prob
omega = NormalRandomVariable(0, 1) # Not used, just here for reference
sample_mean = NormalRandomVariable(0, 1.0/100) # This is the samp. dist. of the sample mean
bound = prob(TODO <= sample_mean, 1000)
{% endhighlight %}

Here, `prob` takes in a (simple) inequality (or equality), repeatedly draws samples of the random variable in question (1000 samples, in this case), evaluates the inequality on the basis of the samples, averages the results (just as before), and (finally) returns a scalar; in other words

$$\mathrm{bound} = \frac{1}{1000} \sum_{i=1}^{1000} 1( TODO \leq \mathrm{sample\_mean}_i ),$$

where $$1(\cdot)$$ denotes the 0-1 indicator function and $$\mathrm{sample\_mean}_i$$ denotes the $$i$$th sample of the `sample_mean` random variable.

## Example: yield-constrained cost minimization

Let's combine all these ideas and code our first stochastic optimization problem using cvxstoc.

Suppose we'd like to choose the parameters $$x \in {\bf R}^n$$ governing a manufacturing process so that our cost $$c^T x$$, where $$c \in {\bf R}^n$$, is minimized, while the parameters lie in a set of allowable values $$S$$; we can model noise in the manufacturing process by expressing this constraint as

\begin{equation}
{\bf Prob}(x+\omega \in S) \geq \eta, \label{eq:chance}
\end{equation}

where $$\omega \in {\bf R}^n$$ is a random vector and $$\eta$$ is a large probability (e.g., 0.95).  Note that \eqref{eq:chance} is similar to \eqref{eq:event}; in general, \eqref{eq:chance} is referred to as a *[chance constraint](http://stanford.edu/class/ee364a/lectures/chance_constr.pdf)* (although, in this context, \eqref{eq:chance} is more often referred to as an *$$\eta$$-yield constraint*).  This leads us to the following optimization problem:

\begin{equation}
\begin{array}{ll}
\mbox{minimize} & c^T x \newline
\mbox{subject to} & {\bf Prob}(x+\omega \in S) \geq \eta
\end{array}
\label{eq:yield}
\end{equation}

with variable $$x$$.

We can directly express \eqref{eq:yield} using cvxstoc as follows ($$S$$ is taken to be an ellipsoid):

{% highlight python %}
from cvxstoc import NormalRandomVariable, prob
import numpy
from cvxpy.atoms import *
from cvxpy.expressions.variables import Variable
from cvxpy import Minimize, Problem

# Create problem data
n = 10
c = numpy.random.randn(n)
P, q, r = numpy.eye(n), numpy.random.randn(n), numpy.random.randn()
mu, Sigma = numpy.zeros(n), 0.1*numpy.eye(n)
omega = NormalRandomVariable(mu, Sigma)
m, eta = 100, 0.95

# Create and solve optimization problem
x = Variable(n)
yield_constr = prob(quad_form(x+omega,P) + (x+omega).T*q + r >= 0, m) <= 1-eta
p = Problem(Minimize(x.T*c), [yield_constr])
p.solve()
{% endhighlight %}

(Much of the syntax for creating and solving the optimization problem follows from [cvxpy](http://cvxpy.readthedocs.org/en/latest/).)

### Stochastic optimization problems

More generally, cvxstoc can handle *convex stochastic optimization problems*, i.e., optimization problems of the form

\begin{equation}
\begin{array}{ll}
\mbox{minimize} & \mathop{\bf E{}} f_0(x,\omega) \newline
\mbox{subject to} & \mathop{\bf E{}} f_i(x,\omega) \leq 0, \quad i=1,\ldots,m \newline
& h_i(x) = 0, \quad i=1,\ldots,p,
\end{array}
\label{eq:sp}
\end{equation}
where $$f_i : {\bf R}^n \times {\bf R}^q \rightarrow {\bf R}, \; i=0,\ldots,m$$ are convex functions in $$x$$ for each value of a random variable $$\omega \in {\bf R}^q$$, and $$h_i : {\bf R}^n \rightarrow {\bf R}, \; i=1,\ldots,p$$ are (deterministic) affine functions; since expectations preserve convexity, the objective and inequality constraint functions in \eqref{eq:sp} are (also) convex in $$x$$, making \eqref{eq:sp} a convex optimization problem.  (See Chaps. 3-4 of [Convex Optimization](http://web.stanford.edu/~boyd/cvxbook/) if you need more background.)

Note that cvxstoc handles manipulating problems (behind the scenes) into standard form (i.e., so you don't have to) and checking convexity by leveraging the [disciplined convex programming framework](http://dcp.stanford.edu/).

## Example: the news vendor problem

Let's now consider a different problem.

Suppose we sell newspapers.  We must decide how much newspaper to stock at the start of each day, so that our profit is maximized, while stocking and return fees (due to insufficient demand) at the end of the day are minimized, in the face of uncertain demand.  Thus, our optimization variables are the number of units of stocked newspaper $$x \in {\bf R}_+$$, the number of units purchased by customers $$y_1 \in {\bf R}_+$$, and the number of unpurchased (surplus) units that we must return $$y_2 \in {\bf R}_+$$.  Our problem data are $$c, s, r \in {\bf R}_{+}$$, which denote the price to stock, sell, and return a unit of newspaper, respectively.  Lastly, we let the random variable $$d \sim \mathrm{Categorical}$$ model the uncertain (newspaper) demand.  This leads us to the following convex stochastic optimization problem:

\begin{equation}
\begin{array}{ll}
	\mbox{minimize} & cx + \mathop{\bf E{}} Q(x) \newline
	\mbox{subject to} & 0 \leq x \leq u, %\notag
\end{array}
\label{eq:news1}
\end{equation}

\begin{equation}
\begin{array}{lll}
	\textrm{where } Q(x) \; = & \underset{y_1, y_2}{\min} & -(sy_1 + ry_2) \newline
	& \textrm{s.t. } & y_1 + y_2 \leq x \\ %\notag \newline
	& & 0 \leq y_1 \leq d \\ %\notag \newline
	& & y_2 \geq 0 %\notag
\end{array}
\label{eq:news2}
\end{equation}

with variable $$x$$.

Note that $$Q$$ is itself the optimal value of *another* convex optimization problem.  The problem in \eqref{eq:news1}-\eqref{eq:news2} is referred to as a *[two-stage stochastic optimization problem](https://en.wikipedia.org/wiki/Stochastic_programming#Two-Stage_Problems)*: \eqref{eq:news1} is referred to as the *first stage problem*, while \eqref{eq:news2} is referred to as the *second stage problem*.

A cvxstoc implementation of \eqref{eq:news1}-\eqref{eq:news2} is as follows:

{% highlight python %}
from cvxstoc import CategoricalRandomVariable, expectation, partial_optimize
import numpy
from cvxpy.atoms import *
from cvxpy.expressions.variables import Variable, NonNegative
from cvxpy import Minimize, Problem

# Create problem data
c, s, r, u = 10, 25, 5, 150
d_probs = [0.3, 0.6, 0.1]
d_vals = [55, 139, 141]
d = CategoricalRandomVariable(d_vals, d_probs)
    
# Create optimization variables
x = NonNegative()
y1, y2 = NonNegative(), NonNegative()

# Create second stage problem
obj = -s*y1 - r*y2
constrs = [y1+y2<=x, y1<=d]
p2 = Problem(Minimize(obj), constrs)
Q = partial_optimize(p2, [y1, y2], [x])

# Create and solve first stage problem
p1 = Problem(Minimize(c*x + expectation(Q(x))), [x<=u])
p1.solve()
{% endhighlight %}

Here, `partial_optimize` takes in a convex optimization `Problem` and a list of `Variable`'s to optimize over --- in this case, `partial_optimize` is given the (second stage) `Problem` named `p2` and told to optimize (only) over the (second stage) variables `y1` and `y2`.

# Where can I learn more?

* The cvxstoc [paper](TODO) contains much more (mathematical) detail as well as examples
* The cvxpy [mailing list](https://groups.google.com/forum/#!forum/cvxpy) is a great place to ask questions (regarding cvxpy as well as cvxstoc) --- please feel free to get in touch!
* Please feel free to grab the [source code](https://github.com/alnurali/cvxstoc) and contribute!