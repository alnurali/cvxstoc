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

* How you can [install](#how-do-i-install-cvxstoc) cvxstoc.
* What you can [do](#what-can-i-do-with-cvxstoc) with cvxstoc (including examples).
* Where you can [learn](#where-can-i-learn-more) more.

Prerequisites:

* You've taken [a course in convex optimization](https://youtu.be/McLq1hEq3UY?list=PL3940DD956CDF0622).
* You've seen some probability and statistics.
* You've seen some [cvxpy](http://cvxpy.readthedocs.org/en/latest/) code.

# How do I install cvxstoc?

On a Mac:

1. Follow the [instructions](http://www.cvxpy.org/en/latest/install/) for installing cvxpy.
2. ...

# What can I do with cvxstoc?

Here, we walk through some basic scenarios that hint at what you can accomplish with cvxstoc; more advanced scenarios are detailed in the cvxstoc [paper](http://stanford.edu/~boyd/papers/dcsp.html).

## Random variables, expectations, and events
Suppose we're interested in a random variable $$\omega\sim \mathrm{Normal}(0, 1)$$ (i.e., $$\omega$$ is a standard normal random variable).  In cvxstoc, we can declare this random variable like so:

{% highlight python %}
from cvxstoc import NormalRandomVariable
omega = NormalRandomVariable(0, 1)
{% endhighlight %}

Now, intuitively, we might [expect](https://en.wikipedia.org/wiki/Law_of_large_numbers#Weak_law) that as we average more and more samples of this random variable (i.e., we compute $$\omega$$'s sample mean on larger and larger samples), the sample mean will converge to zero.  In cvxstoc, we can compute the sample mean like this (repeating some of the previous code):

{% highlight python %}
import cvxpy as cvx
from cvxstoc import NormalRandomVariable, expectation
omega = NormalRandomVariable(0, 1)
sample_mean = expectation(omega, 100).value
{% endhighlight %}

Here, "100" specifies the number of samples to use when computing the sample mean --- we could have specified any (natural) number instead.  As might be expected, `sample_mean` stores the (scalar) sample mean.

Indeed, if we execute this code for various values of the number of samples to use and plot the resulting `sample_mean` values, we get (as expected):

<div style="text-align:center" markdown="1">
![fig.png](fig.png)
</div>

We can see that the sample mean occasionally jumps above zero --- so, one thing we might be curious about is the probability that the sample mean (as a function of the number of samples) indeed lies above zero, i.e.,

\begin{equation}
{\bf Prob}( 0 \leq \textrm{sample\_mean} ). \label{eq:event}
\end{equation}

In cvxstoc, we can compute this like so (by appealing to the [Central Limit Theorem](http://www.stat.cmu.edu/~larry/=stat705/Lecture4.pdf) as well as repeating some of the previous code):

{% highlight python %}
from cvxstoc import NormalRandomVariable, expectation, prob
omega = NormalRandomVariable(0, 1)             # Not used, just here for reference
sample_mean = NormalRandomVariable(0, 1.0/100) # This is the samp. dist. of the sample mean
bound = prob(0 <= sample_mean, 1000).value
print bound                                    # Get something close to 0.5, as expected
{% endhighlight %}

Here, `prob` takes in a (convex) inequality (or equality), draws samples of the random variable in question (1000 samples, in this case), evaluates the inequality on the basis of the samples, averages the results (just as in our experiments with the expected value from earlier), and (finally) returns a scalar; in other words

$$\mathrm{bound} = \frac{1}{1000} \sum_{i=1}^{1000} 1( 0 \leq \mathrm{sample\_mean}_i ),$$

where $$1(\cdot)$$ denotes the zero/one indicator function (one if its argument is true, zero otherwise) and $$\mathrm{sample\_mean}_i$$ denotes the $$i$$th sample of the `sample_mean` random variable.

## Example: yield-constrained cost minimization

Let's combine all these ideas and code our first stochastic optimization problem using cvxstoc.

Suppose we run a factory that makes toys from $$n$$ different kinds of raw materials.  We would like to decide how much of each different kind of raw material to order, which we model as a vector $$x \in {\bf R}^n$$, so that our ordering cost $$c^T x$$, where $$c \in {\bf R}^n$$ is a constant, is minimized, so long as we don't exceed our factory's ordering budget, i.e., $$x$$ lies in some set of allowable values $$S$$.  Suppose further that our ordering process is error-prone: We may receive raw materials $$x + \omega$$, where $$\omega \in {\bf R}^n$$ is some random vector, even if we place an order for just $$x$$; thus, we can express our wish to not exceed our factory's budget as

\begin{equation}
{\bf Prob}(x+\omega \in S) \geq \eta, \label{eq:chance}
\end{equation}

where $$\eta$$ is a large probability (e.g., 0.95).  Note that \eqref{eq:chance} is similar to \eqref{eq:event}; in general, \eqref{eq:chance} is referred to as a *[chance constraint](http://stanford.edu/class/ee364a/lectures/chance_constr.pdf)* (although, in this context, \eqref{eq:chance} is more often referred to as an *$$\eta$$-yield constraint*).  This leads us to the following optimization problem:

\begin{equation}
\begin{array}{ll}
\mbox{minimize} & c^T x \newline
\mbox{subject to} & {\bf Prob}(x+\omega \in S) \geq \eta
\end{array}
\label{eq:yield}
\end{equation}

with variable $$x$$.

We can directly express \eqref{eq:yield} using cvxstoc as follows (let's take $$S$$ to be an [ellipsoid](http://ee263.stanford.edu/lectures/ellipsoids.pdf) for simplicity):

{% highlight python %}
from cvxstoc import NormalRandomVariable, prob
import numpy
from cvxpy.atoms import *
from cvxpy.expressions.variables import Variable
from cvxpy import Minimize, Problem

# Create problem data.
n = 10
c = numpy.random.randn(n)
P, q, r = numpy.eye(n), numpy.random.randn(n), numpy.random.randn()
mu, Sigma = numpy.zeros(n), 0.1*numpy.eye(n)
omega = NormalRandomVariable(mu, Sigma)
m, eta = 100, 0.95

# Create and solve stochastic optimization problem.
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
where $$f_i : {\bf R}^n \times {\bf R}^q \rightarrow {\bf R}, \; i=0,\ldots,m$$ are convex functions in $$x$$ for each value of a random variable $$\omega \in {\bf R}^q$$, and $$h_i : {\bf R}^n \rightarrow {\bf R}, \; i=1,\ldots,p$$ are (deterministic) affine functions; since expectations preserve convexity, the objective and inequality constraint functions in \eqref{eq:sp} are (also) convex in $$x$$, making \eqref{eq:sp} a convex optimization problem.  (See Chaps. 3-4 of [Convex Optimization](http://web.stanford.edu/~boyd/cvxbook/) if you need more background here.)

(Note that cvxstoc handles manipulating problems behind the scenes into standard form so you don't have to, and also checks convexity by leveraging the [disciplined convex programming framework](http://dcp.stanford.edu/).)

## Example: the news vendor problem

Let's now consider a different problem.

Suppose we sell newspapers.  Each morning, we must decide how many newspapers $$x \in [0,b]$$ to acquire, where $$b$$ is our budget (e.g., the maximum number of newspapers we can store); later in the day, we will sell $$y \in {\bf R}_+$$ of these newspapers at a (fixed) price of $$s \in {\bf R}_+$$ dollars per newspaper, and return the rest ($$z \in {\bf R}_+$$) to our supplier at a (fixed) income of $$r \in {\bf R}_+$$, in a proportion that is dictated by the amount of (uncertain) demand $$d \sim \mathrm{Categorical}$$.  We can model our decision-making process by the following optimization problem:
\begin{equation}
\begin{array}{ll}
	\mbox{minimize} & cx + \mathop{\bf E{}} Q(x,d) \newline
	\mbox{subject to} & 0 \leq x \leq b, %\notag
\end{array}
\label{eq:news1}
\end{equation}

\begin{equation}
\begin{array}{lll}
	\textrm{where } Q(x,d) \; = & \underset{y, z}{\min} & -(sy + rz) \newline
	& \textrm{s.t. } & 0 \leq y + z \leq x \\ %\notag \newline
	& & 0 \leq y, z \leq d %\notag
\end{array}
\label{eq:news2}
\end{equation}

with variable $$x$$.

Note that $$Q$$ here is itself the optimal value of *another* convex optimization problem.  The problem in \eqref{eq:news1}-\eqref{eq:news2} is referred to as a *[two-stage stochastic optimization problem](https://en.wikipedia.org/wiki/Stochastic_programming#Two-Stage_Problems)*: \eqref{eq:news1} is referred to as the *first stage problem*, while \eqref{eq:news2} is referred to as the *second stage problem*.

A cvxstoc implementation of \eqref{eq:news1}-\eqref{eq:news2} is as follows:

{% highlight python %}
import cvxpy as cvx
from cvxpy import Minimize, Problem
from cvxpy.expressions.variables import Variable, NonNegative
from cvxpy.transforms import partial_optimize
from cvxstoc import expectation, prob, CategoricalRandomVariable
import numpy

# Create problem data.
c, s, r, b = 10, 25, 5, 150
d_probs = [0.5, 0.5] # [0.3, 0.6, 0.1]
d_vals = [55, 139] # [55, 139, 141]
d = CategoricalRandomVariable(d_vals, d_probs)

# Create optimization variables.
x = NonNegative()
y, z = NonNegative(), NonNegative()

# Create second stage problem.
obj = -s*y - r*z
constrs = [y+z<=x, y<=d, z<=d]
p2 = Problem(Minimize(obj), constrs)
Q = partial_optimize(p2, [y, z], [x])

# Create and solve first stage problem.
p1 = Problem(Minimize(c*x + expectation(Q, num_samples=100)), [x<=b])
p1.solve()
{% endhighlight %}

Here, `partial_optimize` takes in a convex optimization `Problem` and a list of `Variable`'s to optimize over --- in this case, `partial_optimize` is given the (second stage) `Problem` named `p2` and told to optimize (only) over the (second stage) variables `y` and `z`.

# Where can I learn more?

* The cvxstoc [paper](http://stanford.edu/~boyd/papers/dcsp.html) contains much more (mathematical) detail as well as examples.
* The cvxpy [mailing list](https://groups.google.com/forum/#!forum/cvxpy) is a great place to ask questions (regarding cvxpy as well as cvxstoc) --- please feel free to get in touch!
* Please feel free to grab the [source code](https://github.com/alnurali/cvxstoc) and contribute!
