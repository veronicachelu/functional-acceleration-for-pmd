import distrax
import numpy as np
import jax
import jax.numpy as jnp
from collections import defaultdict
import pylab as pl
import numpy as np

def categorical_kl_divergence(p_logits, q_logits):
    """
    Compute categorical KL divergence between two distributions.

    Args:
        p_logits: Logits of the first distribution.
        q_logits: Logits of the second distribution.

    Returns:
        Categorical KL divergence.
    """
    # Create distrax Softmax distributions from logits and compute KL divergence
    return distrax.Softmax(p_logits, 1.).kl_divergence(
        distrax.Softmax(q_logits, 1.))

def log_diff(p_logits, q_logits):
    """
    Compute element-wise difference of log probabilities between two distributions.

    Args:
        p_logits: Logits of the first distribution.
        q_logits: Logits of the second distribution.

    Returns:
        Element-wise difference of log probabilities.
    """
    # Compute log softmax probabilities for each distribution and subtract
    log_probs1 = jax.nn.log_softmax(p_logits, axis=-1)
    log_probs2 = jax.nn.log_softmax(q_logits, axis=-1)
    return log_probs1 - log_probs2

def entropy(p_logits):
    """
    Compute categorical KL divergence between two distributions.

    Args:
        p_logits: Logits of the first distribution.
        q_logits: Logits of the second distribution.

    Returns:
        Categorical KL divergence.
    """
    return distrax.Softmax(p_logits, 1.).entropy()

def add_uniform_noise(rng, q_t, tau_mean):
    """
    Add uniform noise to a distribution.

    Args:
        rng: Random number generator.
        q_t: Distribution to which noise is added.
        tau_mean: Mean of the uniform noise distribution.

    Returns:
        Distribution with added uniform noise.
    """
    # Generate uniform noise and add it to the distribution
    noise = rng.uniform(-tau_mean, tau_mean, q_t.shape)
    inexact_q_t = q_t + noise
    return inexact_q_t

def add_normal_noise(rng, q_t, tau_mean, tau_scale):
    """
    Add normal noise to a distribution.

    Args:
        rng: Random number generator.
        q_t: Distribution to which noise is added.
        tau_mean: Mean of the normal noise distribution.
        tau_scale: Scale of the normal noise distribution.

    Returns:
        Distribution with added normal noise.
    """
    # Generate normal noise and add it to the distribution
    noise = rng.normal(tau_mean, tau_scale, q_t.shape)
    inexact_q_t = q_t + noise
    return inexact_q_t


def load_imports_for_ipynb():
    import numpy as np
    import matplotlib.pyplot as plt
    from jupyterthemes import jtplot
    import importlib

    # Set Jupyter theme
    jtplot.style(theme='grade3', context='paper', ticks=True, grid=False)



    # Update matplotlib font settings
    plt.rcParams.update({'font.size': 12,  # Adjust font size as needed
                         "mathtext.fontset": 'cm'})  # Set font family for math text

    # If you are reloading modules, you may need to use importlib.reload()
    # importlib.reload(module_name)


def sample(w, rng, size=None, u=None):
    """
    Uses the inverse CDF method to return samples drawn from an (unnormalized)
    discrete distribution.
    """
    c = np.cumsum(w)
    if u is None:
        u = rng.uniform(0, 1, size=size)
    return c.searchsorted(u * c[-1])

def random_dist(*size):
    """
    Generate a random conditional distribution which sums to one over the last
    dimension of the input dimensions.
    """
    return np.random.dirichlet(np.ones(size[-1]), size=size[:-1])



class Halfspaces:
    "A x <= b"

    def __init__(self, A, b):
        self.A = np.array(A)
        self.b = np.array(b)
        [self.m, self.n] = self.A.shape
        assert self.b.shape == (self.m,), [A.shape, b.shape]

    def __call__(self, x):
        "Feasible?"
        return self.A @ x <= self.b

    def viz(self, xlim, ylim, ax=None, color='k', alpha=1., lw=1):
        if ax is None: ax = pl.gca()
        assert self.n == 2   # todo: support 1 and 3 dimensional cases.
        for i in range(self.m):
            # a x <= b
            a = self.A[i]
            b = self.b[i]

            # x0 will be the x-axis
            [p,q] = a

            # p*x + q*y <= b
            # y <= (b - p*x)/q

            # y <= (b - p*x)/q
            # -(y*q-b)/p <= x

            xs = np.linspace(*xlim, 2)
            ys = np.linspace(*ylim, 2)
            y2 = (b - p*xs)/q

            # Which side of the line do we fill?  The two-argument arc-tangent
            # function `arctan2(p, q)` gives the angle of the vector from
            # `<0,0>` to the point `<p,q>`.  The a positive angle tells us
            # whether to fill y values above, and a negative angle tells us to
            # fill the y values below.
            #
            # TODO: Handle the special case when the line is completely verical
            # (i.e., q=0)
            if q == 0:
                assert False, 'vertical lines not yet supported.'

            else:
                if np.arctan2(p, q) < 0:
                    y1 = ylim[1]*np.ones_like(xs)
                else:
                    y1 = ylim[0]*np.ones_like(xs)
                ax.plot(xs, y2, alpha=alpha, color=color, lw=lw, ls=":")
                # ax.fill_between(xs, y1, y2, alpha=1, color='grey')

        ax.set_ylim(*ylim)
        ax.set_xlim(*xlim)


