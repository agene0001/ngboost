"""The NGBoost Gamma distribution and scores"""

import numpy as np
import scipy as sp
from scipy.stats import gamma as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
try:
    from numba import njit
    from numba import float64
    from numba import prange
    import math
    from numba.special import digamma, polygamma
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:
    @njit(fastmath=True, cache=True)
    def gamma_logpdf_numba(y, alpha, beta, eps=1e-10):
        """
        Log PDF for Gamma(alpha, beta), parameterized with shape alpha and rate beta.
        scale = 1 / beta
        """
        n = y.shape[0]
        out = np.empty(n)
        for i in range(n):
            if y[i] <= 0.0:
                out[i] = -np.inf
            else:
                out[i] = (
                        alpha[i] * math.log(beta[i])
                        - math.lgamma(alpha[i])
                        + (alpha[i] - 1.0) * math.log(y[i] + eps)
                        - beta[i] * y[i]
                )
        return out

    @njit(fastmath=True, cache=True)
    def gamma_dscore_numba(y, alpha, beta, eps=1e-10):
        """
        Derivatives of -log(PDF) wrt alpha and beta.
        """
        n = y.shape[0]
        out = np.empty((n, 2))
        for i in range(n):
            out[i, 0] = alpha[i] * (digamma(alpha[i]) - math.log(beta[i] * y[i] + eps))
            out[i, 1] = (beta[i] * y[i]) - alpha[i]
        return out

class GammaLogScore(LogScore):
    def score(self, Y):
        if HAS_NUMBA:
            return -gamma_logpdf_numba(Y, self.alpha, self.beta)
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        if HAS_NUMBA:
            return gamma_dscore_numba(Y, self.alpha, self.beta)
        D = np.zeros((len(Y), 2))
        D[:, 0] = self.alpha * (
                sp.special.digamma(self.alpha) - np.log(self.eps + self.beta * Y)
        )
        D[:, 1] = (self.beta * Y) - self.alpha
        return D

    def metric(self):
        FI = np.zeros((self.alpha.shape[0], 2, 2))
        FI[:, 0, 0] = self.alpha**2 * sp.special.polygamma(1, self.alpha)
        FI[:, 1, 1] = self.alpha
        FI[:, 0, 1] = -self.alpha
        FI[:, 1, 0] = -self.alpha
        return FI


class Gamma(RegressionDistn):
    n_params = 2
    scores = [GammaLogScore]

    def __init__(self, params):
        super().__init__(params)
        self.alpha = np.exp(params[0])
        self.beta = np.exp(params[1])
        self.dist = dist(
            a=self.alpha, loc=np.zeros_like(self.alpha), scale=1 / self.beta
        )
        self.eps = 1e-10

    def fit(Y):
        a, _, scale = dist.fit(Y, floc=0)
        return np.array([np.log(a), np.log(1 / scale)])

    def sample(self, m):
        return self.dist.rvs(size=m)
    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"alpha": self.alpha, "beta": self.beta}
