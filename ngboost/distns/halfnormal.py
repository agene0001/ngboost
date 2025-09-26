"""The NGBoost Half-Normal distribution and scores"""

import numpy as np
from scipy.stats import halfnorm as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
try:
    from numba import njit
    import math
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
if HAS_NUMBA:
    @njit(fastmath=True, cache=True)
    def halfnormal_logpdf_numba(y, scale):
        n = y.shape[0]
        out = np.empty(n)
        const = math.log(math.sqrt(2.0) / math.sqrt(math.pi))
        for i in range(n):
            out[i] = const - math.log(scale[i]) - 0.5 * (y[i] ** 2) / (scale[i] ** 2)
        return out

    @njit(fastmath=True, cache=True)
    def halfnormal_dscore_numba(y, scale):
        n = y.shape[0]
        out = np.empty((n, 1))
        for i in range(n):
            out[i, 0] = (scale[i] ** 2 - y[i] ** 2) / (scale[i] ** 2)
        return out

class HalfNormalLogScore(LogScore):
    def score(self, Y):
        if HAS_NUMBA:
            return -halfnormal_logpdf_numba(Y, self.scale)
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        if HAS_NUMBA:
            return halfnormal_dscore_numba(Y, self.scale)
        D = np.zeros((len(Y), 1))
        D[:, 0] = (self.scale**2 - Y**2) / self.scale**2
        return D

    def metric(self):
        FI = 2 * np.ones_like(self.scale)
        return FI[:, np.newaxis, np.newaxis]


class HalfNormal(RegressionDistn):
    """
    Implements the Half-Normal distribution for NGBoost.

    The Half-Normal distribution has one parameter, scale.
    The scipy loc parameter is held constant at zero for this implementation.
    LogScore is supported for the Half-Normal distribution.
    """

    n_params = 1
    scores = [HalfNormalLogScore]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self._params = params
        self.scale = np.exp(params[0])  # scale (sigma)
        self.dist = dist(loc=0, scale=self.scale)

    def fit(Y):
        _loc, scale = dist.fit(Y, floc=0)  # loc held constant
        return np.array([np.log(scale)])

    def sample(self, m):
        return self.dist.rvs(size=m)
    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"loc": np.zeros(shape=self.scale.shape), "scale": self.scale}
