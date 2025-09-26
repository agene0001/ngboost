"""The NGBoost Weibull distribution and scores"""
import numpy as np
from scipy.stats import weibull_min as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
_EULER_GAMMA = 0.5772156649015328606
_FI00_CONST = (np.pi**2 / 6.0) + (1.0 - _EULER_GAMMA) ** 2
if HAS_NUMBA:
    import math

    @njit(fastmath=True, cache=True)
    def weibull_logpdf_numba(y, shape, scale):
        n = y.shape[0]
        out = np.empty(n)
        for i in range(n):
            z = y[i] / scale[i]
            out[i] = (
                    math.log(shape[i])
                    - shape[i] * math.log(scale[i])
                    + (shape[i] - 1.0) * math.log(z)
                    - (z ** shape[i])
            )
        return out

    @njit(fastmath=True, cache=True)
    def weibull_dscore_numba(y, shape, scale):
        n = y.shape[0]
        out = np.empty((n, 2))
        for i in range(n):
            z = y[i] / scale[i]
            z_shape = z ** shape[i]
            shared_term = shape[i] * (z_shape - 1.0)

            out[i, 0] = shared_term * math.log(z) - 1.0  # d/d(shape)
            out[i, 1] = -shared_term                      # d/d(scale)
        return out
class WeibullLogScore(LogScore):
    def score(self, Y):
        if HAS_NUMBA:
            return -weibull_logpdf_numba(Y, self.shape, self.scale)
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        if HAS_NUMBA:
            return weibull_dscore_numba(Y, self.shape, self.scale)
        D = np.zeros((len(Y), 2))
        z = Y / self.scale
        z_shape = z ** self.shape
        shared_term = self.shape * (z_shape - 1)
        D[:, 0] = shared_term * np.log(z) - 1
        D[:, 1] = -shared_term
        return D

    def metric(self):
        n = self.scale.shape[0]
        FI = np.zeros((n, 2, 2))
        FI[:, 0, 0] = _FI00_CONST
        cross = -self.shape * (1.0 - _EULER_GAMMA)
        FI[:, 0, 1] = cross
        FI[:, 1, 0] = cross
        FI[:, 1, 1] = self.shape**2
        return FI


class Weibull(RegressionDistn):
    """
    Implements the Weibull distribution for NGBoost.

    The Weibull distribution has two parameters, shape and scale.
    The scipy loc parameter is held constant for this implementation.
    LogScore is supported for the Weibull distribution.
    """

    n_params = 2
    scores = [WeibullLogScore]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self._params = params
        self.shape, self.scale = np.exp(params[:2])
        self.dist = dist(c=self.shape, loc=0, scale=self.scale)

    def fit(Y):
        shape, _loc, scale = dist.fit(Y, floc=0)  # hold loc constant
        return np.array([np.log(shape), np.log(scale)])

    def sample(self, m):
        return self.dist.rvs(size=m)

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"shape": self.shape, "scale": self.scale}
