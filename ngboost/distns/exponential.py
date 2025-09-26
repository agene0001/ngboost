"""The NGBoost Exponential distribution and scores"""

import numpy as np
import scipy as sp
from scipy.stats import expon as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import CRPScore, LogScore

eps = 1e-10

try:
    from numba import njit
    import math
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:
    @njit(fastmath=True, cache=True)
    def exponential_logpdf_numba(y, scale):
        n = y.shape[0]
        out = np.empty(n)
        for i in range(n):
            if y[i] < 0:
                out[i] = -np.inf
            else:
                out[i] = -math.log(scale[i]) - y[i] / scale[i]
        return out

    @njit(fastmath=True, cache=True)
    def exponential_dscore_numba(event, time, scale):
        """
        d(-loglik)/d log(scale)
        event = 1 → uncensored, 0 → censored
        """
        n = time.shape[0]
        out = np.empty((n, 1))
        for i in range(n):
            if event[i] == 1:  # uncensored
                out[i, 0] = -( -1.0 + time[i] / scale[i] )
            else:  # censored
                out[i, 0] = -( time[i] / scale[i] )
        return out

class ExponentialLogScore(LogScore):
    def score(self, Y):
        E, T = Y["Event"], Y["Time"]
        if HAS_NUMBA:
            # uncensored: logpdf, censored: log survival
            logpdf_vals = exponential_logpdf_numba(T, self.scale)
            cens = (1 - E) * np.log(1 - np.exp(-T / self.scale) + eps)
            uncens = E * logpdf_vals
            return -(cens + uncens)
        else:
            cens = (1 - E) * np.log(1 - self.dist.cdf(T) + eps)
            uncens = E * self.dist.logpdf(T)
            return -(cens + uncens)

    def d_score(self, Y):
        E, T = Y["Event"], Y["Time"]
        if HAS_NUMBA:
            return exponential_dscore_numba(E, T, self.scale)
        else:
            cens = (1 - E) * T.squeeze() / self.scale
            uncens = E * (-1 + T.squeeze() / self.scale)
            return -(cens + uncens).reshape((-1, 1))

    def metric(self):
        FI = np.ones_like(self.scale)
        return FI[:, np.newaxis, np.newaxis]


class ExponentialCRPScore(CRPScore):
    def score(self, Y):
        E, T = Y["Event"], Y["Time"]
        score = T + self.scale * (2 * np.exp(-T / self.scale) - 1.5)
        score[E == 1] -= (
            0.5 * self.scale[E == 1] * np.exp(-2 * T[E == 1] / self.scale[E == 1])
        )
        return score

    def d_score(self, Y):
        E, T = Y["Event"], Y["Time"]
        deriv = 2 * np.exp(-T / self.scale) * (self.scale + T) - 1.5 * self.scale
        deriv[E == 1] -= np.exp(-2 * T[E == 1] / self.scale[E == 1]) * (
            0.5 * self.scale[E == 1] - T[E == 1]
        )
        return deriv.reshape((-1, 1))

    def metric(self):
        M = 0.5 * self.scale[:, np.newaxis, np.newaxis]
        return M


class Exponential(RegressionDistn):
    """
    Implements the exponential distribution for NGBoost.

    The exponential distribution has one parameters, scale. See scipy.stats.expon for details.
    This distribution has both LogScore and CRPScore implemented for it
    and both work with right-censored data
    """

    n_params = 1
    censored_scores = [ExponentialLogScore, ExponentialCRPScore]

    def __init__(self, params):  # pylint: disable=super-init-not-called
        self._params = params
        self.scale = np.exp(params[0])
        self.dist = dist(scale=self.scale)

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    # should implement a `sample()` method

    @property
    def params(self):
        return {"scale": self.scale}

    def fit(Y):
        m, s = sp.stats.expon.fit(Y)
        return np.array([np.log(m + s)])
