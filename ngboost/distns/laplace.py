"""The NGBoost Laplace distribution and scores"""
import numpy as np
from scipy.stats import laplace as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import CRPScore, LogScore
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ----------------------------
# Numba implementations
# ----------------------------
if HAS_NUMBA:
    import math

    @njit(fastmath=True, cache=True)
    def laplace_logpdf_numba(y, loc, scale):
        n = y.shape[0]
        out = np.empty(n)
        inv_scale = 1.0 / scale
        for i in range(n):
            diff = abs(y[i] - loc[i])
            out[i] = -math.log(2.0 * scale[i]) - diff * inv_scale[i]
        return out

    @njit(fastmath=True, cache=True)
    def laplace_dscore_log_numba(y, loc, scale):
        n = y.shape[0]
        out = np.empty((n, 2))
        inv_scale = 1.0 / scale
        for i in range(n):
            diff = loc[i] - y[i]
            abs_diff = abs(diff)
            sign_term = 0.0 if diff == 0.0 else (1.0 if diff > 0 else -1.0)
            out[i, 0] = sign_term * inv_scale[i]          # d/dloc
            out[i, 1] = 1.0 - abs_diff * inv_scale[i]     # d/dscale
        return out

    @njit(fastmath=True, cache=True)
    def laplace_score_crps_numba(y, loc, scale):
        n = y.shape[0]
        out = np.empty(n)
        for i in range(n):
            diff = abs(y[i] - loc[i])
            out[i] = diff + math.exp(-diff / scale[i]) * scale[i] - 0.75 * scale[i]
        return out

    @njit(fastmath=True, cache=True)
    def laplace_dscore_crps_numba(y, loc, scale):
        n = y.shape[0]
        out = np.empty((n, 2))
        for i in range(n):
            diff = y[i] - loc[i]
            abs_diff = abs(diff)
            sign_term = 0.0 if diff == 0.0 else (-1.0 if diff > 0 else 1.0)
            exp_term = math.exp(-abs_diff / scale[i])
            # d/dloc
            out[i, 0] = sign_term * (1.0 - exp_term)
            # d/dscale
            out[i, 1] = exp_term * (scale[i] + abs_diff)
        return out


class LaplaceLogScore(LogScore):
    def score(self, Y):
        if HAS_NUMBA:
            return -laplace_logpdf_numba(Y, self.loc, self.scale)
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        if HAS_NUMBA:
            return laplace_dscore_log_numba(Y, self.loc, self.scale)
        D = np.zeros((len(Y), 2))
        D[:, 0] = np.sign(self.loc - Y) / self.scale
        D[:, 1] = 1 - np.abs(self.loc - Y) / self.scale
        return D

    def metric(self):
        FI = np.zeros((self.loc.shape[0], 2, 2))
        FI[:, 0, 0] = 1 / self.scale**2
        FI[:, 1, 1] = 1
        return FI


class LaplaceCRPScore(CRPScore):
    def score(self, Y):
        if HAS_NUMBA:
            return laplace_score_crps_numba(Y, self.loc, self.scale)
        return (
                np.abs(Y - self.loc)
                + np.exp(-np.abs(Y - self.loc) / self.scale) * self.scale
                - 0.75 * self.scale
        )

    def d_score(self, Y):
        if HAS_NUMBA:
            return laplace_dscore_crps_numba(Y, self.loc, self.scale)
        D = np.zeros((len(Y), 2))
        D[:, 0] = np.sign(self.loc - Y) * (
                1 - np.exp(-np.abs(Y - self.loc) / self.scale)
        )
        D[:, 1] = np.exp(-np.abs(Y - self.loc) / self.scale) * (
                self.scale + np.abs(Y - self.loc)
        )
        return D

    def metric(self):
        FI = np.zeros((self.loc.shape[0], 2, 2))
        FI[:, 0, 0] = 0.5 / self.scale
        FI[:, 1, 1] = 0.25 * self.scale
        return FI


class Laplace(RegressionDistn):

    n_params = 2
    scores = [LaplaceLogScore, LaplaceCRPScore]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self._params = params
        self.loc = params[0]
        self.logscale = params[1]
        self.scale = np.exp(params[1])
        self.dist = dist(loc=self.loc, scale=self.scale)

    def fit(Y):
        m, s = dist.fit(Y)
        return np.array([m, np.log(s)])

    def sample(self, m):
        return self.dist.rvs(size=m)
    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale}
