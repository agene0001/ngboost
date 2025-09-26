"""The NGBoost Normal distribution and scores"""
import math

import numpy as np
import scipy as sp
from scipy.stats import norm as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import CRPScore, LogScore

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit(fastmath=True, cache=True)
    def normal_logpdf_numba(y, loc, scale):
        # Ensure inputs are 1D
        y = y.ravel()
        loc = loc.ravel()
        scale = scale.ravel()

        n = y.shape[0]
        out = np.empty(n)
        inv_var = 1.0 / (scale * scale)
        log_norm = -0.5 * np.log(2.0 * np.pi) - np.log(scale)

        for i in range(n):
            diff = y[i] - loc[i]
            out[i] = log_norm[i] - 0.5 * diff * diff * inv_var[i]
        return out

    @njit(fastmath=True, cache=True)
    def normal_pdf_numba(z):
        return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

    @njit(cache=True)
    def normal_cdf_numba(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    @njit(fastmath=True, cache=True)
    def normal_crps_numba(y, loc, scale):
        # Ensure inputs are 1D
        y = y.ravel()
        loc = loc.ravel()
        scale = scale.ravel()

        out = np.empty_like(y)
        eps = 1e-12
        for i in range(y.shape[0]):
            s = scale[i] if scale[i] > eps else eps
            z = (y[i] - loc[i]) / s
            term = (
                    z * (2 * normal_cdf_numba(z) - 1.0)
                    + 2.0 * normal_pdf_numba(z)
                    - 1.0 / math.sqrt(math.pi)
            )
            out[i] = s * term
        return out

    @njit(fastmath=True, cache=True)
    def normal_dscore_numba(y, loc, var):
        # Ensure inputs are 1D
        y = y.ravel()
        loc = loc.ravel()
        var = var.ravel()

        n = y.shape[0]
        D = np.empty((n, 2))
        for i in range(n):
            diff = loc[i] - y[i]
            D[i, 0] = diff / var[i]
            D[i, 1] = 1.0 - (diff * diff) / var[i]
        return D

    @njit(fastmath=True, cache=True)
    def normal_metric_numba(var):
        # Ensure input is 1D
        var = var.ravel()

        n = var.shape[0]
        FI = np.zeros((n, 2, 2))
        for i in range(n):
            FI[i, 0, 0] = 1.0 / var[i]
            FI[i, 1, 1] = 2.0
        return FI

    @njit(fastmath=True, cache=True)
    def normal_fixedvar_logpdf_numba(y, loc):
        # Ensure inputs are 1D
        y = y.ravel()
        loc = loc.ravel()

        n = y.shape[0]
        out = np.empty(n)
        log_norm = -0.5 * np.log(2.0 * np.pi)
        for i in range(n):
            diff = y[i] - loc[i]
            out[i] = log_norm - 0.5 * diff * diff
        return out


class NormalLogScore(LogScore):
    def score(self, Y):
        if HAS_NUMBA:
            return -normal_logpdf_numba(Y, self.loc, self.scale)
        else:
            return -self.dist.logpdf(Y)

    def d_score(self, Y):
        if HAS_NUMBA:
            return normal_dscore_numba(Y, self.loc, self.var)
        D = np.zeros((len(Y), 2))
        D[:, 0] = (self.loc - Y) / self.var
        D[:, 1] = 1 - ((self.loc - Y) ** 2) / self.var
        return D

    def metric(self):
        if HAS_NUMBA:
            return normal_metric_numba(self.var)
        FI = np.zeros((self.var.shape[0], 2, 2))
        FI[:, 0, 0] = 1 / self.var
        FI[:, 1, 1] = 2
        return FI


class NormalCRPScore(CRPScore):
    def score(self, Y):
        if HAS_NUMBA:
            return normal_crps_numba(Y, self.loc, self.scale)
        else:
            Z = (Y - self.loc) / self.scale
            return self.scale * (
                    Z * (2 * sp.stats.norm.cdf(Z) - 1)
                    + 2 * sp.stats.norm.pdf(Z)
                    - 1 / np.sqrt(np.pi)
            )

    def d_score(self, Y):
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            Z = (Y - self.loc) / self.scale
        D = np.zeros((len(Y), 2))
        D[:, 0] = -(2 * sp.stats.norm.cdf(Z) - 1)
        D[:, 1] = self.score(Y) + (Y - self.loc) * D[:, 0]
        return D

    def metric(self):
        I = np.c_[
            2 * np.ones_like(self.var),
            np.zeros_like(self.var),
            np.zeros_like(self.var),
            self.var,
        ]
        I = I.reshape((self.var.shape[0], 2, 2))
        I = 1 / (2 * np.sqrt(np.pi)) * I
        return I


class Normal(RegressionDistn):
    """
    Implements the normal distribution for NGBoost.

    The normal distribution has two parameters, loc and scale, which are
    the mean and standard deviation, respectively.
    This distribution has both LogScore and CRPScore implemented for it.
    """

    n_params = 2
    scores = [NormalLogScore, NormalCRPScore]

    def __init__(self, params):
        super().__init__(params)
        self.loc = params[0]
        with np.errstate(over='ignore'):
            self.scale = np.exp(params[1])
            self.var = self.scale**2
        self.dist = dist(loc=self.loc, scale=self.scale)

    def fit(Y):
        m, s = sp.stats.norm.fit(Y)
        return np.array([m, np.log(s)])

    def sample(self, m):
        return self.dist.rvs(size=m)

    def __getattr__(
            self, name
    ):  # gives us Normal.mean() required for RegressionDist.predict()
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale}


# ### Fixed Variance Normal ###
class NormalFixedVarLogScore(LogScore):
    def score(self, Y):
        if HAS_NUMBA:
            return -normal_fixedvar_logpdf_numba(Y, self.loc)
        else:
            return -self.dist.logpdf(Y)

    def d_score(self, Y):
        D = np.zeros((len(Y), 1))
        D[:, 0] = (self.loc - Y) / self.var
        return D

    def metric(self):
        FI = np.zeros((self.var.shape[0], 1, 1))
        FI[:, 0, 0] = 1 / self.var + 1e-5
        return FI


class NormalFixedVarCRPScore(CRPScore):
    def score(self, Y):
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            Z = (Y - self.loc) / self.scale
        return self.scale * (
                Z * (2 * sp.stats.norm.cdf(Z) - 1)
                + 2 * sp.stats.norm.pdf(Z)
                - 1 / np.sqrt(np.pi)
        )

    def d_score(self, Y):
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            Z = (Y - self.loc) / self.scale
        D = np.zeros((len(Y), 1))
        D[:, 0] = -(2 * sp.stats.norm.cdf(Z) - 1)
        return D

    def metric(self):
        I = np.c_[2 * np.ones_like(self.var)]
        I = I.reshape((self.var.shape[0], 1, 1))
        I = 1 / (2 * np.sqrt(np.pi)) * I
        return I


class NormalFixedVar(Normal):
    """
    Implements the normal distribution with variance=1 for NGBoost.

    The fixed-variance normal distribution has one parameters, loc which is the mean.
    This distribution has both LogScore and CRPScore implemented for it.
    """

    n_params = 1
    scores = [NormalFixedVarLogScore, NormalFixedVarCRPScore]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self.loc = params[0]
        self.var = np.ones_like(self.loc)
        self.scale = np.ones_like(self.loc)
        self.shape = self.loc.shape
        self.dist = dist(loc=self.loc, scale=self.scale)

    def fit(Y):
        m, _ = sp.stats.norm.fit(Y)
        return m


# ### Fixed Mean Normal ###
class NormalFixedMeanLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        D = np.zeros((len(Y), 1))
        D[:, 0] = 1 - ((self.loc - Y) ** 2) / self.var
        return D

    def metric(self):
        FI = np.zeros((self.var.shape[0], 1, 1))
        FI[:, 0, 0] = 2
        return FI


class NormalFixedMeanCRPScore(CRPScore):
    def score(self, Y):
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            Z = (Y - self.loc) / self.scale
        return self.scale * (
                Z * (2 * sp.stats.norm.cdf(Z) - 1)
                + 2 * sp.stats.norm.pdf(Z)
                - 1 / np.sqrt(np.pi)
        )

    def d_score(self, Y):
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            Z = (Y - self.loc) / self.scale
        D = np.zeros((len(Y), 1))
        D[:, 0] = self.score(Y) + (Y - self.loc) * -1 * (2 * sp.stats.norm.cdf(Z) - 1)
        return D

    def metric(self):
        I = np.c_[self.var]
        I = I.reshape((self.var.shape[0], 1, 1))
        I = 1 / (2 * np.sqrt(np.pi)) * I
        return I


class NormalFixedMean(Normal):
    """
    Implements the normal distribution with mean=0 for NGBoost.

    The fixed-mean normal distribution has one parameter, scale which is the standard deviation.
    This distribution has both LogScore and CRPScore implemented for it.
    """

    n_params = 1
    scores = [NormalFixedMeanLogScore, NormalFixedMeanCRPScore]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self.loc = np.zeros_like(params[0])
        self.scale = np.exp(params[0])
        self.var = self.scale**2
        self.shape = self.loc.shape
        self.dist = dist(loc=self.loc, scale=self.scale)

    def fit(Y):
        _, s = sp.stats.norm.fit(Y)
        return s