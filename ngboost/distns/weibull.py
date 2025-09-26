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
        # Ensure inputs are 1D
        y = y.ravel() if y.ndim > 1 else y
        shape = shape.ravel() if shape.ndim > 1 else shape
        scale = scale.ravel() if scale.ndim > 1 else scale

        n = y.shape[0]
        out = np.empty(n)
        for i in range(n):
            if y[i] <= 0:
                out[i] = -np.inf
            else:
                z = y[i] / scale[i]
                out[i] = (
                        math.log(shape[i])
                        - shape[i] * math.log(scale[i])
                        + (shape[i] - 1.0) * math.log(y[i])
                        - (z ** shape[i])
                )
        return out


class WeibullLogScore(LogScore):
    def score(self, Y):
        if HAS_NUMBA:
            return -weibull_logpdf_numba(Y, self.shape, self.scale)
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        """
        Gradient of negative log score with respect to log-transformed parameters.

        The Weibull log PDF is:
        log_pdf = log(k) - k*log(λ) + (k-1)*log(y) - (y/λ)^k

        Where k = shape = exp(params[0]) and λ = scale = exp(params[1])

        We need:
        d(-log_pdf)/d(log(k)) = -d(log_pdf)/d(log(k))
        d(-log_pdf)/d(log(λ)) = -d(log_pdf)/d(log(λ))
        """
        D = np.zeros((len(Y), 2))
        z = Y / self.scale
        z_k = z ** self.shape
        log_z = np.log(z)

        # Gradient w.r.t. log(shape)
        # d(log_pdf)/d(log(k)) = d(log_pdf)/dk * dk/d(log(k)) = d(log_pdf)/dk * k
        # d(log_pdf)/dk = 1/k - log(λ) + log(y) - (y/λ)^k * log(y/λ)
        # So: d(log_pdf)/d(log(k)) = 1 - k*log(λ) + k*log(y) - k*(y/λ)^k * log(y/λ)
        #                           = 1 + k*log(z) - k*z^k*log(z)
        #                           = 1 + k*log(z)*(1 - z^k)
        D[:, 0] = -(1 + self.shape * log_z * (1 - z_k))

        # Gradient w.r.t. log(scale)
        # d(log_pdf)/d(log(λ)) = d(log_pdf)/dλ * dλ/d(log(λ)) = d(log_pdf)/dλ * λ
        # d(log_pdf)/dλ = -k/λ + k*(y/λ)^k/λ
        # So: d(log_pdf)/d(log(λ)) = -k + k*(y/λ)^k = k*(z^k - 1)
        D[:, 1] = -self.shape * (z_k - 1)

        return D

    def metric(self):
        """
        Fisher Information matrix for log-transformed parameters.

        The Fisher Information is the expected value of the outer product of the score gradient.
        For Weibull distribution with log-transformed parameters.
        """
        n = self.scale.shape[0]
        FI = np.zeros((n, 2, 2))

        # These formulas come from the expectation of the Hessian
        # For log-transformed parameters

        # Var(d(-log_pdf)/d(log(shape)))
        FI[:, 0, 0] = _FI00_CONST

        # Cov(d(-log_pdf)/d(log(shape)), d(-log_pdf)/d(log(scale)))
        cross = -self.shape * (1.0 - _EULER_GAMMA)
        FI[:, 0, 1] = cross
        FI[:, 1, 0] = cross

        # Var(d(-log_pdf)/d(log(scale)))
        FI[:, 1, 1] = self.shape ** 2

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
        # Parameters are stored on log scale
        self.shape,self.scale = np.exp(params[:2])
        # Handle potential shape issues
        if hasattr(self.shape, 'shape') and len(self.shape.shape) > 0:
            self.shape = self.shape.ravel()
        if hasattr(self.scale, 'shape') and len(self.scale.shape) > 0:
            self.scale = self.scale.ravel()
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