import math
import numpy as np
import pytest
from scipy.stats import weibull_min as dist

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ----------------------------
# SciPy versions
# ----------------------------
def weibull_logpdf_scipy(y, shape, scale):
    return dist.logpdf(y, c=shape, scale=scale)


def weibull_dscore_scipy(y, shape, scale):
    D = np.zeros((len(y), 2))
    z = y / scale
    z_shape = z ** shape
    shared_term = shape * (z_shape - 1)
    D[:, 0] = shared_term * np.log(z) - 1
    D[:, 1] = -shared_term
    return D


# ----------------------------
# Numba versions
# ----------------------------
if HAS_NUMBA:
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

            out[i, 0] = shared_term * math.log(z) - 1.0
            out[i, 1] = -shared_term
        return out


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture(scope="module")
def data():
    n = 50_000
    shape = np.full(n, 1.5)
    scale = np.full(n, 2.0)
    y = dist.rvs(c=shape[0], scale=scale[0], size=n)
    return y, shape, scale


# ----------------------------
# Benchmarks
# ----------------------------
def test_weibull_logpdf_scipy(benchmark, data):
    y, shape, scale = data
    benchmark(lambda: weibull_logpdf_scipy(y, shape[0], scale[0]).sum())


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
def test_weibull_logpdf_numba(benchmark, data):
    y, shape, scale = data
    weibull_logpdf_numba(y[:10], shape[:10], scale[:10])  # warmup
    benchmark(lambda: weibull_logpdf_numba(y, shape, scale).sum())


def test_weibull_dscore_scipy(benchmark, data):
    y, shape, scale = data
    benchmark(lambda: weibull_dscore_scipy(y, shape[0], scale[0]))


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
def test_weibull_dscore_numba(benchmark, data):
    y, shape, scale = data
    weibull_dscore_numba(y[:10], shape[:10], scale[:10])  # warmup
    benchmark(lambda: weibull_dscore_numba(y, shape, scale))
