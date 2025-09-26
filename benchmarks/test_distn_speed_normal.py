import math

import numpy as np
import pytest
import scipy as sp
from scipy.stats import norm
from numba import njit

from ngboost.distns.normal import normal_logpdf_numba

# --- Numba implementations ---
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# --- SciPy version (current NGBoost logic) ---
def normal_crps_scipy(y, mu, sigma):
    Z = (y - mu) / sigma
    return sigma * (
            Z * (2 * sp.stats.norm.cdf(Z) - 1)
            + 2 * sp.stats.norm.pdf(Z)
            - 1 / np.sqrt(np.pi)
    )

# --- Numba version ---
if HAS_NUMBA:
    @njit(fastmath=True, cache=True)
    def normal_pdf_numba(z):
        return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

    @njit(cache=True)
    def normal_cdf_numba(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    @njit(fastmath=True, cache=True)
    def normal_crps_numba(y, mu, sigma):
        out = np.empty_like(y)
        for i in range(y.shape[0]):
            z = (y[i] - mu[i]) / sigma[i]
            term = (
                    z * (2 * normal_cdf_numba(z) - 1.0)
                    + 2.0 * normal_pdf_numba(z)
                    - 1.0 / math.sqrt(math.pi)
            )
            out[i] = sigma[i] * term
        return out

    @njit(fastmath=True, cache=True)
    def normal_dscore_numba(y, loc, var):
        n = y.shape[0]
        D = np.empty((n, 2))
        for i in range(n):
            diff = loc[i] - y[i]
            D[i, 0] = diff / var[i]
            D[i, 1] = 1.0 - (diff * diff) / var[i]
        return D

    @njit(fastmath=True, cache=True)
    def normal_metric_numba(var):
        n = var.shape[0]
        FI = np.zeros((n, 2, 2))
        for i in range(n):
            FI[i, 0, 0] = 1.0 / var[i]
            FI[i, 1, 1] = 2.0
        return FI

# --- Fixtures ---
@pytest.fixture(scope="module")
def data():
    n = 50_000  # realistic NGBoost size
    y = np.random.randn(n)
    mu = np.random.randn(n)
    sigma = np.abs(np.random.randn(n)) + 1.0
    return y, mu, sigma
def test_dscore_numpy(benchmark, data):
    y, mu, sigma = data
    var = sigma**2
    def numpy_dscore(y, mu, var):
        D = np.zeros((len(y), 2))
        D[:, 0] = (mu - y) / var
        D[:, 1] = 1 - ((mu - y) ** 2) / var
        return D
    benchmark(numpy_dscore, y, mu, var)


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
def test_dscore_numba(benchmark, data):
    y, mu, sigma = data
    var = sigma**2
    normal_dscore_numba(y[:10], mu[:10], var[:10])  # warm-up
    benchmark(normal_dscore_numba, y, mu, var)


def test_metric_numpy(benchmark, data):
    _, _, sigma = data
    var = sigma**2
    def numpy_metric(var):
        FI = np.zeros((var.shape[0], 2, 2))
        FI[:, 0, 0] = 1 / var
        FI[:, 1, 1] = 2
        return FI
    benchmark(numpy_metric, var)


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
def test_metric_numba(benchmark, data):
    _, _, sigma = data
    var = sigma**2
    normal_metric_numba(var[:10])  # warm-up
    benchmark(normal_metric_numba, var)

# --- Benchmarks ---
def test_crps_scipy(benchmark, data):
    y, mu, sigma = data
    benchmark(normal_crps_scipy, y, mu, sigma)

@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
def test_crps_numba(benchmark, data):
    y, mu, sigma = data
    # warm-up compilation
    normal_crps_numba(y[:10], mu[:10], sigma[:10])
    benchmark(normal_crps_numba, y, mu, sigma)
# --- Benchmarks ---
def test_logpdf_scipy(benchmark, data):
    y, mu, sigma = data
    def run(y, mu, sigma):
        return norm.logpdf(y, loc=mu, scale=sigma).sum()
    benchmark(run, y, mu, sigma)

def test_logpdf_numba(benchmark, data):
    y, mu, sigma = data
    # Warm-up compilation outside benchmark
    normal_logpdf_numba(y[:10], mu[:10], sigma[:10])
    benchmark(normal_logpdf_numba, y, mu, sigma)



