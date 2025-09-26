# benchmarks/test_distn_speed_poisson.py
import numpy as np
import pytest
from scipy.stats import poisson

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from ngboost.distns.poisson import poisson_logpmf_numba

@pytest.fixture(scope="module")
def data():
    n = 50_000
    y = np.random.poisson(lam=5.0, size=n)
    mu = np.full(n, 5.0)
    return y, mu

def test_poisson_logpmf_scipy(benchmark, data):
    y, mu = data
    benchmark(lambda: poisson.logpmf(y, mu).sum())

@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
def test_poisson_logpmf_numba(benchmark, data):
    y, mu = data
    poisson_logpmf_numba(y[:10], mu[:10])  # warmup
    benchmark(lambda: poisson_logpmf_numba(y, mu).sum())
