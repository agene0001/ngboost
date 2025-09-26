import math
import numpy as np
import pytest
from scipy.stats import t as t_dist
from scipy.special import digamma

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ----------------------------
# SciPy versions
# ----------------------------
def t_logpdf_scipy(y, df, loc, scale):
    return t_dist.logpdf(y, df=df, loc=loc, scale=scale)

def t_dscore_scipy(y, df, loc, scale):
    var = scale**2
    num_loc = (df + 1) * (y - loc)
    den = (df * var) + (y - loc) ** 2
    d_loc = -(num_loc / den)

    num_scale = (df + 1) * (y - loc) ** 2
    den_scale = (df * var) + (y - loc) ** 2
    d_scale = 1 - (num_scale / den_scale)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        term_1 = (df / 2) * digamma((df + 1) / 2)
        term_2 = (-df / 2) * digamma(df / 2)
        term_3 = -0.5
        term_4_1 = (-df / 2) * np.log(1 + ((y - loc) ** 2) / (df * var))
        term_4_2_num = (df + 1) * (y - loc) ** 2
        term_4_2_den = 2 * (df * var) * (1 + ((y - loc) ** 2) / (df * var))
        d_df = -(term_1 + term_2 + term_3 + term_4_1 + term_4_2_num / term_4_2_den)

    return np.stack([d_loc, d_scale, d_df], axis=1)

# ----------------------------
# Numba versions
# ----------------------------
if HAS_NUMBA:
    @njit(fastmath=True, cache=True)
    def t_logpdf_numba(y, df, loc, scale):
        n = y.shape[0]
        out = np.empty(n)

        # assume df is constant (all elements the same)
        df_val = df[0]

        c = math.lgamma((df_val + 1.0) / 2.0) - math.lgamma(df_val / 2.0)
        log_norm = c - 0.5 * math.log(df_val * math.pi)

        for i in range(n):
            z = (y[i] - loc[i]) / scale[i]
            out[i] = log_norm - math.log(scale[i]) - ((df_val + 1.0) / 2.0) * math.log(1.0 + (z * z) / df_val)

        return out


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture(scope="module")
def data():
    n = 50_000
    y = np.random.standard_t(df=5, size=n)
    mu = np.zeros(n)
    sigma = np.ones(n)
    df = np.full(n, 5.0)
    return y, df, mu, sigma

# ----------------------------
# Benchmarks
# ----------------------------
def test_t_logpdf_scipy(benchmark, data):
    y, df, mu, sigma = data
    benchmark(lambda: t_logpdf_scipy(y, df[0], mu, sigma).sum())

@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
def test_t_logpdf_numba(benchmark, data):
    y, df, mu, sigma = data
    # warm-up compile
    t_logpdf_numba(y[:10], df[:10], mu[:10], sigma[:10])
    benchmark(lambda: t_logpdf_numba(y, df, mu, sigma).sum())


