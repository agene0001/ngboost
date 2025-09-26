# benchmarks/test_perf_suite.py
import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBRegressor, NGBClassifier, NGBoost
from ngboost.distns import Normal, LogNormal, Gamma, k_categorical, MultivariateNormal
from ngboost.scores import LogScore

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.ensemble import HistGradientBoostingRegressor

def test_fit_histgradientboosting_small(benchmark, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    hist_tree = HistGradientBoostingRegressor(max_depth=3, max_iter=1, l2_regularization=1.0,learning_rate=1.0, early_stopping=False)
    ngb = NGBRegressor(Dist=Normal, Score=LogScore, Base=hist_tree, verbose=False, n_estimators=50)
    benchmark(ngb.fit, X_train, y_train)
def test_fit_decisiontree_small(benchmark, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    tree = DecisionTreeRegressor(
        criterion="friedman_mse",
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )
    ngb = NGBRegressor(Dist=Normal, Score=LogScore, Base=tree, verbose=False, n_estimators=50)
    benchmark(ngb.fit, X_train, y_train)
def test_fit_histgradient_large(benchmark):
    N, P = 20000, 20
    X_train = np.random.randn(N, P)
    y_train = X_train[:, 0] * 0.5 + np.random.randn(N)  # simple signal
    hist_tree = HistGradientBoostingRegressor(
        max_depth=3, max_iter=1, learning_rate=1.0, l2_regularization=1.0,early_stopping=False
    )
    ngb = NGBRegressor(Dist=Normal, Score=LogScore, Base=hist_tree,
                       verbose=False, n_estimators=50)
    benchmark(ngb.fit, X_train, y_train)
def test_fit_decisiontree_large(benchmark):
    N, P = 20000, 20
    X_train = np.random.randn(N, P)
    # simple linear-ish target with noise
    y_train = 0.5 * X_train[:, 0] + np.random.randn(N)

    tree = DecisionTreeRegressor(
        criterion="friedman_mse",
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )
    ngb = NGBRegressor(
        Dist=Normal,
        Score=LogScore,
        Base=tree,
        verbose=False,
        n_estimators=50
    )
    benchmark(ngb.fit, X_train, y_train)
# --- Fixtures ---
@pytest.fixture(scope="module")
def regression_data():
    data = fetch_california_housing()
    X, y = data["data"][:2000], data["target"][:2000]
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture(scope="module")
def classification_data():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X[:1000], y[:1000], test_size=0.2, random_state=42)

def test_fit_parallel(benchmark, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    ngb = NGBRegressor(Dist=Normal, Score=LogScore, verbose=False, n_estimators=50)
    # monkeypatch: add n_jobs parameter
    ngb.fit_base = lambda X, grads, sample_weight=None: NGBoost.fit_base(
        ngb, X, grads, sample_weight, n_jobs=-1
    )
    benchmark(ngb.fit, X_train, y_train)

# --- Regression Benchmarks ---
def test_fit_normal(benchmark, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    ngb = NGBRegressor(Dist=Normal, Score=LogScore, verbose=False, n_estimators=50)
    benchmark(ngb.fit, X_train, y_train)

def test_fit_lognormal(benchmark, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    ngb = NGBRegressor(Dist=LogNormal, Score=LogScore, verbose=False, n_estimators=50)
    benchmark(ngb.fit, X_train, y_train)

def test_fit_gamma(benchmark, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    ngb = NGBRegressor(Dist=Gamma, Score=LogScore, verbose=False, n_estimators=50)
    benchmark(ngb.fit, X_train, y_train)


# --- Classification Benchmarks ---
def test_fit_categorical(benchmark, classification_data):
    X_train, X_test, y_train, y_test = classification_data
    ngb = NGBClassifier(Dist=k_categorical(2), Score=LogScore, verbose=False, n_estimators=50)
    benchmark(ngb.fit, X_train, y_train)

def test_predict_proba_categorical(benchmark, classification_data):
    X_train, X_test, y_train, y_test = classification_data
    ngb = NGBClassifier(Dist=k_categorical(2), Score=LogScore, verbose=False, n_estimators=50)
    ngb.fit(X_train, y_train)
    benchmark(ngb.predict_proba, X_test)


# --- Prediction Benchmarks ---
def test_predict_normal(benchmark, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    ngb = NGBRegressor(Dist=Normal, Score=LogScore, verbose=False, n_estimators=50)
    ngb.fit(X_train, y_train)
    benchmark(ngb.predict, X_test)

def test_pred_dist_lognormal(benchmark, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    ngb = NGBRegressor(Dist=LogNormal, Score=LogScore, verbose=False, n_estimators=50)
    ngb.fit(X_train, y_train)
    benchmark(ngb.pred_dist, X_test)


# --- Multivariate Benchmarks ---
def test_fit_multivariatenormal(benchmark):
    k = 3  # small k for stress-testing matrix ops
    dist = MultivariateNormal(k)

    N = 500
    X_train = np.random.randn(N, k)
    y_cols = [
        np.sin(X_train[:, i]).reshape(-1, 1) + np.random.randn(N, 1)
        for i in range(k)
    ]
    y_train = np.hstack(y_cols)
    X_test = np.random.randn(N, k)

    ngb = NGBRegressor(Dist=dist, Score=LogScore, verbose=False, n_estimators=30)
    benchmark(ngb.fit, X_train, y_train)
