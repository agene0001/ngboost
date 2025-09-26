import cProfile, pstats
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore

# Prepare data
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X[:2000], y[:2000], test_size=0.2)

# Profile NGBRegressor.fit
ngb = NGBRegressor(Dist=Normal, Score=LogScore, verbose=False, n_estimators=50)

profiler = cProfile.Profile()
profiler.enable()
ngb.fit(X_train, y_train)
profiler.disable()

stats = pstats.Stats(profiler).sort_stats("time")
stats.print_stats(20)
