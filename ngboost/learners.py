from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

default_tree_learner = HistGradientBoostingRegressor(
    max_depth=3,
    max_iter=1,               # only one tree per fit
    learning_rate=1.0,        # let NGBoost control the learning rate
    min_samples_leaf=1,
    l2_regularization=1.0,
    random_state=None,
    max_bins=255,             # default, can be tuned
    early_stopping=False,     # don’t let it run its own validation loop
)


default_linear_learner = Ridge(alpha=0.0, random_state=None)
