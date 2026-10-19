import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src import models


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "x3": rng.integers(0, 2, n)})
    t = rng.integers(0, 2, n)
    tau = 1.0 + 1.5 * (X.x1 > 0) + 0.5 * X.x3
    y = 2 + X.x2 + t * tau + rng.normal(size=n)
    return X, t, y.to_numpy(), tau.to_numpy()


@pytest.mark.parametrize("key,params", [
    ("s_learner", {}), ("t_learner", {}), ("x_learner", {"propensity": 0.5}),
    ("dr_learner", {"propensity": 0.5}),
    ("causal_forest", {"n_estimators": 100, "min_samples_leaf": 20}),
])
def test_cate_models_recover_heterogeneity(synthetic, key, params):
    X, t, y, tau = synthetic
    m = models.make_model(key, seed=0, min_samples_leaf=50, **params) if key != "causal_forest" \
        else models.make_model(key, seed=0, **params)
    pred = m.fit(X, t, y).predict(X)
    assert pred.shape == (len(X),)
    assert spearmanr(pred, tau)[0] > 0.5
    assert abs(pred.mean() - tau.mean()) < 0.3


@pytest.mark.parametrize("base", ["hgb", "rf", "xgb"])
@pytest.mark.parametrize("key", ["t_learner", "x_learner", "dr_learner"])
def test_base_learners_plug_into_meta_learners(synthetic, key, base):
    X, t, y, tau = synthetic
    extra = {"propensity": 0.5} if key != "t_learner" else {}
    pred = models.make_model(key, seed=0, base=base, min_samples_leaf=50, **extra).fit(X, t, y).predict(X)
    assert spearmanr(pred, tau)[0] > 0.5
    assert abs(pred.mean() - tau.mean()) < 0.3


def test_unknown_base_learner_fails():
    with pytest.raises(ValueError, match="unknown base learner"):
        models.base_regressor(base="svm")


@pytest.mark.parametrize("base", ["hgb", "rf", "xgb"])
def test_response_model_base_learners(base):
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"x": rng.normal(size=4000)})
    t = rng.integers(0, 2, 4000)
    y = ((X.x > 0) & (t == 1)).astype(int)
    p = models.ResponseModel(seed=0, base=base, min_samples_leaf=20).fit(X, t, y).predict(X)
    assert p[X.x > 1].mean() > p[X.x < -1].mean() + 0.5


def test_causal_forest_intervals(synthetic):
    X, t, y, _ = synthetic
    cf = models.CausalForest(seed=0, n_estimators=100, min_samples_leaf=20).fit(X, t, y)
    lo, hi = cf.predict_interval(X.iloc[:50])
    assert (lo <= hi).all()


def test_response_model_uses_treated_only():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(size=4000)})
    t = rng.integers(0, 2, 4000)
    y = ((X.x > 0) & (t == 1)).astype(int)  # only treated customers ever "buy"
    p = models.ResponseModel(seed=0, min_samples_leaf=20).fit(X, t, y).predict(X)
    assert p[X.x > 1].mean() > 0.8


def test_calibrated_propensity_outputs_dollars(synthetic):
    X, t, y, _ = synthetic
    m = models.CalibratedPropensity(seed=0, propensity=0.5).fit(X, t, np.clip(y - 3, 0, None))
    pred = m.predict(X)
    assert np.isfinite(pred).all() and hasattr(m, "b_")
