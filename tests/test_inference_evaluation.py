import numpy as np
import pandas as pd

from src import evaluation, inference


def _synthetic(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.integers(0, 2, n)
    tau = np.where(x > 0, 1.0, 0.0)
    y = 0.5 * x + t * tau + rng.normal(size=n)
    return x, t, y, tau


def test_diff_in_means_recovers_ate():
    x, t, y, tau = _synthetic()
    r = inference.diff_in_means(y, t)
    assert r["ci_low"] < tau.mean() < r["ci_high"]


def test_lin_adjustment_is_more_precise():
    x, t, y, _ = _synthetic()
    dim = inference.diff_in_means(y, t)
    lin = inference.lin_regression_ate(y, t, pd.DataFrame({"x": x}))
    assert lin["se"] < dim["se"]


def test_holm_monotone_and_bounded():
    adj = inference.holm([0.01, 0.04, 0.03, 0.5])
    assert np.all(adj >= [0.01, 0.04, 0.03, 0.5])
    assert adj.max() <= 1


def test_qini_oracle_beats_random():
    x, t, y, tau = _synthetic()
    oracle = evaluation.qini_coefficient(y, t, tau + 1e-3 * x)
    noise = evaluation.qini_coefficient(y, t, np.random.default_rng(1).random(len(y)))
    assert oracle > 0.05 > abs(noise)


def test_qini_curve_endpoint_is_ate():
    x, t, y, _ = _synthetic()
    c = evaluation.qini_curve(y, t, x)
    dim = inference.diff_in_means(y, t)["ate"]
    # Q(1)/n = mean_T * n_T/n - mean_C * n_T/n = ATE * n_T / n
    assert np.isclose(c["qini"].iloc[-1], dim * t.mean(), rtol=1e-6)


def test_aipw_policy_value_unbiased_for_oracle():
    x, t, y, tau = _synthetic(40000)
    X = pd.DataFrame({"x": x})
    arm = np.where(t == 1, "T", "C")
    g = evaluation.aipw_scores(X, arm, y, ["C", "T"], "C", {"C": 0.5, "T": 0.5}, n_folds=3)["T"]
    gain = evaluation.policy_gain(x > 0, g)
    assert gain["ci_low"] < 0.5 < gain["ci_high"]  # truth: half the population gains 1.0


def test_top_k_mask_size():
    m = evaluation.top_k_mask(np.zeros(1000), 0.3)
    assert m.sum() == 300
