import numpy as np
import pandas as pd
import pytest

from src import benchmark as bm, config, data


def test_equal_tuning_budget_and_roles():
    sizes = {b: len(g) for b, g in bm.GRIDS.items()}
    assert len(set(sizes.values())) == 1, sizes
    assert set(bm.ROLES) == set(bm.GRIDS) == set(bm.ORDER) == {"rf", "xgb", "hgb"}
    assert bm.ROLES["rf"] == "Baseline" and bm.ROLES["xgb"] == "Primary candidate"


@pytest.mark.parametrize("budget", [0.0, 1000.0, 3000.0, None])
def test_policy_contributions_respect_budget(budget):
    rng = np.random.default_rng(0)
    tau, gamma = rng.normal(0.5, 1, 5000), rng.normal(0.5, 30, 5000)
    z, cov = bm.policy_contributions(tau, gamma, budget=budget)
    assert len(z) == 5000
    if budget is not None:
        max_cov = budget / (config.COST_PER_CONTACT * config.POPULATION_SIZE)
        assert cov <= max_cov + 1e-9
    # only customers with margin * tau > cost may be contacted
    assert cov <= (config.MARGIN * tau > config.COST_PER_CONTACT).mean() + 1e-9


@pytest.mark.parametrize("base", ["rf", "xgb", "hgb"])
def test_fit_tlearner_scores_only_heldout_rows(base):
    df = data.load_raw().sample(6000, random_state=0)
    X = data.build_features(df)
    fit = np.arange(len(df)) < 4000
    tau, fit_s, pred_s = bm.fit_tlearner(df, X, fit, ~fit, "womens", base, bm.GRIDS[base][0])
    assert list(tau.index) == list(X.index[~fit])
    assert np.isfinite(tau).all() and fit_s > 0 and pred_s >= 0
