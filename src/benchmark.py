"""Base-learner study: T-learners on Random Forest, XGBoost and HistGradientBoosting.

Protocol (fixed before any result was seen):

* Same data, splits, 12 pre-treatment features and AIPW evaluation for every base learner.
* Same tuning budget: 6 configurations per base learner, scored by the decision
  the model will drive, which is budgeted incremental profit (uplift policy at
  the default budget), averaged over the two campaigns.
* Primary held-out evaluation: **nested cross-validation over all 64,000
  customers.** In each outer fold the configuration is chosen by inner
  cross-validation on the training part only, refit, and the outer fold is
  scored and valued with AIPW scores computed inside that fold. No tuning
  decision ever sees the data it is evaluated on.
* Winner: the highest pooled held-out profit at the default budget, averaged
  over campaigns. Paired intervals against the Random Forest baseline are reported.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config, data, evaluation, models, optimization as opt, pipeline

ROLES = {"rf": "Baseline", "xgb": "Primary candidate", "hgb": "Efficiency benchmark"}
NAMES = {"rf": "Random Forest", "xgb": "XGBoost", "hgb": "HistGradientBoosting"}
ORDER = ["rf", "xgb", "hgb"]

# Equal budget: 6 configurations each, varying leaf size (the shared regularization
# knob) and one family-specific capacity knob.
GRIDS = {
    "rf": [{"min_samples_leaf": l, "max_features": f} for l in (100, 200, 400) for f in (0.33, 0.67)],
    # XGBoost has no early stopping here, so its capacity knob is the number of boosting rounds.
    "xgb": [{"min_samples_leaf": l, "n_estimators": n} for l in (100, 200, 400) for n in (50, 200)],
    "hgb": [{"min_samples_leaf": l, "max_leaf_nodes": m} for l in (100, 200, 400) for m in (7, 15)],
}


def cfg_label(cfg: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in cfg.items())


def fit_tlearner(df, X, fit_mask, score_mask, campaign, base, cfg, seed=config.SEED):
    """Fit one T-learner on (campaign arm + control) rows of ``fit_mask`` and score ``score_mask``.

    Returns (tau, fit_seconds, predict_seconds).
    """
    label = config.CAMPAIGNS[campaign]
    m = np.asarray(fit_mask) & df["segment"].isin([label, config.CONTROL]).to_numpy()
    t = (df.loc[m, "segment"] == label).astype(int).to_numpy()
    y = df.loc[m, config.PRIMARY_OUTCOME].astype(float).to_numpy()
    model = models.make_model("t_learner", seed=seed, base=base, **cfg)
    t0 = time.perf_counter()
    model.fit(X.loc[m], t, y)
    t1 = time.perf_counter()
    tau = model.predict(X.loc[np.asarray(score_mask)])
    t2 = time.perf_counter()
    return pd.Series(tau, index=X.index[np.asarray(score_mask)]), t1 - t0, t2 - t1


def policy_contributions(tau, gamma, budget=config.DEFAULT_BUDGET, margin=config.MARGIN,
                         cost=config.COST_PER_CONTACT, population=config.POPULATION_SIZE):
    """Per-customer held-out profit of the single-campaign uplift policy.

    Contact a customer only when margin * tau > cost, in order of net value,
    until the budget (scaled to the sample) runs out.
    """
    tau, gamma = np.asarray(tau, float), np.asarray(gamma, float)
    cap = None if budget is None else opt.budget_per_customer(budget, population) * len(tau)
    a = opt.allocate(opt.net_value(tau, margin, cost), cost, cap)
    return opt.profit_contributions(a, gamma, margin, cost), float((a >= 0).mean())


def tune(df, X, folds, gamma, base, seed=config.SEED, timings=None):
    """Score every configuration of ``base`` by out-of-fold budgeted profit on ``folds``.

    ``gamma`` holds AIPW scores for the rows in ``folds`` (fold >= 0), each
    computed within its own fold. Returns a table sorted by mean profit.
    """
    rows = []
    for cfg in GRIDS[base]:
        prof = {}
        for camp in config.CAMPAIGNS:
            parts = []
            for k in sorted(f for f in folds.unique() if f >= 0):
                tau, ft, pt = fit_tlearner(df, X, ((folds >= 0) & (folds != k)).to_numpy(),
                                           (folds == k).to_numpy(), camp, base, cfg, seed + k)
                parts.append(tau)
                if timings is not None:
                    timings.append({"base": base, "stage": "tuning", "fit_s": ft, "predict_s": pt})
            tau = pd.concat(parts).sort_index()
            z, _ = policy_contributions(tau, gamma[camp].loc[tau.index])
            prof[camp] = z.mean() * config.POPULATION_SIZE
        rows.append({"base": base, "config": cfg_label(cfg), "cfg": cfg, **prof,
                     "mean_profit": np.mean(list(prof.values()))})
    return pd.DataFrame(rows).sort_values("mean_profit", ascending=False).reset_index(drop=True)


def nested_heldout(df, X, n_outer=5, n_inner=3, seed=config.SEED, verbose=True):
    """Nested cross-validated T-learner scores for every customer and base learner.

    Returns (tau, gamma, chosen, timings):
      tau     DataFrame indexed like df, columns "<base>|<campaign>" (held-out CATE)
      gamma   DataFrame indexed like df, AIPW score per campaign (computed within outer fold)
      chosen  configuration picked by the inner loop, per outer fold and base learner
      timings one row per fit
    """
    outer = data.assign_folds(df, np.ones(len(df), bool), n_folds=n_outer, seed=seed + 11)
    tau = pd.DataFrame(index=df.index)
    gamma = pipeline.cross_fit_gamma(df, X, outer, seed=seed)
    chosen, timings = [], []
    for k in range(n_outer):
        t0 = time.time()
        train = (outer != k).to_numpy()
        inner = data.assign_folds(df, train, n_folds=n_inner, seed=seed + 100 + k)
        g_inner = pipeline.cross_fit_gamma(df, X, inner, seed=seed + 100 + k)
        for base in ORDER:
            table = tune(df, X, inner, g_inner, base, seed=seed + 100 + k, timings=timings)
            cfg = table.loc[0, "cfg"]
            chosen.append({"outer_fold": k, "base": base, "config": cfg_label(cfg),
                           "inner_mean_profit": table.loc[0, "mean_profit"]})
            for camp in config.CAMPAIGNS:
                s, ft, pt = fit_tlearner(df, X, train, (outer == k).to_numpy(), camp, base, cfg, seed + k)
                tau.loc[s.index, f"{base}|{camp}"] = s
                timings.append({"base": base, "stage": "final", "fit_s": ft, "predict_s": pt})
        if verbose:
            print(f"outer fold {k}: done in {time.time() - t0:.0f}s")
    return tau, gamma, pd.DataFrame(chosen), pd.DataFrame(timings)


def policy_table(tau, gamma, budgets=(config.DEFAULT_BUDGET, None), baseline="rf"):
    """Held-out profit per base learner, campaign and budget, with paired differences vs ``baseline``."""
    rows = []
    for budget in budgets:
        for camp in config.CAMPAIGNS:
            g = gamma[camp].to_numpy()
            z = {b: policy_contributions(tau[f"{b}|{camp}"], g, budget) for b in ORDER}
            for b in ORDER:
                r = evaluation.mean_ci(z[b][0])
                row = {"budget": "no limit" if budget is None else f"${budget:,.0f}", "campaign": camp,
                       "base": NAMES[b], "role": ROLES[b], "coverage": z[b][1],
                       "profit": r["estimate"] * config.POPULATION_SIZE,
                       "ci_low": r["ci_low"] * config.POPULATION_SIZE, "ci_high": r["ci_high"] * config.POPULATION_SIZE}
                if b != baseline:
                    d = evaluation.mean_ci(z[b][0] - z[baseline][0])
                    row.update({"vs_baseline": d["estimate"] * config.POPULATION_SIZE,
                                "vs_ci_low": d["ci_low"] * config.POPULATION_SIZE,
                                "vs_ci_high": d["ci_high"] * config.POPULATION_SIZE,
                                "z": d["estimate"] / d["se"]})
                rows.append(row)
    return pd.DataFrame(rows)


def ranking_table(tau, gamma):
    """Uplift-ranking quality per base learner and campaign: AUGC, gain@30%, BLP slope."""
    rows = []
    for camp in config.CAMPAIGNS:
        g = gamma[camp].to_numpy()
        for b in ORDER:
            s = tau[f"{b}|{camp}"].to_numpy()
            gain = evaluation.policy_gain(evaluation.top_k_mask(s, 0.3), g)
            rows.append({"campaign": camp, "base": NAMES[b],
                         "AUGC": evaluation.area_above_random(evaluation.gain_curve(s, g)),
                         "gain@30%": gain["estimate"], "BLP": evaluation.calibration_slope(s, g)["slope"]})
    return pd.DataFrame(rows)
