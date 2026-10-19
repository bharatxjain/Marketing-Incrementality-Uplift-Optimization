"""Held-out evaluation of uplift rankings and targeting policies.

Two families of estimators are used, both valid because assignment
probabilities are known by design:

* Qini / uplift curves on the two-arm subset (the conventional uplift metric).
* AIPW (doubly robust) policy values on the full evaluation population. For
  arm k and the control arm 0, the per-customer score is

      Gamma_ik = mu_k(X) - mu_0(X) + 1[A=k](Y - mu_k(X)) / e_k - 1[A=0](Y - mu_0(X)) / e_0

  and the incremental outcome of a policy that sends arm k to set S is
  (1/n) * sum_{i in S} Gamma_ik. The nuisance models mu are cross-fitted
  *inside* the evaluation partition, so they never see training outcomes and
  no candidate model's predictions enter its own evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

from .models import base_regressor

GRID = np.linspace(0, 1, 101)


# ----------------------------------------------------------------------------
# AIPW scores and policy values
# ----------------------------------------------------------------------------
def aipw_scores(X: pd.DataFrame, arm: pd.Series, y, arms: list[str], control: str,
                propensities: dict[str, float], n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Cross-fitted multi-arm AIPW scores Gamma_ik for every non-control arm."""
    Xa = X.to_numpy(float)
    arm = np.asarray(arm)
    y = np.asarray(y, float)
    mu = {a: np.zeros(len(y)) for a in arms}
    folds = KFold(n_folds, shuffle=True, random_state=seed).split(Xa)
    for k, (tr, te) in enumerate(folds):
        for j, a in enumerate(arms):
            m = tr[arm[tr] == a]
            mu[a][te] = base_regressor(seed + 100 * k + j).fit(Xa[m], y[m]).predict(Xa[te])
    e0 = propensities[control]
    resid0 = (arm == control) * (y - mu[control]) / e0
    out = {}
    for a in arms:
        if a == control:
            continue
        out[a] = mu[a] - mu[control] + (arm == a) * (y - mu[a]) / propensities[a] - resid0
    return pd.DataFrame(out, index=X.index)


def ipw_scores(arm, y, arm_k: str, control: str, propensities: dict[str, float]) -> np.ndarray:
    """Horvitz-Thompson scores; unbiased but higher variance than AIPW."""
    arm, y = np.asarray(arm), np.asarray(y, float)
    return (arm == arm_k) * y / propensities[arm_k] - (arm == control) * y / propensities[control]


def mean_ci(z, alpha: float = 0.05) -> dict:
    z = np.asarray(z, float)
    m, se = z.mean(), z.std(ddof=1) / np.sqrt(len(z))
    q = stats.norm.ppf(1 - alpha / 2)
    return {"estimate": m, "se": se, "ci_low": m - q * se, "ci_high": m + q * se}


def top_k_mask(score, frac: float, seed: int = 0) -> np.ndarray:
    """Boolean mask of the top ``frac`` share by score (random tie-breaking)."""
    score = np.asarray(score, float)
    n = len(score)
    k = int(round(frac * n))
    jitter = np.random.default_rng(seed).random(n) * 1e-12
    order = np.lexsort((jitter, -score))
    mask = np.zeros(n, bool)
    mask[order[:k]] = True
    return mask


def policy_gain(mask, gamma, alpha: float = 0.05) -> dict:
    """Incremental outcome per population customer from treating ``mask``."""
    return mean_ci(np.asarray(mask, float) * np.asarray(gamma, float), alpha)


def gain_at_coverage(scores: dict[str, np.ndarray], gamma, fracs) -> pd.DataFrame:
    """Policy gain (with CI) for each ranking at each coverage level.

    ``Random`` is the analytic expectation of random targeting, frac * ATE.
    """
    gamma = np.asarray(gamma, float)
    rows = []
    for f in fracs:
        rnd = mean_ci(f * gamma)
        rows.append({"policy": "Random", "coverage": f, **rnd})
        for name, s in scores.items():
            rows.append({"policy": name, "coverage": f, **policy_gain(top_k_mask(s, f), gamma)})
    return pd.DataFrame(rows)


def gain_curve(score, gamma, grid=GRID) -> np.ndarray:
    """Cumulative AIPW gain per population customer along the ranking."""
    gamma = np.asarray(gamma, float)
    order = np.argsort(-np.asarray(score, float), kind="stable")
    cum = np.concatenate([[0.0], np.cumsum(gamma[order])]) / len(gamma)
    idx = np.round(grid * len(gamma)).astype(int)
    return cum[idx]


def area_above_random(curve, grid=GRID) -> float:
    """Mean vertical distance between a gain curve and the random line."""
    curve = np.asarray(curve, float)
    return float(np.trapezoid(curve - grid * curve[-1], grid))


# ----------------------------------------------------------------------------
# Qini / uplift curves on a two-arm sample
# ----------------------------------------------------------------------------
def qini_curve(y, t, score, grid=GRID) -> pd.DataFrame:
    """Qini curve Q(phi) = Y_T(phi) - Y_C(phi) * N_T(phi) / N_C(phi).

    Y_T(phi) is the summed outcome of treated units among the top phi share by
    score. Values are divided by the sample size so they read as incremental
    outcome per customer. The uplift curve (diff in means times count) is also
    returned.
    """
    y, t = np.asarray(y, float), np.asarray(t).astype(int)
    order = np.argsort(-np.asarray(score, float), kind="stable")
    y, t = y[order], t[order]
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    yt, yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        qini = np.where(nc > 0, yt - yc * nt / np.maximum(nc, 1), 0.0)
        uplift = np.where((nt > 0) & (nc > 0),
                          (yt / np.maximum(nt, 1) - yc / np.maximum(nc, 1)) * (nt + nc), 0.0)
    n = len(y)
    idx = np.clip(np.round(grid * n).astype(int) - 1, 0, n - 1)
    q = np.where(grid == 0, 0.0, qini[idx]) / n
    u = np.where(grid == 0, 0.0, uplift[idx]) / n
    return pd.DataFrame({"frac": grid, "qini": q, "uplift": u})


def qini_coefficient(y, t, score, grid=GRID) -> float:
    """Area between the Qini curve and the random-targeting diagonal.

    Units: incremental outcome per customer, averaged over coverage. Zero means
    no better than random; it can be negative.
    """
    c = qini_curve(y, t, score, grid)
    return area_above_random(c["qini"].to_numpy(), grid)


def bootstrap_qini(y, t, scores: dict[str, np.ndarray], n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Paired bootstrap of Qini coefficients (same resample for every model)."""
    rng = np.random.default_rng(seed)
    y, t = np.asarray(y, float), np.asarray(t)
    n = len(y)
    rows = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        rows.append({k: qini_coefficient(y[idx], t[idx], np.asarray(s)[idx]) for k, s in scores.items()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Calibration of predicted uplift
# ----------------------------------------------------------------------------
def uplift_by_bin(score, gamma, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs. AIPW-estimated uplift within score quantile bins.

    Bin 1 holds the highest scores. A well-ranked model shows a decreasing
    estimated uplift from bin 1 to bin n_bins.
    """
    score, gamma = np.asarray(score, float), np.asarray(gamma, float)
    ranks = pd.Series(score).rank(method="first", ascending=False)
    bins = pd.qcut(ranks, n_bins, labels=range(1, n_bins + 1)).astype(int).to_numpy()
    rows = []
    for b in range(1, n_bins + 1):
        m = bins == b
        r = mean_ci(gamma[m])
        rows.append({"bin": b, "n": int(m.sum()), "predicted": score[m].mean(),
                     "estimated": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    return pd.DataFrame(rows)


def calibration_slope(score, gamma) -> dict:
    """OLS slope of the AIPW score on predicted CATE.

    A slope near 1 means predictions are on the right scale; near 0 means the
    predicted heterogeneity is not supported by held-out data. This is the
    "best linear predictor" diagnostic of Chernozhukov et al.
    """
    import statsmodels.api as sm

    s = np.asarray(score, float)
    fit = sm.OLS(np.asarray(gamma, float), sm.add_constant(s - s.mean())).fit(cov_type="HC2")
    lo, hi = fit.conf_int()[1]
    return {"slope": fit.params[1], "se": fit.bse[1], "ci_low": lo, "ci_high": hi,
            "p_value": fit.pvalues[1]}
