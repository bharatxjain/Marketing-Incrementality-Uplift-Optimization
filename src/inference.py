"""Experimental (average-effect) inference for a randomized two-arm comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def diff_in_means(y, t, alpha: float = 0.05) -> dict:
    """Difference in means with a Welch (unequal-variance) normal interval.

    This is the primary, design-based ATE estimator for a randomized experiment.
    """
    y = np.asarray(y, float)
    t = np.asarray(t).astype(bool)
    y1, y0 = y[t], y[~t]
    m1, m0 = y1.mean(), y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    ate = m1 - m0
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * stats.norm.sf(abs(ate / se)) if se > 0 else np.nan
    return {
        "n_t": len(y1), "n_c": len(y0), "mean_t": m1, "mean_c": m0,
        "ate": ate, "se": se, "ci_low": ate - z * se, "ci_high": ate + z * se,
        "p_value": p, "rel_lift": ate / m0 if m0 > 0 else np.nan,
    }


def bootstrap_ate(y, t, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap, resampling within arm (preserves the design)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    t = np.asarray(t).astype(bool)
    y1, y0 = y[t], y[~t]
    i1 = rng.integers(0, len(y1), (n_boot, len(y1)))
    i0 = rng.integers(0, len(y0), (n_boot, len(y0)))
    draws = y1[i1].mean(axis=1) - y0[i0].mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"ate": y1.mean() - y0.mean(), "ci_low": lo, "ci_high": hi,
            "se": draws.std(ddof=1), "draws": draws}


def permutation_test(y, t, n_perm: int = 5000, seed: int = 0) -> dict:
    """Randomization-inference p-value for the sharp null of no effect."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    t = np.asarray(t).astype(bool)
    obs = y[t].mean() - y[~t].mean()
    n1, total = t.sum(), y.sum()
    n = len(y)
    null = np.empty(n_perm)
    for b in range(n_perm):
        s1 = y[rng.permutation(n)[:n1]].sum()
        null[b] = s1 / n1 - (total - s1) / (n - n1)
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1)
    return {"ate": obs, "p_value": p, "null": null}


def _independent_columns(Xc: pd.DataFrame) -> list[str]:
    """Greedily keep columns that add rank (drops one level of each one-hot set)."""
    keep, mat = [], np.ones((len(Xc), 1))
    for col in Xc.columns:
        cand = np.column_stack([mat, Xc[col].to_numpy(float)])
        if np.linalg.matrix_rank(cand) > mat.shape[1]:
            keep.append(col)
            mat = cand
    return keep


def lin_regression_ate(y, t, X: pd.DataFrame) -> dict:
    """Lin (2013) regression adjustment: Y ~ T + Xc + T*Xc with HC2 errors.

    With centred covariates the coefficient on T is a consistent ATE estimate
    that is never less precise asymptotically than the difference in means.
    """
    import statsmodels.api as sm

    Xc = X - X.mean()
    Xc = Xc[_independent_columns(Xc)]
    t = np.asarray(t, float)
    design = pd.concat(
        [pd.Series(t, index=X.index, name="T"), Xc,
         Xc.mul(t, axis=0).add_prefix("T_x_")], axis=1)
    design = sm.add_constant(design)
    fit = sm.OLS(np.asarray(y, float), design).fit(cov_type="HC2")
    lo, hi = fit.conf_int().loc["T"]
    return {"ate": fit.params["T"], "se": fit.bse["T"], "ci_low": lo, "ci_high": hi,
            "p_value": fit.pvalues["T"]}


def standardized_mean_difference(x, t) -> float:
    x = np.asarray(x, float)
    t = np.asarray(t).astype(bool)
    pooled = np.sqrt((x[t].var(ddof=1) + x[~t].var(ddof=1)) / 2)
    return (x[t].mean() - x[~t].mean()) / pooled if pooled > 0 else 0.0


def balance_table(X: pd.DataFrame, t) -> pd.DataFrame:
    t = np.asarray(t).astype(bool)
    rows = []
    for col in X.columns:
        x = X[col].to_numpy(float)
        rows.append({"feature": col, "mean_t": x[t].mean(), "mean_c": x[~t].mean(),
                     "smd": standardized_mean_difference(x, t)})
    return pd.DataFrame(rows).set_index("feature")


def srm_test(counts, expected_shares) -> dict:
    """Sample-ratio-mismatch chi-square test against the intended allocation."""
    counts = np.asarray(counts, float)
    exp = np.asarray(expected_shares, float) * counts.sum()
    chi2, p = stats.chisquare(counts, exp)
    return {"chi2": chi2, "p_value": p}


def minimum_detectable_effect(sd_t, sd_c, n_t, n_c, alpha=0.05, power=0.8) -> float:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * np.sqrt(sd_t ** 2 / n_t + sd_c ** 2 / n_c)


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def segment_effects(df: pd.DataFrame, t, outcome: str, by: str, alpha=0.05) -> pd.DataFrame:
    """Descriptive diff-in-means within each level of a pre-treatment variable."""
    t = pd.Series(np.asarray(t), index=df.index)
    rows = []
    for level, g in df.groupby(by, observed=True):
        r = diff_in_means(g[outcome], t.loc[g.index], alpha=alpha)
        r.update({"variable": by, "level": level, "n": len(g)})
        rows.append(r)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"].fillna(1.0))
    return out
