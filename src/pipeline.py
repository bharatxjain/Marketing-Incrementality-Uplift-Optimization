"""Fit-and-score pipeline shared by notebooks 02-06.

Model selection uses K-fold cross-fitting on the development partition: each
fold's customers are scored by models trained on the other folds, and their
AIPW evaluation scores are computed using only that fold. The test partition
is scored once by models refit on the full development set.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config, data, evaluation, models

PROPENSITY_AWARE = {"x_learner", "dr_learner", "calibrated_propensity"}


def fit_and_score(df: pd.DataFrame, X: pd.DataFrame, fit_mask, score_mask, campaign: str,
                  model_specs: dict[str, tuple[str, dict]], outcome: str = config.PRIMARY_OUTCOME,
                  seed: int = config.SEED, include_response: bool = True,
                  verbose: bool = False) -> pd.DataFrame:
    """Fit models for one campaign on ``fit_mask`` and score ``score_mask``.

    Training uses (campaign arm + control) rows only. Scoring covers every
    ``score_mask`` row, all three arms included, so one customer can be
    compared across campaigns. Returns one column per model.
    """
    label = config.CAMPAIGNS[campaign]
    fit_mask = np.asarray(fit_mask) & df["segment"].isin([label, config.CONTROL]).to_numpy()
    Xf = X.loc[fit_mask]
    t = (df.loc[fit_mask, "segment"] == label).astype(int).to_numpy()
    Xs = X.loc[np.asarray(score_mask)]
    e = float(t.mean())
    out = pd.DataFrame(index=Xs.index)

    start = time.time()
    if include_response:
        resp = models.ResponseModel(seed=seed, binary=True).fit(
            Xf, t, df.loc[fit_mask, "conversion"].to_numpy())
        out["Purchase propensity"] = resp.predict(Xs)
    y = df.loc[fit_mask, outcome].astype(float).to_numpy()
    for name, (key, params) in model_specs.items():
        params = dict(params)
        if key in PROPENSITY_AWARE:
            params.setdefault("propensity", e)
        m = models.make_model(key, seed=seed, **params).fit(Xf, t, y)
        out[name] = m.predict(Xs)
        if verbose:
            print(f"  {campaign:6s} {name:22s} fitted ({time.time() - start:5.1f}s)")
    return out


def evaluation_scores(df: pd.DataFrame, X: pd.DataFrame, mask, outcome: str = config.PRIMARY_OUTCOME,
                      seed: int = config.SEED) -> pd.DataFrame:
    """Cross-fitted AIPW scores on one evaluation partition, columns = campaign keys."""
    mask = np.asarray(mask)
    shares = data.arm_shares(df)
    g = evaluation.aipw_scores(X.loc[mask], df.loc[mask, "segment"], df.loc[mask, outcome],
                               config.ARMS, config.CONTROL, shares, seed=seed)
    return g.rename(columns={v: k for k, v in config.CAMPAIGNS.items()})[list(config.CAMPAIGNS)]


def cross_fit(df: pd.DataFrame, X: pd.DataFrame, folds: pd.Series, campaign: str,
              model_specs: dict[str, tuple[str, dict]], outcome: str = config.PRIMARY_OUTCOME,
              seed: int = config.SEED, include_response: bool = True,
              verbose: bool = False) -> pd.DataFrame:
    """Out-of-fold scores for every development row (fold id >= 0)."""
    parts = []
    for k in sorted(f for f in folds.unique() if f >= 0):
        fit = ((folds >= 0) & (folds != k)).to_numpy()
        score = (folds == k).to_numpy()
        parts.append(fit_and_score(df, X, fit, score, campaign, model_specs, outcome,
                                   seed + k, include_response, verbose=False))
        if verbose:
            print(f"  {campaign}: fold {k} done")
    return pd.concat(parts).sort_index()


def cross_fit_gamma(df: pd.DataFrame, X: pd.DataFrame, folds: pd.Series,
                    outcome: str = config.PRIMARY_OUTCOME, seed: int = config.SEED) -> pd.DataFrame:
    """AIPW scores for development rows, each computed within its own fold."""
    parts = [evaluation_scores(df, X, (folds == k).to_numpy(), outcome, seed + k)
             for k in sorted(f for f in folds.unique() if f >= 0)]
    return pd.concat(parts).sort_index()


def summarize_rankings(scores: pd.DataFrame, gamma, coverage: float = 0.3) -> pd.DataFrame:
    """AUGC, gain at a coverage, paired difference vs. random, and BLP slope."""
    gamma = np.asarray(gamma, float)
    rows = []
    for name in scores:
        s = scores[name].to_numpy()
        curve = evaluation.gain_curve(s, gamma)
        mask = evaluation.top_k_mask(s, coverage)
        gain = evaluation.policy_gain(mask, gamma)
        vs_random = evaluation.mean_ci((mask - coverage) * gamma)
        rows.append({
            "model": name,
            "AUGC": evaluation.area_above_random(curve),
            f"gain@{coverage:.0%}": gain["estimate"],
            f"gain@{coverage:.0%}_se": gain["se"],
            "vs_random": vs_random["estimate"], "vs_random_se": vs_random["se"],
            "BLP_slope": evaluation.calibration_slope(s, gamma)["slope"],
        })
    return pd.DataFrame(rows).set_index("model")
