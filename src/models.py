"""Response (purchase-propensity) and uplift / CATE estimators.

All uplift models share one interface::

    model.fit(X, t, y)      # t in {0, 1}, randomized
    model.predict(X)        # estimated CATE  tau(x) = E[Y(1) - Y(0) | X = x]

Estimated CATEs are model predictions, not observed individual effects.

Terminology: "purchase propensity" means P(purchase | X), the quantity a
conventional targeting model scores. It is unrelated to the *assignment*
probability e = P(T = 1), which is known here because treatment was randomized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold


def base_regressor(seed: int = 0, **overrides) -> HistGradientBoostingRegressor:
    params = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                  min_samples_leaf=200, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.15,
                  n_iter_no_change=20, random_state=seed)
    params.update(overrides)
    return HistGradientBoostingRegressor(**params)


def base_classifier(seed: int = 0, **overrides) -> HistGradientBoostingClassifier:
    params = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                  min_samples_leaf=200, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.15,
                  n_iter_no_change=20, random_state=seed)
    params.update(overrides)
    return HistGradientBoostingClassifier(**params)


def _arr(X):
    return X.to_numpy(float) if isinstance(X, pd.DataFrame) else np.asarray(X, float)


class ResponseModel:
    """Conventional targeting: score = predicted outcome *if emailed*.

    Fit on treated customers only, as a marketer would model "who responds to
    our emails". For a binary outcome it returns P(Y = 1 | X, T = 1).
    """

    name = "Purchase propensity"

    def __init__(self, seed: int = 0, binary: bool = True, **params):
        self.binary = binary
        self.model = base_classifier(seed, **params) if binary else base_regressor(seed, **params)

    def fit(self, X, t, y):
        t = np.asarray(t).astype(bool)
        Xa = _arr(X)
        self.model.fit(Xa[t], np.asarray(y)[t])
        return self

    def predict(self, X):
        Xa = _arr(X)
        if self.binary:
            return self.model.predict_proba(Xa)[:, 1]
        return self.model.predict(Xa)


class SLearner:
    """Single outcome model with treatment as a feature; tau = mu(x,1) - mu(x,0)."""

    name = "S-learner"

    def __init__(self, seed: int = 0, **params):
        self.model = base_regressor(seed, **params)

    def fit(self, X, t, y):
        Xa = _arr(X)
        self.model.fit(np.column_stack([Xa, t]), y)
        return self

    def predict(self, X):
        Xa = _arr(X)
        ones, zeros = np.ones(len(Xa)), np.zeros(len(Xa))
        return (self.model.predict(np.column_stack([Xa, ones]))
                - self.model.predict(np.column_stack([Xa, zeros])))


class TLearner:
    """Separate outcome models per arm; tau = mu1(x) - mu0(x)."""

    name = "T-learner"

    def __init__(self, seed: int = 0, **params):
        self.mu0 = base_regressor(seed, **params)
        self.mu1 = base_regressor(seed + 1, **params)

    def fit(self, X, t, y):
        t = np.asarray(t).astype(bool)
        Xa, y = _arr(X), np.asarray(y, float)
        self.mu0.fit(Xa[~t], y[~t])
        self.mu1.fit(Xa[t], y[t])
        return self

    def predict_outcomes(self, X):
        Xa = _arr(X)
        return self.mu0.predict(Xa), self.mu1.predict(Xa)

    def predict(self, X):
        m0, m1 = self.predict_outcomes(X)
        return m1 - m0


class XLearner:
    """Kunzel et al. (2019) X-learner with a known assignment probability.

    Stage 1: mu0, mu1 as in the T-learner.
    Stage 2: imputed effects D1 = Y1 - mu0(X1) and D0 = mu1(X0) - Y0, each
             regressed on X to give tau1, tau0.
    Stage 3: tau(x) = e * tau0(x) + (1 - e) * tau1(x), with e = P(T = 1).
    """

    name = "X-learner"

    def __init__(self, seed: int = 0, propensity: float | None = None,
                 effect_params: dict | None = None, **params):
        self.propensity = propensity
        self.mu0 = base_regressor(seed, **params)
        self.mu1 = base_regressor(seed + 1, **params)
        eff = {**params, **(effect_params or {})}
        self.tau0 = base_regressor(seed + 2, **eff)
        self.tau1 = base_regressor(seed + 3, **eff)

    def fit(self, X, t, y):
        t = np.asarray(t).astype(bool)
        Xa, y = _arr(X), np.asarray(y, float)
        self.e_ = t.mean() if self.propensity is None else self.propensity
        self.mu0.fit(Xa[~t], y[~t])
        self.mu1.fit(Xa[t], y[t])
        d1 = y[t] - self.mu0.predict(Xa[t])
        d0 = self.mu1.predict(Xa[~t]) - y[~t]
        self.tau1.fit(Xa[t], d1)
        self.tau0.fit(Xa[~t], d0)
        return self

    def predict(self, X):
        Xa = _arr(X)
        return self.e_ * self.tau0.predict(Xa) + (1 - self.e_) * self.tau1.predict(Xa)


class DRLearner:
    """Doubly robust learner: regress cross-fitted AIPW pseudo-outcomes on X.

    Gamma_i = mu1(X) - mu0(X) + T (Y - mu1(X)) / e - (1 - T)(Y - mu0(X)) / (1 - e)
    With e known by design, E[Gamma | X] = tau(X) even if mu is misspecified.
    """

    name = "DR-learner"

    def __init__(self, seed: int = 0, propensity: float | None = None, n_folds: int = 3,
                 final_params: dict | None = None, **params):
        self.seed, self.propensity, self.n_folds = seed, propensity, n_folds
        self.params = params
        self.final = base_regressor(seed + 5, **{"min_samples_leaf": 400, **(final_params or {})})

    def pseudo_outcomes(self, X, t, y):
        Xa, t, y = _arr(X), np.asarray(t).astype(int), np.asarray(y, float)
        e = t.mean() if self.propensity is None else self.propensity
        mu0, mu1 = np.zeros(len(y)), np.zeros(len(y))
        for k, (tr, te) in enumerate(KFold(self.n_folds, shuffle=True,
                                           random_state=self.seed).split(Xa)):
            tl = TLearner(self.seed + 10 * k, **self.params).fit(Xa[tr], t[tr], y[tr])
            mu0[te], mu1[te] = tl.predict_outcomes(Xa[te])
        return mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)

    def fit(self, X, t, y):
        self.final.fit(_arr(X), self.pseudo_outcomes(X, t, y))
        return self

    def predict(self, X):
        return self.final.predict(_arr(X))


class CalibratedPropensity:
    """Rank by purchase propensity, size the effect with a causal calibration.

    Motivated by notebook 02, where the purchase-propensity score was a strong
    *ranking* of spend uplift for the Women's e-mail. The model is a
    two-parameter "best linear predictor" of the CATE on a single index:

        tau(x) = a + b * p(x),   p(x) = P(purchase | X = x, emailed)

    ``a`` and ``b`` are fitted by OLS of cross-fitted DR pseudo-outcomes on
    out-of-fold p(x), so neither stage sees its own training labels. The
    ranking equals the propensity ranking whenever b > 0, but the output is in
    outcome units, which profit thresholds and arm comparisons require.
    Low-propensity predictions can fall below zero because the calibration is linear.
    Notebook 03 finds (a, b) too unstable across training samples on this data.
    """

    name = "Calibrated propensity"

    def __init__(self, seed: int = 0, propensity: float | None = None, n_folds: int = 3, **params):
        self.seed, self.propensity, self.n_folds, self.params = seed, propensity, n_folds, params

    def fit(self, X, t, y):
        Xa, t, y = _arr(X), np.asarray(t).astype(int), np.asarray(y, float)
        buy = (y > 0).astype(int)
        p_oof = np.zeros(len(y))
        for k, (tr, te) in enumerate(KFold(self.n_folds, shuffle=True,
                                           random_state=self.seed).split(Xa)):
            m = ResponseModel(self.seed + k, binary=True, **self.params).fit(Xa[tr], t[tr], buy[tr])
            p_oof[te] = m.predict(Xa[te])
        gamma = DRLearner(self.seed, self.propensity, self.n_folds).pseudo_outcomes(Xa, t, y)
        self.b_, self.a_ = np.polyfit(p_oof, gamma, 1)
        self.response = ResponseModel(self.seed, binary=True, **self.params).fit(Xa, t, buy)
        return self

    def predict(self, X):
        return self.a_ + self.b_ * self.response.predict(_arr(X))


class CausalForest:
    """Wrapper around EconML's ``CausalForestDML`` (generalized random forest).

    Uses honest splitting and residualisation (double ML). The treatment model
    is a constant (prior) classifier because assignment was randomized, so
    residualising T only removes its known mean.
    """

    name = "Causal forest"

    def __init__(self, seed: int = 0, n_estimators: int = 500, min_samples_leaf: int = 50,
                 max_depth: int | None = None, max_samples: float = 0.45, cv: int = 3,
                 **outcome_params):
        from econml.dml import CausalForestDML
        from sklearn.dummy import DummyClassifier

        self.est = CausalForestDML(
            model_y=base_regressor(seed, **outcome_params),
            model_t=DummyClassifier(strategy="prior"),
            discrete_treatment=True,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_samples=max_samples,
            honest=True,
            inference=True,
            cv=cv,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, X, t, y):
        self.est.fit(np.asarray(y, float), np.asarray(t).astype(int), X=_arr(X))
        return self

    def predict(self, X):
        return self.est.effect(_arr(X)).ravel()

    def predict_interval(self, X, alpha: float = 0.05):
        lo, hi = self.est.effect_interval(_arr(X), alpha=alpha)
        return lo.ravel(), hi.ravel()

    def feature_importances(self, names):
        return pd.Series(self.est.feature_importances_, index=names).sort_values(ascending=False)


MODEL_REGISTRY = {
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
    "dr_learner": DRLearner,
    "calibrated_propensity": CalibratedPropensity,
    "causal_forest": CausalForest,
}


def make_model(key: str, seed: int = 0, **params):
    return MODEL_REGISTRY[key](seed=seed, **params)
