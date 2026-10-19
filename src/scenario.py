"""Policy scenarios on the frozen held-out scores (used by notebook 04/05 and the dashboard).

Nothing here retrains a model. A scenario changes only the user-supplied economics
(margin, cost, budget) and the campaign option, then re-runs the allocation on
fixed test-set scores and re-evaluates it with fixed test-set AIPW scores.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, evaluation, optimization as opt

CAMPAIGN_OPTIONS = {
    "womens": "Women's e-mail only",
    "mens": "Men's e-mail only",
    "best": "Best campaign per customer",
}
ARM_ORDER = ["mens", "womens"]  # column order of score / gamma matrices
LABELS = {
    "observed": "Observed experimental estimate",
    "heldout": "Held-out policy-value estimate",
    "model": "Model prediction",
    "assumption": "User-supplied business assumption",
    "simulated": "Simulated scenario",
}


def load_artifacts(path=None):
    path = path or config.ARTIFACTS_DIR
    frame = pd.read_csv(path / "test_scores.csv", index_col=0)
    with open(path / "test_metadata.json") as f:
        meta = json.load(f)
    return frame, meta


def score_matrix(frame: pd.DataFrame, model: str) -> np.ndarray:
    return frame[[f"{model}|{a}" for a in ARM_ORDER]].to_numpy(float)


def gamma_matrix(frame: pd.DataFrame, outcome: str = "spend") -> np.ndarray:
    return frame[[f"gamma_{outcome}|{a}" for a in ARM_ORDER]].to_numpy(float)


def _arms(option: str) -> list[int]:
    return [0, 1] if option == "best" else [ARM_ORDER.index(option)]


def contacts_cap(budget: float, cost: float, n: int, population: int) -> float:
    """Total budget scaled to the evaluation sample of ``n`` customers."""
    return opt.budget_per_customer(budget, population) * n


def uplift_policy(frame, model, option, margin, cost, budget, population, positive_only=True):
    """Allocation from a dollar-valued CATE model: net = margin * tau - cost."""
    tau = score_matrix(frame, model)
    net = np.full_like(tau, -np.inf)
    arms = _arms(option)
    net[:, arms] = opt.net_value(tau[:, arms], margin, cost)
    B = None if budget is None else contacts_cap(budget, cost, len(frame), population)
    return opt.allocate(net, cost, B, positive_only=positive_only)


def ranking_policy(frame, model, option, cost, budget, population, ate_by_arm, coverage=None):
    """Fill the budget by ranking on a non-dollar score (e.g. purchase propensity).

    A ranking cannot say whether a contact is worth its cost or which campaign is
    better for a customer, so it spends the whole budget on the campaign with the
    larger average effect.
    """
    arm = _arms(option)[0] if option != "best" else int(np.argmax([ate_by_arm[a] for a in ARM_ORDER]))
    n = len(frame)
    if coverage is None:
        coverage = 1.0 if budget is None else min(1.0, contacts_cap(budget, cost, n, population) / (cost * n))
    s = frame[f"{model}|{ARM_ORDER[arm]}"].to_numpy(float)
    return opt.allocate_top_fraction(s, coverage, arm)


def random_policy_value(gamma, option, margin, cost, coverage, ate_by_arm):
    arm = _arms(option)[0] if option != "best" else int(np.argmax([ate_by_arm[a] for a in ARM_ORDER]))
    z = coverage * (margin * gamma[:, arm] - cost)
    r = evaluation.mean_ci(z)
    inc = evaluation.mean_ci(coverage * gamma[:, arm])
    return {"coverage": coverage, "mens_share": coverage if arm == 0 else 0.0,
            "cost_per_customer": coverage * cost, "inc_spend": inc["estimate"],
            "inc_spend_ci_low": inc["ci_low"], "inc_spend_ci_high": inc["ci_high"],
            "profit": r["estimate"], "profit_se": r["se"], "profit_ci_low": r["ci_low"],
            "profit_ci_high": r["ci_high"]}


def run_scenario(frame, meta, option="womens", margin=config.MARGIN, cost=config.COST_PER_CONTACT,
                 budget=config.DEFAULT_BUDGET, population=config.POPULATION_SIZE) -> pd.DataFrame:
    """Evaluate every policy under one set of assumptions. Values are per ``population`` customers."""
    gamma = gamma_matrix(frame)
    # Campaign choice for non-dollar baselines uses development-set ATEs, never test outcomes.
    ate = meta["dev_ate_spend"]
    n = len(frame)
    cap_cov = 1.0 if budget is None else min(1.0, contacts_cap(budget, cost, n, population) / (cost * n))

    rows = []

    def add(name, kind, res, assign=None):
        rows.append({"policy": name, "kind": kind, **res,
                     "mens_share": res.get("mens_share", float(np.mean(assign == 0)) if assign is not None else 0.0)})

    add("No campaign", "baseline", {"coverage": 0.0, "cost_per_customer": 0.0, "inc_spend": 0.0,
                                    "inc_spend_ci_low": 0.0, "inc_spend_ci_high": 0.0, "profit": 0.0,
                                    "profit_se": 0.0, "profit_ci_low": 0.0, "profit_ci_high": 0.0})
    add("Random", "baseline", random_policy_value(gamma, option, margin, cost, cap_cov, ate))
    for model in meta["ranking_models"]:
        a = ranking_policy(frame, model, option, cost, budget, population, ate)
        add(model, "ranking", opt.evaluate_allocation(a, gamma, margin, cost), a)
    for model in meta["uplift_models"]:
        a = uplift_policy(frame, model, option, margin, cost, budget, population)
        add(model, "uplift", opt.evaluate_allocation(a, gamma, margin, cost), a)

    out = pd.DataFrame(rows).set_index("policy")
    money = ["cost_per_customer", "inc_spend", "inc_spend_ci_low", "inc_spend_ci_high",
             "profit", "profit_se", "profit_ci_low", "profit_ci_high"]
    out[money] = out[money] * population
    out = out.rename(columns={"cost_per_customer": "spend_on_email"})
    out["contacts"] = out["coverage"] * population
    out["roi"] = np.where(out["spend_on_email"] > 0, out["profit"] / out["spend_on_email"], np.nan)
    return out


def budget_curve(frame, meta, option, margin, cost, budgets, population=config.POPULATION_SIZE):
    """Profit of each policy across a grid of budgets."""
    parts = []
    for b in budgets:
        r = run_scenario(frame, meta, option, margin, cost, b, population)
        r["budget"] = b
        parts.append(r.reset_index())
    return pd.concat(parts, ignore_index=True)


def segment_view(frame, meta, variable, model, option, margin, cost, budget,
                 population=config.POPULATION_SIZE) -> pd.DataFrame:
    """Per-segment observed effect, predicted effect and share targeted by the uplift policy."""
    gamma = gamma_matrix(frame)
    assign = uplift_policy(frame, model, option, margin, cost, budget, population)
    tau = score_matrix(frame, model)
    rows = []
    for level, idx in frame.groupby(variable).groups.items():
        m = frame.index.isin(idx)
        row = {"segment": level, "customers": int(m.sum())}
        for i, a in enumerate(ARM_ORDER):
            r = evaluation.mean_ci(gamma[m, i])
            row[f"observed_{a}"], row[f"lo_{a}"], row[f"hi_{a}"] = r["estimate"], r["ci_low"], r["ci_high"]
            row[f"predicted_{a}"] = tau[m, i].mean()
        row["share_targeted"] = float((assign[m] >= 0).mean())
        row["share_mens"] = float((assign[m] == 0).mean())
        row["share_womens"] = float((assign[m] == 1).mean())
        rows.append(row)
    return pd.DataFrame(rows)
