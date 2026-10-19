"""Campaign economics and budget-constrained allocation.

Economic model (all inputs are user-supplied assumptions):

    net_i,k = margin * tau_hat_i,k - cost_k

``tau_hat`` is estimated incremental *spend* in dollars, so ``margin * tau``
is incremental contribution and cost is subtracted exactly once. The
allocation problem is

    max  sum_i sum_k a_ik * net_ik
    s.t. sum_i sum_k a_ik * cost_k <= B,   sum_k a_ik <= 1,   a_ik in {0, 1}
"""
from __future__ import annotations

import numpy as np

from .evaluation import mean_ci


def _check_inputs(cost, budget):
    cost = np.asarray(cost, float)
    if np.any(cost <= 0):
        raise ValueError("cost per contact must be positive")
    if budget is not None and (not np.isfinite(budget) or budget < 0):
        raise ValueError("budget must be a non-negative finite number")
    return cost


def net_value(tau, margin: float, cost):
    """Expected incremental profit of contacting each customer."""
    if margin < 0:
        raise ValueError("margin must be non-negative")
    return margin * np.asarray(tau, float) - np.asarray(cost, float)


def allocate(net, cost, budget: float | None, positive_only: bool = True) -> np.ndarray:
    """Multi-arm greedy allocation.

    ``net`` is an (n, K) matrix of expected incremental profit per arm (or a
    1-D vector for a single campaign). Returns an int array with the chosen arm
    index per customer, -1 for "no contact".

    Each customer is offered their best arm; customers are then admitted in
    order of net value per dollar until the budget is exhausted. With equal
    arm costs (the case here) this is exactly optimal; with unequal costs it is
    the standard ratio heuristic and ``solve_milp`` gives the exact optimum.
    """
    net = np.asarray(net, float)
    if net.ndim == 1:
        net = net[:, None]
    n, K = net.shape
    cost = _check_inputs(np.broadcast_to(cost, (K,)), budget)
    best = net.argmax(axis=1)
    best_net = net[np.arange(n), best]
    best_cost = cost[best]
    eligible = best_net > 0 if positive_only else np.ones(n, bool)
    order = np.argsort(-(best_net / best_cost), kind="stable")
    order = order[eligible[order]]
    assign = np.full(n, -1)
    if budget is None:
        assign[order] = best[order]
        return assign
    spent = np.cumsum(best_cost[order])
    chosen = order[spent <= budget + 1e-9]
    assign[chosen] = best[chosen]
    return assign


def allocate_top_fraction(score, frac: float, arm: int = 0) -> np.ndarray:
    """Contact the top ``frac`` share by a score (used for baseline policies)."""
    from .evaluation import top_k_mask

    assign = np.full(len(score), -1)
    assign[top_k_mask(score, frac)] = arm
    return assign


def solve_milp(net, cost, budget: float) -> np.ndarray:
    """Exact 0/1 solution with SciPy's HiGHS MILP solver (for verification)."""
    from scipy import sparse
    from scipy.optimize import Bounds, LinearConstraint, milp

    net = np.asarray(net, float)
    if net.ndim == 1:
        net = net[:, None]
    n, K = net.shape
    cost = _check_inputs(np.broadcast_to(cost, (K,)), budget)
    c = -net.ravel()  # minimise negative profit
    budget_row = np.tile(cost, n)[None, :]
    one_arm = sparse.kron(sparse.eye(n), np.ones((1, K)), format="csr")
    cons = [LinearConstraint(budget_row, -np.inf, budget),
            LinearConstraint(one_arm, 0, 1)]
    res = milp(c, constraints=cons, integrality=np.ones(n * K), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x.reshape(n, K) > 0.5
    return np.where(x.any(axis=1), x.argmax(axis=1), -1)


def evaluate_allocation(assign, gamma, margin: float, cost, alpha: float = 0.05) -> dict:
    """Held-out incremental profit of an allocation, per population customer.

    ``gamma`` is the (n, K) matrix of AIPW scores in the same arm order as the
    allocation. Profit_i = sum_k 1[a_i = k] (margin * Gamma_ik - cost_k).
    """
    gamma = np.asarray(gamma, float)
    if gamma.ndim == 1:
        gamma = gamma[:, None]
    n, K = gamma.shape
    cost = np.broadcast_to(np.asarray(cost, float), (K,))
    assign = np.asarray(assign)
    z = profit_contributions(assign, gamma, margin, cost)
    inc = np.zeros(n)
    for k in range(K):
        inc[assign == k] = gamma[assign == k, k]
    contacted = assign >= 0
    profit = mean_ci(z, alpha)
    spend = mean_ci(inc, alpha)
    return {
        "coverage": contacted.mean(),
        "cost_per_customer": float(np.where(contacted, cost[np.clip(assign, 0, K - 1)], 0).mean()),
        "inc_spend": spend["estimate"], "inc_spend_ci_low": spend["ci_low"],
        "inc_spend_ci_high": spend["ci_high"],
        "profit": profit["estimate"], "profit_se": profit["se"],
        "profit_ci_low": profit["ci_low"], "profit_ci_high": profit["ci_high"],
    }


def profit_contributions(assign, gamma, margin: float, cost) -> np.ndarray:
    """Per-customer AIPW profit contribution z_i of an allocation."""
    gamma = np.asarray(gamma, float)
    if gamma.ndim == 1:
        gamma = gamma[:, None]
    K = gamma.shape[1]
    cost = np.broadcast_to(np.asarray(cost, float), (K,))
    assign = np.asarray(assign)
    z = np.zeros(len(assign))
    for k in range(K):
        m = assign == k
        z[m] = margin * gamma[m, k] - cost[k]
    return z


def compare_allocations(assign_a, assign_b, gamma, margin: float, cost, alpha: float = 0.05) -> dict:
    """Paired difference in held-out profit (A - B) per population customer."""
    za = profit_contributions(assign_a, gamma, margin, cost)
    zb = profit_contributions(assign_b, gamma, margin, cost)
    return mean_ci(za - zb, alpha)


def budget_per_customer(budget: float, population: int) -> float:
    """Convert a campaign budget for ``population`` customers to a per-customer cap."""
    if population <= 0:
        raise ValueError("population must be positive")
    _check_inputs(1.0, budget)
    return budget / population
