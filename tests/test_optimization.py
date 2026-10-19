import numpy as np
import pytest

from src import optimization as opt


def test_net_value():
    assert np.allclose(opt.net_value([1.0, 0.0], 0.4, 0.1), [0.3, -0.1])


@pytest.mark.parametrize("budget", [0.0, 0.35, 1.0, 5.0])
def test_budget_never_exceeded(budget):
    rng = np.random.default_rng(0)
    net = rng.normal(0.05, 0.2, (200, 2))
    a = opt.allocate(net, 0.1, budget)
    assert (a >= 0).sum() * 0.1 <= budget + 1e-9


def test_zero_budget_contacts_nobody():
    a = opt.allocate(np.ones(10), 0.1, 0.0)
    assert (a == -1).all()


def test_negative_profit_customers_excluded():
    a = opt.allocate(np.array([0.5, -0.2, 0.1]), 0.1, None)
    assert list(a) == [0, -1, 0]


def test_invalid_inputs():
    with pytest.raises(ValueError):
        opt.allocate(np.ones(3), 0.0, 1.0)
    with pytest.raises(ValueError):
        opt.allocate(np.ones(3), 0.1, -1.0)
    with pytest.raises(ValueError):
        opt.budget_per_customer(10, 0)


def test_greedy_matches_milp_with_equal_costs():
    rng = np.random.default_rng(3)
    net = rng.normal(0.0, 0.2, (60, 2))
    greedy = opt.allocate(net, 0.1, 2.0)
    exact = opt.solve_milp(net, 0.1, 2.0)
    value = lambda a: sum(net[i, k] for i, k in enumerate(a) if k >= 0)
    assert np.isclose(value(greedy), value(exact))


def test_evaluate_allocation_accounting():
    gamma = np.array([[2.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    r = opt.evaluate_allocation(np.array([0, 1, -1]), gamma, margin=0.5, cost=0.1)
    assert np.isclose(r["profit"], ((0.5 * 2 - 0.1) + (0.5 * 1 - 0.1)) / 3)
    assert np.isclose(r["coverage"], 2 / 3)
