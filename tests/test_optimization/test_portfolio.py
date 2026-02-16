"""Unit tests for portfolio optimization and robust optimization."""
import numpy as np
import pandas as pd
import pytest

from src.optimization.portfolio_model import build_portfolio_model, solve_portfolio
from src.optimization.robust_opt import (
    build_box_uncertainty_set,
    scenario_analysis,
    worst_case_expected_loss,
)


@pytest.fixture
def small_portfolio():
    """Create a small portfolio problem (10 loans)."""
    rng = np.random.RandomState(42)
    n = 10
    loans = pd.DataFrame({
        "loan_amnt": rng.uniform(5000, 30000, n),
        "purpose": rng.choice(["debt", "credit", "home"], n),
    })
    pd_point = rng.uniform(0.02, 0.15, n)
    pd_low = np.clip(pd_point - 0.05, 0, 1)
    pd_high = np.clip(pd_point + 0.05, 0, 1)
    lgd = np.full(n, 0.45)
    int_rates = rng.uniform(0.05, 0.25, n)
    return loans, pd_point, pd_low, pd_high, lgd, int_rates


# ── build_portfolio_model ──


def test_build_model_creates_pyomo_model(small_portfolio):
    loans, pd_point, pd_low, pd_high, lgd, int_rates = small_portfolio
    model = build_portfolio_model(
        loans, pd_point, pd_low, pd_high, lgd, int_rates,
        total_budget=100_000,
    )
    assert hasattr(model, "x")
    assert hasattr(model, "obj")
    assert hasattr(model, "budget")
    assert hasattr(model, "pd_cap")


def test_build_model_robust_vs_nonrobust(small_portfolio):
    loans, pd_point, pd_low, pd_high, lgd, int_rates = small_portfolio
    model_robust = build_portfolio_model(
        loans, pd_point, pd_low, pd_high, lgd, int_rates, robust=True,
    )
    model_nonrobust = build_portfolio_model(
        loans, pd_point, pd_low, pd_high, lgd, int_rates, robust=False,
    )
    # Both should build successfully
    assert model_robust is not None
    assert model_nonrobust is not None


# ── solve_portfolio ──


def test_solve_portfolio_returns_solution(small_portfolio):
    loans, pd_point, pd_low, pd_high, lgd, int_rates = small_portfolio
    model = build_portfolio_model(
        loans, pd_point, pd_low, pd_high, lgd, int_rates,
        total_budget=500_000, max_portfolio_pd=0.20,
    )
    solution = solve_portfolio(model, time_limit=30)
    assert "allocation" in solution
    assert "objective_value" in solution
    assert "n_funded" in solution
    assert "total_allocated" in solution
    assert "solver_status" in solution


def test_solve_portfolio_budget_constraint(small_portfolio):
    loans, pd_point, pd_low, pd_high, lgd, int_rates = small_portfolio
    budget = 50_000
    model = build_portfolio_model(
        loans, pd_point, pd_low, pd_high, lgd, int_rates,
        total_budget=budget, max_portfolio_pd=0.20,
    )
    solution = solve_portfolio(model, time_limit=30)
    assert solution["total_allocated"] <= budget + 1  # Small tolerance


def test_solve_portfolio_allocations_bounded(small_portfolio):
    loans, pd_point, pd_low, pd_high, lgd, int_rates = small_portfolio
    model = build_portfolio_model(
        loans, pd_point, pd_low, pd_high, lgd, int_rates,
        total_budget=500_000, max_portfolio_pd=0.20,
    )
    solution = solve_portfolio(model, time_limit=30)
    for i, alloc in solution["allocation"].items():
        assert -0.001 <= alloc <= 1.001, f"Allocation {i} out of bounds: {alloc}"


def test_solve_portfolio_objective_finite(small_portfolio):
    loans, pd_point, pd_low, pd_high, lgd, int_rates = small_portfolio
    model = build_portfolio_model(
        loans, pd_point, pd_low, pd_high, lgd, int_rates,
        total_budget=500_000, max_portfolio_pd=0.20,
    )
    solution = solve_portfolio(model, time_limit=30)
    assert np.isfinite(solution["objective_value"])


# ── build_box_uncertainty_set ──


def test_box_uncertainty_set_keys():
    pd_low = np.array([0.01, 0.05])
    pd_high = np.array([0.10, 0.15])
    result = build_box_uncertainty_set(pd_low, pd_high)
    assert "pd_low" in result
    assert "pd_high" in result
    assert "pd_center" in result
    assert "pd_radius" in result


def test_box_uncertainty_set_center():
    pd_low = np.array([0.0, 0.2])
    pd_high = np.array([0.4, 0.6])
    result = build_box_uncertainty_set(pd_low, pd_high)
    np.testing.assert_array_almost_equal(result["pd_center"], [0.2, 0.4])
    np.testing.assert_array_almost_equal(result["pd_radius"], [0.2, 0.2])


def test_box_uncertainty_set_with_lgd():
    pd_low = np.array([0.05])
    pd_high = np.array([0.15])
    lgd_low = np.array([0.30])
    lgd_high = np.array([0.60])
    result = build_box_uncertainty_set(pd_low, pd_high, lgd_low, lgd_high)
    assert "lgd_low" in result
    assert "lgd_center" in result


# ── worst_case_expected_loss ──


def test_worst_case_loss_uses_upper_bounds():
    allocation = np.array([1.0, 1.0])
    loan_amounts = np.array([10000, 20000])
    pd_high = np.array([0.10, 0.20])
    loss = worst_case_expected_loss(allocation, loan_amounts, pd_high)
    # loss = 1*10000*0.10*0.45 + 1*20000*0.20*0.45 = 450 + 1800 = 2250
    assert loss == pytest.approx(2250, abs=1)


def test_worst_case_loss_zero_allocation():
    allocation = np.array([0.0, 0.0])
    loan_amounts = np.array([10000, 20000])
    pd_high = np.array([0.10, 0.20])
    loss = worst_case_expected_loss(allocation, loan_amounts, pd_high)
    assert loss == 0.0


# ── scenario_analysis ──


def test_scenario_analysis_ordering():
    allocation = np.array([1.0, 1.0])
    loan_amounts = np.array([10000, 20000])
    pd_low = np.array([0.05, 0.10])
    pd_point = np.array([0.10, 0.15])
    pd_high = np.array([0.15, 0.20])
    lgd = np.array([0.45, 0.45])
    result = scenario_analysis(allocation, loan_amounts, pd_low, pd_point, pd_high, lgd)
    assert result["best_case"].iloc[0] <= result["expected"].iloc[0]
    assert result["expected"].iloc[0] <= result["worst_case"].iloc[0]
    assert result["range"].iloc[0] > 0
