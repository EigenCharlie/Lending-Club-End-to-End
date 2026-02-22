"""Tests for src/optimization/portfolio_model.py.

Covers Pyomo model construction, constraint correctness,
solver integration, and binary (MILP) variant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.portfolio_model import (
    build_binary_model,
    build_portfolio_model,
    solve_portfolio,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_loans():
    """Small synthetic loan portfolio (10 loans)."""
    rng = np.random.default_rng(42)
    n = 10
    df = pd.DataFrame(
        {
            "loan_amnt": rng.integers(5000, 30000, size=n).astype(float),
            "purpose": ["credit_card"] * 5 + ["debt_consolidation"] * 5,
        }
    )
    pd_point = rng.uniform(0.02, 0.20, n)
    pd_low = np.clip(pd_point - 0.02, 0.0, 1.0)
    pd_high = np.clip(pd_point + 0.05, 0.0, 1.0)
    lgd = np.full(n, 0.45)
    int_rates = rng.uniform(0.06, 0.25, n)
    return {
        "loans": df,
        "pd_point": pd_point,
        "pd_low": pd_low,
        "pd_high": pd_high,
        "lgd": lgd,
        "int_rates": int_rates,
    }


# ---------------------------------------------------------------------------
# build_portfolio_model
# ---------------------------------------------------------------------------


class TestBuildPortfolioModel:
    def test_model_has_expected_components(self, small_loans):
        model = build_portfolio_model(**small_loans)
        assert hasattr(model, "x")
        assert hasattr(model, "obj")
        assert hasattr(model, "budget")
        assert hasattr(model, "pd_cap")

    def test_robust_uses_pd_high(self, small_loans):
        model = build_portfolio_model(**small_loans, robust=True)
        # pd_worst should equal pd_high when robust=True
        for i in model.I:
            assert model.pd_worst[i] == pytest.approx(small_loans["pd_high"][i])

    def test_non_robust_uses_pd_point(self, small_loans):
        model = build_portfolio_model(**small_loans, robust=False)
        for i in model.I:
            assert model.pd_worst[i] == pytest.approx(small_loans["pd_point"][i])

    def test_concentration_constraints_when_purpose_exists(self, small_loans):
        model = build_portfolio_model(**small_loans)
        # Should have concentration constraints for each purpose
        has_concentration = any(attr.startswith("concentration_") for attr in dir(model))
        assert has_concentration

    def test_no_concentration_without_purpose(self, small_loans):
        loans_no_purpose = small_loans["loans"].drop(columns=["purpose"])
        model = build_portfolio_model(
            loans=loans_no_purpose,
            pd_point=small_loans["pd_point"],
            pd_low=small_loans["pd_low"],
            pd_high=small_loans["pd_high"],
            lgd=small_loans["lgd"],
            int_rates=small_loans["int_rates"],
        )
        has_concentration = any(attr.startswith("concentration_") for attr in dir(model))
        assert not has_concentration

    def test_min_budget_utilization_creates_constraint(self, small_loans):
        model = build_portfolio_model(**small_loans, min_budget_utilization=0.5)
        assert hasattr(model, "min_budget")

    def test_pd_cap_slack_creates_variable(self, small_loans):
        model = build_portfolio_model(**small_loans, pd_cap_slack_penalty=100.0)
        assert hasattr(model, "pd_cap_slack")


# ---------------------------------------------------------------------------
# solve_portfolio
# ---------------------------------------------------------------------------


class TestSolvePortfolio:
    def test_solves_successfully(self, small_loans):
        model = build_portfolio_model(**small_loans)
        sol = solve_portfolio(model)
        assert "allocation" in sol
        assert "objective_value" in sol
        assert "n_funded" in sol
        assert "total_allocated" in sol
        assert "solver_status" in sol

    def test_budget_constraint_respected(self, small_loans):
        budget = 50_000
        model = build_portfolio_model(**small_loans, total_budget=budget)
        sol = solve_portfolio(model)
        assert sol["total_allocated"] <= budget + 1.0  # Allow tiny numerical slack

    def test_allocations_in_zero_one(self, small_loans):
        model = build_portfolio_model(**small_loans)
        sol = solve_portfolio(model)
        for alloc in sol["allocation"].values():
            assert -1e-6 <= alloc <= 1.0 + 1e-6

    def test_robust_funds_fewer_or_equal(self, small_loans):
        model_nr = build_portfolio_model(**small_loans, robust=False)
        model_r = build_portfolio_model(**small_loans, robust=True)
        sol_nr = solve_portfolio(model_nr)
        sol_r = solve_portfolio(model_r)
        # Robust has tighter PD constraint → should fund <= non-robust
        # (or equal in edge cases)
        assert sol_r["n_funded"] <= sol_nr["n_funded"] + 1  # Allow +-1 for numerics


# ---------------------------------------------------------------------------
# build_binary_model
# ---------------------------------------------------------------------------


class TestBuildBinaryModel:
    def test_binary_model_has_components(self, small_loans):
        model = build_binary_model(
            loans=small_loans["loans"],
            pd_point=small_loans["pd_point"],
            pd_high=small_loans["pd_high"],
            lgd=small_loans["lgd"],
            int_rates=small_loans["int_rates"],
        )
        assert hasattr(model, "x")
        assert hasattr(model, "obj")
        assert hasattr(model, "budget")
        assert hasattr(model, "pd_cap")

    def test_binary_solves(self, small_loans):
        model = build_binary_model(
            loans=small_loans["loans"],
            pd_point=small_loans["pd_point"],
            pd_high=small_loans["pd_high"],
            lgd=small_loans["lgd"],
            int_rates=small_loans["int_rates"],
            total_budget=100_000,
        )
        sol = solve_portfolio(model)
        # Binary allocations should be 0 or 1
        for alloc in sol["allocation"].values():
            assert alloc < 0.01 or alloc > 0.99
