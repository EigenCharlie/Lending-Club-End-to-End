"""Tests for src/optimization/portfolio_model.py.

Covers Pyomo model construction, constraint correctness,
solver integration, and binary (MILP) variant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.optimization.cuopt_adapter as cuopt_adapter
from src.optimization.portfolio_model import (
    build_binary_model,
    build_portfolio_model,
    compute_effective_pd,
    optimize_portfolio_allocation,
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
    def test_compute_effective_pd_supports_blended_uncertainty(self, small_loans):
        effective = compute_effective_pd(
            small_loans["pd_point"],
            small_loans["pd_high"],
            policy_mode="blended_uncertainty",
            gamma=0.25,
        )
        expected = small_loans["pd_point"] + 0.25 * (
            small_loans["pd_high"] - small_loans["pd_point"]
        )
        assert np.allclose(effective, expected)

    def test_compute_effective_pd_supports_capped_blended_uncertainty(self, small_loans):
        delta = np.clip(small_loans["pd_high"] - small_loans["pd_point"], 0.0, 1.0)
        delta_cap = np.quantile(delta, 0.5)
        effective = compute_effective_pd(
            small_loans["pd_point"],
            small_loans["pd_high"],
            policy_mode="capped_blended_uncertainty",
            gamma=0.5,
            delta_cap_quantile=0.5,
        )
        expected = np.clip(
            small_loans["pd_point"] + 0.5 * np.minimum(delta, delta_cap),
            0.0,
            1.0,
        )
        assert np.allclose(effective, expected)

    def test_compute_effective_pd_supports_tail_blended_uncertainty(self, small_loans):
        delta = np.clip(small_loans["pd_high"] - small_loans["pd_point"], 0.0, 1.0)
        cutoff = np.quantile(delta, 0.9)
        local_delta = np.where(delta >= cutoff, delta, 0.0)
        effective = compute_effective_pd(
            small_loans["pd_point"],
            small_loans["pd_high"],
            policy_mode="tail_blended_uncertainty",
            gamma=0.5,
            tail_focus_quantile=0.9,
        )
        expected = np.clip(small_loans["pd_point"] + 0.5 * local_delta, 0.0, 1.0)
        assert np.allclose(effective, expected)

    def test_compute_effective_pd_supports_segment_tail_blended_uncertainty(self, small_loans):
        labels = np.array(["A|36"] * 5 + ["B|60"] * 5, dtype=object)
        delta = np.clip(small_loans["pd_high"] - small_loans["pd_point"], 0.0, 1.0)
        expected = np.zeros_like(delta)
        for label in np.unique(labels):
            mask = labels == label
            seg_delta = delta[mask]
            cutoff = np.quantile(seg_delta, 0.8)
            expected[mask] = np.where(seg_delta >= cutoff, seg_delta, 0.0)
        effective = compute_effective_pd(
            small_loans["pd_point"],
            small_loans["pd_high"],
            policy_mode="segment_tail_blended_uncertainty",
            gamma=0.5,
            tail_focus_quantile=0.8,
            segment_labels=labels,
            min_segment_size=1,
        )
        expected = np.clip(small_loans["pd_point"] + 0.5 * expected, 0.0, 1.0)
        assert np.allclose(effective, expected)

    def test_compute_effective_pd_supports_segment_relative_tail_blended_uncertainty(
        self, small_loans
    ):
        labels = np.array(["A|36|verified"] * 5 + ["B|60|source"] * 5, dtype=object)
        delta = np.clip(small_loans["pd_high"] - small_loans["pd_point"], 0.0, 1.0)
        rel = delta / np.maximum(small_loans["pd_point"], 1e-4)
        expected = np.zeros_like(delta)
        for label in np.unique(labels):
            mask = labels == label
            seg_delta = delta[mask]
            seg_rel = rel[mask]
            cutoff = np.quantile(seg_rel, 0.8)
            expected[mask] = np.where(seg_rel >= cutoff, seg_delta, 0.0)
        effective = compute_effective_pd(
            small_loans["pd_point"],
            small_loans["pd_high"],
            policy_mode="segment_relative_tail_blended_uncertainty",
            gamma=0.5,
            tail_focus_quantile=0.8,
            segment_labels=labels,
            min_segment_size=1,
        )
        expected = np.clip(small_loans["pd_point"] + 0.5 * expected, 0.0, 1.0)
        assert np.allclose(effective, expected)

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

    def test_invalid_solver_backend_raises(self, small_loans):
        model = build_portfolio_model(**small_loans)
        with pytest.raises(ValueError, match="Unsupported solver_backend"):
            solve_portfolio(model, solver_backend="unknown")

    def test_optimize_portfolio_allocation_dispatches_to_native_cuopt(
        self, small_loans, monkeypatch
    ):
        monkeypatch.setattr(
            cuopt_adapter,
            "solve_portfolio_cuopt_native",
            lambda **kwargs: {
                "allocation": dict.fromkeys(range(len(kwargs["loans"])), 0.0),
                "objective_value": 123.0,
                "n_funded": 0,
                "total_allocated": 0.0,
                "solver_status": "mock-optimal",
                "solver_backend": "cuopt",
                "pd_cap_slack": 0.0,
            },
        )

        sol = optimize_portfolio_allocation(
            solver_backend="cuopt",
            time_limit=30,
            threads=4,
            **small_loans,
        )
        assert sol["solver_backend"] == "cuopt"
        assert sol["objective_value"] == pytest.approx(123.0)


def test_cuopt_native_adapter_accepts_seed_and_presolve(monkeypatch):
    captured: dict[str, object] = {}

    class FakeDataModel:
        def set_csr_constraint_matrix(self, *args, **kwargs):
            return None

        def set_constraint_bounds(self, *args, **kwargs):
            return None

        def set_row_types(self, *args, **kwargs):
            return None

        def set_objective_coefficients(self, *args, **kwargs):
            return None

        def set_maximize(self, *args, **kwargs):
            return None

        def set_variable_lower_bounds(self, *args, **kwargs):
            return None

        def set_variable_upper_bounds(self, *args, **kwargs):
            return None

    class FakeSolverSettings:
        def __init__(self):
            captured["params"] = {}

        def set_parameter(self, name, value):
            captured["params"][name] = value

    class FakeSolution:
        def get_primal_solution(self):
            return np.array([0.25, 0.25, 0.25], dtype=float)

        def get_termination_reason(self):
            return "Optimal"

        def get_primal_objective(self):
            return 42.0

    class FakeLpApi:
        DataModel = FakeDataModel
        SolverSettings = FakeSolverSettings

        @staticmethod
        def Solve(dm, settings):
            _ = (dm, settings)
            return FakeSolution()

    monkeypatch.setattr(cuopt_adapter, "_require_cuopt", lambda: FakeLpApi)

    loans = pd.DataFrame(
        {
            "loan_amnt": [10_000.0, 10_000.0, 10_000.0],
            "purpose": ["credit_card", "debt_consolidation", "credit_card"],
        }
    )
    result = cuopt_adapter.solve_portfolio_cuopt_native(
        loans=loans,
        pd_point=np.array([0.05, 0.06, 0.07], dtype=float),
        pd_high=np.array([0.07, 0.08, 0.09], dtype=float),
        lgd=np.array([0.45, 0.45, 0.45], dtype=float),
        int_rates=np.array([0.12, 0.13, 0.11], dtype=float),
        random_seed=42,
        presolve=1,
    )

    assert result["solver_backend"] == "cuopt"
    assert captured["params"]["random_seed"] == 42
    assert captured["params"]["presolve"] == 1


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
