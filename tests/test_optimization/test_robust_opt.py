"""Tests for src/optimization/robust_opt.py.

Covers box uncertainty sets, worst-case loss, and scenario analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.robust_opt import (
    build_box_uncertainty_set,
    scenario_analysis,
    worst_case_expected_loss,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_arrays():
    """Standard arrays for 5 loans."""
    pd_low = np.array([0.01, 0.05, 0.10, 0.02, 0.08])
    pd_high = np.array([0.05, 0.15, 0.25, 0.08, 0.18])
    pd_point = (pd_low + pd_high) / 2
    lgd = np.full(5, 0.45)
    loan_amnt = np.array([10000, 20000, 15000, 25000, 12000])
    allocation = np.array([1.0, 0.5, 0.0, 1.0, 0.8])
    return {
        "pd_low": pd_low,
        "pd_high": pd_high,
        "pd_point": pd_point,
        "lgd": lgd,
        "loan_amnt": loan_amnt,
        "allocation": allocation,
    }


# ---------------------------------------------------------------------------
# build_box_uncertainty_set
# ---------------------------------------------------------------------------


class TestBuildBoxUncertaintySet:
    def test_contains_required_keys(self, sample_arrays):
        uset = build_box_uncertainty_set(sample_arrays["pd_low"], sample_arrays["pd_high"])
        assert "pd_low" in uset
        assert "pd_high" in uset
        assert "pd_center" in uset
        assert "pd_radius" in uset

    def test_center_is_midpoint(self, sample_arrays):
        uset = build_box_uncertainty_set(sample_arrays["pd_low"], sample_arrays["pd_high"])
        expected_center = (sample_arrays["pd_low"] + sample_arrays["pd_high"]) / 2
        np.testing.assert_array_almost_equal(uset["pd_center"], expected_center)

    def test_radius_is_half_width(self, sample_arrays):
        uset = build_box_uncertainty_set(sample_arrays["pd_low"], sample_arrays["pd_high"])
        expected_radius = (sample_arrays["pd_high"] - sample_arrays["pd_low"]) / 2
        np.testing.assert_array_almost_equal(uset["pd_radius"], expected_radius)

    def test_lgd_bounds_included_when_provided(self, sample_arrays):
        lgd_low = np.full(5, 0.30)
        lgd_high = np.full(5, 0.60)
        uset = build_box_uncertainty_set(
            sample_arrays["pd_low"], sample_arrays["pd_high"], lgd_low, lgd_high
        )
        assert "lgd_low" in uset
        assert "lgd_high" in uset
        assert "lgd_center" in uset
        np.testing.assert_array_almost_equal(uset["lgd_center"], np.full(5, 0.45))

    def test_no_lgd_keys_when_not_provided(self, sample_arrays):
        uset = build_box_uncertainty_set(sample_arrays["pd_low"], sample_arrays["pd_high"])
        assert "lgd_low" not in uset


# ---------------------------------------------------------------------------
# worst_case_expected_loss
# ---------------------------------------------------------------------------


class TestWorstCaseExpectedLoss:
    def test_positive_loss(self, sample_arrays):
        loss = worst_case_expected_loss(
            allocation=sample_arrays["allocation"],
            loan_amounts=sample_arrays["loan_amnt"],
            pd_high=sample_arrays["pd_high"],
            lgd_point=sample_arrays["lgd"],
        )
        assert loss > 0

    def test_zero_allocation_zero_loss(self, sample_arrays):
        loss = worst_case_expected_loss(
            allocation=np.zeros(5),
            loan_amounts=sample_arrays["loan_amnt"],
            pd_high=sample_arrays["pd_high"],
            lgd_point=sample_arrays["lgd"],
        )
        assert loss == pytest.approx(0.0)

    def test_uses_lgd_high_when_provided(self, sample_arrays):
        lgd_high = np.full(5, 0.60)
        loss_high = worst_case_expected_loss(
            allocation=sample_arrays["allocation"],
            loan_amounts=sample_arrays["loan_amnt"],
            pd_high=sample_arrays["pd_high"],
            lgd_high=lgd_high,
        )
        loss_point = worst_case_expected_loss(
            allocation=sample_arrays["allocation"],
            loan_amounts=sample_arrays["loan_amnt"],
            pd_high=sample_arrays["pd_high"],
            lgd_point=sample_arrays["lgd"],
        )
        assert loss_high > loss_point  # 0.60 > 0.45

    def test_defaults_lgd_045_when_none(self, sample_arrays):
        loss = worst_case_expected_loss(
            allocation=sample_arrays["allocation"],
            loan_amounts=sample_arrays["loan_amnt"],
            pd_high=sample_arrays["pd_high"],
        )
        # Should use 0.45 default
        expected = float(
            np.sum(
                sample_arrays["allocation"]
                * sample_arrays["loan_amnt"]
                * sample_arrays["pd_high"]
                * 0.45
            )
        )
        assert loss == pytest.approx(expected)


# ---------------------------------------------------------------------------
# scenario_analysis
# ---------------------------------------------------------------------------


class TestScenarioAnalysis:
    def test_returns_dataframe_with_scenarios(self, sample_arrays):
        result = scenario_analysis(
            allocation=sample_arrays["allocation"],
            loan_amounts=sample_arrays["loan_amnt"],
            pd_low=sample_arrays["pd_low"],
            pd_point=sample_arrays["pd_point"],
            pd_high=sample_arrays["pd_high"],
            lgd=sample_arrays["lgd"],
        )
        assert isinstance(result, pd.DataFrame)
        assert "best_case" in result.columns
        assert "expected" in result.columns
        assert "worst_case" in result.columns
        assert "range" in result.columns

    def test_best_case_leq_expected_leq_worst_case(self, sample_arrays):
        result = scenario_analysis(
            allocation=sample_arrays["allocation"],
            loan_amounts=sample_arrays["loan_amnt"],
            pd_low=sample_arrays["pd_low"],
            pd_point=sample_arrays["pd_point"],
            pd_high=sample_arrays["pd_high"],
            lgd=sample_arrays["lgd"],
        )
        assert result["best_case"].iloc[0] <= result["expected"].iloc[0]
        assert result["expected"].iloc[0] <= result["worst_case"].iloc[0]

    def test_range_equals_difference(self, sample_arrays):
        result = scenario_analysis(
            allocation=sample_arrays["allocation"],
            loan_amounts=sample_arrays["loan_amnt"],
            pd_low=sample_arrays["pd_low"],
            pd_point=sample_arrays["pd_point"],
            pd_high=sample_arrays["pd_high"],
            lgd=sample_arrays["lgd"],
        )
        computed_range = result["worst_case"].iloc[0] - result["best_case"].iloc[0]
        assert result["range"].iloc[0] == pytest.approx(computed_range)

    def test_zero_allocation_all_scenarios_zero(self):
        n = 3
        result = scenario_analysis(
            allocation=np.zeros(n),
            loan_amounts=np.full(n, 10000),
            pd_low=np.full(n, 0.01),
            pd_point=np.full(n, 0.05),
            pd_high=np.full(n, 0.10),
            lgd=np.full(n, 0.45),
        )
        assert result["best_case"].iloc[0] == pytest.approx(0.0)
        assert result["worst_case"].iloc[0] == pytest.approx(0.0)
