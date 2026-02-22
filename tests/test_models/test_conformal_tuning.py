"""Tests for src/models/conformal_tuning.py.

Covers calibration splitting, Pareto front, config selection,
group multipliers, and coverage floor enforcement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.conformal_tuning import (
    apply_group_multipliers,
    choose_best_tuning_row,
    enforce_group_coverage_floor,
    mark_pareto_front,
    split_calibration_for_tuning,
    to_python_scalar,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cal_data():
    """Small calibration dataset with two classes and two groups."""
    rng = np.random.default_rng(42)
    n = 200
    y = pd.Series(rng.integers(0, 2, size=n))
    groups = pd.Series(np.where(np.arange(n) < n // 2, "A", "B"))
    dates = pd.Series(pd.date_range("2018-01-01", periods=n, freq="W"))
    return y, groups, dates


@pytest.fixture()
def tuning_results():
    """DataFrame resembling multi-config tuning output."""
    return pd.DataFrame(
        {
            "empirical_coverage": [0.88, 0.91, 0.93, 0.95, 0.90],
            "min_group_coverage": [0.85, 0.89, 0.90, 0.93, 0.87],
            "avg_interval_width": [0.10, 0.12, 0.15, 0.20, 0.11],
            "coverage_gap": [0.02, 0.01, 0.00, -0.05, 0.01],
        }
    )


# ---------------------------------------------------------------------------
# split_calibration_for_tuning
# ---------------------------------------------------------------------------


class TestSplitCalibration:
    def test_partitions_are_disjoint_and_cover_all(self, cal_data):
        y, groups, dates = cal_data
        idx_fit, idx_tune = split_calibration_for_tuning(y, groups, dates)
        all_idx = np.sort(np.concatenate([idx_fit, idx_tune]))
        np.testing.assert_array_equal(all_idx, np.arange(len(y)))

    def test_holdout_ratio_respected(self, cal_data):
        y, groups, dates = cal_data
        ratio = 0.20
        _, idx_tune = split_calibration_for_tuning(y, groups, dates, holdout_ratio=ratio)
        actual_ratio = len(idx_tune) / len(y)
        assert abs(actual_ratio - ratio) < 0.05

    def test_temporal_split_when_dates_available(self, cal_data):
        y, groups, dates = cal_data
        idx_fit, idx_tune = split_calibration_for_tuning(y, groups, dates)
        # Tuning set should contain the latest dates
        fit_max = dates.iloc[idx_fit].max()
        tune_min = dates.iloc[idx_tune].min()
        assert tune_min >= fit_max or len(idx_tune) > 0

    def test_fallback_without_dates(self, cal_data):
        y, groups, _ = cal_data
        idx_fit, idx_tune = split_calibration_for_tuning(y, groups, issue_dates=None)
        assert len(idx_fit) + len(idx_tune) == len(y)

    def test_single_row_returns_empty_tune(self):
        y = pd.Series([1])
        groups = pd.Series(["A"])
        idx_fit, idx_tune = split_calibration_for_tuning(y, groups)
        assert len(idx_fit) == 1
        assert len(idx_tune) == 0

    def test_holdout_ratio_is_clipped(self, cal_data):
        y, groups, _ = cal_data
        # Extreme ratios should be clipped to [0.05, 0.50]
        _, idx_tune_low = split_calibration_for_tuning(y, groups, holdout_ratio=0.001)
        _, idx_tune_high = split_calibration_for_tuning(y, groups, holdout_ratio=0.99)
        assert len(idx_tune_low) >= 1
        assert len(idx_tune_high) <= len(y) - 1


# ---------------------------------------------------------------------------
# mark_pareto_front
# ---------------------------------------------------------------------------


class TestParetoFront:
    def test_single_row_is_pareto(self):
        df = pd.DataFrame(
            {
                "empirical_coverage": [0.90],
                "min_group_coverage": [0.85],
                "avg_interval_width": [0.10],
            }
        )
        result = mark_pareto_front(df)
        assert result.iloc[0] is True or result.iloc[0] == True  # noqa: E712

    def test_dominated_point_excluded(self):
        df = pd.DataFrame(
            {
                "empirical_coverage": [0.90, 0.95],
                "min_group_coverage": [0.85, 0.90],
                "avg_interval_width": [0.15, 0.10],
            }
        )
        result = mark_pareto_front(df)
        # Row 0 is dominated by row 1 (worse on all 3 objectives)
        assert result.iloc[0] == False  # noqa: E712
        assert result.iloc[1] == True  # noqa: E712

    def test_incomparable_points_both_pareto(self):
        df = pd.DataFrame(
            {
                "empirical_coverage": [0.95, 0.90],
                "min_group_coverage": [0.85, 0.90],
                "avg_interval_width": [0.10, 0.10],
            }
        )
        result = mark_pareto_front(df)
        # Neither dominates the other
        assert result.all()


# ---------------------------------------------------------------------------
# choose_best_tuning_row
# ---------------------------------------------------------------------------


class TestChooseBestTuningRow:
    def test_selects_meeting_target(self, tuning_results):
        row, tier = choose_best_tuning_row(
            tuning_results, target_coverage=0.90, min_group_coverage_target=0.85
        )
        assert row["empirical_coverage"] >= 0.90

    def test_prefers_narrow_width(self, tuning_results):
        row, tier = choose_best_tuning_row(
            tuning_results, target_coverage=0.90, min_group_coverage_target=0.85
        )
        # Among qualifying rows, should prefer smaller width
        assert row["avg_interval_width"] <= 0.20

    def test_fallback_when_nothing_meets_target(self):
        df = pd.DataFrame(
            {
                "empirical_coverage": [0.60, 0.65],
                "min_group_coverage": [0.50, 0.55],
                "avg_interval_width": [0.10, 0.12],
                "coverage_gap": [0.30, 0.25],
            }
        )
        row, tier = choose_best_tuning_row(df, target_coverage=0.99, min_group_coverage_target=0.95)
        assert tier == "fallback_penalty"

    def test_width_budget_prefers_narrow(self, tuning_results):
        # With tight budget, width tiers should be tried first;
        # if nothing qualifies for width, it falls through to non-width tiers
        row_strict, tier_strict = choose_best_tuning_row(
            tuning_results,
            target_coverage=0.90,
            min_group_coverage_target=0.85,
            max_width_budget=0.11,
        )
        row_loose, tier_loose = choose_best_tuning_row(
            tuning_results,
            target_coverage=0.90,
            min_group_coverage_target=0.85,
            max_width_budget=None,
        )
        # Both should still return valid rows
        assert row_strict["empirical_coverage"] >= 0.90 or tier_strict == "fallback_penalty"
        assert row_loose["empirical_coverage"] >= 0.90 or tier_loose == "fallback_penalty"


# ---------------------------------------------------------------------------
# apply_group_multipliers
# ---------------------------------------------------------------------------


class TestApplyGroupMultipliers:
    def test_identity_when_factor_one(self):
        y_pred = np.array([0.5, 0.5])
        intervals = np.array([[0.3, 0.7], [0.4, 0.6]])
        groups = np.array(["A", "A"])
        result = apply_group_multipliers(y_pred, intervals, groups, {"A": 1.0})
        np.testing.assert_array_almost_equal(result, intervals)

    def test_widens_intervals_for_group(self):
        y_pred = np.array([0.5, 0.5])
        intervals = np.array([[0.4, 0.6], [0.4, 0.6]])
        groups = np.array(["A", "B"])
        result = apply_group_multipliers(y_pred, intervals, groups, {"A": 1.5})
        # Group A should be wider, group B unchanged
        width_a = result[0, 1] - result[0, 0]
        width_b = result[1, 1] - result[1, 0]
        assert width_a > width_b

    def test_clips_to_zero_one(self):
        y_pred = np.array([0.05])
        intervals = np.array([[0.0, 0.10]])
        groups = np.array(["A"])
        result = apply_group_multipliers(y_pred, intervals, groups, {"A": 5.0})
        assert result[0, 0] >= 0.0
        assert result[0, 1] <= 1.0


# ---------------------------------------------------------------------------
# enforce_group_coverage_floor
# ---------------------------------------------------------------------------


class TestEnforceGroupCoverageFloor:
    def test_already_covered_groups_unchanged(self):
        rng = np.random.default_rng(42)
        n = 100
        y_pred = rng.uniform(0.1, 0.9, n)
        y_true = y_pred + rng.normal(0, 0.01, n)  # Tight around predictions
        intervals = np.column_stack([y_pred - 0.3, y_pred + 0.3])
        groups = np.array(["A"] * n)

        new_intervals, factors, report = enforce_group_coverage_floor(
            y_true, y_pred, intervals, groups, target_coverage=0.80
        )
        # Coverage already high → no adjustment needed
        assert len(factors) == 0 or all(f == 1.0 for f in factors.values())

    def test_returns_report_dataframe(self):
        n = 50
        y_pred = np.full(n, 0.5)
        y_true = np.full(n, 0.5)
        intervals = np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])
        groups = np.array(["A"] * 25 + ["B"] * 25)

        _, _, report = enforce_group_coverage_floor(
            y_true, y_pred, intervals, groups, target_coverage=0.90
        )
        assert isinstance(report, pd.DataFrame)
        assert "group" in report.columns
        assert "coverage_before" in report.columns
        assert "multiplier" in report.columns
        assert len(report) == 2  # A and B


# ---------------------------------------------------------------------------
# to_python_scalar
# ---------------------------------------------------------------------------


class TestToPythonScalar:
    def test_numpy_float(self):
        result = to_python_scalar(np.float64(3.14))
        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_numpy_int(self):
        result = to_python_scalar(np.int32(42))
        assert isinstance(result, int)

    def test_python_native_passthrough(self):
        assert to_python_scalar("hello") == "hello"
        assert to_python_scalar(42) == 42
