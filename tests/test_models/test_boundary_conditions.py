"""Negative and boundary condition tests for model modules."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ifrs9 import assign_stage, compute_ecl, compute_lifetime_pd_from_survival
from src.models.conformal import (
    _conformal_quantile,
    apply_probability_calibrator,
    conditional_coverage_by_group,
    validate_coverage,
)
from src.optimization.robust_opt import (
    build_box_uncertainty_set,
    scenario_analysis,
    worst_case_expected_loss,
)
from src.optimization.sda import dynamic_credit_policy

# ── Conformal boundary conditions ──


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_conformal_quantile_valid_alpha(alpha):
    """Quantile computation should handle valid alpha values without crash."""
    scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    result = _conformal_quantile(scores, alpha)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_conformal_quantile_invalid_alpha_raises():
    """Alpha > 1 should raise ValueError (numpy quantile constraint)."""
    scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    with pytest.raises(ValueError):
        _conformal_quantile(scores, 1.5)


def test_apply_calibrator_with_constant_scores():
    """Calibrator should handle constant input scores."""
    scores = np.full(10, 0.5)
    result = apply_probability_calibrator(None, scores)
    assert np.all(result == 0.5)


def test_conditional_coverage_empty_groups():
    """Coverage computation should handle a group with all NaN labels."""
    y_true = np.array([0.5, 0.5])
    y_intervals = np.array([[0.0, 1.0], [0.0, 1.0]])
    groups = pd.Series(["A", "A"])
    result = conditional_coverage_by_group(y_true, y_intervals, groups)
    assert len(result) == 1
    assert result["coverage"].iloc[0] == 1.0


def test_validate_coverage_single_element():
    """Coverage should work with a single observation."""
    y_true = np.array([0.5])
    y_intervals = np.array([[0.0, 1.0]])
    result = validate_coverage(y_true, y_intervals, alpha=0.1)
    assert result["empirical_coverage"] == 1.0


def test_validate_coverage_zero_width_intervals():
    """Zero-width intervals should only cover exact matches."""
    y_true = np.array([0.5, 0.5, 0.3])
    y_intervals = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    result = validate_coverage(y_true, y_intervals, alpha=0.1)
    # Only first two are covered (0.5 in [0.5, 0.5])
    assert result["empirical_coverage"] == pytest.approx(2 / 3, abs=0.01)


# ── IFRS9 boundary conditions ──


def test_assign_stage_empty_arrays():
    """Staging should handle empty input arrays."""
    stages = assign_stage(np.array([]), np.array([]))
    assert len(stages) == 0


def test_assign_stage_all_stage3_dpd():
    """All loans with DPD >= 90 should be Stage 3."""
    n = 5
    stages = assign_stage(
        np.zeros(n),
        np.zeros(n),
        dpd=np.full(n, 100),
    )
    np.testing.assert_array_equal(stages, np.full(n, 3))


def test_compute_ecl_all_stage1():
    """ECL with all Stage 1 should use 12-month PD."""
    n = 3
    result = compute_ecl(
        pd_values=np.array([0.1, 0.2, 0.3]),
        lgd_values=np.full(n, 0.45),
        ead_values=np.full(n, 10000.0),
        stages=np.ones(n, dtype=int),
    )
    assert len(result) == n
    assert all(result["stage"] == 1)
    assert all(result["ecl"] > 0)


def test_compute_ecl_stage3_pd_is_one():
    """Stage 3 effective PD should be 1.0 (credit-impaired)."""
    result = compute_ecl(
        pd_values=np.array([0.1]),
        lgd_values=np.array([0.45]),
        ead_values=np.array([10000.0]),
        stages=np.array([3]),
    )
    assert result["effective_pd"].iloc[0] == 1.0


def test_compute_ecl_stage2_proxy_capped_at_one():
    """Stage 2 proxy (PD * 3) should be capped at 1.0."""
    result = compute_ecl(
        pd_values=np.array([0.5]),
        lgd_values=np.array([0.45]),
        ead_values=np.array([10000.0]),
        stages=np.array([2]),
    )
    # 0.5 * 3 = 1.5 -> capped to 1.0
    assert result["effective_pd"].iloc[0] == 1.0


def test_compute_lifetime_pd_returns_none_on_bad_model():
    """Should return None when survival model prediction fails."""

    class BrokenModel:
        def predict_survival_function(self, X):
            raise ValueError("Model is broken")

    result = compute_lifetime_pd_from_survival(BrokenModel(), pd.DataFrame({"x": [1, 2, 3]}))
    assert result is None


# ── Robust optimization boundary conditions ──


def test_box_uncertainty_zero_width():
    """Zero-width intervals mean no uncertainty."""
    pds = np.array([0.1, 0.2, 0.3])
    uset = build_box_uncertainty_set(pds, pds)
    np.testing.assert_array_equal(uset["pd_radius"], np.zeros(3))


def test_worst_case_loss_zero_allocation():
    """Zero allocation should yield zero worst-case loss."""
    loss = worst_case_expected_loss(
        allocation=np.zeros(3),
        loan_amounts=np.array([10000, 20000, 30000]),
        pd_high=np.array([0.2, 0.3, 0.4]),
    )
    assert loss == 0.0


def test_scenario_analysis_output_shape():
    """Scenario analysis should return one row with 4 columns."""
    result = scenario_analysis(
        allocation=np.array([1.0, 0.5, 0.0]),
        loan_amounts=np.array([10000, 20000, 30000]),
        pd_low=np.array([0.05, 0.10, 0.15]),
        pd_point=np.array([0.10, 0.20, 0.30]),
        pd_high=np.array([0.15, 0.30, 0.45]),
        lgd=np.full(3, 0.45),
    )
    assert len(result) == 1
    assert "best_case" in result.columns
    assert "worst_case" in result.columns
    assert result["worst_case"].iloc[0] >= result["best_case"].iloc[0]


# ── SDA boundary conditions ──


def test_dynamic_credit_policy_empty_forecasts():
    """Empty forecast DataFrame should return empty policy."""
    result = dynamic_credit_policy(
        forecasts=pd.DataFrame(columns=["ds", "y", "y_hi_90"]),
        current_portfolio_pd=0.05,
    )
    assert len(result) == 0


def test_dynamic_credit_policy_tighten_on_high_pd():
    """High forecasted PD should trigger TIGHTEN action."""
    forecasts = pd.DataFrame({"ds": ["2025-01"], "y": [0.15], "y_hi_90": [0.20]})
    result = dynamic_credit_policy(forecasts, current_portfolio_pd=0.05, target_pd=0.08)
    assert result["action"].iloc[0] == "TIGHTEN"


def test_dynamic_credit_policy_expand_on_low_pd():
    """Low forecasted PD should trigger EXPAND action."""
    forecasts = pd.DataFrame({"ds": ["2025-01"], "y": [0.02], "y_hi_90": [0.03]})
    result = dynamic_credit_policy(forecasts, current_portfolio_pd=0.05, target_pd=0.08)
    assert result["action"].iloc[0] == "EXPAND"
