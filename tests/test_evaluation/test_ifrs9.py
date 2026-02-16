"""Unit tests for IFRS9 ECL calculation and staging logic."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.ifrs9 import (
    DEFAULT_DPD_THRESHOLD,
    SICR_DPD_THRESHOLD,
    SICR_PD_INCREASE_THRESHOLD,
    assign_stage,
    compute_ecl,
    ecl_with_conformal_range,
)


# ── assign_stage ──


def test_assign_stage_all_stage1():
    """Loans with no SICR should stay Stage 1."""
    pd_orig = np.array([0.05, 0.10, 0.15])
    pd_curr = np.array([0.05, 0.10, 0.15])  # No increase
    stages = assign_stage(pd_orig, pd_curr)
    np.testing.assert_array_equal(stages, [1, 1, 1])


def test_assign_stage_sicr_pd_increase():
    """PD increase > threshold triggers Stage 2."""
    pd_orig = np.array([0.05, 0.05, 0.05])
    pd_curr = np.array([0.05, 0.08, 0.10])  # 2nd and 3rd have SICR
    stages = assign_stage(pd_orig, pd_curr)
    assert stages[0] == 1
    assert stages[1] == 2  # increase = 0.03 > 0.02
    assert stages[2] == 2  # increase = 0.05 > 0.02


def test_assign_stage_dpd_30():
    """DPD >= 30 triggers Stage 2."""
    pd_orig = np.array([0.05, 0.05])
    pd_curr = np.array([0.05, 0.05])
    dpd = np.array([0, 35])
    stages = assign_stage(pd_orig, pd_curr, dpd=dpd)
    assert stages[0] == 1
    assert stages[1] == 2


def test_assign_stage_dpd_90():
    """DPD >= 90 triggers Stage 3."""
    pd_orig = np.array([0.05, 0.05, 0.05])
    pd_curr = np.array([0.05, 0.05, 0.05])
    dpd = np.array([0, 45, 95])
    stages = assign_stage(pd_orig, pd_curr, dpd=dpd)
    assert stages[0] == 1
    assert stages[1] == 2
    assert stages[2] == 3


def test_assign_stage_values_only_123():
    """Stages should only contain 1, 2, or 3."""
    rng = np.random.RandomState(42)
    pd_orig = rng.uniform(0.01, 0.20, 100)
    pd_curr = rng.uniform(0.01, 0.30, 100)
    dpd = rng.choice([0, 10, 35, 60, 95], 100)
    stages = assign_stage(pd_orig, pd_curr, dpd=dpd)
    assert set(stages).issubset({1, 2, 3})


def test_assign_stage_with_conformal_pd_high():
    """pd_high should influence staging for SICR-flagged loans."""
    pd_orig = np.array([0.05, 0.05])
    pd_curr = np.array([0.08, 0.08])  # Both have SICR
    pd_high = np.array([0.08, 0.50])  # 2nd has high uncertainty
    stages = assign_stage(pd_orig, pd_curr, pd_high=pd_high)
    # Both should be Stage 2 (SICR triggered)
    assert stages[0] == 2
    assert stages[1] == 2


# ── compute_ecl ──


def test_compute_ecl_stage1_uses_12m_pd():
    """Stage 1 ECL should use 12-month PD."""
    pd_values = np.array([0.10])
    lgd = np.array([0.45])
    ead = np.array([10000.0])
    stages = np.array([1])
    result = compute_ecl(pd_values, lgd, ead, stages, discount_rate=0.0)
    expected_ecl = 0.10 * 0.45 * 10000  # 450
    assert result["ecl"].iloc[0] == pytest.approx(expected_ecl, abs=1)


def test_compute_ecl_stage3_pd_is_1():
    """Stage 3 ECL should use PD = 1.0 (credit-impaired)."""
    pd_values = np.array([0.10])
    lgd = np.array([0.45])
    ead = np.array([10000.0])
    stages = np.array([3])
    result = compute_ecl(pd_values, lgd, ead, stages, discount_rate=0.0)
    expected_ecl = 1.0 * 0.45 * 10000  # 4500
    assert result["ecl"].iloc[0] == pytest.approx(expected_ecl, abs=1)
    assert result["effective_pd"].iloc[0] == 1.0


def test_compute_ecl_stage2_scaled_pd():
    """Stage 2 ECL should use scaled PD (no lifetime_pd provided)."""
    pd_values = np.array([0.10])
    lgd = np.array([0.45])
    ead = np.array([10000.0])
    stages = np.array([2])
    result = compute_ecl(pd_values, lgd, ead, stages, discount_rate=0.0)
    # Scaled: min(0.10 * 3, 1.0) = 0.30
    expected_ecl = 0.30 * 0.45 * 10000  # 1350
    assert result["ecl"].iloc[0] == pytest.approx(expected_ecl, abs=1)


def test_compute_ecl_stage2_with_lifetime_pd():
    """Stage 2 should use provided lifetime PD."""
    pd_values = np.array([0.10])
    lgd = np.array([0.45])
    ead = np.array([10000.0])
    stages = np.array([2])
    lifetime_pd = np.array([0.25])
    result = compute_ecl(pd_values, lgd, ead, stages, lifetime_pd=lifetime_pd, discount_rate=0.0)
    expected_ecl = 0.25 * 0.45 * 10000  # 1125
    assert result["ecl"].iloc[0] == pytest.approx(expected_ecl, abs=1)


def test_compute_ecl_returns_dataframe():
    pd_values = np.array([0.05, 0.10, 0.50])
    lgd = np.array([0.45, 0.45, 0.45])
    ead = np.array([10000, 20000, 5000])
    stages = np.array([1, 2, 3])
    result = compute_ecl(pd_values, lgd, ead, stages)
    assert isinstance(result, pd.DataFrame)
    assert "stage" in result.columns
    assert "ecl" in result.columns
    assert len(result) == 3


def test_compute_ecl_all_positive():
    """ECL should be non-negative for all stages."""
    rng = np.random.RandomState(42)
    n = 50
    pd_values = rng.uniform(0.01, 0.30, n)
    lgd = np.full(n, 0.45)
    ead = rng.uniform(1000, 50000, n)
    stages = rng.choice([1, 2, 3], n)
    result = compute_ecl(pd_values, lgd, ead, stages)
    assert np.all(result["ecl"] >= 0)


def test_compute_ecl_discount_factor():
    """Discount factor should reduce ECL."""
    pd_values = np.array([0.10])
    lgd = np.array([0.45])
    ead = np.array([10000.0])
    stages = np.array([1])
    ecl_no_discount = compute_ecl(pd_values, lgd, ead, stages, discount_rate=0.0)
    ecl_with_discount = compute_ecl(pd_values, lgd, ead, stages, discount_rate=0.10)
    assert ecl_with_discount["ecl"].iloc[0] < ecl_no_discount["ecl"].iloc[0]


# ── ecl_with_conformal_range ──


def test_ecl_conformal_range_ordering():
    """ECL_low <= ECL_point <= ECL_high."""
    pd_low = np.array([0.05, 0.08])
    pd_point = np.array([0.10, 0.15])
    pd_high = np.array([0.15, 0.22])
    lgd = np.array([0.45, 0.45])
    ead = np.array([10000, 20000])
    stages = np.array([1, 2])
    result = ecl_with_conformal_range(pd_low, pd_point, pd_high, lgd, ead, stages)
    assert np.all(result["ecl_low"] <= result["ecl_point"])
    assert np.all(result["ecl_point"] <= result["ecl_high"])


def test_ecl_conformal_range_positive():
    pd_low = np.array([0.01])
    pd_point = np.array([0.05])
    pd_high = np.array([0.10])
    lgd = np.array([0.45])
    ead = np.array([10000])
    stages = np.array([1])
    result = ecl_with_conformal_range(pd_low, pd_point, pd_high, lgd, ead, stages)
    assert result["ecl_range"].iloc[0] > 0
    assert result["ecl_low"].iloc[0] >= 0


def test_ecl_conformal_range_columns():
    pd_low = np.array([0.05])
    pd_point = np.array([0.10])
    pd_high = np.array([0.15])
    lgd = np.array([0.45])
    ead = np.array([10000])
    stages = np.array([1])
    result = ecl_with_conformal_range(pd_low, pd_point, pd_high, lgd, ead, stages)
    expected_cols = {"stage", "ecl_low", "ecl_point", "ecl_high", "ecl_range"}
    assert expected_cols.issubset(set(result.columns))
