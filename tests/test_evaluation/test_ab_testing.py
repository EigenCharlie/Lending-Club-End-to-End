"""Unit tests for A/B testing simulation framework."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ab_testing import (
    ab_summary,
    compare_strategies,
    power_analysis,
    stratified_split,
)

# ── Power Analysis ──


def test_power_analysis_returns_required_keys():
    """Result should contain all expected keys."""
    result = power_analysis(effect_size=0.05)
    assert "n_per_group" in result
    assert "total_n" in result
    assert "effect_size" in result
    assert "alpha" in result
    assert "power" in result


def test_power_analysis_larger_effect_needs_fewer():
    """Larger effect size should require fewer samples."""
    small = power_analysis(effect_size=0.02)
    large = power_analysis(effect_size=0.10)
    assert large["n_per_group"] < small["n_per_group"]


# ── Stratified Split ──


def test_stratified_split_ratio():
    """Treatment ratio should be approximately respected per stratum."""
    df = pd.DataFrame(
        {
            "grade": ["A"] * 100 + ["B"] * 100,
            "value": range(200),
        }
    )
    control, treatment = stratified_split(df, strata_col="grade", treatment_ratio=0.50)
    assert len(treatment) == pytest.approx(100, abs=5)
    assert len(control) == pytest.approx(100, abs=5)


def test_stratified_split_no_overlap():
    """Control and treatment should partition the full dataset."""
    df = pd.DataFrame(
        {
            "grade": ["A"] * 50 + ["B"] * 50,
            "value": range(100),
        }
    )
    control, treatment = stratified_split(df, strata_col="grade")
    combined = set(control.index) | set(treatment.index)
    overlap = set(control.index) & set(treatment.index)
    assert len(overlap) == 0, "No index overlap allowed"
    assert combined == set(df.index), "Must cover all rows"


# ── Compare Strategies ──


def test_compare_strategies_bootstrap_no_difference():
    """Identical arrays → CI should include 0."""
    rng = np.random.RandomState(42)
    values = rng.normal(0.10, 0.02, 200)
    result = compare_strategies(values, values.copy(), method="bootstrap", seed=42)
    assert result["ci_low"] <= 0 <= result["ci_high"]


def test_compare_strategies_bootstrap_clear_difference():
    """Well-separated arrays → significant."""
    a = np.full(200, 0.05)
    b = np.full(200, 0.15)
    result = compare_strategies(a, b, method="bootstrap", seed=42)
    assert result["diff"] == pytest.approx(0.10)
    assert result["significant"] is True


def test_compare_strategies_all_methods():
    """All three methods should run without error."""
    rng = np.random.RandomState(42)
    a = rng.normal(0.10, 0.02, 100)
    b = rng.normal(0.12, 0.02, 100)
    for method in ["bootstrap", "permutation", "ttest"]:
        result = compare_strategies(a, b, method=method, seed=42)
        assert "p_value" in result
        assert "significant" in result


# ── AB Summary ──


def test_ab_summary_lift_calculation():
    """Known values should produce known lift."""
    control = {"return": 0.10, "loss": 0.05}
    treatment = {"return": 0.12, "loss": 0.04}
    result = ab_summary(control, treatment)
    assert len(result) == 2
    ret_row = result[result["metric"] == "return"].iloc[0]
    assert ret_row["lift_pct"] == pytest.approx(20.0, abs=0.1)
