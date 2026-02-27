"""Unit tests for backtesting functions (Kupiec, Christoffersen)."""

import numpy as np
import pytest

from src.evaluation.backtesting import (
    christoffersen_test,
    kupiec_pof_test,
    population_stability_index,
)

# ── Kupiec POF Test ──


def test_kupiec_perfect_coverage():
    """Zero violations with nominal alpha → should not reject."""
    violations = np.zeros(1000)
    result = kupiec_pof_test(violations, alpha=0.10)
    assert result["n_violations"] == 0
    assert result["violation_rate"] == 0.0
    assert result["nominal_alpha"] == 0.10


def test_kupiec_exact_nominal_rate():
    """Violation rate exactly matching alpha → should not reject."""
    rng = np.random.RandomState(42)
    violations = (rng.random(1000) < 0.10).astype(float)
    result = kupiec_pof_test(violations, alpha=0.10)
    assert result["p_value"] > 0.01  # Should not reject at reasonable level
    assert result["reject"] is False


def test_kupiec_excessive_violations():
    """30% violations vs 10% nominal → should reject."""
    rng = np.random.RandomState(42)
    violations = (rng.random(500) < 0.30).astype(float)
    result = kupiec_pof_test(violations, alpha=0.10)
    assert result["reject"] is True
    assert result["p_value"] < 0.05


def test_kupiec_empty_array():
    """Empty violations array → graceful return."""
    result = kupiec_pof_test(np.array([]), alpha=0.10)
    assert result["n_total"] == 0
    assert result["p_value"] == 1.0
    assert result["reject"] is False


def test_kupiec_all_violations():
    """All violations → should reject."""
    violations = np.ones(100)
    result = kupiec_pof_test(violations, alpha=0.10)
    assert result["violation_rate"] == 1.0
    assert result["reject"] is True


def test_kupiec_returns_all_keys():
    """Check output dict has expected keys."""
    violations = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    result = kupiec_pof_test(violations, alpha=0.10)
    expected_keys = {
        "lr_statistic",
        "p_value",
        "reject",
        "n_violations",
        "n_total",
        "violation_rate",
        "nominal_alpha",
    }
    assert expected_keys == set(result.keys())


# ── Christoffersen Test ──


def test_christoffersen_independent_violations():
    """IID violations at nominal rate → should not reject independence."""
    rng = np.random.RandomState(123)
    violations = (rng.random(2000) < 0.10).astype(float)
    result = christoffersen_test(violations, alpha=0.10)
    # Should not reject independence for truly IID sequence
    assert result["p_ind"] > 0.01
    assert result["reject_ind"] is False


def test_christoffersen_clustered_violations():
    """Clustered violations → should reject independence."""
    # Create strongly clustered pattern: 50 violations in a row, then 450 no violations
    violations = np.zeros(500)
    violations[100:150] = 1.0  # 50 consecutive violations → strong clustering
    result = christoffersen_test(violations, alpha=0.10)
    assert result["reject_ind"] is True


def test_christoffersen_returns_all_keys():
    """Check output dict has expected keys."""
    violations = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0])
    result = christoffersen_test(violations, alpha=0.10)
    expected_keys = {
        "lr_uc",
        "p_uc",
        "reject_uc",
        "lr_ind",
        "p_ind",
        "reject_ind",
        "lr_cc",
        "p_cc",
        "reject_cc",
        "transition_matrix",
    }
    assert expected_keys == set(result.keys())


def test_christoffersen_transition_matrix_sums():
    """Transition counts should sum to n-1."""
    violations = np.array([0, 1, 1, 0, 0, 1, 0, 1, 0, 0])
    result = christoffersen_test(violations, alpha=0.10)
    tm = result["transition_matrix"]
    assert tm["n00"] + tm["n01"] + tm["n10"] + tm["n11"] == len(violations) - 1


def test_christoffersen_single_element():
    """Single-element array → graceful return."""
    violations = np.array([0.0])
    result = christoffersen_test(violations, alpha=0.10)
    assert result["lr_ind"] == 0.0
    assert result["p_ind"] == 1.0


def test_christoffersen_joint_statistic():
    """Joint LR should be sum of UC and IND components."""
    violations = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 0] * 20)
    result = christoffersen_test(violations, alpha=0.10)
    assert result["lr_cc"] == pytest.approx(result["lr_uc"] + result["lr_ind"], abs=1e-6)


# ── PSI (existing function) ──


def test_psi_identical_distributions():
    """Same distribution → PSI ≈ 0."""
    rng = np.random.RandomState(42)
    data = rng.random(1000)
    psi = population_stability_index(data, data)
    assert psi < 0.01


def test_psi_shifted_distribution():
    """Shifted distribution → PSI > 0."""
    rng = np.random.RandomState(42)
    expected = rng.random(1000)
    actual = rng.random(1000) + 0.5
    psi = population_stability_index(expected, actual)
    assert psi > 0.1
