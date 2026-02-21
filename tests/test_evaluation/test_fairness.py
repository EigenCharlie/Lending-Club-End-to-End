"""Unit tests for fairness metrics."""

import numpy as np
import pytest

from src.evaluation.fairness import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equalized_odds_gap,
    fairness_report,
)

# ── Demographic Parity Difference ──


def test_dpd_perfect_parity():
    """Identical positive rates across groups → DPD = 0."""
    y_pred = np.array([1, 0, 1, 1, 0, 1])  # A: 2/3, B: 2/3
    groups = np.array(["A", "A", "A", "B", "B", "B"])
    result = demographic_parity_difference(y_pred, groups)
    assert result["dpd"] == pytest.approx(0.0, abs=1e-9)


def test_dpd_worst_case():
    """One group all positive, other all negative → DPD = 1.0."""
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    groups = np.array(["A", "A", "A", "B", "B", "B"])
    result = demographic_parity_difference(y_pred, groups)
    assert result["dpd"] == pytest.approx(1.0)
    assert result["max_rate_group"] == "A"
    assert result["min_rate_group"] == "B"


def test_dpd_known_values():
    """Known rates: A=0.75, B=0.25 → DPD=0.50."""
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    result = demographic_parity_difference(y_pred, groups)
    assert result["dpd"] == pytest.approx(0.50)


# ── Equalized Odds Gap ──


def test_eo_gap_perfect_parity():
    """Same TPR and FPR across groups → EO gap = 0."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    groups = np.array(["A", "A", "B", "B"])
    result = equalized_odds_gap(y_true, y_pred, groups)
    assert result["eo_gap"] == pytest.approx(0.0, abs=1e-9)


def test_eo_gap_worst_case():
    """Group A perfect, Group B inverted → large gap."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1])
    groups = np.array(["A", "A", "B", "B"])
    result = equalized_odds_gap(y_true, y_pred, groups)
    assert result["tpr_gap"] == pytest.approx(1.0)
    assert result["fpr_gap"] == pytest.approx(1.0)
    assert result["eo_gap"] == pytest.approx(1.0)


# ── Disparate Impact Ratio ──


def test_dir_above_threshold():
    """Equal rates → DIR = 1.0."""
    y_pred = np.array([1, 0, 1, 0])
    groups = np.array(["A", "A", "B", "B"])
    result = disparate_impact_ratio(y_pred, groups)
    assert result["dir"] == pytest.approx(1.0, abs=0.01)


def test_dir_zero_rate_group():
    """Group with 0 positive predictions → DIR ≈ 0."""
    y_pred = np.array([1, 1, 0, 0])
    groups = np.array(["A", "A", "B", "B"])
    result = disparate_impact_ratio(y_pred, groups)
    assert result["dir"] < 0.01


def test_dir_single_group():
    """Single group → DIR = 1.0 (no comparison possible)."""
    y_pred = np.array([1, 0, 1])
    groups = np.array(["A", "A", "A"])
    result = disparate_impact_ratio(y_pred, groups)
    assert result["dir"] == pytest.approx(1.0)


# ── Fairness Report ──


def test_fairness_report_keys():
    """Report should have required columns."""
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_proba = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    groups_dict = {"attr_a": np.array(["X", "X", "X", "Y", "Y", "Y"])}
    result = fairness_report(y_true, y_proba, groups_dict)
    assert "attribute" in result.columns
    assert "dpd" in result.columns
    assert "eo_gap" in result.columns
    assert "dir" in result.columns
    assert "passed_all" in result.columns


def test_fairness_report_multi_attribute():
    """Two attributes should produce two rows."""
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.9, 0.1, 0.8, 0.2])
    groups_dict = {
        "attr_a": np.array(["X", "X", "Y", "Y"]),
        "attr_b": np.array(["P", "Q", "P", "Q"]),
    }
    result = fairness_report(y_true, y_proba, groups_dict)
    assert len(result) == 2
    assert list(result["attribute"]) == ["attr_a", "attr_b"]


def test_fairness_report_passed_flags():
    """Perfect parity should pass all thresholds."""
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.9, 0.1, 0.9, 0.1])
    groups_dict = {"perfect": np.array(["A", "A", "B", "B"])}
    result = fairness_report(y_true, y_proba, groups_dict)
    assert result["passed_all"].iloc[0] is True or result["passed_all"].iloc[0] == True  # noqa: E712
