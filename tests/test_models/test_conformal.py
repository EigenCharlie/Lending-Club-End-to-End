"""Unit tests for conformal prediction utilities."""
import numpy as np
import pandas as pd
import pytest

from src.models.conformal import (
    ProbabilityRegressor,
    _conformal_quantile,
    apply_probability_calibrator,
    conditional_coverage_by_group,
    validate_coverage,
)


# ── ProbabilityRegressor ──


class FakeClassifier:
    """Minimal classifier stub for testing."""

    def predict_proba(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        probs = np.random.RandomState(42).random(n)
        return np.column_stack([1 - probs, probs])


def test_probability_regressor_wraps_classifier():
    clf = FakeClassifier()
    reg = ProbabilityRegressor(clf)
    X = pd.DataFrame({"a": [1, 2, 3]})
    preds = reg.predict(X)
    assert preds.shape == (3,)
    assert np.all(preds >= 0)
    assert np.all(preds <= 1)


def test_probability_regressor_fit_is_noop():
    clf = FakeClassifier()
    reg = ProbabilityRegressor(clf)
    result = reg.fit(None, None)
    assert result is reg


# ── _conformal_quantile ──


def test_conformal_quantile_returns_float():
    scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    q = _conformal_quantile(scores, alpha=0.1)
    assert isinstance(q, float)


def test_conformal_quantile_empty_array():
    q = _conformal_quantile(np.array([]), alpha=0.1)
    assert q == 0.0


def test_conformal_quantile_high_coverage():
    """Lower alpha should give higher quantile."""
    scores = np.random.RandomState(42).random(100)
    q_90 = _conformal_quantile(scores, alpha=0.1)
    q_95 = _conformal_quantile(scores, alpha=0.05)
    assert q_95 >= q_90


# ── apply_probability_calibrator ──


def test_apply_calibrator_none_clips():
    scores = np.array([-0.1, 0.5, 1.2])
    result = apply_probability_calibrator(None, scores)
    assert np.all(result >= 0)
    assert np.all(result <= 1)


def test_apply_calibrator_isotonic():
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit([0.1, 0.5, 0.9], [0, 0.5, 1])
    result = apply_probability_calibrator(iso, np.array([0.3, 0.7]))
    assert result.shape == (2,)
    assert np.all(result >= 0)
    assert np.all(result <= 1)


# ── validate_coverage ──


def test_validate_coverage_perfect():
    """When all points are covered, coverage should be 1.0."""
    y_true = np.array([0.2, 0.5, 0.8])
    y_intervals = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    result = validate_coverage(y_true, y_intervals, alpha=0.1)
    assert result["empirical_coverage"] == 1.0
    assert result["target_coverage"] == 0.9
    assert result["coverage_gap"] == pytest.approx(0.1, abs=0.01)


def test_validate_coverage_none():
    """When no points are covered, coverage should be 0.0."""
    y_true = np.array([0.5, 0.5, 0.5])
    y_intervals = np.array([[0.0, 0.1], [0.0, 0.1], [0.0, 0.1]])
    result = validate_coverage(y_true, y_intervals, alpha=0.1)
    assert result["empirical_coverage"] == 0.0


def test_validate_coverage_returns_all_keys():
    y_true = np.array([0.3, 0.7])
    y_intervals = np.array([[0.1, 0.5], [0.5, 0.9]])
    result = validate_coverage(y_true, y_intervals, alpha=0.1)
    expected_keys = {
        "empirical_coverage",
        "target_coverage",
        "coverage_gap",
        "avg_interval_width",
        "median_interval_width",
    }
    assert expected_keys.issubset(result.keys())


def test_validate_coverage_width_positive():
    y_true = np.array([0.3, 0.7])
    y_intervals = np.array([[0.1, 0.5], [0.5, 0.9]])
    result = validate_coverage(y_true, y_intervals, alpha=0.1)
    assert result["avg_interval_width"] > 0
    assert result["median_interval_width"] > 0


# ── conditional_coverage_by_group ──


def test_conditional_coverage_groups():
    y_true = np.array([0.2, 0.8, 0.3, 0.9])
    y_intervals = np.array([[0.0, 0.5], [0.5, 1.0], [0.0, 0.5], [0.0, 0.5]])
    groups = pd.Series(["A", "A", "B", "B"])
    result = conditional_coverage_by_group(y_true, y_intervals, groups)
    assert len(result) == 2
    assert "coverage" in result.columns
    assert "avg_width" in result.columns
    assert all(result["n"] == 2)


def test_conditional_coverage_handles_nans():
    y_true = np.array([0.5, 0.5])
    y_intervals = np.array([[0.0, 1.0], [0.0, 1.0]])
    groups = pd.Series([None, "B"])
    result = conditional_coverage_by_group(y_true, y_intervals, groups)
    assert len(result) == 2  # UNKNOWN + B


# ── conformal_metrics (from evaluation module) ──


def test_conformal_metrics_consistency():
    from src.evaluation.metrics import conformal_metrics

    rng = np.random.RandomState(42)
    y_true = rng.random(100)
    low = y_true - 0.3
    high = y_true + 0.3
    y_intervals = np.column_stack([low, high])
    result = conformal_metrics(y_true, y_intervals, alpha=0.1)
    assert result["empirical_coverage"] == 1.0  # All covered with ±0.3
    assert result["avg_width"] == pytest.approx(0.6, abs=0.01)
    assert result["coverage_gap"] == pytest.approx(0.1, abs=0.01)


def test_conformal_metrics_partial_coverage():
    from src.evaluation.metrics import conformal_metrics

    y_true = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0])
    low = np.full(10, 0.3)
    high = np.full(10, 0.7)
    y_intervals = np.column_stack([low, high])
    result = conformal_metrics(y_true, y_intervals, alpha=0.1)
    assert 0 < result["empirical_coverage"] < 1
    assert result["avg_width"] == pytest.approx(0.4, abs=0.01)
