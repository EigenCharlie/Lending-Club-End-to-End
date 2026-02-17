"""Unit tests for evaluation metrics."""

import numpy as np

from src.evaluation.metrics import (
    classification_metrics,
    compute_all_metrics,
    conformal_metrics,
    ks_statistic,
    regression_metrics,
)

# ── classification_metrics ──


def test_classification_metrics_keys():
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 100)
    y_prob = rng.random(100)
    result = classification_metrics(y_true, y_prob)
    expected_keys = {"auc_roc", "gini", "brier_score", "ece", "ks_statistic"}
    assert expected_keys == set(result.keys())


def test_classification_metrics_perfect():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    result = classification_metrics(y_true, y_prob)
    assert result["auc_roc"] == 1.0
    assert result["gini"] == 1.0


def test_classification_metrics_bounded():
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 200)
    y_prob = rng.random(200)
    result = classification_metrics(y_true, y_prob)
    assert 0 <= result["auc_roc"] <= 1
    assert -1 <= result["gini"] <= 1
    assert 0 <= result["brier_score"] <= 1
    assert 0 <= result["ece"] <= 1
    assert 0 <= result["ks_statistic"] <= 1


# ── ks_statistic ──


def test_ks_statistic_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    ks = ks_statistic(y_true, y_prob)
    assert ks == 1.0


def test_ks_statistic_bounded():
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 200)
    y_prob = rng.random(200)
    ks = ks_statistic(y_true, y_prob)
    assert 0 <= ks <= 1


# ── regression_metrics ──


def test_regression_metrics_perfect():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    result = regression_metrics(y_true, y_pred)
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["r2"] == 1.0


def test_regression_metrics_keys():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.1, 2.2])
    result = regression_metrics(y_true, y_pred)
    assert set(result.keys()) == {"mae", "rmse", "r2"}


# ── conformal_metrics ──


def test_conformal_metrics_full_coverage():
    y_true = np.array([0.5, 0.5, 0.5])
    y_intervals = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    result = conformal_metrics(y_true, y_intervals, alpha=0.1)
    assert result["empirical_coverage"] == 1.0
    assert result["avg_width"] == 1.0


def test_conformal_metrics_keys():
    y_true = np.array([0.5])
    y_intervals = np.array([[0.0, 1.0]])
    result = conformal_metrics(y_true, y_intervals, alpha=0.1)
    expected = {
        "empirical_coverage",
        "target_coverage",
        "coverage_gap",
        "avg_width",
        "median_width",
        "width_std",
        "width_90th_pct",
    }
    assert expected == set(result.keys())


# ── compute_all_metrics ──


def test_compute_all_metrics_without_conformal():
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 100)
    y_prob = rng.random(100)
    result = compute_all_metrics(y_true, y_prob)
    assert "auc_roc" in result
    assert "empirical_coverage" not in result


def test_compute_all_metrics_with_conformal():
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 100).astype(float)
    y_prob = rng.random(100)
    y_intervals = np.column_stack([y_prob - 0.3, y_prob + 0.3])
    result = compute_all_metrics(y_true, y_prob, y_intervals=y_intervals, alpha=0.1)
    assert "auc_roc" in result
    assert "empirical_coverage" in result
