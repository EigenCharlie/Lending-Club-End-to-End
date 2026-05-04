"""Unit tests for conformal prediction utilities."""

import numpy as np
import pandas as pd
import pytest

from src.models.conformal import (
    ProbabilityRegressor,
    _conformal_quantile,
    apply_probability_calibrator,
    build_mondrian_partition_labels,
    conditional_coverage_by_group,
    create_classification_sets,
    create_classification_sets_mondrian,
    create_cross_conformal_score_intervals,
    create_pd_intervals_mondrian,
    summarize_prediction_sets,
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


# ── Edge Cases ──


def test_conformal_quantile_single_element():
    """Quantile on single-element array should return that element."""
    result = _conformal_quantile(np.array([0.5]), 0.9)
    assert isinstance(result, float)


def test_validate_coverage_inverted_intervals():
    """Intervals where low > high should yield zero coverage."""
    y_true = np.array([0.5, 0.5])
    y_intervals = np.array([[0.8, 0.2], [0.9, 0.1]])  # inverted
    result = validate_coverage(y_true, y_intervals, alpha=0.1)
    assert result["empirical_coverage"] == 0.0


def test_validate_coverage_with_nans_in_y_true():
    """NaN in y_true should not cause crash."""
    y_true = np.array([0.5, np.nan, 0.3])
    y_intervals = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    # Should not raise — coverage calculation handles NaN gracefully
    result = validate_coverage(y_true, y_intervals, alpha=0.1)
    assert "empirical_coverage" in result


def test_conditional_coverage_single_group():
    """One group should produce a single-row result."""
    y_true = np.array([0.2, 0.5, 0.8])
    y_intervals = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    groups = pd.Series(["A", "A", "A"])
    result = conditional_coverage_by_group(y_true, y_intervals, groups)
    assert len(result) == 1
    assert result["coverage"].iloc[0] == 1.0


def test_build_mondrian_partition_labels_score_decile_returns_score_bands():
    y_prob_cal = np.linspace(0.01, 0.99, 20)
    y_prob_eval = np.array([0.05, 0.55, 0.95])

    group_cal, group_eval, meta = build_mondrian_partition_labels(
        y_prob_cal=y_prob_cal,
        y_prob_eval=y_prob_eval,
        partition="score_decile_mondrian",
    )

    assert len(group_cal) == len(y_prob_cal)
    assert len(group_eval) == len(y_prob_eval)
    assert meta["partition"] == "score_decile_mondrian"
    assert meta["score_band_count"] >= 2
    assert all(label.startswith("score_q") for label in group_cal)


def test_build_mondrian_partition_labels_hybrid_falls_back_for_small_groups():
    y_prob_cal = np.linspace(0.01, 0.99, 12)
    y_prob_eval = np.array([0.15, 0.85])
    base_groups_cal = pd.Series(["A"] * 10 + ["B"] * 2)
    base_groups_eval = pd.Series(["A", "B"])

    group_cal, group_eval, meta = build_mondrian_partition_labels(
        y_prob_cal=y_prob_cal,
        y_prob_eval=y_prob_eval,
        partition="grade_x_scoreband_mondrian",
        base_groups_cal=base_groups_cal,
        base_groups_eval=base_groups_eval,
        n_score_bins=4,
        min_group_size=5,
    )

    assert meta["partition"] == "grade_x_scoreband_mondrian"
    assert meta["fallback_groups"]
    assert "GLOBAL" in set(group_eval)
    assert all(isinstance(label, str) for label in group_cal)


def test_build_mondrian_partition_labels_global_only_fallback_uses_global():
    y_prob_cal = np.linspace(0.01, 0.99, 10)
    y_prob_eval = np.array([0.10, 0.90])
    base_groups_cal = pd.Series(["A"] * 8 + ["B"] * 2)
    base_groups_eval = pd.Series(["A", "B"])

    group_cal, group_eval, meta = build_mondrian_partition_labels(
        y_prob_cal=y_prob_cal,
        y_prob_eval=y_prob_eval,
        partition="grade_x_scoreband_mondrian",
        base_groups_cal=base_groups_cal,
        base_groups_eval=base_groups_eval,
        n_score_bins=5,
        min_group_size=20,
        fallback_mode="global_only",
    )

    assert meta["fallback_mode"] == "global_only"
    assert set(group_cal) == {"GLOBAL"}
    assert set(group_eval) == {"GLOBAL"}


def test_summarize_prediction_sets_reports_ambiguity_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    y_sets = np.array(
        [
            [1, 0],  # singleton negative
            [0, 1],  # singleton positive
            [1, 1],  # ambiguous
            [0, 0],  # empty
        ]
    )

    result = summarize_prediction_sets(y_true, y_pred, y_sets)

    assert result["singleton_rate"] == pytest.approx(0.5)
    assert result["ambiguity_rate"] == pytest.approx(0.25)
    assert result["empty_set_rate"] == pytest.approx(0.25)
    assert result["set_coverage"] == pytest.approx(0.75)


def test_create_classification_sets_margin_returns_valid_sets():
    clf = FakeBinaryClassifier(seed=7)
    rng = np.random.RandomState(7)
    X_cal = pd.DataFrame({"a": rng.random(120), "b": rng.random(120)})
    y_cal = pd.Series(rng.randint(0, 2, 120))
    X_test = pd.DataFrame({"a": rng.random(30), "b": rng.random(30)})

    y_pred, y_sets = create_classification_sets(
        classifier=clf,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        alpha=0.10,
        method="margin",
    )

    assert y_pred.shape == (30,)
    assert y_sets.shape == (30, 2)
    assert np.isin(y_sets, [0, 1]).all()


def test_create_classification_sets_mondrian_supports_margin_with_fallback():
    clf = FakeBinaryClassifier(seed=9)
    rng = np.random.RandomState(9)
    X_cal = pd.DataFrame({"a": rng.random(40), "b": rng.random(40)})
    y_cal = pd.Series(rng.randint(0, 2, 40))
    X_test = pd.DataFrame({"a": rng.random(12), "b": rng.random(12)})
    group_cal = pd.Series(["A"] * 30 + ["B"] * 10)
    group_test = pd.Series(["A"] * 6 + ["B"] * 6)

    y_pred, y_sets, diagnostics = create_classification_sets_mondrian(
        classifier=clf,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        group_cal=group_cal,
        group_test=group_test,
        alpha=0.10,
        method="margin",
        min_group_size=25,
    )

    assert y_pred.shape == (12,)
    assert y_sets.shape == (12, 2)
    assert "B" in diagnostics["fallback_groups"]


def test_create_pd_intervals_mondrian_supports_score_scale_family():
    clf = FakeBinaryClassifier(seed=11)
    rng = np.random.RandomState(11)
    X_cal = pd.DataFrame({"a": rng.random(120), "b": rng.random(120)})
    y_cal = pd.Series(rng.randint(0, 2, 120))
    X_test = pd.DataFrame({"a": rng.random(24), "b": rng.random(24)})
    group_cal = pd.Series(["A"] * 60 + ["B"] * 60)
    group_test = pd.Series(["A"] * 12 + ["B"] * 12)

    y_pred, y_intervals, diagnostics = create_pd_intervals_mondrian(
        classifier=clf,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        group_cal=group_cal,
        group_test=group_test,
        alpha=0.10,
        min_group_size=20,
        score_scale_family="bernoulli_sqrt_clipped_0.02",
    )

    assert y_pred.shape == (24,)
    assert y_intervals.shape == (24, 2)
    assert diagnostics["score_scale_family"] == "bernoulli_sqrt_clipped_0.02"


def test_cross_conformal_score_intervals_output_shape():
    rng = np.random.RandomState(42)
    y_cal = rng.binomial(1, 0.25, size=80)
    y_prob_cal = rng.uniform(0.05, 0.8, size=80)
    y_prob_test = rng.uniform(0.05, 0.8, size=20)

    y_pred, y_intervals = create_cross_conformal_score_intervals(
        y_cal=y_cal,
        y_prob_cal=y_prob_cal,
        y_prob_test=y_prob_test,
        alpha=0.1,
        cv=3,
    )

    assert y_pred.shape == (20,)
    assert y_intervals.shape == (20, 2)
    assert np.all(y_pred >= 0.0)
    assert np.all(y_pred <= 1.0)
    assert np.all(y_intervals[:, 0] <= y_intervals[:, 1])
    assert np.all(y_intervals >= 0.0)
    assert np.all(y_intervals <= 1.0)


# ── Residual Intervals (bootstrap-style benchmark) ──


def test_residual_intervals_output_shape():
    from src.models.conformal import create_residual_intervals

    clf = FakeClassifier()
    X_cal = pd.DataFrame({"a": np.random.RandomState(0).random(100)})
    y_cal = pd.Series(np.random.RandomState(0).random(100))
    X_test = pd.DataFrame({"a": np.random.RandomState(1).random(20)})

    y_pred, y_intervals = create_residual_intervals(clf, X_cal, y_cal, X_test, alpha=0.1)
    assert y_pred.shape == (20,)
    assert y_intervals.shape == (20, 2)


def test_residual_intervals_width_positive():
    from src.models.conformal import create_residual_intervals

    clf = FakeClassifier()
    X_cal = pd.DataFrame({"a": np.random.RandomState(0).random(100)})
    y_cal = pd.Series(np.random.RandomState(0).random(100))
    X_test = pd.DataFrame({"a": np.random.RandomState(1).random(20)})

    _, y_intervals = create_residual_intervals(clf, X_cal, y_cal, X_test, alpha=0.1)
    widths = y_intervals[:, 1] - y_intervals[:, 0]
    assert np.all(widths > 0)


def test_residual_intervals_narrower_with_higher_alpha():
    from src.models.conformal import create_residual_intervals

    clf = FakeClassifier()
    X_cal = pd.DataFrame({"a": np.random.RandomState(0).random(200)})
    y_cal = pd.Series(np.random.RandomState(0).random(200))
    X_test = pd.DataFrame({"a": np.random.RandomState(1).random(30)})

    _, iv_90 = create_residual_intervals(clf, X_cal, y_cal, X_test, alpha=0.10)
    _, iv_50 = create_residual_intervals(clf, X_cal, y_cal, X_test, alpha=0.50)

    w_90 = (iv_90[:, 1] - iv_90[:, 0]).mean()
    w_50 = (iv_50[:, 1] - iv_50[:, 0]).mean()
    assert w_90 > w_50  # 90% intervals should be wider than 50%


# ── Venn-Abers ──


class FakeBinaryClassifier:
    """Classifier stub that returns probabilities close to the true label."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self._classes = np.array([0, 1])

    @property
    def classes_(self):
        return self._classes

    def predict_proba(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        probs = self.rng.random(n) * 0.5 + 0.25  # between 0.25 and 0.75
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def test_venn_abers_output_shape():
    """Venn-Abers should return point predictions and p0/p1 arrays."""
    pytest.importorskip("venn_abers")
    from src.models.conformal import create_pd_intervals_venn_abers

    clf = FakeBinaryClassifier(seed=42)
    rng = np.random.RandomState(42)
    n_cal, n_test = 200, 50
    X_cal = pd.DataFrame({"a": rng.random(n_cal), "b": rng.random(n_cal)})
    y_cal = pd.Series(rng.randint(0, 2, n_cal))
    X_test = pd.DataFrame({"a": rng.random(n_test), "b": rng.random(n_test)})

    y_pred, p0, p1 = create_pd_intervals_venn_abers(clf, X_cal, y_cal, X_test)
    assert y_pred.shape == (n_test,)
    assert p0.shape == (n_test,)
    assert p1.shape == (n_test,)


def test_venn_abers_bounds_valid():
    """p0 <= p1 and both in [0, 1]."""
    pytest.importorskip("venn_abers")
    from src.models.conformal import create_pd_intervals_venn_abers

    clf = FakeBinaryClassifier(seed=123)
    rng = np.random.RandomState(123)
    n_cal, n_test = 300, 80
    X_cal = pd.DataFrame({"a": rng.random(n_cal), "b": rng.random(n_cal)})
    y_cal = pd.Series(rng.randint(0, 2, n_cal))
    X_test = pd.DataFrame({"a": rng.random(n_test), "b": rng.random(n_test)})

    y_pred, p0, p1 = create_pd_intervals_venn_abers(clf, X_cal, y_cal, X_test)
    assert np.all(p0 >= 0)
    assert np.all(p1 <= 1)
    assert np.all(p0 <= p1)
    assert np.all(y_pred >= p0)
    assert np.all(y_pred <= p1)
