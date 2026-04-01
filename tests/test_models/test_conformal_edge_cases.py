"""Edge-case tests for conformal prediction and new features.

Covers:
- All-positive / all-negative calibration sets
- Degenerate distributions (near-constant features)
- Very small calibration sets
- Brier decomposition
- PSI feature filter
- Monotonic constraints resolver
- sklearn Pipeline wrapper
- Survival conformal intervals
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import brier_score_decomposition
from src.models.conformal import _conformal_quantile, validate_coverage

# ── Conformal edge cases ──────────────────────────────────────────────


def test_conformal_all_positive_calibration():
    """Calibration set with all y=1 should not crash."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    q = _conformal_quantile(scores, alpha=0.1)
    assert 0 <= q <= 1.0


def test_conformal_all_negative_calibration():
    """Calibration set with all y=0 should not crash."""
    scores = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    q = _conformal_quantile(scores, alpha=0.1)
    assert 0 <= q <= 1.0


def test_conformal_very_small_calibration():
    """Calibration set with 2 observations."""
    scores = np.array([0.1, 0.9])
    q = _conformal_quantile(scores, alpha=0.1)
    assert np.isfinite(q)


def test_conformal_constant_scores():
    """All conformity scores identical."""
    scores = np.full(100, 0.5)
    q = _conformal_quantile(scores, alpha=0.1)
    assert q == pytest.approx(0.5)


def test_validate_coverage_all_covered():
    """All observations within intervals → 100% coverage."""
    y_true = np.array([0.3, 0.5, 0.7])
    intervals = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    result = validate_coverage(y_true, intervals, alpha=0.1)
    assert result["empirical_coverage"] == pytest.approx(1.0)


def test_validate_coverage_none_covered():
    """All observations outside intervals → 0% coverage."""
    y_true = np.array([0.0, 0.0, 0.0])
    intervals = np.array([[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]])
    result = validate_coverage(y_true, intervals, alpha=0.1)
    assert result["empirical_coverage"] == pytest.approx(0.0)


# ── Brier decomposition ──────────────────────────────────────────────


def test_brier_decomposition_perfect_predictions():
    """Perfect predictions: reliability=0, resolution=uncertainty."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    y_prob = y_true.astype(float)
    result = brier_score_decomposition(y_true, y_prob, n_bins=5)
    assert result["reliability"] == pytest.approx(0.0, abs=0.01)
    assert result["brier_decomposed"] == pytest.approx(0.0, abs=0.01)


def test_brier_decomposition_constant_predictions():
    """Constant predictions at base rate: reliability=0, resolution=0."""
    y_true = np.array([0, 0, 0, 1, 1])
    base_rate = y_true.mean()
    y_prob = np.full_like(y_true, base_rate, dtype=float)
    result = brier_score_decomposition(y_true, y_prob, n_bins=3)
    assert result["reliability"] == pytest.approx(0.0, abs=0.02)
    assert result["resolution"] == pytest.approx(0.0, abs=0.02)
    assert result["uncertainty"] == pytest.approx(base_rate * (1 - base_rate), abs=0.01)


def test_brier_decomposition_components_sum():
    """Verify: Brier = Reliability - Resolution + Uncertainty."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 500)
    y_prob = np.clip(y_true + rng.normal(0, 0.3, 500), 0, 1)
    result = brier_score_decomposition(y_true, y_prob, n_bins=10)
    expected_brier = result["reliability"] - result["resolution"] + result["uncertainty"]
    assert result["brier_decomposed"] == pytest.approx(expected_brier, abs=0.01)


def test_brier_decomposition_shares_nonnegative():
    """Miscalibration and discrimination shares should be non-negative."""
    rng = np.random.RandomState(99)
    y_true = rng.randint(0, 2, 200)
    y_prob = np.clip(rng.random(200), 0.01, 0.99)
    result = brier_score_decomposition(y_true, y_prob)
    assert result["miscalibration_share"] >= 0
    assert result["discrimination_share"] >= 0


# ── PSI feature filter ────────────────────────────────────────────────


def test_filter_high_psi_stable():
    """Features with same distribution should be stable."""
    from src.evaluation.backtesting import filter_high_psi_features

    rng = np.random.RandomState(42)
    train = pd.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(5, 2, 1000)})
    test = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(5, 2, 500)})
    result = filter_high_psi_features(train, test, ["a", "b"])
    assert len(result["stable_features"]) == 2
    assert len(result["drifted_features"]) == 0


def test_filter_high_psi_detects_drift():
    """Feature with shifted distribution should be flagged."""
    from src.evaluation.backtesting import filter_high_psi_features

    rng = np.random.RandomState(42)
    train = pd.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(0, 1, 1000)})
    # Shift 'b' by 5 standard deviations → extreme drift
    test = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(5, 1, 500)})
    result = filter_high_psi_features(train, test, ["a", "b"], psi_threshold=0.25)
    assert "b" in result["drifted_features"]
    assert "a" in result["stable_features"]


# ── Monotonic constraints ────────────────────────────────────────────


def test_resolve_monotonic_constraints_basic():
    """Direct config dict produces correct constraint string."""
    from src.models.pd_model import resolve_monotonic_constraints

    features = ["int_rate", "annual_inc", "dti", "grade_woe"]
    constraints = {"int_rate": 1, "annual_inc": -1, "dti": 1}
    result = resolve_monotonic_constraints(features, constraints_config=constraints)
    assert result == "1,-1,1,0"


def test_resolve_monotonic_constraints_empty():
    """No constraints configured returns None."""
    from src.models.pd_model import resolve_monotonic_constraints

    features = ["a", "b", "c"]
    result = resolve_monotonic_constraints(features, constraints_config={})
    assert result is None


def test_resolve_monotonic_constraints_all_zero():
    """All features unconstrained returns None."""
    from src.models.pd_model import resolve_monotonic_constraints

    features = ["a", "b"]
    result = resolve_monotonic_constraints(features, constraints_config={"x": 1})
    assert result is None


# ── sklearn Pipeline wrapper ─────────────────────────────────────────


def test_pd_pipeline_feature_selector():
    """Feature selector extracts and orders columns correctly."""
    from src.models.pd_pipeline import _make_feature_selector

    selector = _make_feature_selector(["b", "a", "c"])
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "extra": [7, 8]})
    result = selector.transform(df)
    assert list(result.columns) == ["b", "a", "c"]
    assert result["b"].tolist() == [3, 4]


def test_pd_pipeline_feature_selector_missing_col():
    """Missing column filled with default value."""
    from src.models.pd_pipeline import _make_feature_selector

    selector = _make_feature_selector(["a", "missing"], fill_value=-1.0)
    df = pd.DataFrame({"a": [1, 2]})
    result = selector.transform(df)
    assert list(result.columns) == ["a", "missing"]
    assert result["missing"].tolist() == [-1.0, -1.0]


def test_catboost_sklearn_adapter_fit_predict():
    """CatBoostSklearnAdapter can fit and predict."""
    from src.models.pd_pipeline import CatBoostSklearnAdapter

    rng = np.random.RandomState(42)
    X = pd.DataFrame({"a": rng.random(100), "b": rng.random(100)})
    y = rng.randint(0, 2, 100)

    adapter = CatBoostSklearnAdapter(
        catboost_params={"iterations": 10, "verbose": 0, "allow_writing_files": False}
    )
    adapter.fit(X, y)
    proba = adapter.predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)


# ── Survival conformal intervals ─────────────────────────────────────


def test_survival_conformal_intervals_shape():
    """Survival conformal should return arrays of correct shape."""
    from src.models.survival import compute_lifetime_pd_conformal_intervals

    class FakeSurvivalModel:
        def predict_survival_function(self, X):
            """Return fake survival functions."""
            n = len(X)
            return [
                lambda t, s=s: max(0.0, 1.0 - s * 0.001 * t)
                for s in np.random.RandomState(42).random(n)
            ]

    model = FakeSurvivalModel()
    rng = np.random.RandomState(42)
    n_cal, n_test = 100, 30
    X_cal = pd.DataFrame({"a": rng.random(n_cal)})
    y_event = rng.choice([True, False], n_cal, p=[0.3, 0.7])
    y_time = rng.uniform(100, 2000, n_cal)
    X_test = pd.DataFrame({"a": rng.random(n_test)})

    result = compute_lifetime_pd_conformal_intervals(
        model,
        X_cal,
        y_event,
        y_time,
        X_test,
        horizon_months=60,
        confidence_level=0.90,
    )
    assert result["pd_lifetime_point"].shape == (n_test,)
    assert result["pd_lifetime_low"].shape == (n_test,)
    assert result["pd_lifetime_high"].shape == (n_test,)
    assert np.all(result["pd_lifetime_low"] <= result["pd_lifetime_high"])
    assert np.all(result["pd_lifetime_low"] >= 0)
    assert np.all(result["pd_lifetime_high"] <= 1)
