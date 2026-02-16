"""Unit tests for PD model training and calibration."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.models.pd_model import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    get_available_features,
    train_baseline,
)
from src.models.calibration import (
    calibrate_isotonic,
    expected_calibration_error,
    evaluate_calibration,
)


@pytest.fixture
def binary_dataset():
    """Create a synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42,
        flip_y=0.15,
    )
    cols = ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"]
    X_train = pd.DataFrame(X[:300], columns=cols)
    X_test = pd.DataFrame(X[300:], columns=cols)
    y_train = pd.Series(y[:300])
    y_test = pd.Series(y[300:])
    return X_train, y_train, X_test, y_test


# ── get_available_features ──


def test_get_available_features_filters_correctly():
    df = pd.DataFrame({"loan_amnt": [1], "annual_inc": [2], "fake_col": [3]})
    result = get_available_features(df)
    assert "loan_amnt" in result
    assert "annual_inc" in result
    assert "fake_col" not in result


def test_get_available_features_empty_df():
    df = pd.DataFrame({"unrelated": [1]})
    result = get_available_features(df)
    assert result == []


# ── train_baseline ──


def test_baseline_returns_model_and_metrics(binary_dataset):
    X_train, y_train, X_test, y_test = binary_dataset
    model, metrics = train_baseline(X_train, y_train, X_test, y_test)
    assert hasattr(model, "predict_proba")
    assert "auc_roc" in metrics
    assert metrics["model_type"] == "logistic_regression"


def test_baseline_auc_above_random(binary_dataset):
    X_train, y_train, X_test, y_test = binary_dataset
    _, metrics = train_baseline(X_train, y_train, X_test, y_test)
    assert metrics["auc_roc"] > 0.5, "AUC should be better than random"


def test_baseline_probabilities_bounded(binary_dataset):
    X_train, y_train, X_test, y_test = binary_dataset
    model, _ = train_baseline(X_train, y_train, X_test, y_test)
    probs = model.predict_proba(X_test)[:, 1]
    assert np.all(probs >= 0), "Probabilities must be >= 0"
    assert np.all(probs <= 1), "Probabilities must be <= 1"
    assert not np.any(np.isnan(probs)), "No NaN probabilities allowed"


# ── Calibration ──


def test_expected_calibration_error_perfect():
    """Perfect calibration should have ECE ~ 0."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ece = expected_calibration_error(y_true, y_prob, n_bins=5)
    assert ece < 0.01


def test_expected_calibration_error_bounded():
    """ECE should be between 0 and 1."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=200)
    y_prob = rng.random(200)
    ece = expected_calibration_error(y_true, y_prob)
    assert 0 <= ece <= 1


def test_calibrate_isotonic_output_bounded():
    """Isotonic calibrator should produce probabilities in [0, 1]."""
    rng = np.random.RandomState(42)
    y_cal = rng.randint(0, 2, size=200).astype(float)
    proba_cal = rng.random(200)
    iso = calibrate_isotonic(y_cal, proba_cal)
    calibrated = iso.predict(np.linspace(0, 1, 50))
    assert np.all(calibrated >= 0)
    assert np.all(calibrated <= 1)


def test_evaluate_calibration_returns_dict(binary_dataset):
    X_train, y_train, X_test, y_test = binary_dataset
    model, _ = train_baseline(X_train, y_train, X_test, y_test)
    probs = model.predict_proba(X_test)[:, 1]
    result = evaluate_calibration(y_test.values, probs, name="test")
    assert "ece" in result
    assert "brier_score" in result
    assert result["ece"] >= 0
    assert 0 <= result["brier_score"] <= 1
