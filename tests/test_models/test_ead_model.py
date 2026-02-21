"""Unit tests for EAD regression model."""

import numpy as np
import pandas as pd
import pytest

from src.models.ead_model import train_ead_model


@pytest.fixture
def ead_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Synthetic EAD dataset (defaults only)."""
    rng = np.random.RandomState(42)
    n_train, n_test = 400, 100

    X_train = pd.DataFrame(
        {
            "loan_amnt": rng.uniform(1000, 50000, n_train),
            "int_rate": rng.uniform(5, 30, n_train),
            "installment": rng.uniform(50, 1500, n_train),
        }
    )
    X_test = pd.DataFrame(
        {
            "loan_amnt": rng.uniform(1000, 50000, n_test),
            "int_rate": rng.uniform(5, 30, n_test),
            "installment": rng.uniform(50, 1500, n_test),
        }
    )
    # EAD is typically a fraction of loan amount
    y_train = pd.Series(X_train["loan_amnt"].values * rng.uniform(0.3, 0.95, n_train), name="ead")
    y_test = pd.Series(X_test["loan_amnt"].values * rng.uniform(0.3, 0.95, n_test), name="ead")

    return X_train, y_train, X_test, y_test


def test_train_returns_model_and_metrics(ead_dataset):
    """Training should return a regressor and metrics dict."""
    X_train, y_train, X_test, y_test = ead_dataset
    model, metrics = train_ead_model(X_train, y_train, X_test, y_test)
    assert hasattr(model, "predict")
    assert "ead_mae" in metrics
    assert "ead_r2" in metrics


def test_r2_above_zero(ead_dataset):
    """R-squared should be positive for a reasonable model."""
    X_train, y_train, X_test, y_test = ead_dataset
    _, metrics = train_ead_model(X_train, y_train, X_test, y_test)
    assert metrics["ead_r2"] > 0, "R² should be positive"


def test_predictions_positive(ead_dataset):
    """EAD predictions should be positive (exposure cannot be negative)."""
    X_train, y_train, X_test, y_test = ead_dataset
    model, _ = train_ead_model(X_train, y_train, X_test, y_test)
    preds = model.predict(X_test)
    assert np.all(preds > 0), "EAD predictions should be positive"


def test_custom_params(ead_dataset):
    """Custom parameters should be accepted."""
    X_train, y_train, X_test, y_test = ead_dataset
    model, metrics = train_ead_model(
        X_train,
        y_train,
        X_test,
        y_test,
        params={"iterations": 50, "depth": 4},
    )
    assert hasattr(model, "predict")
    assert np.isfinite(metrics["ead_mae"])


def test_metrics_are_finite(ead_dataset):
    """All metrics should be finite numbers."""
    X_train, y_train, X_test, y_test = ead_dataset
    _, metrics = train_ead_model(X_train, y_train, X_test, y_test)
    for key, value in metrics.items():
        assert np.isfinite(value), f"{key} is not finite: {value}"
