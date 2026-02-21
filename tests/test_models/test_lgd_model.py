"""Unit tests for LGD two-stage model."""

import numpy as np
import pandas as pd
import pytest

from src.models.lgd_model import predict_two_stage, train_two_stage_lgd


@pytest.fixture
def lgd_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Synthetic LGD dataset with mix of zero and positive LGD values."""
    rng = np.random.RandomState(42)
    n_train, n_test = 400, 100

    X_train = pd.DataFrame(
        {
            "loan_amnt": rng.uniform(1000, 50000, n_train),
            "int_rate": rng.uniform(5, 30, n_train),
            "dti": rng.uniform(0, 40, n_train),
        }
    )
    X_test = pd.DataFrame(
        {
            "loan_amnt": rng.uniform(1000, 50000, n_test),
            "int_rate": rng.uniform(5, 30, n_test),
            "dti": rng.uniform(0, 40, n_test),
        }
    )
    # ~30% have zero LGD (full recovery), rest have LGD in (0, 1)
    y_train_raw = rng.beta(2, 5, n_train)
    y_train_raw[rng.random(n_train) < 0.3] = 0.0
    y_train = pd.Series(y_train_raw, name="lgd")

    y_test_raw = rng.beta(2, 5, n_test)
    y_test_raw[rng.random(n_test) < 0.3] = 0.0
    y_test = pd.Series(y_test_raw, name="lgd")

    return X_train, y_train, X_test, y_test


def test_train_returns_models_and_metrics(lgd_dataset):
    """Training should return classifier, regressor, and metrics dict."""
    X_train, y_train, X_test, y_test = lgd_dataset
    clf, reg, metrics = train_two_stage_lgd(X_train, y_train, X_test, y_test)
    assert hasattr(clf, "predict_proba")
    assert hasattr(reg, "predict")
    assert "stage1_auc" in metrics
    assert "lgd_mae" in metrics
    assert "lgd_rmse" in metrics


def test_stage1_auc_above_random(lgd_dataset):
    """Stage 1 classifier should do better than random."""
    X_train, y_train, X_test, y_test = lgd_dataset
    _, _, metrics = train_two_stage_lgd(X_train, y_train, X_test, y_test)
    assert metrics["stage1_auc"] > 0.5


def test_predictions_bounded_zero_one(lgd_dataset):
    """Two-stage predictions should be in [0, 1]."""
    X_train, y_train, X_test, y_test = lgd_dataset
    clf, reg, _ = train_two_stage_lgd(X_train, y_train, X_test, y_test)
    preds = predict_two_stage(clf, reg, X_test)
    assert preds.shape == (len(X_test),)
    assert np.all(preds >= 0), "LGD predictions must be >= 0"
    assert np.all(preds <= 1), "LGD predictions must be <= 1"


def test_metrics_are_finite(lgd_dataset):
    """All metrics should be finite numbers."""
    X_train, y_train, X_test, y_test = lgd_dataset
    _, _, metrics = train_two_stage_lgd(X_train, y_train, X_test, y_test)
    for key, value in metrics.items():
        assert np.isfinite(value), f"{key} is not finite: {value}"


def test_mae_reasonable(lgd_dataset):
    """MAE should be below 1.0 for a reasonable model."""
    X_train, y_train, X_test, y_test = lgd_dataset
    _, _, metrics = train_two_stage_lgd(X_train, y_train, X_test, y_test)
    assert metrics["lgd_mae"] < 1.0
