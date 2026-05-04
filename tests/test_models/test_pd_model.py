"""Unit tests for PD model training and calibration."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.models.calibration import (
    calibrate_isotonic,
    evaluate_calibration,
    expected_calibration_error,
)
from src.models.optuna_tuning import SEARCH_SPACE_VERSION, resolve_optuna_study_name
from src.models.pd_model import (
    get_available_features,
    temporal_train_val_split,
    train_baseline,
    train_catboost_default,
    train_catboost_tuned_optuna,
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


@pytest.fixture
def catboost_dataset():
    """Synthetic dataset with one categorical column for CatBoost tests."""
    rng = np.random.RandomState(7)
    n = 700
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    bucket = np.where(x1 + x2 > 0.5, "high", np.where(x1 + x2 < -0.5, "low", "mid"))
    signal = 1.2 * x1 - 0.7 * x2 + (bucket == "high") * 0.8 + (bucket == "low") * (-0.5)
    y = (signal + rng.normal(0, 0.8, n) > 0).astype(int)

    df = pd.DataFrame(
        {
            "issue_d": pd.date_range("2014-01-01", periods=n, freq="D"),
            "x1": x1,
            "x2": x2,
            "bucket": bucket.astype(str),
            "target": y,
        }
    )
    train_df = df.iloc[:560].copy().reset_index(drop=True)
    test_df = df.iloc[560:].copy().reset_index(drop=True)
    return train_df, test_df


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


def test_temporal_train_val_split_keeps_order(catboost_dataset):
    train_df, _ = catboost_dataset
    fit_df, val_df = temporal_train_val_split(train_df, val_fraction=0.2, date_col="issue_d")
    assert not fit_df.empty
    assert not val_df.empty
    assert fit_df["issue_d"].max() <= val_df["issue_d"].min()


def test_catboost_tuned_and_default_predictions_differ(catboost_dataset):
    train_df, test_df = catboost_dataset
    fit_df, val_df = temporal_train_val_split(train_df, val_fraction=0.2, date_col="issue_d")

    X_fit = fit_df[["x1", "x2", "bucket"]].copy()
    y_fit = fit_df["target"].astype(int)
    X_val = val_df[["x1", "x2", "bucket"]].copy()
    y_val = val_df["target"].astype(int)
    X_test = test_df[["x1", "x2", "bucket"]].copy()
    y_test = test_df["target"].astype(int)

    cb_default, _ = train_catboost_default(
        X_fit,
        y_fit,
        X_val,
        y_val,
        X_test=X_test,
        y_test=y_test,
        cat_features=["bucket"],
        params={"iterations": 120, "early_stopping_rounds": 25, "verbose": 0},
    )
    cb_tuned, tuned_metrics = train_catboost_tuned_optuna(
        X_fit,
        y_fit,
        X_val,
        y_val,
        X_test=X_test,
        y_test=y_test,
        cat_features=["bucket"],
        base_params={"iterations": 120, "early_stopping_rounds": 25},
        n_trials=5,
        sampler="tpe",
        pruner="median",
        timeout_minutes=0,
    )

    y_default = cb_default.predict_proba(X_test)[:, 1]
    y_tuned = cb_tuned.predict_proba(X_test)[:, 1]

    assert tuned_metrics["hpo_trials_executed"] >= 1
    assert tuned_metrics["validation_auc"] >= 0.5
    assert not np.allclose(y_default, y_tuned), (
        "Tuned and default CatBoost should produce different predictions when HPO is active."
    )


def test_resolve_optuna_study_name_appends_search_space_suffix_once():
    resolved = resolve_optuna_study_name("pd_catboost_optuna_temporal")
    assert resolved.endswith(f"__{SEARCH_SPACE_VERSION}")
    assert resolve_optuna_study_name(resolved) == resolved


def test_local_refine_best_params_are_materialized_for_catboost():
    df = pd.DataFrame(
        {
            "issue_d": pd.date_range("2019-01-01", periods=120, freq="D"),
            "x1": np.linspace(0.0, 1.0, 120),
            "x2": np.linspace(1.0, 0.0, 120),
            "bucket": np.where(np.arange(120) % 2 == 0, "A", "B"),
            "target": (np.arange(120) % 3 == 0).astype(int),
        }
    )
    train_df = df.iloc[:80].copy()
    val_df = df.iloc[80:100].copy()
    test_df = df.iloc[100:].copy()

    X_fit = train_df[["x1", "x2", "bucket"]].copy()
    y_fit = train_df["target"].astype(int)
    X_val = val_df[["x1", "x2", "bucket"]].copy()
    y_val = val_df["target"].astype(int)
    X_test = test_df[["x1", "x2", "bucket"]].copy()
    y_test = test_df["target"].astype(int)

    _, tuned_metrics = train_catboost_tuned_optuna(
        X_fit,
        y_fit,
        X_val,
        y_val,
        X_test=X_test,
        y_test=y_test,
        cat_features=["bucket"],
        base_params={"iterations": 60, "early_stopping_rounds": 15},
        n_trials=2,
        sampler="tpe",
        pruner="median",
        timeout_minutes=0,
        search_space_mode="local_refine",
        local_refine_space={
            "enqueue_base_trial": False,
            "iterations": {"choices": [60]},
            "learning_rate": {"choices": [0.05]},
            "depth": {"choices": [4]},
            "l2_leaf_reg": {"choices": [3.0]},
            "min_data_in_leaf": {"choices": [20]},
            "random_strength": {"choices": [1e-6]},
            "border_count": {"choices": [64]},
            "subsample": {"choices": [0.8]},
            "leaf_estimation_iterations": {"choices": [2]},
            "feature_weights": {"x1": [1.2]},
            "first_feature_use_penalties": {"x2": [0.5]},
            "penalties_coefficient": [1.25],
        },
    )

    resolved = tuned_metrics["best_params_resolved"]
    assert "feature_weight__x1" not in resolved
    assert "first_use_penalty__x2" not in resolved
    assert resolved["feature_weights"]["x1"] == pytest.approx(1.2)
    assert resolved["first_feature_use_penalties"]["x2"] == pytest.approx(0.5)


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
