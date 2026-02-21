"""Unit tests for survival analysis models."""

import numpy as np
import pandas as pd
import pytest

from src.models.survival import (
    generate_lifetime_pd_curve,
    make_survival_target,
    train_cox_ph,
    train_random_survival_forest,
)


@pytest.fixture
def survival_dataset() -> pd.DataFrame:
    """Synthetic survival dataset with time-to-event and censoring."""
    rng = np.random.RandomState(42)
    n = 500
    x1 = rng.normal(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    # Hazard increases with x1
    true_time = np.exp(3 - 0.5 * x1 + rng.gumbel(0, 0.3, n))
    censor_time = rng.exponential(30, n)
    time_to_event = np.minimum(true_time, censor_time)
    event_observed = (true_time <= censor_time).astype(int)

    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "time_to_event": np.clip(time_to_event, 1, 1000),
            "event_observed": event_observed,
        }
    )


# ── make_survival_target ──


def test_make_survival_target_dtype(survival_dataset):
    """Structured array should have (event, time) fields."""
    y = make_survival_target(survival_dataset, event_col="event_observed", time_col="time_to_event")
    assert y.dtype.names == ("event", "time")
    assert y["event"].dtype == bool
    assert len(y) == len(survival_dataset)


def test_make_survival_target_values(survival_dataset):
    """Target values should match source DataFrame."""
    y = make_survival_target(survival_dataset, event_col="event_observed", time_col="time_to_event")
    assert y["event"].sum() == survival_dataset["event_observed"].sum()


# ── train_cox_ph ──


def test_cox_ph_returns_model_and_metrics(survival_dataset):
    """Cox PH should return a fitted model and concordance index."""
    cph, metrics = train_cox_ph(
        survival_dataset,
        duration_col="time_to_event",
        event_col="event_observed",
        feature_cols=["x1", "x2"],
    )
    assert hasattr(cph, "predict_survival_function")
    assert "concordance_index" in metrics
    assert 0 < metrics["concordance_index"] < 1


def test_cox_ph_concordance_above_random(survival_dataset):
    """Concordance index should be above 0.5 (better than random)."""
    _, metrics = train_cox_ph(
        survival_dataset,
        duration_col="time_to_event",
        event_col="event_observed",
        feature_cols=["x1", "x2"],
    )
    assert metrics["concordance_index"] > 0.5


# ── train_random_survival_forest ──


def test_rsf_returns_model_and_metrics(survival_dataset):
    """RSF should return a fitted model and C-index."""
    X = survival_dataset[["x1", "x2"]]
    y = make_survival_target(survival_dataset, event_col="event_observed", time_col="time_to_event")
    n_train = 400
    rsf, metrics = train_random_survival_forest(
        X.iloc[:n_train],
        y[:n_train],
        X.iloc[n_train:],
        y[n_train:],
        n_estimators=50,
    )
    assert hasattr(rsf, "predict_survival_function")
    assert "c_index" in metrics
    assert 0 < metrics["c_index"] < 1


def test_rsf_c_index_above_random(survival_dataset):
    """RSF C-index should beat random baseline."""
    X = survival_dataset[["x1", "x2"]]
    y = make_survival_target(survival_dataset, event_col="event_observed", time_col="time_to_event")
    n_train = 400
    _, metrics = train_random_survival_forest(
        X.iloc[:n_train],
        y[:n_train],
        X.iloc[n_train:],
        y[n_train:],
        n_estimators=50,
    )
    assert metrics["c_index"] > 0.5


# ── generate_lifetime_pd_curve ──


def test_lifetime_pd_curve_shape(survival_dataset):
    """PD curve should have n_loans rows and one column per time point."""
    X = survival_dataset[["x1", "x2"]]
    y = make_survival_target(survival_dataset, event_col="event_observed", time_col="time_to_event")
    rsf, _ = train_random_survival_forest(
        X.iloc[:400], y[:400], X.iloc[400:], y[400:], n_estimators=50
    )
    times = np.array([30, 60, 90])
    pd_curves = generate_lifetime_pd_curve(rsf, X.iloc[400:], times=times)
    assert pd_curves.shape == (100, 3)
    assert all(pd_curves.columns == ["PD_t30", "PD_t60", "PD_t90"])


def test_lifetime_pd_monotonically_increasing(survival_dataset):
    """Lifetime PD should generally increase over time (non-decreasing)."""
    X = survival_dataset[["x1", "x2"]]
    y = make_survival_target(survival_dataset, event_col="event_observed", time_col="time_to_event")
    rsf, _ = train_random_survival_forest(
        X.iloc[:400], y[:400], X.iloc[400:], y[400:], n_estimators=50
    )
    # Use times within the synthetic data's domain (~1-1000, but RSF
    # domain is bounded by observed event times which are much smaller)
    times = np.array([5, 15, 30, 60])
    pd_curves = generate_lifetime_pd_curve(rsf, X.iloc[400:], times=times)
    # Mean PD should increase over time horizons
    means = pd_curves.mean()
    assert means.iloc[-1] >= means.iloc[0], "Mean PD should increase with horizon"
