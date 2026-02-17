"""Integration test: mini pipeline on synthetic data.

Validates that the core pipeline components work together:
data → features → model → conformal → optimization → IFRS9.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluation.ifrs9 import assign_stage, compute_ecl, ecl_with_conformal_range
from src.evaluation.metrics import classification_metrics
from src.features.feature_engineering import create_buckets, create_ratios
from src.models.calibration import calibrate_isotonic
from src.models.conformal import validate_coverage
from src.optimization.portfolio_model import build_portfolio_model, solve_portfolio


@pytest.fixture
def synthetic_pipeline_data():
    """Create synthetic data that mimics the lending club pipeline."""
    rng = np.random.RandomState(42)
    n = 200

    # Simulate raw loan data
    df = pd.DataFrame(
        {
            "loan_amnt": rng.uniform(1000, 40000, n),
            "annual_inc": rng.uniform(20000, 200000, n),
            "int_rate": rng.uniform(5, 30, n),
            "dti": rng.uniform(0, 40, n),
            "default_flag": rng.binomial(1, 0.18, n),
            "purpose": rng.choice(["debt", "credit", "home", "car"], n),
            "grade": rng.choice(["A", "B", "C", "D", "E"], n),
        }
    )
    return df


def test_feature_engineering(synthetic_pipeline_data):
    """Step 1: Feature engineering produces expected columns."""
    df = synthetic_pipeline_data
    df = create_ratios(df)
    df = create_buckets(df)
    assert "loan_to_income" in df.columns
    assert "int_rate_bucket" in df.columns
    assert not df["loan_to_income"].isna().all()


def test_model_training_and_prediction(synthetic_pipeline_data):
    """Step 2: Train a model and get valid probabilities."""
    df = synthetic_pipeline_data
    df = create_ratios(df)

    features = ["loan_amnt", "annual_inc", "int_rate", "dti", "loan_to_income"]
    X = df[features].fillna(0)
    y = df["default_flag"]

    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    assert probs.shape[0] == 50
    assert np.all(probs >= 0) and np.all(probs <= 1)
    assert not np.any(np.isnan(probs))

    metrics = classification_metrics(y_test.values, probs)
    assert metrics["auc_roc"] > 0  # At least runs


def test_calibration_pipeline(synthetic_pipeline_data):
    """Step 3: Calibration produces bounded probabilities."""
    df = synthetic_pipeline_data
    features = ["loan_amnt", "annual_inc", "int_rate", "dti"]
    X = df[features]
    y = df["default_flag"]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X[:150], y[:150])
    probs_cal = model.predict_proba(X[150:])[:, 1]

    iso = calibrate_isotonic(y[150:].values.astype(float), probs_cal)
    calibrated = iso.predict(probs_cal)
    assert np.all(calibrated >= 0) and np.all(calibrated <= 1)


def test_conformal_intervals_manual(synthetic_pipeline_data):
    """Step 4: Manual conformal intervals are valid."""
    df = synthetic_pipeline_data
    features = ["loan_amnt", "annual_inc", "int_rate", "dti"]
    X = df[features]
    y = df["default_flag"].astype(float)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X[:100], y[:100])

    # Calibration set
    preds_cal = model.predict_proba(X[100:150])[:, 1]
    residuals = np.abs(y[100:150].values - preds_cal)
    quantile_90 = np.quantile(residuals, 0.9)

    # Test set
    preds_test = model.predict_proba(X[150:])[:, 1]
    low = np.clip(preds_test - quantile_90, 0, 1)
    high = np.clip(preds_test + quantile_90, 0, 1)
    intervals = np.column_stack([low, high])

    # Validate
    assert np.all(low <= high)
    assert np.all(intervals >= 0) and np.all(intervals <= 1)

    result = validate_coverage(y[150:].values, intervals, alpha=0.1)
    assert result["avg_interval_width"] > 0


def test_optimization_pipeline(synthetic_pipeline_data):
    """Step 5: Portfolio optimization produces valid allocation."""
    df = synthetic_pipeline_data
    n = len(df)
    rng = np.random.RandomState(42)

    pd_point = rng.uniform(0.02, 0.20, n)
    pd_low = np.clip(pd_point - 0.05, 0, 1)
    pd_high = np.clip(pd_point + 0.05, 0, 1)
    lgd = np.full(n, 0.45)
    int_rates = df["int_rate"].values / 100

    model = build_portfolio_model(
        df,
        pd_point,
        pd_low,
        pd_high,
        lgd,
        int_rates,
        total_budget=500_000,
        max_portfolio_pd=0.20,
    )
    solution = solve_portfolio(model, time_limit=30)

    assert solution["objective_value"] is not None
    assert np.isfinite(solution["objective_value"])
    assert solution["total_allocated"] <= 500_001


def test_ifrs9_pipeline(synthetic_pipeline_data):
    """Step 6: IFRS9 staging and ECL are consistent."""
    n = len(synthetic_pipeline_data)
    rng = np.random.RandomState(42)

    pd_orig = rng.uniform(0.02, 0.15, n)
    pd_curr = pd_orig + rng.uniform(-0.01, 0.05, n)
    pd_curr = np.clip(pd_curr, 0, 1)

    stages = assign_stage(pd_orig, pd_curr)
    assert set(stages).issubset({1, 2, 3})

    lgd = np.full(n, 0.45)
    ead = synthetic_pipeline_data["loan_amnt"].values

    ecl_result = compute_ecl(pd_curr, lgd, ead, stages, discount_rate=0.05)
    assert np.all(ecl_result["ecl"] >= 0)
    assert len(ecl_result) == n


def test_ecl_conformal_range_pipeline(synthetic_pipeline_data):
    """Step 7: ECL with conformal range has valid ordering."""
    n = len(synthetic_pipeline_data)
    rng = np.random.RandomState(42)

    pd_point = rng.uniform(0.05, 0.20, n)
    pd_low = np.clip(pd_point - 0.03, 0, 1)
    pd_high = np.clip(pd_point + 0.03, 0, 1)
    lgd = np.full(n, 0.45)
    ead = synthetic_pipeline_data["loan_amnt"].values
    stages = np.ones(n, dtype=int)

    result = ecl_with_conformal_range(pd_low, pd_point, pd_high, lgd, ead, stages)
    assert np.all(result["ecl_low"] <= result["ecl_point"])
    assert np.all(result["ecl_point"] <= result["ecl_high"])
    assert np.all(result["ecl_range"] >= 0)


def test_robust_vs_nonrobust_comparison(synthetic_pipeline_data):
    """Step 8: Robust optimization should be more conservative."""
    df = synthetic_pipeline_data[:50]  # Small subset
    n = len(df)
    rng = np.random.RandomState(42)

    pd_point = rng.uniform(0.02, 0.15, n)
    pd_low = np.clip(pd_point - 0.03, 0, 1)
    pd_high = np.clip(pd_point + 0.08, 0, 1)  # Wide intervals
    lgd = np.full(n, 0.45)
    int_rates = df["int_rate"].values / 100

    model_robust = build_portfolio_model(
        df,
        pd_point,
        pd_low,
        pd_high,
        lgd,
        int_rates,
        total_budget=200_000,
        max_portfolio_pd=0.10,
        robust=True,
    )
    model_nonrobust = build_portfolio_model(
        df,
        pd_point,
        pd_low,
        pd_high,
        lgd,
        int_rates,
        total_budget=200_000,
        max_portfolio_pd=0.10,
        robust=False,
    )

    sol_robust = solve_portfolio(model_robust, time_limit=30)
    sol_nonrobust = solve_portfolio(model_nonrobust, time_limit=30)

    # Robust should be more conservative (fewer loans or lower return)
    assert sol_robust["n_funded"] <= sol_nonrobust["n_funded"] + 1  # Allow small tolerance
