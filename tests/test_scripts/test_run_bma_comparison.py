"""Tests for scripts/run_bma_comparison.py — unit tests for BMA interval computation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_test_predictions(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    y_true = rng.binomial(1, 0.22, size=n).astype(float)
    return pd.DataFrame(
        {
            "loan_id": [f"L{i}" for i in range(n)],
            "y_true": y_true,
            "y_prob_lr": rng.beta(2, 8, size=n).astype(np.float32),
            "y_prob_cb_default": rng.beta(2, 7, size=n).astype(np.float32),
            "y_prob_cb_tuned": rng.beta(2, 7, size=n).astype(np.float32),
            "grade": rng.choice(["A", "B", "C", "D"], size=n),
        }
    )


def test_bma_coverage_hits_nominal() -> None:
    """BMA intervals must achieve exactly the nominal coverage by construction."""
    from scripts.run_bma_comparison import _compute_bma_intervals

    df = _make_test_predictions(n=500)
    result = _compute_bma_intervals(df, weighting="equal", nominal_coverage=0.90)
    actual_coverage = float(result["bma_covered_90"].mean())
    # The empirical quantile construction guarantees exact nominal coverage on the same data
    assert abs(actual_coverage - 0.90) < 0.005, (
        f"Expected coverage ≈ 90%, got {actual_coverage:.4%}"
    )


def test_bma_output_columns() -> None:
    from scripts.run_bma_comparison import _compute_bma_intervals

    df = _make_test_predictions(n=200)
    result = _compute_bma_intervals(df, weighting="equal")
    required = {
        "bma_mean",
        "bma_std",
        "bma_low_90",
        "bma_high_90",
        "bma_width_90",
        "bma_covered_90",
    }
    assert required.issubset(result.columns)


def test_bma_intervals_clipped_to_unit() -> None:
    """All BMA bounds must be in [0, 1]."""
    from scripts.run_bma_comparison import _compute_bma_intervals

    df = _make_test_predictions(n=300)
    result = _compute_bma_intervals(df, weighting="equal")
    assert (result["bma_low_90"] >= 0).all()
    assert (result["bma_high_90"] <= 1).all()
    assert (result["bma_width_90"] >= 0).all()


def test_bma_auc_weights_bounded() -> None:
    """AUC-based weights should also produce valid intervals."""
    from scripts.run_bma_comparison import _compute_bma_intervals

    df = _make_test_predictions(n=400)
    result = _compute_bma_intervals(df, weighting="auc", nominal_coverage=0.90)
    actual_coverage = float(result["bma_covered_90"].mean())
    assert abs(actual_coverage - 0.90) < 0.005


def test_grade_coverage_all_grades_present() -> None:
    from scripts.run_bma_comparison import (
        _compute_bma_intervals,
        _conformal_covered,
        _grade_coverage,
    )

    df = _make_test_predictions(n=400, seed=3)
    # Make fake conformal columns for testing
    rng = np.random.RandomState(5)
    df["pd_low_90"] = rng.uniform(0.0, 0.1, len(df)).astype(np.float32)
    df["pd_high_90"] = rng.uniform(0.7, 1.0, len(df)).astype(np.float32)
    df["width_90"] = df["pd_high_90"] - df["pd_low_90"]

    bma = _compute_bma_intervals(df, weighting="equal")
    bma["cp_covered_90"] = _conformal_covered(df)
    bma["width_90"] = df["width_90"].values

    grade_cov = _grade_coverage(bma)
    assert set(grade_cov["grade"]) == {"A", "B", "C", "D"}
    assert (grade_cov["bma_coverage"] >= 0).all()
    assert (grade_cov["cp_coverage"] >= 0).all()
