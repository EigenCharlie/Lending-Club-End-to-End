"""Tests for scripts/run_sicr_conformal.py — unit tests for key components.

Tests the core functions independently of real data artifacts.
Covers ECL computation, SICR grid logic, and alpha sensitivity scaling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_conformal_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic conformal intervals dataframe mimicking the test set."""
    rng = np.random.RandomState(seed)
    grades = rng.choice(["A", "B", "C", "D", "E"], size=n)
    return pd.DataFrame(
        {
            "y_true": rng.binomial(1, 0.22, size=n),
            "y_pred": rng.beta(2, 8, size=n).astype(np.float32),
            "width_90": rng.beta(1, 2, size=n).astype(np.float32),
            "grade": grades,
            "loan_amnt": rng.uniform(5_000, 35_000, size=n).astype(np.float32),
        }
    )


GRADE_LIFETIME_PD = {
    "A": 0.178,
    "B": 0.266,
    "C": 0.352,
    "D": 0.456,
    "E": 0.634,
    "ALL": 0.380,
}

LGD = 0.40


# ── _ecl_additional_vectorized ────────────────────────────────────────────────


def test_ecl_additional_empty_mask_returns_zero() -> None:
    from scripts.run_sicr_conformal import _ecl_additional_vectorized

    df = _make_conformal_df()
    mask = pd.Series([False] * len(df), index=df.index)
    ecl = _ecl_additional_vectorized(df, mask, GRADE_LIFETIME_PD, LGD)
    assert ecl == 0.0


def test_ecl_additional_all_mask_positive() -> None:
    from scripts.run_sicr_conformal import _ecl_additional_vectorized

    df = _make_conformal_df()
    # Ensure lifetime > 12m so ECL is positive: set y_pred very small
    df["y_pred"] = 0.01
    mask = pd.Series([True] * len(df), index=df.index)
    ecl = _ecl_additional_vectorized(df, mask, GRADE_LIFETIME_PD, LGD)
    assert ecl > 0.0, "ECL should be positive when lifetime PD >> 12m PD"


def test_ecl_additional_non_negative() -> None:
    """ECL is always >= 0 (clipped at lower=0)."""
    from scripts.run_sicr_conformal import _ecl_additional_vectorized

    df = _make_conformal_df()
    mask = pd.Series([True] * len(df), index=df.index)
    ecl = _ecl_additional_vectorized(df, mask, GRADE_LIFETIME_PD, LGD)
    assert ecl >= 0.0


# ── _run_sicr_grid ─────────────────────────────────────────────────────────────


def test_sicr_grid_output_shape() -> None:
    """Grid should have n_pd_thresholds × n_width_thresholds rows."""
    from scripts.run_sicr_conformal import PD_THRESHOLDS, WIDTH_THRESHOLDS, _run_sicr_grid

    df = _make_conformal_df(n=300)
    result = _run_sicr_grid(df, GRADE_LIFETIME_PD, LGD)
    expected_rows = len(PD_THRESHOLDS) * len(WIDTH_THRESHOLDS)
    assert len(result) == expected_rows, f"Expected {expected_rows} rows, got {len(result)}"


def test_sicr_grid_required_columns() -> None:
    from scripts.run_sicr_conformal import _run_sicr_grid

    df = _make_conformal_df(n=200)
    result = _run_sicr_grid(df, GRADE_LIFETIME_PD, LGD)
    required = {
        "pd_threshold",
        "width_threshold",
        "n_additional_sicr",
        "precision_width",
        "recall_width_of_missed",
        "f1_width",
        "ecl_additional",
    }
    assert required.issubset(result.columns), f"Missing columns: {required - set(result.columns)}"


def test_sicr_grid_additivity_monotone() -> None:
    """Higher width threshold → fewer additional SICR loans (monotone decreasing)."""
    from scripts.run_sicr_conformal import _run_sicr_grid

    df = _make_conformal_df(n=500, seed=42)
    result = _run_sicr_grid(df, GRADE_LIFETIME_PD, LGD)
    # For a fixed pd_threshold, n_additional should be non-increasing with width_threshold
    for pd_thr in result["pd_threshold"].unique():
        sub = result[result["pd_threshold"] == pd_thr].sort_values("width_threshold")
        n_add = sub["n_additional_sicr"].values
        assert (np.diff(n_add) <= 0).all(), (
            f"n_additional not monotone for pd_thr={pd_thr}: {n_add}"
        )


def test_sicr_grid_f1_nonnegative() -> None:
    """All finite F1 values should be in [0, 1]."""
    from scripts.run_sicr_conformal import _run_sicr_grid

    df = _make_conformal_df(n=400)
    result = _run_sicr_grid(df, GRADE_LIFETIME_PD, LGD)
    f1_finite = result["f1_width"].dropna()
    assert ((f1_finite >= 0) & (f1_finite <= 1)).all(), f"F1 out of [0,1]: {f1_finite.describe()}"


# ── _run_alpha_sensitivity ────────────────────────────────────────────────────


def _make_mondrian_sweep() -> pd.DataFrame:
    """Synthetic Mondrian alpha sweep pareto."""
    alphas = [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]
    # avg_width decreases as alpha increases (higher alpha = less conservative = narrower)
    avg_widths = [0.99, 0.98, 0.97, 0.95, 0.77, 0.76, 0.73, 0.59]
    return pd.DataFrame(
        {
            "alpha": alphas,
            "avg_width": avg_widths,
            "empirical_coverage": [1 - a + 0.01 for a in alphas],
            "min_group_coverage": [1 - a - 0.01 for a in alphas],
            "method": ["mondrian"] * len(alphas),
        }
    )


def test_alpha_sensitivity_output_rows() -> None:
    """Output should have one row per alpha level in the sweep."""
    from scripts.run_sicr_conformal import _run_alpha_sensitivity

    df = _make_conformal_df(n=300, seed=1)
    sweep = _make_mondrian_sweep()
    result = _run_alpha_sensitivity(df, sweep, 0.20, 0.30, GRADE_LIFETIME_PD, LGD)
    assert len(result) == len(sweep), f"Expected {len(sweep)} rows, got {len(result)}"


def test_alpha_sensitivity_ecl_increases_lower_alpha() -> None:
    """Lower alpha (wider intervals) → more loans reclassified → higher ECL."""
    from scripts.run_sicr_conformal import _run_alpha_sensitivity

    # Use a wide width distribution so scaling matters
    rng = np.random.RandomState(7)
    df = pd.DataFrame(
        {
            "y_true": rng.binomial(1, 0.22, size=500),
            "y_pred": rng.uniform(0.05, 0.18, size=500).astype(np.float32),
            "width_90": rng.uniform(0.20, 0.80, size=500).astype(np.float32),
            "grade": rng.choice(["A", "B", "C"], size=500),
            "loan_amnt": rng.uniform(5_000, 25_000, size=500).astype(np.float32),
        }
    )
    sweep = _make_mondrian_sweep()
    result = _run_alpha_sensitivity(df, sweep, 0.20, 0.30, GRADE_LIFETIME_PD, LGD)
    result = result.sort_values("alpha").reset_index(drop=True)
    ecl = result["ecl_additional"].values
    # ECL at alpha=0.01 should be >= ECL at alpha=0.20
    assert ecl[0] >= ecl[-1], (
        f"Expected ECL to increase at lower alpha: alpha=0.01 gives ${ecl[0]:.0f}, "
        f"alpha=0.20 gives ${ecl[-1]:.0f}"
    )
