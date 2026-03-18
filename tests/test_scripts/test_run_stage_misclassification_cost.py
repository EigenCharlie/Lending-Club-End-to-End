"""Tests for scripts/run_stage_misclassification_cost.py — Stage 1/2 cost analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

GRADE_PD = {
    "A": 0.178,
    "B": 0.266,
    "C": 0.352,
    "D": 0.456,
    "E": 0.634,
    "ALL": 0.380,
}
LGD = 0.40


def _make_conf_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "y_true": rng.binomial(1, 0.22, size=n),
            "y_pred": rng.beta(2, 8, size=n).astype(np.float32),
            "width_90": rng.beta(1, 2, size=n).astype(np.float32),
            "grade": rng.choice(["A", "B", "C", "D"], size=n),
            "loan_amnt": rng.uniform(5000, 35000, size=n).astype(np.float32),
        }
    )


def test_provision_increment_non_negative() -> None:
    """Provision increment (Stage 2 - Stage 1) must be >= 0."""
    from scripts.run_stage_misclassification_cost import _provision_increment

    df = _make_conf_df()
    mask = pd.Series([True] * len(df), index=df.index)
    increments = _provision_increment(df, mask, GRADE_PD, LGD)
    assert (increments >= 0).all(), "Provision increment should always be non-negative"


def test_provision_increment_empty_mask_returns_empty() -> None:
    from scripts.run_stage_misclassification_cost import _provision_increment

    df = _make_conf_df()
    mask = pd.Series([False] * len(df), index=df.index)
    result = _provision_increment(df, mask, GRADE_PD, LGD)
    assert len(result) == 0


def test_cost_row_keys() -> None:
    """Cost row should have all required keys."""
    from scripts.run_stage_misclassification_cost import _compute_cost_row

    df = _make_conf_df(n=300)
    row = _compute_cost_row(
        df, pd_thr=0.20, w_thr=None, grade_pd=GRADE_PD, lgd=LGD, w_fn=5.0, w_fp=1.0
    )
    required = {"n_fn", "n_fp", "fn_cost", "fp_cost", "total_cost", "fn_rate", "fp_rate"}
    assert required.issubset(row.keys())


def test_total_cost_is_weighted_sum() -> None:
    """total_cost = w_fn * fn_cost + w_fp * fp_cost."""
    from scripts.run_stage_misclassification_cost import _compute_cost_row

    df = _make_conf_df(n=400, seed=7)
    w_fn, w_fp = 5.0, 1.0
    row = _compute_cost_row(
        df, pd_thr=0.20, w_thr=None, grade_pd=GRADE_PD, lgd=LGD, w_fn=w_fn, w_fp=w_fp
    )
    expected = w_fn * row["fn_cost"] + w_fp * row["fp_cost"]
    assert abs(row["total_cost"] - expected) < 1.0, (
        f"total_cost mismatch: {row['total_cost']:.2f} vs {expected:.2f}"
    )


def test_higher_pd_threshold_more_fn() -> None:
    """Higher PD threshold → fewer Stage 2 predictions → more FN (missed defaults)."""
    from scripts.run_stage_misclassification_cost import _compute_cost_row

    df = _make_conf_df(n=600, seed=42)
    row_low = _compute_cost_row(
        df, pd_thr=0.10, w_thr=None, grade_pd=GRADE_PD, lgd=LGD, w_fn=5.0, w_fp=1.0
    )
    row_high = _compute_cost_row(
        df, pd_thr=0.30, w_thr=None, grade_pd=GRADE_PD, lgd=LGD, w_fn=5.0, w_fp=1.0
    )
    assert row_high["n_fn"] >= row_low["n_fn"], (
        f"Higher pd_thr should have more FN: {row_high['n_fn']} vs {row_low['n_fn']}"
    )


def test_width_trigger_increases_stage2_count() -> None:
    """Adding width trigger should stage at least as many loans as pure PD."""
    from scripts.run_stage_misclassification_cost import _compute_cost_row

    df = _make_conf_df(n=500, seed=3)
    row_pd_only = _compute_cost_row(
        df, pd_thr=0.20, w_thr=None, grade_pd=GRADE_PD, lgd=LGD, w_fn=5.0, w_fp=1.0
    )
    row_combined = _compute_cost_row(
        df, pd_thr=0.20, w_thr=0.30, grade_pd=GRADE_PD, lgd=LGD, w_fn=5.0, w_fp=1.0
    )
    assert row_combined["n_stage2_pred"] >= row_pd_only["n_stage2_pred"]
