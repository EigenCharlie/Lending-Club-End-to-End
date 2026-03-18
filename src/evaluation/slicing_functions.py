"""Slicing functions for per-cohort model evaluation in credit risk.

Defines data slices (grade, loan_amnt, dti, issue_year) and a function
to compute per-slice AUC, PR-AUC, Brier score, default_rate, and count.
These slices connect with Mondrian conformal partitions — the same grade
partitions used for conditional coverage can be evaluated for discriminatory
power (AUC) per cohort, revealing where the model is less precise.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

SLICES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "grade_A": lambda df: df["grade"] == "A",
    "grade_B": lambda df: df["grade"] == "B",
    "grade_C": lambda df: df["grade"] == "C",
    "grade_D_F": lambda df: df["grade"].isin(["D", "E", "F"]),
    "small_loan": lambda df: df["loan_amnt"] < 5_000,
    "large_loan": lambda df: df["loan_amnt"] >= 25_000,
    "high_dti": lambda df: df["dti"] > 30,
    # OOT test covers 2018-01 to 2020-09 only; split within OOT window.
    "early_cohort": lambda df: df["issue_year"] == 2018,
    "late_cohort": lambda df: df["issue_year"] >= 2019,
}

MIN_SLICE_SIZE = 100


def compute_slice_metrics(
    df: pd.DataFrame,
    y_true_col: str = "default_flag",
    y_prob_col: str = "pd_calibrated",
    slices: dict[str, Callable[[pd.DataFrame], pd.Series]] | None = None,
) -> list[dict[str, object]]:
    """Compute per-slice AUC, PR-AUC, Brier score, default rate, and count.

    Args:
        df: DataFrame with target, predicted probability, and slice feature columns.
            Required feature columns depend on the active slices (grade, loan_amnt,
            dti, issue_year by default).
        y_true_col: Column name for the binary target (1 = default).
        y_prob_col: Column name for the predicted default probability.
        slices: Dict of slice_name → boolean mask function. Defaults to SLICES.

    Returns:
        List of dicts, one per slice plus one for "overall", each containing:
        slice, count, default_rate, auc_roc, pr_auc, brier_score, skipped.
    """
    active_slices = slices if slices is not None else SLICES
    y_true_all = df[y_true_col].to_numpy(dtype=float)
    y_prob_all = df[y_prob_col].to_numpy(dtype=float)

    results: list[dict[str, object]] = []
    results.append(_slice_metrics("overall", y_true_all, y_prob_all))

    for name, mask_fn in active_slices.items():
        try:
            mask = mask_fn(df).to_numpy(dtype=bool)
        except (KeyError, AttributeError):
            results.append({"slice": name, "count": 0, "skipped": True, "reason": "column_missing"})
            continue
        n = int(mask.sum())
        if n < MIN_SLICE_SIZE:
            results.append({"slice": name, "count": n, "skipped": True, "reason": "too_small"})
            continue
        results.append(_slice_metrics(name, y_true_all[mask], y_prob_all[mask]))

    return results


def _slice_metrics(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, object]:
    n = len(y_true)
    result: dict[str, object] = {
        "slice": name,
        "count": n,
        "skipped": False,
        "default_rate": float(y_true.mean()) if n > 0 else None,
    }
    if len(np.unique(y_true)) < 2:
        result.update(
            {"auc_roc": None, "pr_auc": None, "brier_score": None, "reason": "single_class"}
        )
        return result
    try:
        result["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        result["auc_roc"] = None
    try:
        result["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        result["pr_auc"] = None
    try:
        result["brier_score"] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        result["brier_score"] = None
    return result
