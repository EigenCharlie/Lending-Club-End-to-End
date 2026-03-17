"""Analyze conformal uncertainty cohorts for loan deferral decision support.

Uses the conformal interval width (pd_high_90 - pd_low_90) from Mondrian
conformal prediction as a distribution-free uncertainty proxy. Loans with
wide intervals are those the model is least confident about and are prime
candidates for deferral to human review.

Deferral rule: width_90 > p95 AND pd_point > 0.35 → refer to human underwriting.

This is an adaptation of active learning uncertainty sampling to operational
credit risk — conformal width provides the same signal as margin sampling but
with finite-sample coverage guarantees (Angelopoulos & Bates, 2023).

Usage:
    uv run python scripts/analyze_uncertainty_cohorts.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

INTERVALS_PATH = Path("data/processed/conformal_intervals_mondrian.parquet")
OUTPUT_PATH = Path("models/uncertainty_cohort_status.json")
SCHEMA_VERSION = "2026-03-16.1"
DEFERRAL_PERCENTILE = 95
DEFERRAL_PROB_THRESHOLD = 0.35


def _compute_quintile_metrics(
    df: pd.DataFrame,
    uncertainty_col: str = "width_90",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    n_quintiles: int = 5,
) -> list[dict[str, object]]:
    """Compute per-quintile metrics for uncertainty bins.

    Args:
        df: DataFrame with conformal intervals, labels, and predictions.
        uncertainty_col: Column with uncertainty values (interval widths).
        y_true_col: Binary target column (1 = default).
        y_pred_col: Point probability estimate column.
        n_quintiles: Number of quantile bins.

    Returns:
        List of per-quintile metric dicts ordered from most certain to least.
    """
    df = df.copy()
    # Use labels=False first to let duplicates="drop" reduce bin count freely,
    # then map integer bin indices to descriptive names.
    bin_codes, bin_edges = pd.qcut(
        df[uncertainty_col], q=n_quintiles, labels=False, retbins=True, duplicates="drop"
    )
    actual_n = len(bin_edges) - 1
    labels = []
    for i in range(actual_n):
        if i == 0:
            labels.append("Q1_certero")
        elif i == actual_n - 1:
            labels.append(f"Q{actual_n}_incierto")
        else:
            labels.append(f"Q{i + 1}")
    df["uncertainty_bin"] = pd.Categorical(
        [labels[int(c)] if pd.notna(c) else None for c in bin_codes],
        categories=labels,
        ordered=True,
    )

    results = []
    for bin_label, grp in df.groupby("uncertainty_bin", observed=True):
        n = len(grp)
        y_t = grp[y_true_col].to_numpy(dtype=float)
        y_p = grp[y_pred_col].to_numpy(dtype=float)
        row: dict[str, object] = {
            "quintile": str(bin_label),
            "count": n,
            "default_rate": float(y_t.mean()),
            "mean_pd_point": float(y_p.mean()),
            "mean_uncertainty": float(grp[uncertainty_col].mean()),
            "p25_uncertainty": float(grp[uncertainty_col].quantile(0.25)),
            "p75_uncertainty": float(grp[uncertainty_col].quantile(0.75)),
        }
        if "pd_low_90" in grp.columns and "pd_high_90" in grp.columns:
            low = grp["pd_low_90"].to_numpy(dtype=float)
            high = grp["pd_high_90"].to_numpy(dtype=float)
            covered = (y_t >= low) & (y_t <= high)
            row["conformal_coverage_90"] = float(covered.mean())
        if len(np.unique(y_t)) >= 2:
            from sklearn.metrics import roc_auc_score

            try:
                row["auc_roc"] = float(roc_auc_score(y_t, y_p))
            except Exception:
                row["auc_roc"] = None
        else:
            row["auc_roc"] = None
        results.append(row)

    return results


def _compute_deferral_stats(
    df: pd.DataFrame,
    uncertainty_col: str = "width_90",
    y_pred_col: str = "y_pred",
    deferral_percentile: int = DEFERRAL_PERCENTILE,
    deferral_prob_threshold: float = DEFERRAL_PROB_THRESHOLD,
) -> dict[str, object]:
    """Compute deferral rule statistics.

    Deferral rule: uncertainty > p{deferral_percentile} AND pd_point > threshold.

    Args:
        df: DataFrame with conformal intervals and predictions.
        uncertainty_col: Column with interval widths.
        y_pred_col: Point probability estimate column.
        deferral_percentile: Uncertainty percentile cutoff.
        deferral_prob_threshold: Minimum PD for deferral.

    Returns:
        Dict with deferral counts, rates, and thresholds.
    """
    p_threshold = float(np.percentile(df[uncertainty_col], deferral_percentile))
    # Use >= so that loans AT the percentile threshold are included.
    # When p95 == 1.0 (probability interval maxes out), this correctly
    # captures the maximally uncertain loans (full [0,1] interval).
    high_uncertainty = df[uncertainty_col] >= p_threshold
    high_prob = df[y_pred_col] > deferral_prob_threshold
    deferred = high_uncertainty & high_prob

    n_total = len(df)
    n_deferred = int(deferred.sum())

    return {
        "deferral_rule": (
            f"width_90 > p{deferral_percentile} AND pd_point > {deferral_prob_threshold}"
        ),
        "uncertainty_p95_threshold": float(p_threshold),
        "n_total": n_total,
        "n_high_uncertainty_only": int(high_uncertainty.sum()),
        "n_deferred": n_deferred,
        "deferral_rate": float(n_deferred / n_total) if n_total > 0 else 0.0,
        "deferral_percentile": deferral_percentile,
        "deferral_prob_threshold": deferral_prob_threshold,
    }


def main(
    intervals_path: str | Path = INTERVALS_PATH,
    output_path: str | Path = OUTPUT_PATH,
) -> dict:
    """Run uncertainty cohort analysis and write status JSON.

    Args:
        intervals_path: Path to conformal_intervals_mondrian.parquet.
        output_path: Destination for uncertainty_cohort_status.json.

    Returns:
        Status dict written to output_path.
    """
    intervals_path = Path(intervals_path)
    output_path = Path(output_path)

    if not intervals_path.exists():
        raise FileNotFoundError(f"Missing intervals artifact: {intervals_path}")

    df = pd.read_parquet(intervals_path)
    logger.info(f"Loaded {len(df):,} rows from {intervals_path}")

    required_cols = {"width_90", "y_true", "y_pred", "pd_low_90", "pd_high_90"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in intervals parquet: {missing}")

    quintile_metrics = _compute_quintile_metrics(df)
    logger.info(f"Computed quintile metrics across {len(quintile_metrics)} uncertainty bins")

    deferral_stats = _compute_deferral_stats(df)
    logger.info(
        f"Deferral rate: {deferral_stats['deferral_rate']:.2%} "
        f"({deferral_stats['n_deferred']:,} / {deferral_stats['n_total']:,} loans)"
    )

    grade_stats: list[dict[str, object]] = []
    if "grade" in df.columns:
        for grade, grp in df.groupby("grade", observed=True):
            grade_stats.append(
                {
                    "grade": str(grade),
                    "count": len(grp),
                    "mean_uncertainty": float(grp["width_90"].mean()),
                    "median_uncertainty": float(grp["width_90"].median()),
                    "p95_uncertainty": float(grp["width_90"].quantile(0.95)),
                    "default_rate": float(grp["y_true"].mean()),
                }
            )

    uncertainty_summary = {
        "mean": float(df["width_90"].mean()),
        "median": float(df["width_90"].median()),
        "std": float(df["width_90"].std()),
        "p5": float(df["width_90"].quantile(0.05)),
        "p25": float(df["width_90"].quantile(0.25)),
        "p75": float(df["width_90"].quantile(0.75)),
        "p95": float(df["width_90"].quantile(0.95)),
        "p99": float(df["width_90"].quantile(0.99)),
    }

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "n_observations": len(df),
        "uncertainty_col": "width_90",
        "uncertainty_summary": uncertainty_summary,
        "quintile_metrics": quintile_metrics,
        "deferral_stats": deferral_stats,
        "grade_uncertainty_stats": grade_stats,
        "methodology": (
            "Conformal interval width (pd_high_90 - pd_low_90) as distribution-free "
            "uncertainty proxy. Deferral rule: width_90 > p95 AND pd_point > 0.35 "
            "→ defer to human underwriting. Adapts uncertainty sampling from active "
            "learning (Settles 2012) to conformal prediction guarantees. "
            "Connects with Mondrian grade partitions for group-conditional analysis."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved uncertainty cohort status: {output_path}")
    return status


if __name__ == "__main__":
    main()
