"""Compare baseline vs CATE-adjusted portfolio optimization.

Loads conformal intervals and CATE estimates from the OOT test set,
runs the optimizer twice (with and without CATE adjustment), and
saves a comparison of portfolio metrics.

Usage:
    uv run python scripts/optimize_cate_portfolio.py
    uv run python scripts/optimize_cate_portfolio.py --delta_rate -1.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.optimization.causal_portfolio import build_cate_adjusted_portfolio


def _parse_percent_series(s: pd.Series, default: float = 0.12) -> np.ndarray:
    """Convert a column that may be string '12.5%' or float 12.5 to decimal."""
    if pd.api.types.is_numeric_dtype(s):
        arr = s.to_numpy(dtype=float)
        # If values look like percentages (> 1), convert to decimal
        if np.nanmedian(arr) > 1:
            arr = arr / 100.0
        return np.nan_to_num(arr, nan=default)
    return (
        s.astype(str)
        .str.strip()
        .str.rstrip("%")
        .pipe(pd.to_numeric, errors="coerce")
        .div(100)
        .fillna(default)
        .to_numpy(dtype=float)
    )


def main(
    delta_rate: float = -1.0,
    total_budget: float = 1_000_000,
    max_portfolio_pd: float = 0.10,
    max_candidates: int = 5_000,
    uncertainty_aversion: float = 0.0,
) -> None:
    """Run baseline vs CATE-adjusted portfolio comparison."""
    # Load test data
    data_dir = Path("data/processed")
    test_path = data_dir / "test_fe.parquet"
    intervals_path = data_dir / "conformal_intervals_mondrian.parquet"
    cate_path = data_dir / "cate_estimates.parquet"

    for p in [test_path, intervals_path, cate_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    test_df = pd.read_parquet(test_path)
    intervals = pd.read_parquet(intervals_path)
    cate_df = pd.read_parquet(cate_path)

    logger.info(f"Loaded: test={len(test_df)}, intervals={len(intervals)}, cate={len(cate_df)}")

    # Join CATE to test set — CATE estimated on train, so match by grade median
    if "id" in test_df.columns and "id" in cate_df.columns:
        merged = test_df[["id"]].merge(cate_df[["id", "cate"]], on="id", how="left")
        n_direct = merged["cate"].notna().sum()
        if n_direct > 0:
            cate_df = merged
            cate_df["cate"] = cate_df["cate"].fillna(0.0)
            logger.info(f"Joined CATE by id: {n_direct} direct matches")
        elif "grade" in test_df.columns and "grade" in cate_df.columns:
            # No direct ID overlap (CATE on train, test is OOT) → impute by grade median
            grade_cate = cate_df.groupby("grade")["cate"].median().to_dict()
            cate_vals = test_df["grade"].map(grade_cate).fillna(0.0).values
            cate_df = pd.DataFrame({"cate": cate_vals})
            logger.info(
                f"No ID overlap — imputed CATE by grade median "
                f"({len(grade_cate)} grades, {(cate_vals != 0).sum()} non-zero)"
            )
        else:
            cate_df = pd.DataFrame({"cate": np.zeros(len(test_df))})
            logger.warning("No ID overlap and no grade column — using zero CATE")
    else:
        logger.warning("No 'id' column — falling back to row-index alignment")
        n = min(len(test_df), len(cate_df))
        cate_df = cate_df.iloc[:n].reset_index(drop=True)

    # Align test_df and intervals by row index (same OOT set), cap at max_candidates
    n = min(len(test_df), len(intervals), len(cate_df), max_candidates)
    logger.info(f"Using {n} candidates (max_candidates={max_candidates})")
    test_df = test_df.iloc[:n].reset_index(drop=True)
    intervals = intervals.iloc[:n].reset_index(drop=True)
    cate_df = cate_df.iloc[:n].reset_index(drop=True)

    # Extract arrays
    # Map column names: conformal intervals use y_pred, pd_low_90, pd_high_90
    pd_col = next(
        (c for c in ["pd_calibrated", "y_pred"] if c in intervals.columns), intervals.columns[0]
    )
    low_col = next((c for c in ["pd_low", "pd_low_90"] if c in intervals.columns), None)
    high_col = next((c for c in ["pd_high", "pd_high_90"] if c in intervals.columns), None)
    pd_point = intervals[pd_col].values
    pd_low = intervals[low_col].values if low_col else pd_point * 0.8
    pd_high = intervals[high_col].values if high_col else pd_point * 1.3
    cate = cate_df["cate"].values
    lgd = np.full(n, 0.45)  # standard LGD assumption
    int_rates = (
        _parse_percent_series(test_df["int_rate"])
        if "int_rate" in test_df.columns
        else np.full(n, 0.12)
    )

    # Run comparison
    result = build_cate_adjusted_portfolio(
        loans=test_df,
        pd_point=pd_point,
        pd_low=pd_low,
        pd_high=pd_high,
        cate=cate,
        lgd=lgd,
        int_rates=int_rates,
        delta_rate=delta_rate,
        total_budget=total_budget,
        max_portfolio_pd=max_portfolio_pd,
        uncertainty_aversion=uncertainty_aversion,
    )

    # Save comparison
    comparison_path = data_dir / "cate_portfolio_comparison.parquet"
    result["comparison_df"].to_parquet(comparison_path, index=False)
    logger.info(f"Saved comparison: {comparison_path}")

    # Save status JSON
    status = {
        "delta_rate": delta_rate,
        "baseline_objective": result["baseline"]["objective_value"],
        "cate_adjusted_objective": result["cate_adjusted"]["objective_value"],
        "baseline_n_funded": result["baseline"]["n_funded"],
        "cate_adjusted_n_funded": result["cate_adjusted"]["n_funded"],
        "objective_change_pct": float(
            (result["cate_adjusted"]["objective_value"] - result["baseline"]["objective_value"])
            / (abs(result["baseline"]["objective_value"]) + 1e-6)
            * 100
        ),
    }
    status_path = Path("models/cate_portfolio_status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    logger.info(f"Saved status: {status_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CATE-adjusted portfolio comparison")
    parser.add_argument("--delta_rate", type=float, default=-1.0)
    parser.add_argument("--total_budget", type=float, default=1_000_000)
    parser.add_argument("--max_portfolio_pd", type=float, default=0.10)
    parser.add_argument("--max_candidates", type=int, default=5_000)
    parser.add_argument("--uncertainty_aversion", type=float, default=0.0)
    args = parser.parse_args()
    main(
        delta_rate=args.delta_rate,
        total_budget=args.total_budget,
        max_portfolio_pd=args.max_portfolio_pd,
        max_candidates=args.max_candidates,
        uncertainty_aversion=args.uncertainty_aversion,
    )
