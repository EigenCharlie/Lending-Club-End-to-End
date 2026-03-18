"""Conformal prediction alpha sweep for Paper Estrella Pareto frontier.

For each alpha level, generates split-conformal PD intervals (global and/or
Mondrian by grade), computes coverage/width diagnostics, and estimates a
simplified portfolio return proxy to trace the coverage-vs-return Pareto frontier.

Usage:
    uv run python scripts/run_alpha_sweep.py
    uv run python scripts/run_alpha_sweep.py --mondrian
    uv run python scripts/run_alpha_sweep.py --run-tag estrella-sweep-v1 --mondrian
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger

from src.models.conformal import (
    create_pd_intervals,
    create_pd_intervals_mondrian,
    validate_coverage,
)
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag

SCHEMA_VERSION = "2026-03-16.1"

ALPHAS = [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]

# Portfolio proxy: loans with pd_high below this threshold are "low-risk eligible".
# 0.20 corresponds to the upper CP interval staying below grade-B risk territory.
# (The operational pipeline uses per-grade alpha tuning which can reach <0.10 for grade A;
# a fixed-alpha sweep with alpha=0.10 gives Grade A min pd_high ~0.15, so 0.10 yields 0 eligible.)
LOW_RISK_THRESHOLD = 0.20

# Assumed average annual interest margin for portfolio return proxy (basis points)
AVG_INTEREST_MARGIN = 0.05  # 5%


def _load_model() -> CatBoostClassifier:
    """Load canonical CatBoost PD model."""
    model_path = Path("models/pd_canonical.cbm")
    if not model_path.exists():
        raise FileNotFoundError(f"Canonical model not found: {model_path}")
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    logger.info(f"Loaded CatBoost model from {model_path}")
    return model


def _load_calibrator():
    """Load canonical probability calibrator."""
    cal_path = Path("models/pd_canonical_calibrator.pkl")
    if not cal_path.exists():
        logger.warning(f"Calibrator not found at {cal_path}, proceeding without calibration")
        return None
    with open(cal_path, "rb") as f:
        calibrator = pickle.load(f)
    logger.info(f"Loaded calibrator from {cal_path}")
    return calibrator


def _load_contract() -> dict:
    """Load model contract for feature names."""
    contract_path = Path("models/pd_model_contract.json")
    if not contract_path.exists():
        raise FileNotFoundError(f"Model contract not found: {contract_path}")
    with open(contract_path) as f:
        return json.load(f)


def _load_data(
    feature_cols: list[str],
    model: Any,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load calibration and test data with feature subsetting.

    Returns:
        X_cal, y_cal, X_test, y_test, int_rate_test, grade_cal, grade_test
    """
    cal_path = Path("data/processed/calibration_fe.parquet")
    test_path = Path("data/processed/test_fe.parquet")
    for p in [cal_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")

    cal = pd.read_parquet(cal_path)
    test = pd.read_parquet(test_path)
    logger.info(f"Loaded calibration ({len(cal):,} rows) and test ({len(test):,} rows)")

    # Use model's own cat feature indices for type resolution
    cat_idx = set(model.get_cat_feature_indices())
    cat_names = {feature_cols[i] for i in cat_idx if i < len(feature_cols)}

    X_cal = cal[feature_cols].copy()
    X_test = test[feature_cols].copy()
    for col in feature_cols:
        if col in cat_names:
            X_cal[col] = X_cal[col].astype("string").fillna("UNKNOWN").astype(str)
            X_test[col] = X_test[col].astype("string").fillna("UNKNOWN").astype(str)
        else:
            X_cal[col] = pd.to_numeric(X_cal[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
    y_cal = cal["default_flag"]
    y_test = test["default_flag"]

    # Interest rate for portfolio return proxy
    int_rate_test = (
        pd.to_numeric(test["int_rate"], errors="coerce").fillna(0.12) / 100.0
        if "int_rate" in test.columns
        else pd.Series(np.full(len(test), AVG_INTEREST_MARGIN), index=test.index)
    )

    # Grade for Mondrian conformal
    grade_cal = (
        cal["grade"].fillna("UNKNOWN").astype(str)
        if "grade" in cal.columns
        else pd.Series(["UNKNOWN"] * len(cal))
    )
    grade_test = (
        test["grade"].fillna("UNKNOWN").astype(str)
        if "grade" in test.columns
        else pd.Series(["UNKNOWN"] * len(test))
    )

    return X_cal, y_cal, X_test, y_test, int_rate_test, grade_cal, grade_test


def _compute_portfolio_proxy(
    y_pred: np.ndarray,
    y_intervals: np.ndarray,
    int_rate: np.ndarray,
) -> dict[str, float]:
    """Compute simplified portfolio return proxy.

    Approach:
    - Identify loans where pd_high < LOW_RISK_THRESHOLD (eligible pool)
    - Portfolio return = sum of (interest_margin - expected_loss) for eligible loans
    - Expected loss (point) = pd_point * LGD_assumed (0.40)
    - Expected loss (worst) = pd_high * LGD_assumed (0.40)
    """
    lgd = 0.40
    pd_high = y_intervals[:, 1]

    eligible_mask = pd_high < LOW_RISK_THRESHOLD
    n_eligible = int(eligible_mask.sum())
    n_total = len(y_pred)
    eligible_pct = n_eligible / n_total if n_total > 0 else 0.0

    if n_eligible == 0:
        return {
            "n_eligible": 0,
            "n_total": n_total,
            "eligible_pct": 0.0,
            "portfolio_return_point": 0.0,
            "portfolio_return_worst": 0.0,
            "avg_pd_eligible": float("nan"),
            "avg_pd_high_eligible": float("nan"),
            "avg_int_rate_eligible": float("nan"),
        }

    int_rate_arr = np.asarray(int_rate, dtype=float)
    margin_eligible = int_rate_arr[eligible_mask]
    pd_point_eligible = y_pred[eligible_mask]
    pd_high_eligible = pd_high[eligible_mask]

    # Per-loan net return: interest margin minus expected loss
    net_return_point = (margin_eligible - pd_point_eligible * lgd).sum()
    net_return_worst = (margin_eligible - pd_high_eligible * lgd).sum()

    return {
        "n_eligible": n_eligible,
        "n_total": n_total,
        "eligible_pct": float(eligible_pct),
        "portfolio_return_point": float(net_return_point),
        "portfolio_return_worst": float(net_return_worst),
        "avg_pd_eligible": float(pd_point_eligible.mean()),
        "avg_pd_high_eligible": float(pd_high_eligible.mean()),
        "avg_int_rate_eligible": float(margin_eligible.mean()),
    }


def run_sweep(
    model: CatBoostClassifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    int_rate_test: pd.Series,
    alphas: list[float],
    mondrian: bool = False,
    grade_cal: pd.Series | None = None,
    grade_test: pd.Series | None = None,
    calibrator: Any | None = None,
) -> list[dict]:
    """Run conformal interval sweep across alpha levels.

    Args:
        model: Fitted CatBoost classifier.
        X_cal: Calibration features.
        y_cal: Calibration labels.
        X_test: Test features.
        y_test: Test labels.
        int_rate_test: Interest rate series for test loans.
        alphas: List of alpha (significance) levels.
        mondrian: If True, use Mondrian (grade-conditional) intervals.
        grade_cal: Grade labels for calibration set (required if mondrian=True).
        grade_test: Grade labels for test set (required if mondrian=True).

    Returns:
        List of dicts, one per alpha, with coverage and portfolio metrics.
    """
    results: list[dict] = []
    y_test_arr = np.asarray(y_test, dtype=float)
    int_rate_arr = np.asarray(int_rate_test, dtype=float)
    method_label = "mondrian" if mondrian else "global"

    for alpha in alphas:
        logger.info(f"--- [{method_label}] Alpha = {alpha:.2f} (target = {1 - alpha:.0%}) ---")

        if mondrian:
            if grade_cal is None or grade_test is None:
                raise ValueError("grade_cal and grade_test required for Mondrian sweep")
            y_pred, y_intervals, diag = create_pd_intervals_mondrian(
                model,
                X_cal,
                y_cal,
                X_test,
                grade_cal,
                grade_test,
                alpha=alpha,
                calibrator=calibrator,
            )
            # Compute per-group empirical coverage from y_test
            g_test_arr = np.asarray(grade_test, dtype=str)
            covered = (y_test_arr >= y_intervals[:, 0]) & (y_test_arr <= y_intervals[:, 1])
            group_coverages = {}
            for g in sorted(set(g_test_arr)):
                mask = g_test_arr == g
                if mask.sum() > 0:
                    group_coverages[f"cov_{g}"] = float(covered[mask].mean())
            min_group_cov = (
                float(min(group_coverages.values())) if group_coverages else float("nan")
            )
        else:
            y_pred, y_intervals = create_pd_intervals(
                model,
                X_cal,
                y_cal,
                X_test,
                alpha=alpha,
                calibrator=calibrator,
            )
            group_coverages = {}
            min_group_cov = float("nan")

        # Coverage diagnostics
        cov_metrics = validate_coverage(y_test_arr, y_intervals, alpha=alpha)

        # Width diagnostics
        widths = y_intervals[:, 1] - y_intervals[:, 0]
        width_stats = {
            "avg_width": float(widths.mean()),
            "median_width": float(np.median(widths)),
            "min_width": float(widths.min()),
            "max_width": float(widths.max()),
            "std_width": float(widths.std()),
        }

        # Portfolio proxy
        portfolio = _compute_portfolio_proxy(y_pred, y_intervals, int_rate_arr)

        row = {
            "alpha": alpha,
            "confidence_level": 1.0 - alpha,
            "method": method_label,
            "min_group_coverage": min_group_cov,
            **cov_metrics,
            **width_stats,
            **portfolio,
            **group_coverages,
        }
        results.append(row)

        logger.info(
            f"  coverage={cov_metrics['empirical_coverage']:.4f}, "
            f"avg_width={width_stats['avg_width']:.4f}, "
            f"min_group_cov={min_group_cov:.4f}, "
            f"eligible={portfolio['n_eligible']:,}/{portfolio['n_total']:,} "
            f"({portfolio['eligible_pct']:.1%})"
        )

    return results


def main() -> int:
    """Entry point for conformal alpha sweep."""
    parser = argparse.ArgumentParser(
        description="Conformal prediction alpha sweep for Pareto frontier"
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Run tag for artifact provenance tracking",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Comma-separated alpha values (overrides default sweep)",
    )
    parser.add_argument(
        "--mondrian",
        action="store_true",
        default=False,
        help="Run Mondrian (grade-conditional) sweep instead of global",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        default=False,
        help="Run both global and Mondrian sweeps (combined output)",
    )
    args = parser.parse_args()

    run_tag = resolve_run_tag(args.run_tag, allow_untracked=True)
    logger.info(f"Alpha sweep starting | run_tag={run_tag}")

    # Parse custom alphas if provided
    alphas = ALPHAS
    if args.alphas:
        alphas = [float(a.strip()) for a in args.alphas.split(",")]
        logger.info(f"Custom alphas: {alphas}")

    # Load artifacts
    model = _load_model()
    calibrator = _load_calibrator()
    contract = _load_contract()
    feature_cols = contract["feature_names"]

    X_cal, y_cal, X_test, y_test, int_rate_test, grade_cal, grade_test = _load_data(
        feature_cols, model
    )

    # Determine which sweeps to run
    run_global = not args.mondrian or args.both
    run_mondrian = args.mondrian or args.both

    all_results: list[dict] = []

    if run_global:
        logger.info("=== Running GLOBAL conformal sweep ===")
        global_results = run_sweep(
            model,
            X_cal,
            y_cal,
            X_test,
            y_test,
            int_rate_test,
            alphas,
            mondrian=False,
            calibrator=calibrator,
        )
        all_results.extend(global_results)

    if run_mondrian:
        logger.info("=== Running MONDRIAN (grade-conditional) conformal sweep ===")
        mondrian_results = run_sweep(
            model,
            X_cal,
            y_cal,
            X_test,
            y_test,
            int_rate_test,
            alphas,
            mondrian=True,
            grade_cal=grade_cal,
            grade_test=grade_test,
            calibrator=calibrator,
        )
        all_results.extend(mondrian_results)

    # Save Pareto parquet
    pareto_df = pd.DataFrame(all_results)
    method_suffix = (
        "_mondrian" if (args.mondrian and not args.both) else ("_both" if args.both else "")
    )
    pareto_path = Path(f"data/processed/alpha_sweep_pareto{method_suffix}.parquet")
    pareto_path.parent.mkdir(parents=True, exist_ok=True)
    pareto_df.to_parquet(pareto_path, index=False)
    logger.info(f"Saved Pareto frontier: {pareto_path} ({len(pareto_df)} rows)")

    # Save status JSON
    metadata = build_artifact_metadata(
        schema_version=SCHEMA_VERSION,
        run_tag=run_tag,
        allow_untracked=True,
        extra={
            "alphas": alphas,
            "n_alphas": len(alphas),
            "n_cal": len(X_cal),
            "n_test": len(X_test),
            "low_risk_threshold": LOW_RISK_THRESHOLD,
            "avg_interest_margin": AVG_INTEREST_MARGIN,
            "mondrian": run_mondrian,
            "global": run_global,
        },
    )

    # Compute summaries per method
    summaries: dict[str, object] = {}
    for method in pareto_df["method"].unique():
        mdf = pareto_df[pareto_df["method"] == method]
        best_idx = mdf["empirical_coverage"].idxmax()
        tight_idx = mdf["avg_width"].idxmin()
        elig_idx = mdf["n_eligible"].idxmax()
        summaries[method] = {
            "best_coverage_alpha": float(mdf.loc[best_idx, "alpha"]),
            "best_coverage": float(mdf.loc[best_idx, "empirical_coverage"]),
            "tightest_avg_width_alpha": float(mdf.loc[tight_idx, "alpha"]),
            "tightest_avg_width": float(mdf.loc[tight_idx, "avg_width"]),
            "most_eligible_alpha": float(mdf.loc[elig_idx, "alpha"]),
            "most_eligible_n": int(mdf.loc[elig_idx, "n_eligible"]),
        }

    status = {
        **metadata,
        "pareto_path": str(pareto_path),
        "summaries": summaries,
    }
    status_path = Path("models/alpha_sweep_status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2, default=str)
    logger.info(f"Saved status: {status_path}")

    # Log summary table
    logger.info("=== Alpha Sweep Summary ===")
    summary_cols = [
        c
        for c in [
            "method",
            "alpha",
            "empirical_coverage",
            "min_group_coverage",
            "avg_width",
            "n_eligible",
            "eligible_pct",
            "portfolio_return_point",
            "portfolio_return_worst",
        ]
        if c in pareto_df.columns
    ]
    logger.info("\n" + pareto_df[summary_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
