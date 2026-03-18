"""SICR trigger optimization and ECL sensitivity to confidence level (alpha).

Combines two complementary analyses:

Part 1 — SICR Trigger Optimization:
    Sweeps over conformal interval width thresholds to find the optimal
    SICR trigger.  For each threshold, loans whose conformal width exceeds
    the threshold are flagged as Stage 2 (SICR).  Precision and recall of
    the SICR signal with respect to actual defaults are recorded.

Part 2 — ECL Sensitivity to Alpha:
    Scales the existing 90 % Mondrian conformal intervals to approximate
    other confidence levels (alpha) and computes the corresponding ECL
    range (low / point / high).  Uses a Gaussian quantile-ratio scaling
    factor: width_alpha = width_90 * z(1 − alpha/2) / z(0.95).

Outputs:
    data/processed/sicr_trigger_optimization.parquet
    data/processed/ecl_alpha_sensitivity.parquet
    models/sicr_ecl_sensitivity_status.json

Usage:
    uv run python scripts/run_sicr_ecl_sensitivity.py
    uv run python scripts/run_sicr_ecl_sensitivity.py --base-lgd 0.45 --n-thresholds 16
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.ifrs9 import (  # noqa: F401
    assign_stage,
    compute_ecl,
    ecl_with_conformal_range,
)
from src.models.conformal_artifacts import load_conformal_intervals

SCHEMA_VERSION = "2026-03-16.1"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_intervals() -> pd.DataFrame:
    """Load Mondrian conformal intervals (canonical artifact)."""
    df, path, is_legacy = load_conformal_intervals(allow_legacy_fallback=False)
    logger.info("Loaded conformal intervals: {} ({:,} rows)", path, len(df))
    return df


def _load_test_fe() -> pd.DataFrame:
    """Load feature-engineered test set."""
    path = ROOT / "data" / "processed" / "test_fe.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Expected test FE at {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded test_fe: {} ({:,} rows)", path, len(df))
    return df


# ---------------------------------------------------------------------------
# Part 1: SICR Trigger Optimization
# ---------------------------------------------------------------------------


def run_sicr_sweep(
    intervals: pd.DataFrame,
    test_fe: pd.DataFrame,
    n_thresholds: int = 16,
) -> pd.DataFrame:
    """Sweep conformal width thresholds and evaluate SICR signal quality.

    Args:
        intervals: Conformal intervals with y_pred, pd_high_90 columns.
        test_fe: Test set with default_flag column.
        n_thresholds: Number of threshold steps between 0.05 and 0.80.

    Returns:
        DataFrame with one row per threshold containing SICR metrics.
    """
    n = min(len(intervals), len(test_fe))
    if len(intervals) != len(test_fe):
        logger.warning(
            "Length mismatch: intervals={:,}, test_fe={:,}. Using first {:,} rows.",
            len(intervals),
            len(test_fe),
            n,
        )

    pd_high = intervals["pd_high_90"].to_numpy(dtype=float)[:n]
    pd_low = intervals["pd_low_90"].to_numpy(dtype=float)[:n]
    default_flag = test_fe["default_flag"].to_numpy(dtype=int)[:n]

    # Conformal width: full interval width (pd_high - pd_low)
    width = pd_high - pd_low

    thresholds = np.linspace(0.05, 0.80, n_thresholds)
    total_defaults = int(default_flag.sum())
    rows: list[dict] = []

    for thr in thresholds:
        sicr_mask = width > thr
        n_stage2 = int(sicr_mask.sum())
        n_stage1 = int(n - n_stage2)

        defaults_in_stage2 = int(default_flag[sicr_mask].sum()) if n_stage2 > 0 else 0
        defaults_in_stage1 = int(default_flag[~sicr_mask].sum()) if n_stage1 > 0 else 0

        stage2_default_rate = float(defaults_in_stage2 / n_stage2) if n_stage2 > 0 else 0.0
        stage1_default_rate = float(defaults_in_stage1 / n_stage1) if n_stage1 > 0 else 0.0

        precision = float(defaults_in_stage2 / n_stage2) if n_stage2 > 0 else 0.0
        recall = float(defaults_in_stage2 / total_defaults) if total_defaults > 0 else 0.0
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        rows.append(
            {
                "width_threshold": float(thr),
                "n_stage1": n_stage1,
                "n_stage2": n_stage2,
                "stage2_share": float(n_stage2 / n) if n > 0 else 0.0,
                "stage1_default_rate": stage1_default_rate,
                "stage2_default_rate": stage2_default_rate,
                "defaults_in_stage2": defaults_in_stage2,
                "defaults_in_stage1": defaults_in_stage1,
                "precision_sicr": precision,
                "recall_defaults": recall,
                "f1_sicr": f1,
            }
        )

    result = pd.DataFrame(rows)
    logger.info(
        "SICR sweep complete: {} thresholds, best F1={:.4f} at threshold={:.3f}",
        n_thresholds,
        result["f1_sicr"].max(),
        result.loc[result["f1_sicr"].idxmax(), "width_threshold"],
    )
    return result


# ---------------------------------------------------------------------------
# Part 2: ECL Sensitivity to Alpha
# ---------------------------------------------------------------------------


def run_ecl_alpha_sensitivity(
    intervals: pd.DataFrame,
    test_fe: pd.DataFrame,
    base_lgd: float = 0.45,
) -> pd.DataFrame:
    """Compute ECL range at different confidence levels using Gaussian scaling.

    Scales the existing 90 % intervals (alpha = 0.10) to approximate intervals
    at other alpha levels using the ratio of Gaussian quantiles:
        width_alpha = width_90 * z(1 - alpha/2) / z(0.95)

    Args:
        intervals: Conformal intervals with y_pred, pd_low_90, pd_high_90.
        test_fe: Test set with default_flag, loan_amnt columns.
        base_lgd: Base LGD assumption for ECL computation.

    Returns:
        DataFrame with one row per alpha level containing ECL metrics.
    """
    n = min(len(intervals), len(test_fe))

    pd_point = np.clip(intervals["y_pred"].to_numpy(dtype=float)[:n], 0.0, 1.0)
    pd_low_90 = np.clip(intervals["pd_low_90"].to_numpy(dtype=float)[:n], 0.0, 1.0)
    pd_high_90 = np.clip(intervals["pd_high_90"].to_numpy(dtype=float)[:n], 0.0, 1.0)
    width_90 = pd_high_90 - pd_low_90

    loan_amnt = test_fe["loan_amnt"].to_numpy(dtype=float)[:n]
    lgd = np.full(n, base_lgd, dtype=float)

    # Reference quantile for alpha = 0.10 (90 % two-sided)
    z_ref = float(norm.ppf(0.95))

    alphas = [0.05, 0.10, 0.15, 0.20]
    rows: list[dict] = []

    # Suppress per-loan IFRS9 logger spam during the sweep.
    logger.disable("src.evaluation.ifrs9")

    for alpha in alphas:
        z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
        scale_factor = z_alpha / z_ref
        confidence_pct = (1.0 - alpha) * 100.0

        width_alpha = width_90 * scale_factor
        half_width = width_alpha / 2.0

        pd_low_alpha = np.clip(pd_point - half_width, 0.0, 1.0)
        pd_high_alpha = np.clip(pd_point + half_width, 0.0, 1.0)

        # Use basic staging: treat all loans as Stage 1 for ECL range comparison.
        # This isolates the alpha effect from staging interactions.
        stages = np.ones(n, dtype=int)

        ecl_range = ecl_with_conformal_range(
            pd_low=pd_low_alpha,
            pd_point=pd_point,
            pd_high=pd_high_alpha,
            lgd=lgd,
            ead=loan_amnt,
            stages=stages,
        )

        total_ecl_low = float(ecl_range["ecl_low"].sum())
        total_ecl_point = float(ecl_range["ecl_point"].sum())
        total_ecl_high = float(ecl_range["ecl_high"].sum())
        total_ecl_range = total_ecl_high - total_ecl_low

        rows.append(
            {
                "alpha": float(alpha),
                "confidence_pct": confidence_pct,
                "scale_factor": scale_factor,
                "mean_width_alpha": float(np.mean(width_alpha)),
                "median_width_alpha": float(np.median(width_alpha)),
                "total_ecl_low": total_ecl_low,
                "total_ecl_point": total_ecl_point,
                "total_ecl_high": total_ecl_high,
                "total_ecl_range": total_ecl_range,
                "ecl_range_pct_of_point": float(
                    total_ecl_range / total_ecl_point * 100 if total_ecl_point > 0 else 0.0
                ),
                "n_loans": n,
                "base_lgd": base_lgd,
            }
        )

    logger.enable("src.evaluation.ifrs9")

    result = pd.DataFrame(rows)
    logger.info("ECL alpha sensitivity complete: {} levels swept", len(alphas))
    for _, row in result.iterrows():
        logger.info(
            "  alpha={:.2f} ({:.0f}%): ECL range={:,.0f} ({:.1f}% of point)",
            row["alpha"],
            row["confidence_pct"],
            row["total_ecl_range"],
            row["ecl_range_pct_of_point"],
        )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(base_lgd: float = 0.45, n_thresholds: int = 16) -> None:
    """Run SICR trigger optimization and ECL alpha sensitivity analysis."""
    intervals = _load_intervals()
    test_fe = _load_test_fe()

    # Part 1: SICR trigger sweep
    sicr_df = run_sicr_sweep(intervals, test_fe, n_thresholds=n_thresholds)

    # Part 2: ECL alpha sensitivity
    ecl_alpha_df = run_ecl_alpha_sensitivity(intervals, test_fe, base_lgd=base_lgd)

    # --- Persist outputs ---
    data_dir = ROOT / "data" / "processed"
    model_dir = ROOT / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    sicr_path = data_dir / "sicr_trigger_optimization.parquet"
    ecl_alpha_path = data_dir / "ecl_alpha_sensitivity.parquet"
    status_path = model_dir / "sicr_ecl_sensitivity_status.json"

    sicr_df.to_parquet(sicr_path, index=False)
    logger.info("Saved SICR trigger optimization: {}", sicr_path)

    ecl_alpha_df.to_parquet(ecl_alpha_path, index=False)
    logger.info("Saved ECL alpha sensitivity: {}", ecl_alpha_path)

    # Build combined status summary
    best_idx = int(sicr_df["f1_sicr"].idxmax())
    best_row = sicr_df.iloc[best_idx]

    ecl_90_row = ecl_alpha_df.loc[ecl_alpha_df["alpha"] == 0.10].iloc[0]

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "status": "success",
        "sicr_optimization": {
            "n_thresholds": n_thresholds,
            "best_threshold": float(best_row["width_threshold"]),
            "best_f1": float(best_row["f1_sicr"]),
            "best_precision": float(best_row["precision_sicr"]),
            "best_recall": float(best_row["recall_defaults"]),
            "best_stage2_share": float(best_row["stage2_share"]),
            "best_stage2_default_rate": float(best_row["stage2_default_rate"]),
        },
        "ecl_alpha_sensitivity": {
            "alphas_swept": [float(a) for a in ecl_alpha_df["alpha"]],
            "ecl_point_total": float(ecl_90_row["total_ecl_point"]),
            "ecl_range_at_90pct": float(ecl_90_row["total_ecl_range"]),
            "ecl_range_at_95pct": float(
                ecl_alpha_df.loc[ecl_alpha_df["alpha"] == 0.05, "total_ecl_range"].iloc[0]
            ),
            "ecl_range_at_80pct": float(
                ecl_alpha_df.loc[ecl_alpha_df["alpha"] == 0.20, "total_ecl_range"].iloc[0]
            ),
            "base_lgd": base_lgd,
        },
        "artifacts": {
            "sicr_trigger_optimization": str(sicr_path),
            "ecl_alpha_sensitivity": str(ecl_alpha_path),
        },
    }

    status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    logger.info("Saved status: {}", status_path)

    # Final summary
    logger.info(
        "Best SICR threshold: {:.3f} (F1={:.4f}, precision={:.4f}, recall={:.4f})",
        status["sicr_optimization"]["best_threshold"],
        status["sicr_optimization"]["best_f1"],
        status["sicr_optimization"]["best_precision"],
        status["sicr_optimization"]["best_recall"],
    )
    logger.info(
        "ECL range at 90%%: {:,.0f} ({:.1f}%% of point {:,.0f})",
        status["ecl_alpha_sensitivity"]["ecl_range_at_90pct"],
        float(ecl_90_row["ecl_range_pct_of_point"]),
        status["ecl_alpha_sensitivity"]["ecl_point_total"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SICR trigger optimization + ECL sensitivity to alpha"
    )
    parser.add_argument(
        "--base-lgd",
        type=float,
        default=0.45,
        help="Base LGD assumption (default: 0.45)",
    )
    parser.add_argument(
        "--n-thresholds",
        type=int,
        default=16,
        help="Number of SICR width thresholds to sweep (default: 16)",
    )
    args = parser.parse_args()
    raise SystemExit(main(base_lgd=args.base_lgd, n_thresholds=args.n_thresholds))
