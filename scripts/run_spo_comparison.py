"""Run SPO+ decision regret comparison for Paper Estrella.

Compares decision regret of:
1. Two-stage classic: predict PD point estimates, optimize portfolio
2. Robust conformal: predict PD intervals, optimize with worst-case
3. SPO+ (if pyepo/torch available): train to minimize decision regret directly

This script consumes canonical artifacts and produces
``models/spo_comparison_status.json`` for the Paper Estrella Streamlit page.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.artifact_metadata import build_artifact_metadata

SCHEMA_VERSION = "2026-03-16.1"


def _load_canonical_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load test predictions, conformal intervals, and portfolio allocation."""
    data_dir = ROOT / "data" / "processed"

    test_preds = pd.read_parquet(data_dir / "test_predictions.parquet")
    logger.info("Loaded test_predictions: {} rows", len(test_preds))

    intervals = pd.read_parquet(data_dir / "conformal_intervals_mondrian.parquet")
    logger.info("Loaded conformal_intervals_mondrian: {} rows", len(intervals))

    robustness = pd.read_parquet(data_dir / "portfolio_robustness_summary.parquet")
    logger.info("Loaded portfolio_robustness_summary: {} rows", len(robustness))

    return test_preds, intervals, robustness


def _compute_two_stage_regret(
    y_true: np.ndarray,
    y_pred_point: np.ndarray,
) -> dict[str, float]:
    """Compute proxy decision regret for two-stage (point estimate) approach.

    Uses a simplified regret metric: mean absolute difference between
    the ranking induced by predictions vs true values, weighted by true cost.
    """
    pred_order = np.argsort(y_pred_point)
    true_order = np.argsort(y_true)

    # Rank correlation as proxy for decision alignment
    n = len(y_true)
    pred_ranks = np.empty(n, dtype=float)
    pred_ranks[pred_order] = np.arange(n, dtype=float)
    true_ranks = np.empty(n, dtype=float)
    true_ranks[true_order] = np.arange(n, dtype=float)

    rank_regret = float(np.mean(np.abs(pred_ranks - true_ranks)) / n)
    mae = float(np.mean(np.abs(y_pred_point - y_true)))

    return {
        "approach": "two_stage_point",
        "rank_regret": rank_regret,
        "mae": mae,
        "description": "Point PD estimate, optimize directly on predictions",
    }


def _compute_robust_regret(
    y_true: np.ndarray,
    y_pred_point: np.ndarray,
    y_pred_high: np.ndarray,
) -> dict[str, float]:
    """Compute proxy decision regret for robust conformal approach.

    The robust approach uses pd_high as worst-case, which is more conservative.
    """
    n = len(y_true)
    pred_order = np.argsort(y_pred_high)
    true_order = np.argsort(y_true)

    pred_ranks = np.empty(n, dtype=float)
    pred_ranks[pred_order] = np.arange(n, dtype=float)
    true_ranks = np.empty(n, dtype=float)
    true_ranks[true_order] = np.arange(n, dtype=float)

    rank_regret = float(np.mean(np.abs(pred_ranks - true_ranks)) / n)
    mae_point = float(np.mean(np.abs(y_pred_point - y_true)))
    mae_high = float(np.mean(np.abs(y_pred_high - y_true)))
    coverage = float(np.mean(y_true <= y_pred_high))

    return {
        "approach": "robust_conformal",
        "rank_regret": rank_regret,
        "mae_point": mae_point,
        "mae_high": mae_high,
        "worst_case_coverage": coverage,
        "description": "Conformal pd_high as worst-case, optimize conservatively",
    }


def _try_spo_comparison(
    y_true: np.ndarray,
    y_pred_point: np.ndarray,
) -> dict[str, object] | None:
    """Attempt SPO+ comparison if pyepo and torch are available."""
    try:
        import torch  # noqa: F401, I001
        from pyepo.func import SPOPlus  # noqa: F401

        logger.info("PyEPO and torch available. SPO+ comparison enabled.")
    except ImportError:
        logger.warning(
            "PyEPO or torch not available. "
            "SPO+ comparison skipped. Install with: pip install pyepo torch"
        )
        return {
            "approach": "spo_plus",
            "status": "skipped",
            "reason": "pyepo or torch not installed",
            "description": "SPO+ trains model to minimize decision regret directly",
        }

    # SPO+ requires a proper optimization model setup.
    # For now, return a placeholder indicating the capability exists.
    logger.info(
        "SPO+ full training requires optimization model setup. Returning capability assessment."
    )
    return {
        "approach": "spo_plus",
        "status": "capability_available",
        "reason": "pyepo and torch installed; full training requires optmodel setup",
        "description": "SPO+ trains neural net with decision-focused loss (Elmachtoub & Grigas 2022)",
        "reference_code": "src/optimization/spo_integration.py",
    }


def _compute_portfolio_economics(
    robustness: pd.DataFrame,
) -> dict[str, object]:
    """Extract portfolio economics from robustness summary."""
    if robustness.empty:
        return {"status": "no_data"}

    result: dict[str, object] = {}
    for _, row in robustness.iterrows():
        tol = float(row.get("risk_tolerance", 0))
        result[f"tol_{tol:.2f}"] = {
            "risk_tolerance": tol,
            "robust_return": float(row.get("best_robust_return", 0)),
            "nonrobust_return": float(row.get("baseline_nonrobust_return", 0)),
            "price_of_robustness": float(row.get("price_of_robustness", 0)),
            "price_of_robustness_pct": float(row.get("price_of_robustness_pct", 0)),
        }
    return result


def main() -> int:
    """Run SPO+ decision regret comparison."""
    parser = argparse.ArgumentParser(description="SPO+ decision regret comparison")
    parser.add_argument(
        "--run-tag",
        default=f"spo-comparison-{datetime.now(UTC).strftime('%Y-%m-%d-%H%M%S')}",
    )
    parser.add_argument("--sample-size", type=int, default=50_000)
    args = parser.parse_args()

    run_tag = str(args.run_tag).strip()
    sample_size = int(args.sample_size)

    logger.info("Starting SPO+ comparison | run_tag={}", run_tag)

    test_preds, intervals, robustness = _load_canonical_artifacts()

    # Align predictions with true labels
    target_col = "default_flag" if "default_flag" in test_preds.columns else "y_true"
    y_true = test_preds[target_col].to_numpy(dtype=float)
    y_pred_col = "y_prob_final" if "y_prob_final" in test_preds.columns else "y_prob_cb_tuned"
    y_pred_point = test_preds[y_pred_col].to_numpy(dtype=float)

    # Get conformal upper bound
    if "pd_high_90" in intervals.columns:
        y_pred_high = intervals["pd_high_90"].to_numpy(dtype=float)
    elif "y_pred" in intervals.columns:
        y_pred_high = intervals["y_pred"].to_numpy(dtype=float) + 0.1  # fallback
    else:
        y_pred_high = y_pred_point + 0.1

    # Align sizes
    n = min(len(y_true), len(y_pred_high), sample_size)
    y_true = y_true[:n]
    y_pred_point = y_pred_point[:n]
    y_pred_high = y_pred_high[:n]

    logger.info("Using {} samples for comparison", n)

    # Compute regret metrics
    two_stage = _compute_two_stage_regret(y_true, y_pred_point)
    robust = _compute_robust_regret(y_true, y_pred_point, y_pred_high)
    spo = _try_spo_comparison(y_true, y_pred_point)
    economics = _compute_portfolio_economics(robustness)

    # Build output
    output = {
        **build_artifact_metadata(
            schema_version=SCHEMA_VERSION,
            run_tag=run_tag,
        ),
        "sample_size": n,
        "prediction_column": y_pred_col,
        "two_stage": two_stage,
        "robust_conformal": robust,
        "spo_plus": spo,
        "portfolio_economics": economics,
        "summary": {
            "two_stage_rank_regret": two_stage["rank_regret"],
            "robust_rank_regret": robust["rank_regret"],
            "robust_vs_two_stage_pct": float(
                (two_stage["rank_regret"] - robust["rank_regret"])
                / max(two_stage["rank_regret"], 1e-9)
                * 100
            ),
        },
    }

    out_path = ROOT / "models" / "spo_comparison_status.json"
    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    logger.info("Written SPO+ comparison to {}", out_path)

    logger.info(
        "Two-stage rank regret: {:.6f} | Robust rank regret: {:.6f} | Improvement: {:.1f}%",
        two_stage["rank_regret"],
        robust["rank_regret"],
        output["summary"]["robust_vs_two_stage_pct"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
