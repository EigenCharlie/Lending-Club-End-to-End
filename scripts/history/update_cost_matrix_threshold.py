"""Compute optimal cost-matrix threshold on the real OOT test set.

Runs optimal_threshold_cost_matrix() on test_predictions.parquet and
writes the result into models/threshold_semantics.json as cost_matrix_result.

Default costs:
  fn_cost = 1.0  (missing a default → LGD × EAD, normalised to 1)
  fp_cost = 0.12 (rejecting a good borrower → foregone ~12% interest)

Usage:
    uv run python scripts/history/update_cost_matrix_threshold.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.utils.threshold_semantics import (
    load_threshold_semantics,
    optimal_threshold_cost_matrix,
    write_threshold_semantics,
)

TEST_PREDS_PATH = Path("data/processed/test_predictions.parquet")
SEMANTICS_PATH = Path("models/threshold_semantics.json")
SCHEMA_VERSION = "2026-03-16.1"


def main(
    fn_cost: float = 1.0,
    fp_cost: float = 0.12,
) -> dict:
    """Run cost-matrix analysis on OOT test set and update threshold_semantics.json."""
    if not TEST_PREDS_PATH.exists():
        raise FileNotFoundError(f"Missing: {TEST_PREDS_PATH}")

    test_preds = pd.read_parquet(TEST_PREDS_PATH)
    logger.info("Loaded {:,} test predictions", len(test_preds))

    y_true = test_preds["y_true"].values
    y_prob = test_preds["pd_calibrated"].values

    result = optimal_threshold_cost_matrix(
        y_true, y_prob, fn_cost=fn_cost, fp_cost=fp_cost, n_steps=199
    )
    result["generated_at_utc"] = datetime.now(tz=UTC).isoformat()
    result["n_observations"] = len(test_preds)
    result["dataset"] = "oot_test_2018_2020"

    logger.info(
        "Cost-matrix optimal threshold: {:.3f} (min_cost={:.1f}, fn/fp ratio={:.1f})",
        result["optimal_threshold"],
        result["min_cost"],
        result["fn_fp_ratio"],
    )
    logger.info(
        "Cost at t=0.35 (operational): {:.1f}  |  Cost at optimal: {:.1f}  |  "
        "Savings from optimal: {:.1f} ({:.1%})",
        result["cost_at_035"],
        result["min_cost"],
        result["cost_at_035"] - result["min_cost"],
        (result["cost_at_035"] - result["min_cost"]) / result["cost_at_035"]
        if result["cost_at_035"] > 0
        else 0.0,
    )

    # Persist into threshold_semantics.json
    current = load_threshold_semantics(SEMANTICS_PATH)
    updated = write_threshold_semantics(
        pd_internal_selected_threshold=current.get("pd_internal_selected_threshold"),
        pd_internal_fallback_threshold=current.get("pd_internal_fallback_threshold"),
        fairness_primary_threshold=current.get("fairness_primary_threshold"),
        decision_policy_global_threshold=current.get("decision_policy_global_threshold"),
        source_artifacts={"test_predictions": str(TEST_PREDS_PATH)},
        run_tag="paper-grade-cost-matrix",
        path=SEMANTICS_PATH,
        extra={"cost_matrix_result": result},
    )

    logger.info("Updated threshold_semantics.json with cost_matrix_result")
    return updated


if __name__ == "__main__":
    main()
