"""Export PR-AUC, Recall@0.35, and F1@0.35 to data/processed/model_comparison.json.

Reads the OOT test predictions, computes full classification metrics via
src.evaluation.metrics.classification_metrics(), and patches the
final_test_metrics block in model_comparison.json.

Usage:
    uv run python scripts/history/export_pr_auc_metrics.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.evaluation.metrics import classification_metrics

TEST_PREDS_PATH = Path("data/processed/test_predictions.parquet")
MODEL_COMPARISON_PATH = Path("data/processed/model_comparison.json")


def main() -> dict:
    """Compute PR-AUC / Recall / F1 on test set and patch model_comparison.json."""
    if not TEST_PREDS_PATH.exists():
        raise FileNotFoundError(f"Missing: {TEST_PREDS_PATH}")
    if not MODEL_COMPARISON_PATH.exists():
        raise FileNotFoundError(f"Missing: {MODEL_COMPARISON_PATH}")

    test_preds = pd.read_parquet(TEST_PREDS_PATH)
    logger.info("Loaded {:,} test predictions", len(test_preds))

    metrics = classification_metrics(
        test_preds["y_true"].values,
        test_preds["pd_calibrated"].values,
        threshold=0.35,
    )
    logger.info(
        "AUC={:.4f} | PR-AUC={:.4f} | Recall@0.35={:.4f} | F1@0.35={:.4f}",
        metrics["auc_roc"],
        metrics["pr_auc"],
        metrics["recall_at_0p35"],
        metrics["f1_at_0p35"],
    )

    with open(MODEL_COMPARISON_PATH, encoding="utf-8") as f:
        mc = json.load(f)

    existing = mc.get("final_test_metrics", {})
    existing.update(
        {
            "pr_auc": metrics["pr_auc"],
            "recall_at_0p35": metrics["recall_at_0p35"],
            "f1_at_0p35": metrics["f1_at_0p35"],
            "ks_statistic": metrics["ks_statistic"],
            "metrics_updated_at": datetime.now(tz=UTC).isoformat(),
        }
    )
    mc["final_test_metrics"] = existing

    MODEL_COMPARISON_PATH.write_text(json.dumps(mc, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Patched final_test_metrics in {}", MODEL_COMPARISON_PATH)
    return mc["final_test_metrics"]


if __name__ == "__main__":
    main()
