"""Run fairness audit across multiple protected attributes.

Computes demographic parity, equalized odds, and disparate impact
for each attribute defined in the fairness policy config.

Usage:
    uv run python scripts/run_fairness_audit.py
    uv run python scripts/run_fairness_audit.py --config configs/fairness_policy.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.evaluation.fairness import fairness_report


def _load_config(config_path: str) -> dict:
    """Load fairness policy YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_groups_dict(
    df: pd.DataFrame,
    attributes: list[dict],
) -> dict[str, np.ndarray]:
    """Build groups dict from config attribute definitions."""
    groups_dict: dict[str, np.ndarray] = {}

    for attr in attributes:
        name = attr["name"]
        col = attr["column"]

        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in data, skipping attribute '{name}'")
            continue

        if attr.get("binning") == "quartile":
            groups_dict[name] = (
                pd.qcut(df[col], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
                .astype(str)
                .values
            )
        else:
            groups_dict[name] = df[col].astype(str).values

    return groups_dict


def main(config_path: str = "configs/fairness_policy.yaml") -> None:
    """Run the fairness audit pipeline."""
    cfg = _load_config(config_path)
    policy = cfg["policy"]
    artifacts = cfg["artifacts"]
    output = cfg["output"]

    # Load test predictions and test data
    pred_path = Path(artifacts["test_predictions_path"])
    data_path = Path(artifacts["test_data_path"])

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing test predictions: {pred_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing test data: {data_path}")

    preds = pd.read_parquet(pred_path)
    data = pd.read_parquet(data_path)

    # Extract y_true and y_pred_proba
    y_true_col = "default_flag"
    y_proba_col = "y_pred_proba" if "y_pred_proba" in preds.columns else "pd_calibrated"

    if y_true_col not in data.columns:
        raise KeyError(f"Missing target column '{y_true_col}' in test data")
    if y_proba_col not in preds.columns:
        raise KeyError(
            f"Missing probability column in predictions. Available: {list(preds.columns)}"
        )

    # Align lengths (both should be OOT test set)
    n = min(len(preds), len(data))
    y_true = data[y_true_col].values[:n]
    y_proba = preds[y_proba_col].values[:n]

    logger.info(f"Loaded {n} observations for fairness audit")

    # Build groups from attributes config
    groups_dict = _build_groups_dict(data.iloc[:n], cfg["attributes"])

    if not groups_dict:
        logger.error("No valid attributes found for fairness audit")
        return

    # Run fairness report
    report = fairness_report(
        y_true=y_true,
        y_pred_proba=y_proba,
        groups_dict=groups_dict,
        threshold=policy["prediction_threshold"],
        dpd_threshold=policy["dpd_threshold"],
        eo_gap_threshold=policy["eo_gap_threshold"],
        dir_threshold=policy["dir_threshold"],
    )

    # Save audit parquet
    audit_path = Path(output["audit_parquet"])
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(audit_path, index=False)
    logger.info(f"Saved fairness audit: {audit_path}")

    # Build and save status JSON
    overall_pass = bool(report["passed_all"].all())
    status = {
        "overall_pass": overall_pass,
        "n_attributes": len(report),
        "n_passed": int(report["passed_all"].sum()),
        "attributes": report.to_dict(orient="records"),
        "thresholds": {
            "dpd": policy["dpd_threshold"],
            "eo_gap": policy["eo_gap_threshold"],
            "dir": policy["dir_threshold"],
        },
    }

    status_path = Path(output["status_json"])
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, default=str)
    logger.info(f"Saved fairness status: {status_path}")

    pass_label = "PASS" if overall_pass else "FAIL"
    logger.info(
        f"Fairness audit: {pass_label} ({status['n_passed']}/{status['n_attributes']} attributes)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fairness audit")
    parser.add_argument("--config", default="configs/fairness_policy.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
