"""Generate per-slice performance metrics and IsolationForest anomaly report.

Loads test predictions and training features, computes:
- Per-slice AUC / PR-AUC / Brier using grade, loan_amnt, dti, cohort slices
- IsolationForest on training features to flag OOD loans in the test set
- Combined status artifact: models/pd_slice_performance.json

Usage:
    uv run python scripts/history/generate_slice_anomaly_report.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.slicing_functions import compute_slice_metrics
from src.models.anomaly import (
    compute_anomaly_report,
    save_isolation_forest,
    train_isolation_forest,
)
from src.utils.artifact_metadata import build_artifact_metadata

TRAIN_FE_PATH = Path("data/processed/train_fe.parquet")
TEST_FE_PATH = Path("data/processed/test_fe.parquet")
TEST_PREDS_PATH = Path("data/processed/test_predictions.parquet")
CONTRACT_PATH = Path("models/pd_model_contract.json")
OUTPUT_PATH = Path("models/pd_slice_performance.json")
ISO_MODEL_PATH = Path("models/isolation_forest.pkl")
SCHEMA_VERSION = "2026-03-16.1"


def _load_features_for_iso(df: pd.DataFrame, contract_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Select numeric features available in the feature contract."""
    if not contract_path.exists():
        logger.warning("Contract not found at {}; using all numeric columns", contract_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return df[numeric_cols].fillna(0.0), numeric_cols

    with open(contract_path, encoding="utf-8") as f:
        contract = json.load(f)

    feature_names: list[str] = contract.get("feature_names", [])
    available = [c for c in feature_names if c in df.columns]
    # Isolate numeric-only — IsolationForest does not handle categoricals
    numeric = df[available].select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric].fillna(0.0), numeric


def main() -> dict:
    """Run slice performance + anomaly report and write models/pd_slice_performance.json."""
    # ------------------------------------------------------------------
    # 1. Load artifacts
    # ------------------------------------------------------------------
    if not TEST_PREDS_PATH.exists():
        raise FileNotFoundError(f"Missing: {TEST_PREDS_PATH}")
    if not TEST_FE_PATH.exists():
        raise FileNotFoundError(f"Missing: {TEST_FE_PATH}")

    test_preds = pd.read_parquet(TEST_PREDS_PATH)
    test_fe = pd.read_parquet(TEST_FE_PATH)
    logger.info("test_preds: {:,} rows | test_fe: {:,} rows", len(test_preds), len(test_fe))

    # Merge predictions into test_fe (both have the same row order)
    if "pd_calibrated" not in test_fe.columns:
        test_fe = test_fe.copy()
        test_fe["pd_calibrated"] = test_preds["pd_calibrated"].values
        test_fe["default_flag"] = test_preds["y_true"].values

    # Derive issue_year for cohort slices
    if "issue_year" not in test_fe.columns and "issue_d" in test_fe.columns:
        test_fe["issue_year"] = pd.to_datetime(test_fe["issue_d"], errors="coerce").dt.year

    # ------------------------------------------------------------------
    # 2. Per-slice metrics
    # ------------------------------------------------------------------
    logger.info("Computing per-slice metrics...")
    slice_results = compute_slice_metrics(
        test_fe,
        y_true_col="default_flag",
        y_prob_col="pd_calibrated",
    )
    logger.info("Slices evaluated: {}", len(slice_results))

    # ------------------------------------------------------------------
    # 3. IsolationForest: train on train_fe, score on test_fe
    # ------------------------------------------------------------------
    iso_report: dict = {}
    if TRAIN_FE_PATH.exists():
        logger.info("Loading train_fe for IsolationForest training...")
        train_fe = pd.read_parquet(TRAIN_FE_PATH)
        X_train, feature_names = _load_features_for_iso(train_fe, CONTRACT_PATH)
        X_test, _ = _load_features_for_iso(test_fe, CONTRACT_PATH)
        # Align to common columns
        common = [c for c in feature_names if c in X_test.columns]
        X_train_fit = X_train[common]
        X_test_score = X_test[common]
        logger.info(
            "Training IsolationForest on {:,} rows, {} features", len(X_train_fit), len(common)
        )
        iso = train_isolation_forest(X_train_fit, contamination=0.05, random_state=42)
        save_isolation_forest(iso, ISO_MODEL_PATH)
        iso_report = compute_anomaly_report(iso, X_test_score, label="oot_test")
        iso_report["n_features"] = len(common)
        iso_report["train_rows"] = len(X_train_fit)
        logger.info(
            "Anomaly rate in OOT test: {:.2%} ({:,} / {:,})",
            iso_report["anomaly_rate"],
            iso_report["n_anomalies"],
            iso_report["n_total"],
        )
    else:
        logger.warning("train_fe.parquet not found at {}; skipping IsolationForest", TRAIN_FE_PATH)
        iso_report = {"skipped": True, "reason": "train_fe_not_found"}

    # ------------------------------------------------------------------
    # 4. Build and save status artifact
    # ------------------------------------------------------------------
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "n_observations": len(test_fe),
        "slice_performance": slice_results,
        "isolation_forest": iso_report,
        **build_artifact_metadata(
            schema_version=SCHEMA_VERSION,
            run_tag="paper-grade-slice-anomaly",
            require_explicit=True,
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved slice + anomaly report: {}", OUTPUT_PATH)
    return status


if __name__ == "__main__":
    main()
