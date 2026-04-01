"""Interpret PD backtesting results in materiality terms, not just p-values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.evaluation.pd_validation_interpretation import (
    quarter_materiality_report,
    rare_event_summary,
    summarize_slice_materiality,
    validation_interpretation_status,
)
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag
from src.utils.baseline_registry import resolve_official_baseline_run_tag

SCHEMA_VERSION = "2026-03-30.1"
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")


def main(run_tag: str | None = None) -> None:
    resolved_run_tag = resolve_run_tag(
        run_tag,
        fallback_candidates=[resolve_official_baseline_run_tag()],
        require_explicit=True,
    )
    backtesting = json.loads((MODEL_DIR / "pd_backtesting_status.json").read_text(encoding="utf-8"))
    rare_event_status = json.loads(
        (MODEL_DIR / "pd_rare_event_calibration_status.json").read_text(encoding="utf-8")
    )
    grade_df = pd.read_parquet(DATA_DIR / "pd_backtesting_by_grade.parquet")
    band_df = pd.read_parquet(DATA_DIR / "pd_backtesting_by_band.parquet")
    rare_event_report = pd.read_parquet(DATA_DIR / "pd_rare_event_calibration_report.parquet")
    predictions = pd.read_parquet(DATA_DIR / "test_predictions.parquet")
    meta = pd.read_parquet(DATA_DIR / "test_fe.parquet")

    quarter_report = quarter_materiality_report(predictions, meta)
    slice_materiality = summarize_slice_materiality(grade_df, band_df)
    rare_event = rare_event_summary(rare_event_status, rare_event_report)
    payload = validation_interpretation_status(
        overall_backtesting=backtesting.get("summary", {}),
        slice_materiality=slice_materiality,
        quarter_report=quarter_report,
        rare_event=rare_event,
    )

    quarter_path = DATA_DIR / "pd_backtesting_quarter_materiality.parquet"
    quarter_report.to_parquet(quarter_path, index=False)
    out_path = MODEL_DIR / "pd_validation_interpretation_status.json"
    out = {
        **payload,
        "quarter_rows": quarter_report.head(12).to_dict(orient="records"),
        "artifacts": {"quarter_materiality_path": str(quarter_path)},
        **build_artifact_metadata(
            schema_version=SCHEMA_VERSION,
            run_tag=resolved_run_tag,
            require_explicit=True,
        ),
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info(
        "PD validation interpretation saved: {} (severity={}, signal_type={})",
        out_path,
        out["severity"],
        out["signal_type"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpret PD backtesting in materiality terms")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()
    main(run_tag=args.run_tag)
