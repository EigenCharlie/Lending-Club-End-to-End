"""Run bootstrap-based PD validation diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.evaluation.backtesting import bootstrap_pd_gap_test, bootstrap_slice_gap_report
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag
from src.utils.baseline_registry import resolve_official_baseline_run_tag

SCHEMA_VERSION = "2026-03-30.1"
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")


def _coerce_issue_quarter(meta: pd.DataFrame) -> pd.Series:
    if "issue_quarter" in meta.columns:
        return meta["issue_quarter"].astype("string")
    if "issue_d" in meta.columns:
        return pd.to_datetime(meta["issue_d"], errors="coerce").dt.to_period("Q").astype("string")
    return pd.Series(["unknown"] * len(meta), index=meta.index, dtype="string")


def main(run_tag: str | None = None) -> None:
    resolved_run_tag = resolve_run_tag(
        run_tag,
        fallback_candidates=[resolve_official_baseline_run_tag()],
        require_explicit=True,
    )
    preds = pd.read_parquet(DATA_DIR / "test_predictions.parquet")
    meta = pd.read_parquet(DATA_DIR / "test_fe.parquet")
    score_col = "pd_calibrated" if "pd_calibrated" in preds.columns else "y_prob_final"
    if score_col not in preds.columns:
        raise KeyError("Expected `pd_calibrated` or `y_prob_final` in test_predictions.parquet.")

    frame = pd.DataFrame(
        {
            "default_flag": pd.to_numeric(meta["default_flag"], errors="coerce"),
            "pd_score": pd.to_numeric(preds[score_col], errors="coerce"),
            "issue_quarter": _coerce_issue_quarter(meta),
        }
    )
    if "grade" in meta.columns:
        frame["grade"] = meta["grade"].astype("string")

    overall = bootstrap_pd_gap_test(
        frame["default_flag"].to_numpy(dtype=float),
        frame["pd_score"].to_numpy(dtype=float),
        n_boot=2_000,
        max_sample_size=10_000,
    )
    quarter_df = bootstrap_slice_gap_report(
        frame,
        group_col="issue_quarter",
        target_col="default_flag",
        score_col="pd_score",
        min_rows=150,
        n_boot=1_000,
        max_sample_size=5_000,
    )
    grade_df = (
        bootstrap_slice_gap_report(
            frame,
            group_col="grade",
            target_col="default_flag",
            score_col="pd_score",
            min_rows=200,
            n_boot=1_000,
            max_sample_size=5_000,
        )
        if "grade" in frame.columns
        else pd.DataFrame()
    )
    slices = (
        pd.concat([quarter_df, grade_df], ignore_index=True)
        if not quarter_df.empty or not grade_df.empty
        else pd.DataFrame()
    )
    slice_path = DATA_DIR / "bootstrap_validation_slices.parquet"
    slice_path.parent.mkdir(parents=True, exist_ok=True)
    slices.to_parquet(slice_path, index=False)

    excluded_zero = int((~slices["zero_inside_ci"].astype(bool)).sum()) if not slices.empty else 0
    severe_slices = (
        int((slices["materiality"].astype(str).isin({"high", "severe"})).sum())
        if not slices.empty
        else 0
    )
    severity = "pass"
    if (
        not bool(overall["zero_inside_ci"]) and float(overall["abs_gap_bp"]) >= 50.0
    ) or severe_slices >= 2:
        severity = "diagnostic_fail"
    elif excluded_zero > 0 or float(overall["abs_gap_bp"]) >= 25.0:
        severity = "warning"

    payload = {
        "diagnostic_only": True,
        "overall_pass": bool(severity != "diagnostic_fail"),
        "severity": severity,
        "signal_type": "bootstrap_gap_materiality",
        "summary": {
            **overall,
            "n_slice_rows": int(len(slices)),
            "slice_ci_exclusions": excluded_zero,
            "high_or_severe_slices": severe_slices,
        },
        "top_slice_rows": slices.head(15).to_dict(orient="records") if not slices.empty else [],
        "artifacts": {"slice_report_path": str(slice_path)},
        **build_artifact_metadata(
            schema_version=SCHEMA_VERSION,
            run_tag=resolved_run_tag,
            require_explicit=True,
        ),
    }
    out_path = MODEL_DIR / "bootstrap_validation_status.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Bootstrap validation diagnostics saved: {} (severity={}, abs_gap_bp={:.1f})",
        out_path,
        severity,
        float(overall["abs_gap_bp"]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bootstrap-based PD validation diagnostics")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()
    main(run_tag=args.run_tag)
