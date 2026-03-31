"""Run diagnostic-only calibration remapping comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.evaluation.calibration_mapping import calibration_mapping_candidates_report
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
            "pd_calibrated": pd.to_numeric(preds[score_col], errors="coerce"),
            "issue_quarter": _coerce_issue_quarter(meta),
        }
    )
    if "issue_d" in meta.columns:
        frame["issue_d"] = pd.to_datetime(meta["issue_d"], errors="coerce")
    if "grade" in meta.columns:
        frame["grade"] = meta["grade"].astype("string")

    report = calibration_mapping_candidates_report(frame)
    report_path = DATA_DIR / "calibration_mapping_candidates.parquet"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(report_path, index=False)

    current = (
        report.loc[report["candidate_id"] == "current_identity"].iloc[0].to_dict()
        if not report.empty and (report["candidate_id"] == "current_identity").any()
        else {}
    )
    best = report.iloc[0].to_dict() if not report.empty else {}
    promising = bool(
        best
        and current
        and str(best.get("candidate_id")) != "current_identity"
        and float(current.get("abs_global_gap_bp", 0.0)) - float(best.get("abs_global_gap_bp", 0.0))
        >= 10.0
        and float(best.get("brier_score", 1.0)) <= float(current.get("brier_score", 1.0)) + 0.002
        and float(best.get("ece", 1.0)) <= float(current.get("ece", 1.0)) + 0.002
        and int(best.get("material_quarter_breaches", 0))
        <= int(current.get("material_quarter_breaches", 0))
    )
    severity = "pass"
    if promising:
        severity = "warning"
    elif current and (
        float(current.get("abs_global_gap_bp", 0.0)) >= 100.0
        or int(current.get("material_quarter_breaches", 0)) >= 2
    ):
        severity = "diagnostic_fail"

    payload = {
        "diagnostic_only": True,
        "overall_pass": bool(severity != "diagnostic_fail"),
        "severity": severity,
        "promising_candidate_exists": promising,
        "recommendation": "shadow_candidate" if promising else "keep_current_calibrator",
        "best_candidate": best,
        "current_candidate": current,
        "top_candidates": report.head(10).to_dict(orient="records") if not report.empty else [],
        "artifacts": {"candidate_report_path": str(report_path)},
        **build_artifact_metadata(
            schema_version=SCHEMA_VERSION,
            run_tag=resolved_run_tag,
            require_explicit=True,
        ),
    }
    out_path = MODEL_DIR / "calibration_mapping_status.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Calibration mapping diagnostics saved: {} (severity={}, best_candidate={})",
        out_path,
        severity,
        best.get("candidate_id", "none") if best else "none",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run diagnostic-only calibration remap comparisons"
    )
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()
    main(run_tag=args.run_tag)
