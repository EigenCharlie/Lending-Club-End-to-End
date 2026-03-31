"""Run ADSFCR-inspired IFRS9 diagnostics after the official IFRS9 sensitivity stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.evaluation.ifrs9_diagnostics import (
    adf_power_diagnostic,
    recursive_regression_paths,
    recursive_regression_status,
    scenario_interval_uncertainty,
    scenario_sign_coherence,
    sensitivity_surface_summary,
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

    ts_full = pd.read_parquet(DATA_DIR / "time_series_full.parquet")
    ifrs9_summary = pd.read_parquet(DATA_DIR / "ifrs9_scenario_summary.parquet")
    ifrs9_grid = pd.read_parquet(DATA_DIR / "ifrs9_sensitivity_grid.parquet")
    ts_scenarios = pd.read_parquet(DATA_DIR / "ts_ifrs9_scenarios.parquet")

    recursive_path = recursive_regression_paths(
        ts_full,
        time_col="ds",
        target_col="default_rate",
        feature_cols=["avg_int_rate", "avg_dti", "avg_loan_amnt"],
        min_window=36,
        weight_col="loan_count",
    )
    recursive_status = recursive_regression_status(recursive_path, min_sign_match_share=0.75)
    recursive_detail = recursive_status.pop("detail", pd.DataFrame())

    adf_status = adf_power_diagnostic(
        ts_full["default_rate"],
        n_simulations=120,
        candidate_phis=(0.80, 0.90, 0.95, 0.98),
        random_state=42,
    )
    coherence_df = scenario_sign_coherence(ifrs9_summary)
    coherence_pass = bool(
        not coherence_df.empty and coherence_df["overall_pass"].astype(bool).all()
    )
    uncertainty_status = scenario_interval_uncertainty(ts_scenarios)
    sensitivity_status = sensitivity_surface_summary(ifrs9_grid)

    recursive_path_path = DATA_DIR / "ifrs9_recursive_regression_paths.parquet"
    recursive_summary_path = DATA_DIR / "ifrs9_recursive_regression_summary.parquet"
    coherence_path = DATA_DIR / "ifrs9_sign_coherence.parquet"
    for path in [recursive_path_path, recursive_summary_path, coherence_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    recursive_path.to_parquet(recursive_path_path, index=False)
    pd.DataFrame(recursive_detail if isinstance(recursive_detail, pd.DataFrame) else []).to_parquet(
        recursive_summary_path, index=False
    )
    coherence_df.to_parquet(coherence_path, index=False)

    payload = {
        "diagnostic_only": True,
        "overall_pass": bool(
            recursive_status.get("overall_pass", False)
            and coherence_pass
            and bool(adf_status.get("available", False))
        ),
        "summary": {
            "recursive_regression_overall_pass": bool(recursive_status.get("overall_pass", False)),
            "recursive_n_features": int(recursive_status.get("n_features", 0)),
            "recursive_min_sign_match_share": float(
                recursive_status.get("min_sign_match_share", 0.0)
            ),
            "recursive_max_sign_flips": int(recursive_status.get("max_sign_flips", 0)),
            "adf_available": bool(adf_status.get("available", False)),
            "adf_pvalue_level": adf_status.get("adf_pvalue_level"),
            "adf_pvalue_diff1": adf_status.get("adf_pvalue_diff1"),
            "near_unit_root_power": adf_status.get("near_unit_root_power"),
            "adequate_near_unit_root_power": bool(
                adf_status.get("adequate_near_unit_root_power", False)
            ),
            "scenario_sign_coherence_pass": coherence_pass,
            "n_sign_coherence_checks": int(len(coherence_df)),
            "scenario_interval_uncertainty_available": bool(
                uncertainty_status.get("available", False)
            ),
            "mean_width_90": uncertainty_status.get("mean_width_90"),
            "mean_relative_width_90": uncertainty_status.get("mean_relative_width_90"),
            "ifrs9_sensitivity_available": bool(sensitivity_status.get("available", False)),
            "dominant_ifrs9_driver": sensitivity_status.get("dominant_driver"),
        },
        "recursive_regression": recursive_status,
        "adf_power": adf_status,
        "sign_coherence_checks": coherence_df.to_dict(orient="records"),
        "scenario_uncertainty": uncertainty_status,
        "sensitivity_surface": sensitivity_status,
        "artifacts": {
            "recursive_paths_path": str(recursive_path_path),
            "recursive_summary_path": str(recursive_summary_path),
            "sign_coherence_path": str(coherence_path),
        },
        **build_artifact_metadata(
            schema_version=SCHEMA_VERSION,
            run_tag=resolved_run_tag,
            require_explicit=True,
        ),
    }
    out_path = MODEL_DIR / "ifrs9_diagnostics_status.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "IFRS9 diagnostics saved: {} (recursive_pass={}, coherence_pass={}, near_unit_root_power={})",
        out_path,
        payload["summary"]["recursive_regression_overall_pass"],
        coherence_pass,
        payload["summary"]["near_unit_root_power"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADSFCR-inspired IFRS9 diagnostics")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()
    main(run_tag=args.run_tag)
