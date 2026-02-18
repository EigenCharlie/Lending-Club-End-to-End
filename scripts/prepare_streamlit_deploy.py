"""Build a lightweight Streamlit deployment bundle for public sharing.

Usage:
    uv run python scripts/prepare_streamlit_deploy.py --clean
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dist" / "streamlit_deploy"

DATASET_SHAPE_ASSETS: list[tuple[str, str]] = [
    ("data/raw/Loan_status_2007-2020Q3.csv", "csv"),
    ("data/interim/lending_club_cleaned.parquet", "parquet"),
    ("data/processed/train.parquet", "parquet"),
    ("data/processed/calibration.parquet", "parquet"),
    ("data/processed/test.parquet", "parquet"),
    ("data/processed/train_fe.parquet", "parquet"),
    ("data/processed/loan_master.parquet", "parquet"),
    ("data/processed/time_series.parquet", "parquet"),
    ("data/processed/ead_dataset.parquet", "parquet"),
    ("data/processed/obt_loan_features.parquet", "parquet"),
]

REQUIRED_DIRS = [
    "streamlit_app",
    "reports/notebook_images",
    "dbt_project/models",
]

OPTIONAL_DIRS = [
    ".streamlit",
]

REQUIRED_FILES = [
    "requirements.streamlit.txt",
    "docs/LCDataDictionary.xlsx",
    "feature_repo/feature_views.py",
    "feature_repo/feature_services.py",
    "dbt_project/target/manifest.json",
    "dbt_project/target/run_results.json",
    "data/processed/eda_summary.json",
    "data/processed/feature_importance_iv.json",
    "data/processed/model_comparison.json",
    "data/processed/pipeline_summary.json",
    "data/processed/dataset_dictionary.json",
    "data/processed/macro_context.json",
    "data/processed/dataset_shapes_summary.json",
    "data/processed/calibration_curve_data.parquet",
    "data/processed/cate_estimates.parquet",
    "data/processed/causal_policy_grade_summary.parquet",
    "data/processed/causal_policy_rule_candidates.parquet",
    "data/processed/causal_policy_rule_selected.parquet",
    "data/processed/causal_policy_segment_summary.parquet",
    "data/processed/causal_policy_simulation.parquet",
    "data/processed/conformal_backtest_monthly.parquet",
    "data/processed/conformal_backtest_monthly_grade.parquet",
    "data/processed/conformal_group_metrics_mondrian.parquet",
    "data/processed/conformal_intervals.parquet",
    "data/processed/conformal_intervals_mondrian.parquet",
    "data/processed/conformal_policy_checks.parquet",
    "data/processed/hazard_ratios.parquet",
    "data/processed/ifrs9_input_quality.parquet",
    "data/processed/ifrs9_scenario_grade_summary.parquet",
    "data/processed/ifrs9_scenario_summary.parquet",
    "data/processed/ifrs9_sensitivity_grid.parquet",
    "data/processed/km_curve_data.parquet",
    "data/processed/lifetime_pd_table.parquet",
    "data/processed/loan_master.parquet",
    "data/processed/portfolio_allocations.parquet",
    "data/processed/portfolio_robustness_frontier.parquet",
    "data/processed/portfolio_robustness_summary.parquet",
    "data/processed/roc_curve_data.parquet",
    "data/processed/roi_by_grade.parquet",
    "data/processed/roi_by_grade_term.parquet",
    "data/processed/state_aggregates.parquet",
    "data/processed/test_predictions.parquet",
    "data/processed/time_series.parquet",
    "data/processed/ts_cv_stats.parquet",
    "data/processed/ts_forecasts.parquet",
    "data/processed/ts_ifrs9_scenarios.parquet",
    "data/processed/runtime_status.json",
    "models/conformal_policy_status.json",
    "models/pd_model_contract.json",
    "models/conformal_results_mondrian.pkl",
]

OPTIONAL_FILES = [
    "dbt_project/target/catalog.json",
    "data/lending_club.duckdb",
    "data/processed/efficient_frontier.parquet",
    "data/processed/ifrs9_ecl_comparison.parquet",
    "data/processed/modeva_governance_checks.parquet",
    "data/processed/modeva_governance_drift_ks.parquet",
    "data/processed/modeva_governance_drift_psi.parquet",
    "data/processed/modeva_governance_fairness.parquet",
    "data/processed/modeva_governance_metrics.parquet",
    "data/processed/modeva_governance_robustness.parquet",
    "data/processed/modeva_governance_slicing_accuracy.parquet",
    "data/processed/modeva_governance_slicing_robustness.parquet",
    "data/processed/pd_model_contract_validation.parquet",
    "data/processed/shap_raw_top20.parquet",
    "data/processed/shap_summary.parquet",
    "models/modeva_governance_status.json",
]


def _human_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{size_bytes}B"


def _copy_file(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return src.stat().st_size


def _copy_dir(src: Path, dst: Path) -> int:
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )
    return sum(p.stat().st_size for p in src.rglob("*") if p.is_file())


def _build_dataset_shapes_summary() -> dict[str, dict[str, int | None]]:
    summary: dict[str, dict[str, int | None]] = {}
    for rel_path, fmt in DATASET_SHAPE_ASSETS:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            continue
        if fmt == "csv":
            cols = int(pd.read_csv(path, nrows=0, low_memory=False).shape[1])
            summary[rel_path] = {"rows": None, "cols": cols}
        else:
            metadata = pq.ParquetFile(path).metadata
            summary[rel_path] = {
                "rows": int(metadata.num_rows),
                "cols": int(metadata.num_columns),
            }
    return summary


def _write_bundle_notes(output_dir: Path) -> None:
    notes = """# Streamlit Deploy Bundle

This folder was generated by `scripts/prepare_streamlit_deploy.py`.

## Run locally

```bash
pip install -r requirements.streamlit.txt
streamlit run streamlit_app/app.py
```

## Deploy in Streamlit Community Cloud

- Push this folder as a GitHub repository.
- In Streamlit Cloud, choose `streamlit_app/app.py` as entrypoint.
- Optionally set `GROK_API_KEY` in app secrets.
"""
    (output_dir / "DEPLOY_NOTES.md").write_text(notes, encoding="utf-8")


def build_bundle(output_dir: Path, clean: bool, strict: bool, skip_duckdb: bool) -> None:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_bytes = 0
    missing_required: list[str] = []
    missing_optional: list[str] = []

    for rel_dir in REQUIRED_DIRS:
        src = PROJECT_ROOT / rel_dir
        dst = output_dir / rel_dir
        if not src.exists():
            missing_required.append(rel_dir)
            continue
        copied_bytes += _copy_dir(src, dst)

    for rel_dir in OPTIONAL_DIRS:
        src = PROJECT_ROOT / rel_dir
        dst = output_dir / rel_dir
        if not src.exists():
            continue
        copied_bytes += _copy_dir(src, dst)

    for rel_file in REQUIRED_FILES:
        src = PROJECT_ROOT / rel_file
        dst = output_dir / rel_file
        if not src.exists():
            missing_required.append(rel_file)
            continue
        copied_bytes += _copy_file(src, dst)

    optional_files = [
        f for f in OPTIONAL_FILES if not (skip_duckdb and f == "data/lending_club.duckdb")
    ]
    for rel_file in optional_files:
        src = PROJECT_ROOT / rel_file
        dst = output_dir / rel_file
        if not src.exists():
            missing_optional.append(rel_file)
            continue
        copied_bytes += _copy_file(src, dst)

    # Always regenerate shape summary from local assets when possible.
    summary = _build_dataset_shapes_summary()
    summary_path = output_dir / "data" / "processed" / "dataset_shapes_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _write_bundle_notes(output_dir)

    print(f"Bundle output: {output_dir}")
    print(f"Approx copied size: {_human_size(copied_bytes)}")

    if missing_optional:
        print("Optional files not found:")
        for item in missing_optional:
            print(f"  - {item}")

    if missing_required:
        print("Missing required files:")
        for item in missing_required:
            print(f"  - {item}")
        if strict:
            raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare a lightweight Streamlit deployment bundle"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the deploy bundle",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output directory before copying",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any required artifact is missing",
    )
    parser.add_argument(
        "--skip-duckdb",
        action="store_true",
        help="Do not copy data/lending_club.duckdb",
    )

    args = parser.parse_args()
    build_bundle(
        output_dir=args.output,
        clean=args.clean,
        strict=args.strict,
        skip_duckdb=args.skip_duckdb,
    )
