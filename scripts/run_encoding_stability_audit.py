"""Run ADSFCR-inspired encoding and binning stability diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from src.evaluation.encoding_stability import bucket_stability_report, woe_stability_report
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag
from src.utils.baseline_registry import resolve_official_baseline_run_tag
from src.utils.io_utils import read_split_with_fe_fallback

SCHEMA_VERSION = "2026-03-30.1"


def _load_cfg(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main(config_path: str = "configs/pd_model.champion.yaml", run_tag: str | None = None) -> None:
    cfg = _load_cfg(config_path)
    data_cfg = dict(cfg.get("data") or {})
    train_path = str(data_cfg.get("train_path", "data/processed/train_fe.parquet"))
    test_path = str(data_cfg.get("test_path", "data/processed/test_fe.parquet"))
    resolved_run_tag = resolve_run_tag(
        run_tag,
        fallback_candidates=[resolve_official_baseline_run_tag()],
        require_explicit=True,
    )

    train_df = read_split_with_fe_fallback(train_path)
    test_df = read_split_with_fe_fallback(test_path)
    if "default_flag" not in train_df.columns or "default_flag" not in test_df.columns:
        raise KeyError(
            "Missing 'default_flag' in train/test FE splits for encoding stability audit."
        )

    woe_report = woe_stability_report(train_df, test_df, target_col="default_flag")
    bucket_report = bucket_stability_report(train_df, test_df, target_col="default_flag")

    data_dir = Path("data/processed")
    model_dir = Path("models")
    woe_path = data_dir / "woe_encoding_stability.parquet"
    bucket_path = data_dir / "bucket_binning_stability.parquet"
    status_path = model_dir / "encoding_stability_status.json"
    woe_report.to_parquet(woe_path, index=False)
    bucket_report.to_parquet(bucket_path, index=False)

    payload = {
        "diagnostic_only": True,
        "overall_pass": bool(
            woe_report["overall_pass"].astype(bool).all() if not woe_report.empty else False
        )
        and bool(
            bucket_report["overall_pass"].astype(bool).all() if not bucket_report.empty else True
        ),
        "summary": {
            "n_woe_features": int(len(woe_report)),
            "n_bucket_features": int(len(bucket_report)),
            "woe_failures": int((~woe_report["overall_pass"].astype(bool)).sum())
            if not woe_report.empty
            else 0,
            "bucket_failures": int((~bucket_report["overall_pass"].astype(bool)).sum())
            if not bucket_report.empty
            else 0,
            "max_woe_psi": float(woe_report["psi"].max()) if not woe_report.empty else 0.0,
            "max_bucket_category_psi": float(bucket_report["category_psi"].max())
            if not bucket_report.empty
            else 0.0,
        },
        "top_woe_instabilities": woe_report.head(10).to_dict(orient="records"),
        "top_bucket_instabilities": bucket_report.head(10).to_dict(orient="records"),
        "artifacts": {
            "woe_report_path": str(woe_path),
            "bucket_report_path": str(bucket_path),
        },
        "config_path": str(config_path),
        **build_artifact_metadata(
            schema_version=SCHEMA_VERSION,
            run_tag=resolved_run_tag,
            require_explicit=True,
        ),
    }
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Encoding stability audit saved: {} (woe_failures={}, bucket_failures={})",
        status_path,
        payload["summary"]["woe_failures"],
        payload["summary"]["bucket_failures"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ADSFCR-inspired encoding stability diagnostics"
    )
    parser.add_argument("--config", default="configs/pd_model.champion.yaml")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()
    main(config_path=args.config, run_tag=args.run_tag)
