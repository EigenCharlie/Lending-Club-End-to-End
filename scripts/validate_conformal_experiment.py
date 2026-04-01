"""Backtest and validate a namespaced conformal experiment.

Usage:
    uv run python scripts/validate_conformal_experiment.py \
        --namespace conformal_v3_focus_2026_03_26 \
        --run-tag conformal-v3-recovery-2026-03-26
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from loguru import logger

from scripts.backtest_conformal_coverage import main as backtest_main
from scripts.validate_conformal_policy import main as validate_main


def _build_paths(namespace: str) -> dict[str, Path]:
    ns = str(namespace).strip().replace("/", "_")
    data_dir = Path("data/processed/conformal_gap") / ns
    models_dir = Path("models/conformal_gap") / ns
    return {
        "data_dir": data_dir,
        "models_dir": models_dir,
        "intervals": data_dir / "conformal_intervals_mondrian.parquet",
        "group_metrics": data_dir / "conformal_group_metrics_mondrian.parquet",
        "backtest_monthly": data_dir / "conformal_backtest_monthly.parquet",
        "backtest_alerts": data_dir / "conformal_backtest_alerts.parquet",
        "results": models_dir / "conformal_results_mondrian.pkl",
        "policy_status": models_dir / "conformal_policy_status.json",
        "policy_checks": data_dir / "conformal_policy_checks.parquet",
        "policy_config": models_dir / "conformal_policy_validation.yaml",
    }


def main(
    *,
    namespace: str,
    run_tag: str | None = None,
    base_config_path: str = "configs/conformal_policy.yaml",
) -> None:
    paths = _build_paths(namespace)
    missing = [
        path
        for key, path in paths.items()
        if key in {"intervals", "group_metrics", "results"} and not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing namespaced conformal artifacts required for validation: "
            + ", ".join(str(p) for p in missing)
        )

    with open(base_config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["artifacts"]["conformal_results_path"] = str(paths["results"])
    cfg["artifacts"]["group_metrics_path"] = str(paths["group_metrics"])
    cfg["artifacts"]["backtest_monthly_path"] = str(paths["backtest_monthly"])
    cfg["artifacts"]["backtest_alerts_path"] = str(paths["backtest_alerts"])
    cfg["artifacts"]["intervals_path"] = str(paths["intervals"])
    cfg["output"]["policy_status_json"] = str(paths["policy_status"])
    cfg["output"]["policy_checks_parquet"] = str(paths["policy_checks"])

    paths["models_dir"].mkdir(parents=True, exist_ok=True)
    paths["data_dir"].mkdir(parents=True, exist_ok=True)
    with open(paths["policy_config"], "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    logger.info(f"Running namespaced backtest for conformal experiment: {namespace}")
    backtest_main(intervals_path=str(paths["intervals"]), output_dir=str(paths["data_dir"]))

    logger.info(f"Running namespaced policy validation for conformal experiment: {namespace}")
    validate_main(config_path=str(paths["policy_config"]), run_tag=run_tag)

    logger.info(f"Experiment validation complete: {paths['policy_status']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--base-config", default="configs/conformal_policy.yaml")
    args = parser.parse_args()
    main(namespace=args.namespace, run_tag=args.run_tag, base_config_path=args.base_config)
