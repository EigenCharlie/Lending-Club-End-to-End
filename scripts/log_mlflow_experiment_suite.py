"""Backfill the full thesis experiment suite into DagsHub MLflow.

This script reads existing project artifacts and logs one run per domain:
- end_to_end snapshot
- pd_model
- conformal
- causal_policy
- ifrs9
- optimization
- survival
- time_series
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from loguru import logger

from src.utils.mlflow_utils import init_dagshub

ROOT = Path(__file__).resolve().parents[1]
MAX_ARTIFACT_MB = int(os.getenv("MLFLOW_MAX_ARTIFACT_MB", "64"))


def _load_json(path: str) -> dict[str, Any]:
    with open(ROOT / path, encoding="utf-8") as f:
        return json.load(f)


def _load_pickle(path: str) -> dict[str, Any]:
    with open(ROOT / path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in pickle artifact: {path}")
    return obj


def _to_metrics(data: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_to_metrics(value, prefix=f"{name}_"))
            continue
        if isinstance(value, bool):
            out[name] = float(int(value))
            continue
        if isinstance(value, (int, float)):
            value_f = float(value)
            if math.isfinite(value_f):
                out[name] = value_f
    return out


def _to_params(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif isinstance(value, (list, tuple, set)):
            out[key] = ",".join(str(v) for v in value)
        else:
            out[key] = str(value)
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _log_run(
    experiment_name: str,
    run_name: str,
    metrics: dict[str, float],
    params: dict[str, Any],
    tags: dict[str, str],
    artifacts: list[str],
) -> str:
    mlflow.set_experiment(experiment_name)

    clean_metrics = {k: float(v) for k, v in metrics.items() if math.isfinite(float(v))}
    clean_params = _to_params(params)

    with mlflow.start_run(run_name=run_name) as run:
        if clean_params:
            mlflow.log_params(clean_params)
        if clean_metrics:
            mlflow.log_metrics(clean_metrics)
        if tags:
            mlflow.set_tags(tags)

        for rel_path in artifacts:
            path = ROOT / rel_path
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb > MAX_ARTIFACT_MB:
                    logger.warning(
                        f"Artifact too large for tracking run ({size_mb:.1f}MB), skipped: {rel_path}"
                    )
                    continue
                mlflow.log_artifact(str(path))
            else:
                logger.warning(f"Artifact missing, skipped: {rel_path}")

        logger.info(
            f"Logged run: experiment={experiment_name}, run_id={run.info.run_id}, "
            f"metrics={len(clean_metrics)}, params={len(clean_params)}, artifacts={len(artifacts)}"
        )
        return run.info.run_id


def _log_pd(timestamp: str, common_tags: dict[str, str]) -> str:
    record = _load_pickle("models/pd_training_record.pkl")
    comparison = _load_json("data/processed/model_comparison.json")

    metrics = {
        **_to_metrics(record.get("final_test_metrics", {}), prefix="test_"),
        "optuna_best_auc": float(record.get("optuna_best_auc", 0.0)),
    }
    params = {
        "best_calibration": record.get("best_calibration"),
        "best_model": comparison.get("best_model"),
        "optuna_n_trials": comparison.get("optuna_n_trials"),
    }
    tags = {
        **common_tags,
        "domain": "pd_model",
    }
    artifacts = [
        "models/pd_canonical.cbm",
        "models/pd_canonical_calibrator.pkl",
        "models/pd_training_record.pkl",
        "data/processed/model_comparison.json",
        "configs/pd_model.yaml",
    ]
    return _log_run(
        experiment_name="lending_club/pd_model",
        run_name=f"artifact_backfill_pd_{timestamp}",
        metrics=metrics,
        params=params,
        tags=tags,
        artifacts=artifacts,
    )


def _log_conformal(timestamp: str, common_tags: dict[str, str]) -> str:
    status = _load_json("models/conformal_policy_status.json")
    checks_total = float(status.get("checks_total", 1) or 1)
    checks_passed = float(status.get("checks_passed", 0))
    overall_pass = bool(status.get("overall_pass", False))

    metrics = {
        "coverage_90": float(status.get("coverage_90", 0.0)),
        "coverage_95": float(status.get("coverage_95", 0.0)),
        "avg_width_90": float(status.get("avg_width_90", 0.0)),
        "min_group_coverage_90": float(status.get("min_group_coverage_90", 0.0)),
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "checks_passed_ratio": checks_passed / checks_total,
        "overall_pass": 1.0 if overall_pass else 0.0,
        "critical_alerts": float(status.get("critical_alerts", 0)),
        "warning_alerts": float(status.get("warning_alerts", 0)),
    }
    params = {
        "checks_total": int(status.get("checks_total", 0)),
        "overall_pass": overall_pass,
        "policy_config": status.get("policy_config", ""),
    }
    tags = {
        **common_tags,
        "domain": "conformal",
    }
    artifacts = [
        "models/conformal_policy_status.json",
        "models/conformal_results_mondrian.pkl",
        "data/processed/conformal_intervals_mondrian.parquet",
        "data/processed/conformal_group_metrics_mondrian.parquet",
    ]
    return _log_run(
        experiment_name="lending_club/conformal",
        run_name=f"artifact_backfill_conformal_{timestamp}",
        metrics=metrics,
        params=params,
        tags=tags,
        artifacts=artifacts,
    )


def _log_causal(timestamp: str, common_tags: dict[str, str]) -> str:
    summary = _load_pickle("models/causal_summary.pkl")
    policy = _load_pickle("models/causal_policy_summary.pkl")
    rule = _load_json("models/causal_policy_rule.json")
    selected = rule.get("selected_metrics", {})
    overall = policy.get("overall", {})

    metrics = {
        **_to_metrics(summary),
        **_to_metrics({"policy_total_net_value": overall.get("total_net_value", 0.0)}),
        **_to_metrics({"policy_discount_share": overall.get("discount_share", 0.0)}),
        **_to_metrics({"bootstrap_p05_net": selected.get("bootstrap_p05_net", 0.0)}),
        **_to_metrics({"action_rate": selected.get("action_rate", 0.0)}),
    }
    params = {
        "treatment": summary.get("treatment"),
        "selected_rule": rule.get("selected_rule"),
        "selection_reason": rule.get("selection_reason"),
    }
    tags = {
        **common_tags,
        "domain": "causal_policy",
    }
    artifacts = [
        "models/causal_summary.pkl",
        "models/causal_policy_summary.pkl",
        "models/causal_policy_rule.json",
        "data/processed/cate_estimates.parquet",
        "data/processed/causal_policy_simulation.parquet",
        "data/processed/causal_policy_rule_selected.parquet",
    ]
    return _log_run(
        experiment_name="lending_club/causal_policy",
        run_name=f"artifact_backfill_causal_{timestamp}",
        metrics=metrics,
        params=params,
        tags=tags,
        artifacts=artifacts,
    )


def _log_ifrs9(timestamp: str, common_tags: dict[str, str]) -> str:
    summary = _load_pickle("models/ifrs9_sensitivity_summary.pkl")
    scenarios = summary.get("scenario_summary", [])
    baseline = next((s for s in scenarios if s.get("scenario") == "baseline"), {})
    severe = next((s for s in scenarios if s.get("scenario") == "severe"), {})
    baseline_ecl = float(baseline.get("total_ecl", 0.0))
    severe_ecl = float(severe.get("total_ecl", 0.0))
    uplift = (severe_ecl / baseline_ecl - 1.0) * 100.0 if baseline_ecl > 0 else 0.0

    metrics = {
        "baseline_total_ecl": baseline_ecl,
        "severe_total_ecl": severe_ecl,
        "severe_uplift_pct": uplift,
        "min_total_ecl_grid": float(
            summary.get("sensitivity_extremes", {}).get("min_total_ecl", 0.0)
        ),
        "max_total_ecl_grid": float(
            summary.get("sensitivity_extremes", {}).get("max_total_ecl", 0.0)
        ),
    }
    params = {
        "n_scenarios": len(scenarios),
        "pd_orig_source": summary.get("input_quality", {}).get("source_pd_orig", ""),
        "dpd_source": summary.get("input_quality", {}).get("source_dpd", ""),
    }
    tags = {
        **common_tags,
        "domain": "ifrs9",
    }
    artifacts = [
        "models/ifrs9_sensitivity_summary.pkl",
        "data/processed/ifrs9_scenario_summary.parquet",
        "data/processed/ifrs9_scenario_grade_summary.parquet",
        "data/processed/ifrs9_sensitivity_grid.parquet",
    ]
    return _log_run(
        experiment_name="lending_club/ifrs9",
        run_name=f"artifact_backfill_ifrs9_{timestamp}",
        metrics=metrics,
        params=params,
        tags=tags,
        artifacts=artifacts,
    )


def _log_optimization(timestamp: str, common_tags: dict[str, str]) -> str:
    robustness = _load_pickle("models/portfolio_robustness_results.pkl")
    pipeline = _load_pickle("models/pipeline_results.pkl")

    metrics = {
        "robust_return_pipeline": float(pipeline.get("robust_return", 0.0)),
        "nonrobust_return_pipeline": float(pipeline.get("nonrobust_return", 0.0)),
        "price_of_robustness_pipeline": float(pipeline.get("price_of_robustness", 0.0)),
    }
    for row in robustness.get("summary_rows", []):
        rt = float(row.get("risk_tolerance", 0.0))
        suffix = f"risk_{int(round(rt * 100)):02d}"
        metrics[f"{suffix}_best_robust_return"] = float(row.get("best_robust_return", 0.0))
        metrics[f"{suffix}_price_of_robustness_pct"] = float(
            row.get("price_of_robustness_pct", 0.0)
        )
        metrics[f"{suffix}_best_robust_funded"] = float(row.get("best_robust_funded", 0.0))

    params = {
        "risk_grid": robustness.get("risk_grid", []),
        "aversion_grid": robustness.get("aversion_grid", []),
        "n_candidates": robustness.get("n_candidates", 0),
    }
    tags = {
        **common_tags,
        "domain": "optimization",
    }
    artifacts = [
        "models/portfolio_robustness_results.pkl",
        "data/processed/portfolio_robustness_frontier.parquet",
        "data/processed/portfolio_robustness_summary.parquet",
        "models/pipeline_results.pkl",
    ]
    return _log_run(
        experiment_name="lending_club/optimization",
        run_name=f"artifact_backfill_optimization_{timestamp}",
        metrics=metrics,
        params=params,
        tags=tags,
        artifacts=artifacts,
    )


def _log_survival(timestamp: str, common_tags: dict[str, str]) -> str:
    summary = _load_pickle("models/survival_summary.pkl")
    metrics = _to_metrics(summary)
    params = {
        "cox_features": summary.get("cox_features", []),
        "rsf_sample_size": summary.get("rsf_sample_size", 0),
    }
    tags = {
        **common_tags,
        "domain": "survival",
    }
    artifacts = [
        "models/survival_summary.pkl",
        "models/cox_ph_model.pkl",
        "models/rsf_model.pkl",
        "data/processed/km_curve_data.parquet",
    ]
    return _log_run(
        experiment_name="lending_club/survival",
        run_name=f"artifact_backfill_survival_{timestamp}",
        metrics=metrics,
        params=params,
        tags=tags,
        artifacts=artifacts,
    )


def _log_time_series(timestamp: str, common_tags: dict[str, str]) -> str:
    forecasts = pd.read_parquet(ROOT / "data/processed/ts_forecasts.parquet")
    point_candidates = [
        c
        for c in forecasts.columns
        if c not in {"unique_id", "ds"}
        and not c.endswith("-lo-90")
        and not c.endswith("-hi-90")
        and not c.endswith("-lo-95")
        and not c.endswith("-hi-95")
    ]
    primary_model = (
        "lgbm"
        if "lgbm" in point_candidates
        else (point_candidates[0] if point_candidates else None)
    )

    if primary_model is not None:
        lo_90_col = f"{primary_model}-lo-90"
        hi_90_col = f"{primary_model}-hi-90"
        width_90 = (
            forecasts[hi_90_col].astype(float) - forecasts[lo_90_col].astype(float)
            if lo_90_col in forecasts.columns and hi_90_col in forecasts.columns
            else pd.Series([0.0] * len(forecasts))
        )
        primary_mean = float(forecasts[primary_model].astype(float).mean())
    else:
        width_90 = pd.Series([0.0] * len(forecasts))
        primary_mean = 0.0

    metrics = {
        "n_forecast_rows": float(len(forecasts)),
        "horizon_months": float(forecasts["ds"].nunique()),
        "primary_mean_forecast": primary_mean,
        "primary_width_90_mean": float(width_90.mean()),
        "autoarima_mean_forecast": float(forecasts["AutoARIMA"].mean())
        if "AutoARIMA" in forecasts.columns
        else 0.0,
    }
    params = {
        "series_id": str(forecasts["unique_id"].iloc[0]) if len(forecasts) else "unknown",
        "models": ",".join(point_candidates),
        "primary_model": primary_model or "none",
    }
    tags = {
        **common_tags,
        "domain": "time_series",
    }
    artifacts = [
        "data/processed/ts_forecasts.parquet",
        "data/processed/ts_cv_stats.parquet",
        "data/processed/ts_ifrs9_scenarios.parquet",
    ]
    return _log_run(
        experiment_name="lending_club/time_series",
        run_name=f"artifact_backfill_time_series_{timestamp}",
        metrics=metrics,
        params=params,
        tags=tags,
        artifacts=artifacts,
    )


def _log_end_to_end(timestamp: str, common_tags: dict[str, str]) -> str:
    audit = _load_json("reports/project_audit_snapshot.json")
    summary = _load_json("data/processed/pipeline_summary.json")

    metrics = {}
    metrics.update(_to_metrics(summary.get("pipeline", {}), prefix="pipeline_"))
    metrics.update(_to_metrics(summary.get("pd_model", {}), prefix="pd_"))
    metrics.update(_to_metrics(summary.get("conformal", {}), prefix="conformal_"))
    metrics.update(_to_metrics(summary.get("survival", {}), prefix="survival_"))
    metrics.update(_to_metrics(summary.get("dataset", {}), prefix="dataset_"))
    metrics.update(_to_metrics(audit.get("ifrs9", {}), prefix="ifrs9_"))

    params = {
        "notebooks_executed": len(audit.get("notebooks", [])),
        "git_sha": common_tags["git_sha"],
        "source": "artifact_backfill",
    }
    tags = {
        **common_tags,
        "domain": "end_to_end",
    }
    artifacts = [
        "reports/project_audit_snapshot.json",
        "data/processed/pipeline_summary.json",
        "models/pipeline_results.pkl",
        "dvc.yaml",
        "dvc.lock",
    ]
    return _log_run(
        experiment_name="lending_club/end_to_end",
        run_name=f"artifact_backfill_end_to_end_{timestamp}",
        metrics=metrics,
        params=params,
        tags=tags,
        artifacts=artifacts,
    )


def main(repo_owner: str, repo_name: str) -> None:
    init_dagshub(repo_owner=repo_owner, repo_name=repo_name, enable_dvc=False)
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
    common_tags = {
        "project": "lending-club-end-to-end",
        "git_sha": _git_sha(),
        "sync_mode": "artifact_backfill",
        "logged_at_utc": timestamp,
    }

    run_ids = {
        "end_to_end": _log_end_to_end(timestamp, common_tags),
        "pd_model": _log_pd(timestamp, common_tags),
        "conformal": _log_conformal(timestamp, common_tags),
        "causal_policy": _log_causal(timestamp, common_tags),
        "ifrs9": _log_ifrs9(timestamp, common_tags),
        "optimization": _log_optimization(timestamp, common_tags),
        "survival": _log_survival(timestamp, common_tags),
        "time_series": _log_time_series(timestamp, common_tags),
    }

    logger.info("MLflow suite logged successfully")
    for name, run_id in run_ids.items():
        logger.info(f"{name}: {run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-owner",
        default=os.getenv("DAGSHUB_USER", "EigenCharlie94"),
        help="DagsHub user/org owner.",
    )
    parser.add_argument(
        "--repo-name",
        default=os.getenv("DAGSHUB_REPO", "Lending-Club-End-to-End"),
        help="DagsHub repository name.",
    )
    args = parser.parse_args()
    main(repo_owner=args.repo_owner, repo_name=args.repo_name)
