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
import hashlib
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
PRIMARY_BASELINE_REGISTRY_PATH = (
    ROOT / "configs" / "baselines" / "canonical_operational_baseline.json"
)
LEGACY_BASELINE_REGISTRY_PATH = ROOT / "configs" / "baselines" / "core_official_baseline.json"
BASELINE_REGISTRY_PATH = PRIMARY_BASELINE_REGISTRY_PATH


def _load_json(path: str) -> dict[str, Any]:
    with open(ROOT / path, encoding="utf-8") as f:
        return json.load(f)


def _load_pickle(path: str) -> dict[str, Any]:
    with open(ROOT / path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in pickle artifact: {path}")
    return obj


def _load_dependency_summary() -> list[dict[str, Any]]:
    path = ROOT / "reports" / "dependency_summary.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _project_audit_snapshot_rel_path() -> str | None:
    for rel_path in (
        "reports/project_audit_snapshot.json",
        "reports/history/project_audit_snapshot.json",
    ):
        if (ROOT / rel_path).exists():
            return rel_path
    return None


def _load_project_audit_snapshot() -> dict[str, Any]:
    rel_path = _project_audit_snapshot_rel_path()
    if not rel_path:
        return {}
    try:
        return _load_json(rel_path)
    except Exception:
        return {}


def _nested_numeric(payload: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    if isinstance(current, dict):
        current = current.get("value", default)
    try:
        value = float(current if current is not None else default)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


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
        if isinstance(value, int | float):
            value_f = float(value)
            if math.isfinite(value_f):
                out[name] = value_f
    return out


def _to_params(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            out[key] = value
        elif isinstance(value, list | tuple | set):
            out[key] = ",".join(str(v) for v in value)
        else:
            out[key] = str(value)
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _short_file_sha256(rel_path: str, length: int = 12) -> str:
    path = ROOT / rel_path
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def _dvc_remote_backend() -> str:
    cfg_candidates = [ROOT / ".dvc/config.local", ROOT / ".dvc/config"]
    current_remote = None
    for cfg_path in cfg_candidates:
        if not cfg_path.exists():
            continue
        for raw_line in cfg_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("["):
                current_remote = "dagshub" if 'remote "dagshub"' in line else None
                continue
            if current_remote == "dagshub" and line.startswith("url ="):
                url = line.split("=", 1)[1].strip()
                if url.startswith("s3://"):
                    return "s3"
                if url.startswith("http://") or url.startswith("https://"):
                    return "http"
    return "unknown"


def _resolve_official_baseline_run_tag(cli_value: str | None = None) -> str:
    if cli_value:
        return str(cli_value).strip()
    env_value = os.getenv("OFFICIAL_BASELINE_RUN_TAG", "").strip()
    if env_value:
        return env_value
    for path in (
        BASELINE_REGISTRY_PATH,
        PRIMARY_BASELINE_REGISTRY_PATH,
        LEGACY_BASELINE_REGISTRY_PATH,
    ):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        value = str(payload.get("official_run_tag", "")).strip()
        if value:
            return value
    return "unknown"


def _configure_tracking_non_interactive(repo_owner: str, repo_name: str) -> None:
    token = os.getenv("DAGSHUB_USER_TOKEN") or os.getenv("DAGSHUB_TOKEN")
    if token:
        init_dagshub(repo_owner=repo_owner, repo_name=repo_name, enable_dvc=False)
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        return

    fallback_dir = ROOT / "reports" / "mlruns"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    fallback_uri = os.getenv("MLFLOW_FALLBACK_TRACKING_URI", f"file:{fallback_dir.resolve()}")
    mlflow.set_tracking_uri(fallback_uri)
    logger.warning(
        "DAGSHUB token missing. Running MLflow suite in non-interactive local mode "
        f"(tracking_uri={fallback_uri})."
    )


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
    clean_tags = dict(tags or {})
    clean_tags.setdefault("pipeline_scope", experiment_name.split("/")[-1])

    with mlflow.start_run(run_name=run_name) as run:
        if clean_params:
            mlflow.log_params(clean_params)
        if clean_metrics:
            mlflow.log_metrics(clean_metrics)
        if clean_tags:
            mlflow.set_tags(clean_tags)

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
    feature_manifest = (
        pd.read_parquet(ROOT / "data/processed/feature_manifest_v2.parquet")
        if (ROOT / "data/processed/feature_manifest_v2.parquet").exists()
        else pd.DataFrame()
    )

    metrics = {
        **_to_metrics(record.get("final_test_metrics", {}), prefix="test_"),
        "optuna_best_auc": float(record.get("optuna_best_auc", 0.0)),
        "feature_manifest_rows": float(len(feature_manifest)),
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
        "models/pd_training_status.json",
        "data/processed/model_comparison.json",
        "data/processed/feature_manifest_v2.parquet",
        "data/processed/brier_score_decomposition.json",
        "reports/figures/calibration/murphy_diagram.png",
        "configs/pd_model.yaml",
    ]

    # Log CatBoost model with signature when available
    model_path = ROOT / "models" / "pd_canonical.cbm"
    catboost_logged = False
    if model_path.exists():
        try:
            import mlflow.catboost
            from catboost import CatBoostClassifier
            from mlflow.models.signature import infer_signature

            cb_model = CatBoostClassifier()
            cb_model.load_model(str(model_path))

            # Use test predictions for signature inference
            test_preds_path = ROOT / "data" / "processed" / "test_predictions.parquet"
            sig = None
            if test_preds_path.exists():
                try:
                    test_preds = pd.read_parquet(test_preds_path)
                    sig = infer_signature(
                        pd.DataFrame({"y_prob": test_preds["y_prob_final"].head(5)}),
                        test_preds["y_prob_final"].head(5).values,
                    )
                except Exception:
                    pass
            catboost_logged = True
        except ImportError:
            catboost_logged = False

    run_name = f"artifact_backfill_pd_{timestamp}"
    mlflow.set_experiment("lending_club/pd_model")
    clean_metrics = {k: float(v) for k, v in metrics.items() if math.isfinite(float(v))}
    clean_params = _to_params(params)
    clean_tags = {**tags, "pipeline_scope": "pd_model"}

    with mlflow.start_run(run_name=run_name) as run:
        if clean_params:
            mlflow.log_params(clean_params)
        if clean_metrics:
            mlflow.log_metrics(clean_metrics)
        if clean_tags:
            mlflow.set_tags(clean_tags)

        # Log CatBoost model natively with signature
        if catboost_logged:
            try:
                mlflow.catboost.log_model(
                    cb_model,
                    artifact_path="pd_catboost",
                    signature=sig,
                )
                logger.info("Logged CatBoost model to MLflow with native catboost flavor")
            except Exception as exc:
                logger.warning(f"CatBoost model logging failed: {exc}")

        for rel_path in artifacts:
            path = ROOT / rel_path
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb <= MAX_ARTIFACT_MB:
                    mlflow.log_artifact(str(path))

        logger.info(f"Logged PD run: run_id={run.info.run_id}")
        return run.info.run_id


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
        "models/causal_effect_status.json",
        "models/causal_overlap_status.json",
        "models/causal_sensitivity_status.json",
        "models/causal_estimator_selection_status.json",
        "models/causal_effect_runtime_status.json",
        "models/causal_policy_simulation_runtime_status.json",
        "models/causal_policy_validation_runtime_status.json",
        "models/causal_policy_oot_runtime_status.json",
        "models/causal_policy_summary.pkl",
        "models/causal_policy_rule.json",
        "models/causal_policy_oot_status.json",
        "data/processed/cate_estimates.parquet",
        "data/processed/cate_estimates_oot.parquet",
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
        "models/ifrs9_runtime_status.json",
        "data/processed/ifrs9_scenario_summary.parquet",
        "data/processed/ifrs9_scenario_grade_summary.parquet",
        "data/processed/ifrs9_sensitivity_grid.parquet",
        "data/processed/ifrs9_input_quality.parquet",
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
        "models/portfolio_research_policy.json",
        "models/portfolio_optimization_runtime_status.json",
        "models/portfolio_tradeoff_runtime_status.json",
        "data/processed/portfolio_robustness_frontier.parquet",
        "data/processed/portfolio_robustness_summary.parquet",
        "data/processed/champion_candidate_universe.parquet",
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
        "models/survival_runtime_status.json",
        "models/cox_ph_model.pkl",
        "models/rsf_model.pkl",
        "data/processed/lifetime_pd_table.parquet",
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
    metrics_path = ROOT / "data/processed/ts_backtest_metrics.parquet"
    predictions_path = ROOT / "data/processed/ts_backtest_predictions.parquet"
    status_path = ROOT / "models/time_series_status.json"
    diagnostics_path = ROOT / "data/processed/ts_diagnostics.json"

    backtest_metrics = pd.read_parquet(metrics_path) if metrics_path.exists() else pd.DataFrame()
    backtest_predictions = (
        pd.read_parquet(predictions_path) if predictions_path.exists() else pd.DataFrame()
    )
    status = _load_json("models/time_series_status.json") if status_path.exists() else {}
    diagnostics = (
        _load_json("data/processed/ts_diagnostics.json") if diagnostics_path.exists() else {}
    )

    point_candidates = sorted(
        c
        for c in forecasts.columns
        if c not in {"unique_id", "ds", "y", "point_model", "interval_model", "official_status"}
        and not c.startswith("y_")
        and not c.endswith("-lo-90")
        and not c.endswith("-hi-90")
        and not c.endswith("-lo-95")
        and not c.endswith("-hi-95")
    )
    point_champion = status.get("point_champion", {})
    interval_champion = status.get("interval_champion", {})
    forecast_point_model = (
        str(forecasts["point_model"].iloc[0])
        if "point_model" in forecasts.columns and len(forecasts)
        else "none"
    )
    forecast_interval_model = (
        str(forecasts["interval_model"].iloc[0])
        if "interval_model" in forecasts.columns and len(forecasts)
        else forecast_point_model
    )
    primary_model = str(point_champion.get("model") or forecast_point_model)
    interval_model = str(interval_champion.get("model") or forecast_interval_model)

    point_row = backtest_metrics.loc[backtest_metrics["model"] == primary_model]
    interval_row = backtest_metrics.loc[backtest_metrics["model"] == interval_model]
    point_metrics = point_row.iloc[0].to_dict() if not point_row.empty else {}
    interval_metrics = interval_row.iloc[0].to_dict() if not interval_row.empty else {}

    width_90 = (
        forecasts["y_hi_90"].astype(float) - forecasts["y_lo_90"].astype(float)
        if {"y_lo_90", "y_hi_90"}.issubset(forecasts.columns)
        else pd.Series([0.0] * len(forecasts))
    )
    primary_mean = float(forecasts["y"].astype(float).mean()) if "y" in forecasts.columns else 0.0
    official_status = (
        str(forecasts["official_status"].iloc[0])
        if "official_status" in forecasts.columns and len(forecasts)
        else "unknown"
    )
    recent_actual_mean = status.get("summary", {}).get("recent_actual_mean_12m")
    seasonal_strength = _nested_numeric(
        diagnostics,
        "seasonal_strength",
        default=_nested_numeric(diagnostics, "stl", "seasonal_strength", default=0.0),
    )
    variance_ratio = _nested_numeric(diagnostics, "variance_ratio", default=0.0)

    metrics = {
        "n_forecast_rows": float(len(forecasts)),
        "horizon_months": float(forecasts["ds"].nunique()),
        "n_backtest_rows": float(len(backtest_predictions)),
        "n_models_evaluated": float(backtest_metrics["model"].nunique())
        if not backtest_metrics.empty
        else 0.0,
        "primary_mean_forecast": primary_mean,
        "primary_width_90_mean": float(width_90.mean()),
        "recent_actual_mean_12m": float(recent_actual_mean)
        if recent_actual_mean is not None
        else 0.0,
        "point_champion_mae": float(point_metrics.get("mae", 0.0) or 0.0),
        "point_champion_mase": float(point_metrics.get("mase", 0.0) or 0.0),
        "point_champion_rmsse": float(point_metrics.get("rmsse", 0.0) or 0.0),
        "point_champion_fva_mae_pct": float(point_metrics.get("fva_mae_pct", 0.0) or 0.0),
        "interval_champion_coverage_90": float(interval_metrics.get("coverage_90", 0.0) or 0.0),
        "interval_champion_coverage_gap_90": float(
            interval_metrics.get("coverage_gap_90", 0.0) or 0.0
        ),
        "interval_champion_winkler_90": float(interval_metrics.get("winkler_90", 0.0) or 0.0),
        "interval_champion_avg_width_90": float(
            interval_metrics.get("avg_interval_width_90", 0.0) or 0.0
        ),
        "point_promotable": float(bool(point_champion.get("promotable", False))),
        "interval_promotable": float(bool(interval_champion.get("promotable", False))),
        "status_pass": float(str(status.get("status", "")) == "pass"),
        "seasonal_strength": seasonal_strength,
        "variance_ratio": variance_ratio,
    }
    params = {
        "series_id": str(forecasts["unique_id"].iloc[0]) if len(forecasts) else "unknown",
        "models": ",".join(point_candidates),
        "point_model": primary_model or "none",
        "interval_model": interval_model or "none",
        "status": str(status.get("status", "unknown")),
        "official_status": official_status,
        "point_promotable": bool(point_champion.get("promotable", False)),
        "interval_promotable": bool(interval_champion.get("promotable", False)),
        "exogenous_enabled": bool(status.get("config", {}).get("exogenous_enabled", False)),
    }
    tags = {
        **common_tags,
        "domain": "time_series",
        "time_series_status": str(status.get("status", "unknown")),
    }
    artifacts = [
        "configs/time_series.yaml",
        "data/processed/time_series_full.parquet",
        "data/processed/time_series_panel.parquet",
        "data/processed/ts_backtest_predictions.parquet",
        "data/processed/ts_backtest_metrics.parquet",
        "data/processed/ts_forecasts.parquet",
        "data/processed/ts_cv_stats.parquet",
        "data/processed/ts_ifrs9_scenarios.parquet",
        "data/processed/ts_diagnostics.json",
        "data/processed/ts_panel_forecasts.parquet",
        "models/time_series_status.json",
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
    audit = _load_project_audit_snapshot()
    summary = _load_json("data/processed/pipeline_summary.json")
    dependency_summary = _load_dependency_summary()
    feature_manifest = (
        pd.read_parquet(ROOT / "data/processed/feature_manifest_v2.parquet")
        if (ROOT / "data/processed/feature_manifest_v2.parquet").exists()
        else pd.DataFrame()
    )

    metrics = {}
    metrics.update(_to_metrics(summary.get("pipeline", {}), prefix="pipeline_"))
    metrics.update(_to_metrics(summary.get("pd_model", {}), prefix="pd_"))
    metrics.update(_to_metrics(summary.get("conformal", {}), prefix="conformal_"))
    metrics.update(_to_metrics(summary.get("survival", {}), prefix="survival_"))
    metrics.update(_to_metrics(summary.get("dataset", {}), prefix="dataset_"))
    metrics.update(_to_metrics(audit.get("ifrs9", {}), prefix="ifrs9_"))
    metrics["dependency_count"] = float(len(dependency_summary))
    metrics["dependency_with_usage_count"] = float(
        sum(1 for row in dependency_summary if int(row.get("file_count", 0)) > 0)
    )
    metrics["feature_manifest_rows"] = float(len(feature_manifest))

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
        "data/processed/pipeline_summary.json",
        "data/processed/feature_manifest_v2.parquet",
        "models/pipeline_results.pkl",
        "dvc.yaml",
        "dvc.lock",
    ]
    audit_artifact = _project_audit_snapshot_rel_path()
    if audit_artifact:
        artifacts.insert(0, audit_artifact)
    if (ROOT / "reports" / "dependency_summary.json").exists():
        artifacts.insert(1 if audit_artifact else 0, "reports/dependency_summary.json")
    return _log_run(
        experiment_name="lending_club/end_to_end",
        run_name=f"artifact_backfill_end_to_end_{timestamp}",
        metrics=metrics,
        params=params,
        tags=tags,
        artifacts=artifacts,
    )


def main(repo_owner: str, repo_name: str, official_baseline_run_tag: str | None = None) -> None:
    _configure_tracking_non_interactive(repo_owner=repo_owner, repo_name=repo_name)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
    data_version = _short_file_sha256("dvc.lock")
    dvc_backend = _dvc_remote_backend()
    baseline_tag = _resolve_official_baseline_run_tag(official_baseline_run_tag)
    common_tags = {
        "project": "lending-club-end-to-end",
        "git_sha": _git_sha(),
        "sync_mode": "artifact_backfill",
        "logged_at_utc": timestamp,
        "data_version": data_version,
        "dvc_remote_backend": dvc_backend,
        "official_baseline_run_tag": baseline_tag,
    }
    suite_sync_run_id = ""

    # Keep per-domain experiments unchanged, but create a suite parent run to improve traceability.
    mlflow.set_experiment("lending_club/suite_sync")
    with mlflow.start_run(run_name=f"artifact_backfill_suite_sync_{timestamp}") as suite_run:
        suite_sync_run_id = suite_run.info.run_id
        mlflow.log_params(
            {
                "repo_owner": repo_owner,
                "repo_name": repo_name,
                "sync_mode": "artifact_backfill",
                "domains_expected": 8,
                "official_baseline_run_tag": baseline_tag,
            }
        )
        mlflow.set_tags(
            {
                **common_tags,
                "domain": "suite_sync",
                "pipeline_scope": "suite_sync",
            }
        )
        logger.info(f"Suite sync run started: {suite_sync_run_id}")

    common_tags = {
        **common_tags,
        "suite_sync_run_id": suite_sync_run_id,
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

    # Re-open suite sync run to attach child run links and completion summary.
    if suite_sync_run_id:
        mlflow.set_experiment("lending_club/suite_sync")
        with mlflow.start_run(run_id=suite_sync_run_id):
            mlflow.log_metrics({"child_run_count": float(len(run_ids))})
            mlflow.set_tags({f"child_{name}_run_id": run_id for name, run_id in run_ids.items()})

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
    parser.add_argument(
        "--official-baseline-run-tag",
        default=None,
        help=(
            "Tag of the frozen official baseline to attach to suite and child runs. "
            "Defaults to OFFICIAL_BASELINE_RUN_TAG env var or baseline registry."
        ),
    )
    args = parser.parse_args()
    main(
        repo_owner=args.repo_owner,
        repo_name=args.repo_name,
        official_baseline_run_tag=args.official_baseline_run_tag,
    )
