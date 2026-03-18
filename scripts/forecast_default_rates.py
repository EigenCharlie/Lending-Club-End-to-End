"""Forecast monthly default rates with governed backtests and status artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.build_datasets import (
    build_time_series,
    build_time_series_panel,
    clean_raw_columns,
    load_historical_time_series_source,
)
from src.models.time_series import (
    benchmark_mapie_time_series_intervals,
    build_canonical_forecast_frame,
    build_future_covariates_contract,
    build_ifrs9_temporal_scenarios,
    build_status_payload,
    compute_forecastability_diagnostics,
    compute_forecastability_report,
    compute_point_ensemble_weights,
    evaluate_hierarchical_reconciliation,
    forecast_panel_bottom_up,
    forecast_portfolio_models,
    infer_run_tag,
    load_future_covariates,
    load_time_series_config,
    run_portfolio_backtest,
    select_time_series_champions,
)
from src.utils.artifact_metadata import resolve_run_tag

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")


def _panel_has_irregular_monthly_grid(panel: pd.DataFrame) -> bool:
    """Return True when bottom-level panel series contain gaps or duplicate months."""
    if panel.empty or "series_level" not in panel.columns or "unique_id" not in panel.columns:
        return True
    bottom = panel.loc[panel["series_level"] == "grade_term", ["unique_id", "ds"]].copy()
    if bottom.empty:
        return True
    bottom["ds"] = pd.to_datetime(bottom["ds"], errors="coerce")
    bottom = bottom.dropna(subset=["ds"])
    for _, group in bottom.groupby("unique_id", sort=False):
        ordered = group.sort_values("ds").reset_index(drop=True)
        if bool(ordered.duplicated(subset=["ds"]).any()):
            return True
        if len(ordered) <= 1:
            continue
        expected = pd.date_range(ordered["ds"].min(), ordered["ds"].max(), freq="MS")
        if len(expected) != len(ordered):
            return True
    return False


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Saved {}", path)


def _resolve_output_paths(config: dict) -> dict[str, Path]:
    outputs = config.get("outputs", {})
    return {key: Path(str(value)) for key, value in outputs.items()}


def _rebuild_history_artifacts() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate governed time-series artifacts from canonical split files."""
    source = clean_raw_columns(load_historical_time_series_source(DATA_DIR / "train.parquet"))
    portfolio = build_time_series(source)
    panel = build_time_series_panel(source)

    portfolio.to_parquet(DATA_DIR / "time_series_full.parquet", index=False)
    panel.to_parquet(DATA_DIR / "time_series_panel.parquet", index=False)
    if not (DATA_DIR / "time_series.parquet").exists():
        portfolio.to_parquet(DATA_DIR / "time_series.parquet", index=False)

    logger.warning(
        "Regenerated missing time-series artifacts from canonical splits: {} rows portfolio, {} rows panel",
        len(portfolio),
        len(panel),
    )
    return portfolio, panel


def _load_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    full_path = DATA_DIR / "time_series_full.parquet"
    panel_path = DATA_DIR / "time_series_panel.parquet"
    if not full_path.exists() or not panel_path.exists():
        portfolio, panel = _rebuild_history_artifacts()
    else:
        portfolio = pd.read_parquet(full_path)
        panel = pd.read_parquet(panel_path)
        if _panel_has_irregular_monthly_grid(panel):
            logger.warning(
                "Detected stale/irregular time_series_panel artifacts; rebuilding canonical history."
            )
            portfolio, panel = _rebuild_history_artifacts()
    logger.info("Loaded time_series_full: {}", portfolio.shape)
    logger.info("Loaded time_series_panel: {}", panel.shape)
    return portfolio, panel


def main(config_path: str = "configs/time_series.yaml", horizon: int | None = None) -> None:
    config = load_time_series_config(config_path)
    if horizon is not None:
        config["horizon"] = int(horizon)
    output_paths = _resolve_output_paths(config)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    portfolio_history, panel_history = _load_history()
    forecastability_report, forecastability_status = compute_forecastability_report(
        panel_history,
        season_length=int(config.get("season_length", 12)),
        forecastability_cfg=config.get("forecastability", {}),
    )
    if not forecastability_report.empty:
        forecastability_report.to_parquet(output_paths["forecastability_report_path"], index=False)
    _write_json(output_paths["forecastability_status_path"], forecastability_status)
    future_covariates_contract = build_future_covariates_contract(
        portfolio_history[["unique_id", "ds", "y"]],
        horizon=int(config["horizon"]),
        freq=str(config.get("freq", "MS")),
        macro_context_path=DATA_DIR / "macro_context.json",
    )
    future_covariates_contract.to_parquet(DATA_DIR / "ts_future_covariates.parquet", index=False)
    future_covariates = load_future_covariates(config)
    backtest_predictions, backtest_metrics = run_portfolio_backtest(
        portfolio_history[["unique_id", "ds", "y"]],
        config,
        future_covariates=future_covariates,
    )
    if backtest_metrics.empty:
        raise RuntimeError("Time-series backtest produced no metrics; cannot continue.")

    champions = select_time_series_champions(backtest_metrics, config)
    point_weights = compute_point_ensemble_weights(backtest_metrics, config)
    future_forecasts = forecast_portfolio_models(
        portfolio_history[["unique_id", "ds", "y"]],
        config,
        future_covariates=future_covariates,
        point_weights=point_weights,
    )
    canonical_forecasts = build_canonical_forecast_frame(future_forecasts, champions)
    scenarios = build_ifrs9_temporal_scenarios(canonical_forecasts)
    panel_forecasts, panel_status = forecast_panel_bottom_up(panel_history, config)
    hierarchy_eval, hierarchy_status = evaluate_hierarchical_reconciliation(panel_history, config)
    diagnostics = compute_forecastability_diagnostics(
        portfolio_history[["unique_id", "ds", "y"]],
        season_length=int(config.get("season_length", 12)),
        enable_kpss=bool(config.get("diagnostics", {}).get("enable_kpss", True)),
        enable_entropy=bool(config.get("diagnostics", {}).get("enable_entropy", True)),
        enable_variance_ratio=bool(
            config.get("diagnostics", {}).get("enable_variance_ratio", True)
        ),
    )
    interval_benchmark = benchmark_mapie_time_series_intervals(
        portfolio_history[["unique_id", "ds", "y"]],
        confidence_level=0.90,
        evaluation_size=int(
            (config.get("interval_policy", {}) or {}).get(
                "mapie_evaluation_size",
                max(int(config.get("horizon", 12)), 24),
            )
        ),
        macro_context_path=DATA_DIR / "macro_context.json",
        estimator_name=str(
            (config.get("interval_policy", {}) or {}).get("mapie_estimator", "linear")
        ),
    )

    run_tag = resolve_run_tag(infer_run_tag(), require_explicit=True)
    generated_at = datetime.now(tz=UTC).isoformat()
    diagnostics_payload = {
        "schema_version": "2026-03-07.1",
        "generated_at_utc": generated_at,
        "run_tag": run_tag,
        **diagnostics,
    }

    artifacts = {
        "config_path": str(Path(config_path)),
        "time_series_full_path": "data/processed/time_series_full.parquet",
        "time_series_panel_path": "data/processed/time_series_panel.parquet",
        "backtest_predictions_path": str(output_paths["backtest_predictions_path"]),
        "backtest_metrics_path": str(output_paths["backtest_metrics_path"]),
        "forecasts_path": str(output_paths["forecasts_path"]),
        "scenarios_path": str(output_paths["scenarios_path"]),
        "diagnostics_path": str(output_paths["diagnostics_path"]),
        "panel_forecasts_path": str(output_paths["panel_forecasts_path"]),
        "status_path": str(output_paths["status_path"]),
        "forecastability_report_path": str(output_paths["forecastability_report_path"]),
        "forecastability_status_path": str(output_paths["forecastability_status_path"]),
        "hierarchical_eval_path": str(output_paths["hierarchical_eval_path"]),
        "hierarchy_status_path": str(output_paths["hierarchy_status_path"]),
        "interval_eval_path": str(output_paths["interval_eval_path"]),
    }
    status_payload = build_status_payload(
        config=config,
        metrics=backtest_metrics,
        champions=champions,
        diagnostics=diagnostics_payload,
        panel_status=panel_status,
        future_covariates=future_covariates,
        residual_predictions=backtest_predictions,
        artifacts=artifacts,
        run_tag=run_tag,
    )
    status_payload["generated_at_utc"] = generated_at
    status_payload["schema_version"] = "2026-03-07.1"
    status_payload["forecastability_summary"] = forecastability_status
    status_payload["interval_benchmark"] = interval_benchmark
    status_payload["exogenous_contract_available"] = bool(not future_covariates_contract.empty)
    status_payload["exogenous_contract_version"] = "minimal_macro_covariates_v1"
    status_payload["exogenous_active"] = bool(config.get("exogenous", {}).get("enabled", False))
    status_payload["ensemble_weights"] = point_weights
    backtest_methods = (
        list(backtest_metrics["model"].astype(str).unique()) if not backtest_metrics.empty else []
    )
    status_payload["candidate_methods_tested"] = [
        *backtest_methods,
        *list(interval_benchmark.get("candidate_methods_tested", [])),
    ]
    status_payload["rolling_coverage_summary"] = {
        "official_interval_model": champions["interval"]["model"],
        "official_interval_coverage_gap_90": champions["interval"].get("coverage_gap_90"),
        "mapie_best_method": interval_benchmark.get("best_method"),
        "mapie_best_summary": interval_benchmark.get("rolling_coverage_summary", {}),
    }
    best_benchmark_row = next(
        (
            row
            for row in interval_benchmark.get("results", [])
            if str(row.get("method", "")) == str(interval_benchmark.get("best_method", ""))
        ),
        {},
    )
    status_payload["rolling_coverage_by_horizon"] = {
        "official_interval_model": {
            "coverage_gap_90": champions["interval"].get("coverage_gap_90"),
            "avg_interval_width_90": champions["interval"].get("avg_interval_width_90"),
            "winkler_90": champions["interval"].get("winkler_90"),
            "wis_90": champions["interval"].get("wis_90"),
            "pinball_90": champions["interval"].get("pinball_90"),
        },
        "adaptive_best_method": {
            "method": interval_benchmark.get("best_method"),
            "coverage_gap_90": best_benchmark_row.get("coverage_gap_90"),
            "avg_interval_width_90": best_benchmark_row.get("avg_interval_width_90"),
            "winkler_90": best_benchmark_row.get("winkler_90"),
            "wis_90": best_benchmark_row.get("wis_90"),
            "pinball_90": best_benchmark_row.get("pinball_90"),
            "rolling_coverage_summary": best_benchmark_row.get("rolling_coverage_summary", {}),
        },
    }
    status_payload["hierarchy_reconciliation"] = hierarchy_status
    status_payload["interval_policy"] = {
        "eligible_interval_families": list(
            (config.get("interval_policy", {}) or {}).get("eligible_interval_families", [])
        ),
        "max_coverage_gap": float(
            (config.get("interval_policy", {}) or {}).get("max_coverage_gap", 0.03)
        ),
        "max_winkler_90": (config.get("interval_policy", {}) or {}).get("max_winkler_90"),
    }
    status_payload["interval_selector_reason"] = (
        "interval champion selected from eligible families under the governed coverage-gap policy; "
        "adaptive and conformal statistical candidates remain diagnostic unless they satisfy the same thresholds."
    )
    status_payload["adaptive_method_status"] = {
        "best_method": interval_benchmark.get("best_method"),
        "candidate_methods_tested": interval_benchmark.get("candidate_methods_tested", []),
        "promotion_ready": bool(
            interval_benchmark.get("best_method")
            and bool((status_payload.get("interval_champion", {}) or {}).get("promotable", False))
        ),
        "notes": "Adaptive methods remain diagnostic until they beat the official interval policy.",
    }
    status_payload["final_interval_decision"] = {
        "status": (
            "promoted"
            if bool((status_payload.get("interval_champion", {}) or {}).get("promotable", False))
            else "research_only"
        ),
        "official_family": str(
            (config.get("interval_policy", {}) or {}).get("official_family", "statistical")
        ),
        "reason": status_payload["interval_selector_reason"],
    }
    interval_eval = backtest_metrics.loc[
        :,
        [
            col
            for col in [
                "model",
                "family",
                "interval_subfamily",
                "coverage_90",
                "coverage_gap_90",
                "avg_interval_width_90",
                "winkler_90",
                "wis_90",
                "pinball_90",
                "point_eligible",
                "interval_eligible",
            ]
            if col in backtest_metrics.columns
        ],
    ].copy()
    if interval_benchmark.get("results"):
        adaptive_rows = pd.DataFrame(interval_benchmark.get("results", [])).rename(
            columns={"method": "model"}
        )
        adaptive_rows["model"] = (
            adaptive_rows["model"].astype(str).map(lambda value: f"MAPIE_{value.upper()}")
        )
        adaptive_rows["family"] = "adaptive"
        adaptive_rows["interval_subfamily"] = "adaptive"
        adaptive_rows["point_eligible"] = False
        adaptive_rows["interval_eligible"] = True
        interval_eval = pd.concat([interval_eval, adaptive_rows], ignore_index=True, sort=False)

    backtest_predictions.to_parquet(output_paths["backtest_predictions_path"], index=False)
    backtest_predictions.to_parquet(output_paths["cv_stats_path"], index=False)
    backtest_metrics.to_parquet(output_paths["backtest_metrics_path"], index=False)
    canonical_forecasts.to_parquet(output_paths["forecasts_path"], index=False)
    scenarios.to_parquet(output_paths["scenarios_path"], index=False)
    panel_forecasts.to_parquet(output_paths["panel_forecasts_path"], index=False)
    interval_eval.to_parquet(output_paths["interval_eval_path"], index=False)
    if not hierarchy_eval.empty:
        hierarchy_eval.to_parquet(output_paths["hierarchical_eval_path"], index=False)
    _write_json(output_paths["diagnostics_path"], diagnostics_payload)
    _write_json(output_paths["hierarchy_status_path"], hierarchy_status)
    _write_json(output_paths["status_path"], status_payload)
    if output_paths["status_path"] != MODEL_DIR / "time_series_status.json":
        _write_json(output_paths["research_status_path"], status_payload)

    logger.info(
        "Time-series champions: point={} (promotable={}), interval={} (promotable={})",
        champions["point"]["model"],
        champions["point"]["promotable"],
        champions["interval"]["model"],
        champions["interval"]["promotable"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/time_series.yaml")
    parser.add_argument("--horizon", type=int, default=None)
    args = parser.parse_args()
    main(config_path=args.config, horizon=args.horizon)
