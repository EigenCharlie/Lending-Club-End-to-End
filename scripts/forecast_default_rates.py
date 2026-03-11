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
    build_canonical_forecast_frame,
    build_ifrs9_temporal_scenarios,
    build_status_payload,
    compute_forecastability_diagnostics,
    forecast_panel_bottom_up,
    forecast_portfolio_models,
    infer_run_tag,
    load_future_covariates,
    load_time_series_config,
    run_portfolio_backtest,
    select_time_series_champions,
)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Saved {}", path)


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
    logger.info("Loaded time_series_full: {}", portfolio.shape)
    logger.info("Loaded time_series_panel: {}", panel.shape)
    return portfolio, panel


def main(config_path: str = "configs/time_series.yaml", horizon: int | None = None) -> None:
    config = load_time_series_config(config_path)
    if horizon is not None:
        config["horizon"] = int(horizon)
    future_covariates = load_future_covariates(config)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    portfolio_history, panel_history = _load_history()
    backtest_predictions, backtest_metrics = run_portfolio_backtest(
        portfolio_history[["unique_id", "ds", "y"]],
        config,
        future_covariates=future_covariates,
    )
    if backtest_metrics.empty:
        raise RuntimeError("Time-series backtest produced no metrics; cannot continue.")

    champions = select_time_series_champions(backtest_metrics, config)
    future_forecasts = forecast_portfolio_models(
        portfolio_history[["unique_id", "ds", "y"]],
        config,
        future_covariates=future_covariates,
    )
    canonical_forecasts = build_canonical_forecast_frame(future_forecasts, champions)
    scenarios = build_ifrs9_temporal_scenarios(canonical_forecasts)
    panel_forecasts, panel_status = forecast_panel_bottom_up(panel_history, config)
    diagnostics = compute_forecastability_diagnostics(
        portfolio_history[["unique_id", "ds", "y"]],
        season_length=int(config.get("season_length", 12)),
        enable_kpss=bool(config.get("diagnostics", {}).get("enable_kpss", True)),
        enable_entropy=bool(config.get("diagnostics", {}).get("enable_entropy", True)),
        enable_variance_ratio=bool(
            config.get("diagnostics", {}).get("enable_variance_ratio", True)
        ),
    )

    run_tag = infer_run_tag()
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
        "backtest_predictions_path": "data/processed/ts_backtest_predictions.parquet",
        "backtest_metrics_path": "data/processed/ts_backtest_metrics.parquet",
        "forecasts_path": "data/processed/ts_forecasts.parquet",
        "scenarios_path": "data/processed/ts_ifrs9_scenarios.parquet",
        "diagnostics_path": "data/processed/ts_diagnostics.json",
        "panel_forecasts_path": "data/processed/ts_panel_forecasts.parquet",
        "status_path": "models/time_series_status.json",
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

    backtest_predictions.to_parquet(DATA_DIR / "ts_backtest_predictions.parquet", index=False)
    backtest_predictions.to_parquet(DATA_DIR / "ts_cv_stats.parquet", index=False)
    backtest_metrics.to_parquet(DATA_DIR / "ts_backtest_metrics.parquet", index=False)
    canonical_forecasts.to_parquet(DATA_DIR / "ts_forecasts.parquet", index=False)
    scenarios.to_parquet(DATA_DIR / "ts_ifrs9_scenarios.parquet", index=False)
    panel_forecasts.to_parquet(DATA_DIR / "ts_panel_forecasts.parquet", index=False)
    _write_json(DATA_DIR / "ts_diagnostics.json", diagnostics_payload)
    _write_json(MODEL_DIR / "time_series_status.json", status_payload)

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
