"""Forecast monthly default rates with conformal intervals.

Usage: uv run python scripts/forecast_default_rates.py --horizon 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from src.models.time_series import train_baseline_forecasters, train_ml_forecaster


def main(horizon: int = 12):
    ts = pd.read_parquet("data/processed/time_series.parquet")
    logger.info(f"Loaded time series: {ts.shape}")

    sf_model, baseline_forecasts = train_baseline_forecasters(ts, horizon=horizon)
    logger.info(f"Baseline forecasts head:\n{baseline_forecasts.head()}")

    mlf_model, ml_forecasts = train_ml_forecaster(ts, horizon=horizon)
    logger.info(f"ML forecasts with CP head:\n{ml_forecasts.head()}")

    # Join baseline + ML forecasts on keys for a single output artifact.
    merged = baseline_forecasts.merge(
        ml_forecasts,
        on=["unique_id", "ds"],
        how="outer",
        suffixes=("", "_ml"),
    )

    out_path = Path("data/processed/ts_forecasts.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    logger.info(f"Forecasts saved to {out_path} ({merged.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()
    main(args.horizon)
