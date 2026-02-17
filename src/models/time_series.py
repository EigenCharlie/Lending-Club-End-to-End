"""Time series forecasting with conformal prediction intervals.

Uses Nixtla ecosystem: statsforecast 2.x (baselines), mlforecast 1.x (ML + CP).
Hierarchical reconciliation for grade-to-portfolio forecasts.

API notes (2025-2026 versions):
  - statsforecast 2.0: df not in constructor; passed to .fit()/.forecast()
  - mlforecast 1.0: class-based lag transforms (RollingMean, not tuples)
  - hierarchicalforecast 1.4: reconcile() requires tags param; S renamed to S_df
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


def _require_ts_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return canonical frame with the columns Nixtla always requires.

    Restricting to [unique_id, ds, y] prevents accidental exogenous handling
    that would otherwise require passing future `X_df` at prediction time.
    """
    required = ["unique_id", "ds", "y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required time-series columns: {missing}")
    return df[required].copy()


def train_baseline_forecasters(
    df: pd.DataFrame,
    horizon: int = 12,
    freq: str = "MS",
    levels: list[int] | None = None,
) -> tuple[Any, pd.DataFrame]:
    """Train statistical baselines: AutoARIMA, AutoETS, AutoTheta, SeasonalNaive.

    Args:
        df: DataFrame with columns [unique_id, ds, y].
        horizon: Forecast horizon in periods.
        freq: Frequency string (MS = month start).
        levels: Confidence levels for intervals (e.g., [90, 95]).

    Returns:
        Tuple of (fitted StatsForecast object, forecasts DataFrame).
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, SeasonalNaive

    if levels is None:
        levels = [90, 95]

    df_model = _require_ts_columns(df)
    models = [
        AutoARIMA(season_length=12),
        AutoETS(season_length=12),
        AutoTheta(season_length=12),
        SeasonalNaive(season_length=12),
    ]
    sf = StatsForecast(models=models, freq=freq, n_jobs=1)
    sf.fit(df_model)
    forecasts = sf.predict(h=horizon, level=levels)

    logger.info(
        f"Baseline forecasts: {forecasts.shape}, horizon={horizon}, "
        f"input_cols={list(df_model.columns)}"
    )
    return sf, forecasts


def train_ml_forecaster(
    df: pd.DataFrame,
    horizon: int = 12,
    freq: str = "MS",
    lags: list[int] | None = None,
    n_windows: int = 5,
    levels: list[int] | None = None,
) -> tuple[Any, pd.DataFrame]:
    """Train ML forecaster (LightGBM) with conformal prediction intervals.

    Args:
        df: DataFrame with columns [unique_id, ds, y].
        horizon: Forecast horizon.
        freq: Frequency string.
        lags: Lag features to create.
        n_windows: Backtesting windows for conformal calibration (5-6 for ~120 obs).
        levels: Confidence levels for intervals.

    Returns:
        Tuple of (fitted MLForecast, forecasts_df).
    """
    from lightgbm import LGBMRegressor
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import (
        ExponentiallyWeightedMean,
        RollingMean,
        RollingStd,
    )
    from mlforecast.utils import PredictionIntervals

    if lags is None:
        lags = [1, 2, 3, 6, 12]
    if levels is None:
        levels = [90, 95]

    df_model = _require_ts_columns(df)
    mlf = MLForecast(
        models={"lgbm": LGBMRegressor(n_estimators=100, learning_rate=0.1, verbose=-1)},
        freq=freq,
        lags=lags,
        lag_transforms={
            1: [
                RollingMean(window_size=3),
                RollingMean(window_size=6),
                RollingMean(window_size=12),
            ],
            3: [RollingStd(window_size=6)],
            6: [ExponentiallyWeightedMean(alpha=0.3)],
        },
        date_features=["month"],
    )

    mlf.fit(
        df_model,
        prediction_intervals=PredictionIntervals(n_windows=n_windows, h=horizon),
    )

    forecasts = mlf.predict(h=horizon, level=levels)
    logger.info(f"ML forecasts with CP intervals: {forecasts.shape}")
    return mlf, forecasts


def reconcile_hierarchical(
    forecasts: pd.DataFrame,
    S_df: pd.DataFrame,
    tags: dict[str, np.ndarray],
    method: str = "MinTrace",
    Y_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Reconcile hierarchical forecasts (grade -> portfolio).

    Args:
        forecasts: Base forecasts DataFrame.
        S_df: Summation matrix DataFrame.
        tags: Dict mapping level name -> array of unique_ids at that level.
        method: Reconciliation method (MinTrace, BottomUp).
        Y_df: Training data (required for insample methods like mint_shrink).

    Returns:
        Reconciled forecasts.
    """
    from hierarchicalforecast.core import HierarchicalReconciliation
    from hierarchicalforecast.methods import BottomUp, MinTrace

    reconcilers = {"MinTrace": MinTrace(method="ols"), "BottomUp": BottomUp()}
    reconciler = reconcilers.get(method, MinTrace(method="ols"))

    hrec = HierarchicalReconciliation(reconcilers=[reconciler])
    reconciled = hrec.reconcile(
        Y_hat_df=forecasts,
        tags=tags,
        S_df=S_df,
        Y_df=Y_df,
    )

    logger.info(f"Hierarchical reconciliation ({method}): {reconciled.shape}")
    return reconciled
