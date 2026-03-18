"""Governed time-series forecasting utilities for monthly default-rate planning."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy import stats

from src.evaluation.backtesting import ks_two_sample_test, winkler_interval_score
from src.evaluation.metrics import forecast_backtest_metrics
from src.utils.artifact_metadata import resolve_run_tag

DEFAULT_CONFIG_PATH = Path("configs/time_series.yaml")


def _require_ts_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["unique_id", "ds", "y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required time-series columns: {missing}")
    out = df[required].copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["ds", "y"]).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return out


def _build_catboost_regressor(
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    loss_function: str = "RMSE",
    random_seed: int = 42,
) -> Any:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function=loss_function,
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
    )


def load_time_series_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    cfg_path = Path(path)
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    payload.setdefault("horizon", 12)
    payload.setdefault("freq", "MS")
    payload.setdefault("season_length", 12)
    payload.setdefault("rolling_origin", {})
    payload["rolling_origin"].setdefault("min_train_periods", 72)
    payload["rolling_origin"].setdefault("step_months", 1)
    payload["rolling_origin"].setdefault("embargo_periods", payload["horizon"])
    payload["rolling_origin"].setdefault("max_windows", 36)
    payload.setdefault("point_champion", {})
    payload["point_champion"].setdefault("primary_metric", "mase")
    payload["point_champion"].setdefault("tiebreakers", ["rmsse", "abs_bias"])
    payload["point_champion"].setdefault("must_beat_seasonal_naive", True)
    payload["point_champion"].setdefault("max_rmsse_vs_seasonal_naive", 1.0)
    payload.setdefault("interval_policy", {})
    payload["interval_policy"].setdefault("target_level", 90)
    payload["interval_policy"].setdefault("max_coverage_gap", 0.03)
    payload["interval_policy"].setdefault("tiebreakers", ["winkler_90", "avg_interval_width_90"])
    payload["interval_policy"].setdefault("official_family", "statistical")
    payload["interval_policy"].setdefault(
        "eligible_interval_families",
        [str(payload["interval_policy"].get("official_family", "statistical"))],
    )
    payload["interval_policy"].setdefault("ml_intervals_diagnostic_only_if_unpromoted", True)
    payload["interval_policy"].setdefault("include_conformal_statistical", False)
    payload["interval_policy"].setdefault("conformal_n_windows", 3)
    payload["interval_policy"].setdefault("max_winkler_90", None)
    payload["interval_policy"].setdefault("mapie_estimator", "linear")
    payload["interval_policy"].setdefault("mapie_evaluation_size", max(int(payload["horizon"]), 24))
    payload.setdefault("exogenous", {})
    payload["exogenous"].setdefault("enabled", False)
    payload["exogenous"].setdefault(
        "future_covariates_path", "data/processed/ts_future_covariates.parquet"
    )
    payload["exogenous"].setdefault("required_columns", ["ds"])
    payload.setdefault("models", {})
    payload["models"].setdefault(
        "statistical",
        ["SeasonalNaive", "AutoARIMA", "AutoETS", "AutoTheta"],
    )
    payload["models"].setdefault("challengers", ["SARIMAX", "STL_CatBoost"])
    payload["models"].setdefault("panel_global", "GlobalCatBoostPanel")
    payload.setdefault("panel", {})
    payload["panel"].setdefault("bottom_level", "grade_term")
    payload.setdefault("diagnostics", {})
    payload["diagnostics"].setdefault("enable_kpss", True)
    payload["diagnostics"].setdefault("enable_entropy", True)
    payload["diagnostics"].setdefault("enable_variance_ratio", True)
    payload["diagnostics"].setdefault("enable_residual_drift", True)
    payload.setdefault("forecastability", {})
    payload["forecastability"].setdefault("enabled", True)
    payload["forecastability"].setdefault("intermittent_adi_threshold", 1.32)
    payload["forecastability"].setdefault("intermittent_cv2_threshold", 0.49)
    payload["forecastability"].setdefault(
        "routing_levels",
        ["portfolio", "grade", "grade_term"],
    )
    payload.setdefault("interval_metrics", {})
    payload["interval_metrics"].setdefault("include_wis", True)
    payload["interval_metrics"].setdefault("include_pinball", True)
    payload.setdefault("ensemble", {})
    payload["ensemble"].setdefault("enabled", False)
    payload["ensemble"].setdefault(
        "candidates",
        ["AutoARIMA", "AutoETS", "SARIMAX", "STL_CatBoost"],
    )
    payload["ensemble"].setdefault("weight_metric", "mase")
    payload["ensemble"].setdefault("model_name", "WeightedEnsemble")
    payload.setdefault("hierarchy_reconciliation", {})
    payload["hierarchy_reconciliation"].setdefault("enabled", False)
    payload["hierarchy_reconciliation"].setdefault("methods", ["BottomUp", "TopDown", "MinTrace"])
    payload["hierarchy_reconciliation"].setdefault(
        "target_columns", ["default_count", "loan_count"]
    )
    payload["hierarchy_reconciliation"].setdefault("evaluation_horizon", payload["horizon"])
    payload.setdefault("outputs", {})
    payload["outputs"].setdefault(
        "backtest_predictions_path", "data/processed/ts_backtest_predictions.parquet"
    )
    payload["outputs"].setdefault(
        "backtest_metrics_path", "data/processed/ts_backtest_metrics.parquet"
    )
    payload["outputs"].setdefault("cv_stats_path", "data/processed/ts_cv_stats.parquet")
    payload["outputs"].setdefault("forecasts_path", "data/processed/ts_forecasts.parquet")
    payload["outputs"].setdefault("scenarios_path", "data/processed/ts_ifrs9_scenarios.parquet")
    payload["outputs"].setdefault("diagnostics_path", "data/processed/ts_diagnostics.json")
    payload["outputs"].setdefault(
        "panel_forecasts_path", "data/processed/ts_panel_forecasts.parquet"
    )
    payload["outputs"].setdefault("status_path", "models/time_series_status.json")
    payload["outputs"].setdefault(
        "forecastability_report_path", "data/processed/ts_forecastability_report.parquet"
    )
    payload["outputs"].setdefault(
        "forecastability_status_path", "models/time_series_forecastability_status.json"
    )
    payload["outputs"].setdefault("interval_eval_path", "data/processed/ts_interval_eval.parquet")
    payload["outputs"].setdefault(
        "hierarchical_eval_path", "data/processed/ts_hierarchical_eval.parquet"
    )
    payload["outputs"].setdefault(
        "hierarchy_status_path", "models/time_series_hierarchy_status.json"
    )
    payload["outputs"].setdefault("research_status_path", "models/time_series_research_status.json")
    payload.setdefault("research_backlog", ["ACI", "EnbPI", "OnlineConformal"])
    return payload


def load_future_covariates(config: dict[str, Any]) -> pd.DataFrame:
    exog_cfg = config.get("exogenous", {})
    if not bool(exog_cfg.get("enabled", False)):
        return pd.DataFrame()
    path = Path(str(exog_cfg.get("future_covariates_path", "")).strip())
    if not path.exists():
        raise FileNotFoundError(
            f"Exogenous forecasting is enabled but future covariates are missing: {path}"
        )
    cov = pd.read_parquet(path)
    required_cols = list(exog_cfg.get("required_columns", ["ds"]))
    missing = [col for col in required_cols if col not in cov.columns]
    if missing:
        raise ValueError(f"Future covariates missing required columns: {missing}")
    cov = cov.copy()
    cov["ds"] = pd.to_datetime(cov["ds"], errors="coerce")
    cov = cov.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    return cov


def build_backtest_cutoffs(
    ds: Iterable[pd.Timestamp],
    *,
    horizon: int,
    min_train_periods: int,
    step_months: int,
    max_windows: int | None = None,
) -> list[pd.Timestamp]:
    ds_index = pd.Index(pd.to_datetime(list(ds), errors="coerce")).dropna().sort_values().unique()
    cutoffs: list[pd.Timestamp] = []
    max_start = len(ds_index) - horizon
    if max_start <= min_train_periods:
        return cutoffs
    for idx in range(min_train_periods - 1, max_start, max(int(step_months), 1)):
        cutoffs.append(pd.Timestamp(ds_index[idx]))
    if max_windows and len(cutoffs) > int(max_windows):
        cutoffs = cutoffs[-int(max_windows) :]
    return cutoffs


def _seasonal_error_scales(y: np.ndarray, season_length: int) -> tuple[float, float]:
    series = np.asarray(y, dtype=float)
    period = int(max(season_length, 1))
    diffs = np.diff(series) if len(series) <= period else series[period:] - series[:-period]
    if diffs.size == 0:
        diffs = np.diff(series)
    mae_scale = float(np.mean(np.abs(diffs))) if diffs.size else 1.0
    rmse_scale = float(np.sqrt(np.mean(np.square(diffs)))) if diffs.size else 1.0
    return max(mae_scale, 1e-8), max(rmse_scale, 1e-8)


def _extract_interval_columns(
    frame: pd.DataFrame,
    model_name: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    lo90 = (
        frame[f"{model_name}-lo-90"].to_numpy(dtype=float)
        if f"{model_name}-lo-90" in frame.columns
        else None
    )
    hi90 = (
        frame[f"{model_name}-hi-90"].to_numpy(dtype=float)
        if f"{model_name}-hi-90" in frame.columns
        else None
    )
    lo95 = (
        frame[f"{model_name}-lo-95"].to_numpy(dtype=float)
        if f"{model_name}-lo-95" in frame.columns
        else None
    )
    hi95 = (
        frame[f"{model_name}-hi-95"].to_numpy(dtype=float)
        if f"{model_name}-hi-95" in frame.columns
        else None
    )
    return lo90, hi90, lo95, hi95


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _pinball_loss(y_true: np.ndarray, y_quantile: np.ndarray, tau: float) -> float:
    truth = np.asarray(y_true, dtype=float)
    quantile = np.asarray(y_quantile, dtype=float)
    error = truth - quantile
    return float(np.mean(np.maximum(tau * error, (tau - 1.0) * error)))


def _weighted_interval_score(
    y_true: np.ndarray,
    point_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    alpha: float,
) -> float:
    interval_score = winkler_interval_score(
        np.asarray(y_true, dtype=float),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        alpha=alpha,
    )
    median_component = np.abs(np.asarray(y_true, dtype=float) - np.asarray(point_pred, dtype=float))
    wis = (0.5 * median_component) + ((alpha / 2.0) * interval_score)
    return float(np.mean(wis) / max(0.5 + (alpha / 2.0), 1e-12))


def _adi_cv2(values: np.ndarray) -> tuple[float | None, float | None]:
    series = np.asarray(values, dtype=float)
    non_zero_idx = np.flatnonzero(series > 0)
    if non_zero_idx.size == 0:
        return None, None
    adi = float(len(series)) if non_zero_idx.size == 1 else float(np.mean(np.diff(non_zero_idx)))
    positive = series[series > 0]
    mean_positive = float(np.mean(positive))
    if mean_positive <= 0:
        return adi, None
    cv2 = float((np.std(positive, ddof=0) / max(mean_positive, 1e-12)) ** 2)
    return adi, cv2


def _classify_intermittency(adi: float | None, cv2: float | None) -> str:
    if adi is None or cv2 is None:
        return "insufficient_signal"
    if adi < 1.32 and cv2 < 0.49:
        return "smooth"
    if adi >= 1.32 and cv2 < 0.49:
        return "intermittent"
    if adi < 1.32 and cv2 >= 0.49:
        return "erratic"
    return "lumpy"


def _route_time_series_candidate(
    *,
    series_level: str,
    intermittency_class: str,
    spectral_entropy: float | None,
    residual_drift_pvalue: float | None,
) -> str:
    if series_level == "grade_term" and intermittency_class in {
        "intermittent",
        "erratic",
        "lumpy",
    }:
        return "intermittent_counts"
    if (
        spectral_entropy is not None
        and spectral_entropy >= 0.75
        or (residual_drift_pvalue is not None and residual_drift_pvalue < 0.05)
    ):
        return "exogenous_challenger"
    return "structured_statistical"


def _normal_interval(
    point: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = np.sqrt(np.arange(1, len(point) + 1, dtype=float))
    z90 = 1.6448536269514722
    z95 = 1.959963984540054
    lo90 = point - z90 * sigma * scale
    hi90 = point + z90 * sigma * scale
    lo95 = point - z95 * sigma * scale
    hi95 = point + z95 * sigma * scale
    return lo90, hi90, lo95, hi95


def _conf_int_bounds(conf_int: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract lower/upper bounds from statsmodels conf_int output."""
    if hasattr(conf_int, "iloc"):
        lower = np.asarray(conf_int.iloc[:, 0], dtype=float)
        upper = np.asarray(conf_int.iloc[:, 1], dtype=float)
        return lower, upper
    arr = np.asarray(conf_int, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Unexpected conf_int shape: {arr.shape}")
    return np.asarray(arr[:, 0], dtype=float), np.asarray(arr[:, 1], dtype=float)


def _statsforecast_model_instances(model_names: list[str], season_length: int = 12) -> list[Any]:
    from statsforecast.models import (
        ADIDA,
        IMAPA,
        TSB,
        AutoARIMA,
        AutoETS,
        AutoTheta,
        CrostonSBA,
        SeasonalNaive,
    )

    registry = {
        "AutoARIMA": AutoARIMA(season_length=season_length),
        "AutoETS": AutoETS(season_length=season_length),
        "AutoTheta": AutoTheta(season_length=season_length),
        "SeasonalNaive": SeasonalNaive(season_length=season_length),
        "ADIDA": ADIDA(),
        "CrostonSBA": CrostonSBA(),
        "IMAPA": IMAPA(),
        "TSB": TSB(alpha_d=0.2, alpha_p=0.2),
    }
    return [registry[name] for name in model_names if name in registry]


def _rename_statsforecast_columns(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if not suffix:
        return frame
    renamed: dict[str, str] = {}
    protected = {"unique_id", "ds"}
    for col in frame.columns:
        if col in protected:
            continue
        if "-lo-" in col or "-hi-" in col:
            base, tail = col.split("-", 1)
            renamed[col] = f"{base}{suffix}-{tail}"
        else:
            renamed[col] = f"{col}{suffix}"
    return frame.rename(columns=renamed)


def train_baseline_forecasters(
    df: pd.DataFrame,
    horizon: int = 12,
    freq: str = "MS",
    levels: list[int] | None = None,
    model_names: list[str] | None = None,
    conformal_windows: int | None = None,
    model_suffix: str = "",
) -> tuple[Any, pd.DataFrame]:
    from statsforecast import StatsForecast
    from statsforecast.utils import ConformalIntervals

    if levels is None:
        levels = [90, 95]
    if model_names is None:
        model_names = ["AutoARIMA", "AutoETS", "AutoTheta", "SeasonalNaive"]

    df_model = _require_ts_columns(df)
    models = _statsforecast_model_instances(model_names, season_length=12)
    sf = StatsForecast(models=models, freq=freq, n_jobs=1)
    prediction_intervals = (
        ConformalIntervals(n_windows=max(int(conformal_windows), 2), h=horizon)
        if conformal_windows is not None and int(conformal_windows) >= 2
        else None
    )
    forecasts = sf.forecast(
        h=horizon, df=df_model, level=levels, prediction_intervals=prediction_intervals
    )
    forecasts = _rename_statsforecast_columns(forecasts, model_suffix)
    return sf, forecasts


def train_ml_forecaster(
    df: pd.DataFrame,
    horizon: int = 12,
    freq: str = "MS",
    lags: list[int] | None = None,
    n_windows: int = 5,
    levels: list[int] | None = None,
) -> tuple[Any, pd.DataFrame]:
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import ExponentiallyWeightedMean, RollingMean, RollingStd
    from mlforecast.utils import PredictionIntervals

    if lags is None:
        lags = [1, 2, 3, 6, 12]
    if levels is None:
        levels = [90, 95]

    df_model = _require_ts_columns(df)
    mlf = MLForecast(
        models={
            "CatBoost": _build_catboost_regressor(
                iterations=150,
                learning_rate=0.05,
                depth=6,
            )
        },
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
    return mlf, forecasts


def fit_sarimax_forecaster(
    train_df: pd.DataFrame,
    horizon: int,
    *,
    exog_train: pd.DataFrame | None = None,
    exog_future: pd.DataFrame | None = None,
    season_length: int = 12,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] | None = None,
) -> pd.DataFrame:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    ts_df = _require_ts_columns(train_df)
    if seasonal_order is None:
        seasonal_order = (1, 1, 1, season_length)

    y = ts_df["y"].to_numpy(dtype=float)
    model = SARIMAX(
        y,
        exog=exog_train.to_numpy(dtype=float)
        if exog_train is not None and not exog_train.empty
        else None,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    forecast_res = result.get_forecast(
        steps=horizon,
        exog=exog_future.to_numpy(dtype=float)
        if exog_future is not None and not exog_future.empty
        else None,
    )
    conf90 = forecast_res.conf_int(alpha=0.10)
    conf95 = forecast_res.conf_int(alpha=0.05)
    lo90, hi90 = _conf_int_bounds(conf90)
    lo95, hi95 = _conf_int_bounds(conf95)
    future_ds = pd.date_range(
        ts_df["ds"].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    frame = pd.DataFrame(
        {
            "unique_id": ts_df["unique_id"].iloc[0],
            "ds": future_ds,
            "SARIMAX": np.asarray(forecast_res.predicted_mean, dtype=float),
            "SARIMAX-lo-90": lo90,
            "SARIMAX-hi-90": hi90,
            "SARIMAX-lo-95": lo95,
            "SARIMAX-hi-95": hi95,
        }
    )
    return frame


def fit_stl_catboost_forecaster(
    train_df: pd.DataFrame,
    horizon: int,
    *,
    season_length: int = 12,
    exog_train: pd.DataFrame | None = None,
    exog_future: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from statsmodels.tsa.seasonal import STL

    ts_df = _require_ts_columns(train_df)
    y = ts_df["y"].to_numpy(dtype=float)
    stl = STL(y, period=season_length, robust=True)
    fitted = stl.fit()

    trend = np.asarray(fitted.trend, dtype=float)
    seasonal = np.asarray(fitted.seasonal, dtype=float)
    resid = y - trend - seasonal

    idx = np.arange(len(trend), dtype=float)
    slope, intercept = np.polyfit(idx, trend, deg=1)
    future_idx = np.arange(len(trend), len(trend) + horizon, dtype=float)
    trend_fc = intercept + slope * future_idx

    seasonal_cycle = (
        seasonal[-season_length:]
        if len(seasonal) >= season_length
        else np.resize(seasonal, season_length)
    )
    seasonal_fc = np.resize(seasonal_cycle, horizon)

    residual_fc = np.zeros(horizon, dtype=float)
    residual_sigma = (
        float(np.std(resid[-min(len(resid), season_length * 2) :], ddof=1))
        if len(resid) > 1
        else 0.01
    )
    residual_sigma = max(residual_sigma, 0.005)
    try:
        from mlforecast import MLForecast
        from mlforecast.lag_transforms import ExponentiallyWeightedMean, RollingMean, RollingStd

        resid_df = pd.DataFrame({"unique_id": ts_df["unique_id"], "ds": ts_df["ds"], "y": resid})
        exog_cols: list[str] = []
        if exog_train is not None and not exog_train.empty:
            exog_train_frame = exog_train.copy()
            exog_train_frame = (
                exog_train_frame.reset_index()
                if "ds" not in exog_train_frame.columns
                else exog_train_frame
            )
            if "ds" not in exog_train_frame.columns and "index" in exog_train_frame.columns:
                exog_train_frame = exog_train_frame.rename(columns={"index": "ds"})
            exog_train_frame["ds"] = pd.to_datetime(exog_train_frame["ds"], errors="coerce")
            exog_cols = [col for col in exog_train_frame.columns if col not in {"ds", "month"}]
            resid_df = resid_df.merge(exog_train_frame[["ds", *exog_cols]], on="ds", how="left")
        mlf = MLForecast(
            models={
                "resid": _build_catboost_regressor(
                    iterations=120,
                    learning_rate=0.05,
                    depth=4,
                )
            },
            freq="MS",
            lags=[1, 2, 3, 6, 12],
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
        mlf.fit(resid_df, static_features=[])
        future_exog = None
        if exog_future is not None and not exog_future.empty:
            future_exog = exog_future.copy()
            future_exog = (
                future_exog.reset_index() if "ds" not in future_exog.columns else future_exog
            )
            if "ds" not in future_exog.columns and "index" in future_exog.columns:
                future_exog = future_exog.rename(columns={"index": "ds"})
            future_exog["unique_id"] = ts_df["unique_id"].iloc[0]
            keep_cols = [
                "unique_id",
                "ds",
                *[col for col in future_exog.columns if col not in {"unique_id", "ds", "month"}],
            ]
            future_exog = future_exog[keep_cols]
        resid_pred = mlf.predict(h=horizon, X_df=future_exog)
        residual_fc = resid_pred["resid"].to_numpy(dtype=float)
        residual_sigma = max(float(np.std(resid - np.mean(resid))), residual_sigma)
    except Exception as exc:
        logger.warning("STL_CatBoost residual learner fallback activated: {}", exc)
        tail = resid[-season_length:] if len(resid) >= season_length else resid
        if tail.size:
            residual_fc = np.resize(tail, horizon)

    point = np.clip(trend_fc + seasonal_fc + residual_fc, 0.0, 1.0)
    lo90, hi90, lo95, hi95 = _normal_interval(point, residual_sigma)
    future_ds = pd.date_range(
        ts_df["ds"].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    return pd.DataFrame(
        {
            "unique_id": ts_df["unique_id"].iloc[0],
            "ds": future_ds,
            "STL_CatBoost": point,
            "STL_CatBoost-lo-90": np.clip(lo90, 0.0, 1.0),
            "STL_CatBoost-hi-90": np.clip(hi90, 0.0, 1.0),
            "STL_CatBoost-lo-95": np.clip(lo95, 0.0, 1.0),
            "STL_CatBoost-hi-95": np.clip(hi95, 0.0, 1.0),
        }
    )


def _window_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lo_90: np.ndarray | None,
    y_hi_90: np.ndarray | None,
    y_lo_95: np.ndarray | None,
    y_hi_95: np.ndarray | None,
    season_mae_scale: float,
    season_rmse_scale: float,
) -> dict[str, float]:
    metrics = forecast_backtest_metrics(
        forecast_values=y_pred,
        actual_values=y_true,
        forecast_lo=y_lo_90,
        forecast_hi=y_hi_90,
    )
    metrics["mase"] = float(np.mean(np.abs(y_true - y_pred) / season_mae_scale))
    metrics["rmsse"] = float(np.sqrt(np.mean(np.square(y_true - y_pred))) / season_rmse_scale)
    metrics["abs_bias"] = float(abs(metrics.get("mean_bias", 0.0)))
    if y_lo_90 is not None and y_hi_90 is not None:
        metrics["coverage_90"] = float(((y_true >= y_lo_90) & (y_true <= y_hi_90)).mean())
        metrics["avg_interval_width_90"] = float((y_hi_90 - y_lo_90).mean())
        metrics["winkler_90"] = float(
            np.mean(winkler_interval_score(y_true, y_lo_90, y_hi_90, alpha=0.10))
        )
        metrics["pinball_90"] = float(
            (_pinball_loss(y_true, y_lo_90, 0.05) + _pinball_loss(y_true, y_hi_90, 0.95)) / 2.0
        )
        metrics["wis_90"] = float(
            _weighted_interval_score(y_true, y_pred, y_lo_90, y_hi_90, alpha=0.10)
        )
    if y_lo_95 is not None and y_hi_95 is not None:
        metrics["coverage_95"] = float(((y_true >= y_lo_95) & (y_true <= y_hi_95)).mean())
        metrics["avg_interval_width_95"] = float((y_hi_95 - y_lo_95).mean())
        metrics["winkler_95"] = float(
            np.mean(winkler_interval_score(y_true, y_lo_95, y_hi_95, alpha=0.05))
        )
    return metrics


def diebold_mariano_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    lag: int = 1,
) -> dict[str, float | bool]:
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    n = int(d.size)
    if n < 3:
        return {"dm_stat": 0.0, "p_value": 1.0, "reject": False, "n": float(n)}

    d_mean = float(np.mean(d))
    gamma0 = float(np.var(d, ddof=1))
    var_hat = gamma0
    max_lag = min(max(int(lag), 1), n - 1)
    for k in range(1, max_lag + 1):
        cov = float(np.cov(d[:-k], d[k:], ddof=1)[0, 1])
        weight = 1.0 - (k / (max_lag + 1))
        var_hat += 2.0 * weight * cov
    var_hat = max(var_hat / n, 1e-12)
    stat = d_mean / math.sqrt(var_hat)
    p_value = float(2.0 * stats.t.sf(abs(stat), df=max(n - 1, 1)))
    return {
        "dm_stat": float(stat),
        "p_value": p_value,
        "reject": bool(p_value < 0.05),
        "n": float(n),
    }


def compute_forecastability_diagnostics(
    history: pd.DataFrame,
    *,
    season_length: int = 12,
    enable_kpss: bool = True,
    enable_entropy: bool = True,
    enable_variance_ratio: bool = True,
) -> dict[str, Any]:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

    series = _require_ts_columns(history)
    y = series["y"].to_numpy(dtype=float)
    diagnostics: dict[str, Any] = {
        "n_periods": int(len(series)),
        "series_start": str(series["ds"].min().date()),
        "series_end": str(series["ds"].max().date()),
        "recent_actual_mean_12m": float(series["y"].tail(12).mean()),
    }

    try:
        adf_res = adfuller(y, autolag="AIC")
        diagnostics["adf"] = {
            "statistic": float(adf_res[0]),
            "p_value": float(adf_res[1]),
            "lags_used": int(adf_res[2]),
        }
    except Exception as exc:
        diagnostics["adf"] = {"error": str(exc)}

    if enable_kpss:
        try:
            kpss_res = kpss(y, nlags="auto")
            diagnostics["kpss"] = {
                "statistic": float(kpss_res[0]),
                "p_value": float(kpss_res[1]),
                "lags_used": int(kpss_res[2]),
            }
        except Exception as exc:
            diagnostics["kpss"] = {"error": str(exc)}

    try:
        diagnostics["acf"] = {
            f"lag_{lag}": float(val)
            for lag, val in enumerate(acf(y, nlags=min(12, len(y) - 1), fft=False))
            if lag > 0
        }
    except Exception as exc:
        diagnostics["acf"] = {"error": str(exc)}

    try:
        pacf_vals = pacf(y, nlags=min(12, max(len(y) // 2 - 1, 1)))
        diagnostics["pacf"] = {
            f"lag_{lag}": float(val) for lag, val in enumerate(pacf_vals) if lag > 0
        }
    except Exception as exc:
        diagnostics["pacf"] = {"error": str(exc)}

    try:
        stl = STL(y, period=season_length, robust=True).fit()
        remainder = np.asarray(stl.resid, dtype=float)
        seasonal = np.asarray(stl.seasonal, dtype=float)
        trend = np.asarray(stl.trend, dtype=float)
        diagnostics["stl"] = {
            "seasonal_strength": float(
                max(0.0, 1.0 - (np.var(remainder) / max(np.var(remainder + seasonal), 1e-12)))
            ),
            "trend_strength": float(
                max(0.0, 1.0 - (np.var(remainder) / max(np.var(remainder + trend), 1e-12)))
            ),
        }
    except Exception as exc:
        diagnostics["stl"] = {"error": str(exc)}

    if enable_variance_ratio:
        diff1 = np.diff(y, n=1)
        k = min(season_length, max(len(y) // 4, 2))
        diffk = y[k:] - y[:-k] if len(y) > k else np.array([])
        if diff1.size and diffk.size:
            vr = float(np.var(diffk, ddof=1) / max(k * np.var(diff1, ddof=1), 1e-12))
            diagnostics["variance_ratio"] = {"k": int(k), "value": vr}

    if enable_entropy:
        diagnostics["spectral_entropy"] = float(_spectral_entropy(y))
        diagnostics["permutation_entropy"] = float(_permutation_entropy(y, order=3, delay=1))

    return diagnostics


def compute_forecastability_report(
    panel_history: pd.DataFrame,
    *,
    season_length: int = 12,
    forecastability_cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = forecastability_cfg or {}
    allowed_levels = {
        str(level) for level in cfg.get("routing_levels", ["portfolio", "grade", "grade_term"])
    }
    adi_threshold = float(cfg.get("intermittent_adi_threshold", 1.32))
    cv2_threshold = float(cfg.get("intermittent_cv2_threshold", 0.49))

    if panel_history.empty:
        empty = pd.DataFrame()
        return empty, {
            "available": False,
            "reason": "empty_panel_history",
            "series_evaluated": 0,
        }

    rows: list[dict[str, Any]] = []
    work = panel_history.copy()
    work["ds"] = pd.to_datetime(work["ds"], errors="coerce")
    work = work.dropna(subset=["ds"]).sort_values(["series_level", "unique_id", "ds"])
    for (series_level, unique_id), group in work.groupby(["series_level", "unique_id"], sort=True):
        if str(series_level) not in allowed_levels:
            continue
        rate_frame = group.loc[:, ["unique_id", "ds", "default_rate"]].rename(
            columns={"default_rate": "y"}
        )
        rate_diag = compute_forecastability_diagnostics(
            rate_frame,
            season_length=season_length,
            enable_kpss=True,
            enable_entropy=True,
            enable_variance_ratio=True,
        )
        count_diag: dict[str, Any] = {}
        adi = None
        cv2 = None
        intermittency_class = "not_applicable"
        if "default_count" in group.columns:
            counts = group["default_count"].to_numpy(dtype=float)
            adi, cv2 = _adi_cv2(counts)
            intermittency_class = _classify_intermittency(adi, cv2)
            count_diag = {
                "default_count_mean": float(np.mean(counts)),
                "default_count_std": float(np.std(counts, ddof=0)),
            }
        residual_drift = rate_diag.get("residual_drift", {}) if isinstance(rate_diag, dict) else {}
        residual_drift_pvalue = None
        if isinstance(residual_drift, dict):
            residual_drift_pvalue = _safe_float(residual_drift.get("p_value"))
        spectral_entropy = _safe_float(rate_diag.get("spectral_entropy"))
        route = _route_time_series_candidate(
            series_level=str(series_level),
            intermittency_class=intermittency_class,
            spectral_entropy=spectral_entropy,
            residual_drift_pvalue=residual_drift_pvalue,
        )
        if adi is not None and adi < adi_threshold and cv2 is not None and cv2 < cv2_threshold:
            intermittency_bucket = "smooth"
        elif adi is not None and cv2 is not None:
            intermittency_bucket = intermittency_class
        else:
            intermittency_bucket = intermittency_class

        rows.append(
            {
                "series_level": str(series_level),
                "unique_id": str(unique_id),
                "n_periods": int(rate_diag.get("n_periods", len(group))),
                "recent_actual_mean_12m": _safe_float(rate_diag.get("recent_actual_mean_12m")),
                "acf_1": _safe_float((rate_diag.get("acf", {}) or {}).get("lag_1")),
                "acf_12": _safe_float((rate_diag.get("acf", {}) or {}).get("lag_12")),
                "seasonal_strength": _safe_float(
                    (rate_diag.get("stl", {}) or {}).get("seasonal_strength")
                ),
                "trend_strength": _safe_float(
                    (rate_diag.get("stl", {}) or {}).get("trend_strength")
                ),
                "spectral_entropy": spectral_entropy,
                "permutation_entropy": _safe_float(rate_diag.get("permutation_entropy")),
                "variance_ratio_12": _safe_float(
                    (rate_diag.get("variance_ratio", {}) or {}).get("value")
                ),
                "adi": adi,
                "cv2": cv2,
                "intermittency_class": intermittency_bucket,
                "route": route,
                "default_count_mean": _safe_float(count_diag.get("default_count_mean")),
                "default_count_std": _safe_float(count_diag.get("default_count_std")),
            }
        )

    report = pd.DataFrame(rows).sort_values(["series_level", "unique_id"]).reset_index(drop=True)
    status = {
        "available": not report.empty,
        "series_evaluated": int(len(report)),
        "routes": report["route"].value_counts(dropna=False).to_dict() if not report.empty else {},
        "intermittency": (
            report["intermittency_class"].value_counts(dropna=False).to_dict()
            if not report.empty
            else {}
        ),
        "levels": report["series_level"].value_counts(dropna=False).to_dict()
        if not report.empty
        else {},
    }
    return report, status


def _spectral_entropy(y: np.ndarray) -> float:
    centered = np.asarray(y, dtype=float) - float(np.mean(y))
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    total = float(np.sum(spectrum))
    if total <= 0:
        return 0.0
    probs = spectrum / total
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log2(probs)))
    max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
    return entropy / max(max_entropy, 1e-12)


def _permutation_entropy(y: np.ndarray, order: int = 3, delay: int = 1) -> float:
    series = np.asarray(y, dtype=float)
    n = len(series)
    window = delay * (order - 1)
    if n <= window:
        return 0.0
    patterns: defaultdict[tuple[int, ...], int] = defaultdict(int)
    for start in range(n - window):
        subseq = series[start : start + window + 1 : delay]
        pattern = tuple(np.argsort(subseq))
        patterns[pattern] += 1
    counts = np.asarray(list(patterns.values()), dtype=float)
    probs = counts / np.sum(counts)
    entropy = -float(np.sum(probs * np.log2(probs)))
    max_entropy = math.log2(math.factorial(order))
    return entropy / max(max_entropy, 1e-12)


def _resolve_interval_family(model_name: str) -> str:
    lower = str(model_name).lower()
    if lower.startswith("mapie_"):
        return "adaptive"
    if lower.endswith("_cf"):
        return "statistical"
    if lower in {"sarimax", "stl_catboost"}:
        return "challenger"
    return "statistical"


def _resolve_interval_subfamily(model_name: str) -> str:
    lower = str(model_name).lower()
    if lower.startswith("mapie_"):
        return "adaptive"
    if lower.endswith("_cf"):
        return "conformal_statistical"
    if lower in {"sarimax", "stl_catboost"}:
        return "challenger"
    return "native_statistical"


def compute_point_ensemble_weights(
    metrics: pd.DataFrame, config: dict[str, Any]
) -> dict[str, float]:
    ensemble_cfg = config.get("ensemble", {})
    candidates = list(ensemble_cfg.get("candidates", []))
    metric_name = str(ensemble_cfg.get("weight_metric", "mase"))
    if metrics.empty or not candidates or metric_name not in metrics.columns:
        return {}
    point_eligible = (
        metrics["point_eligible"].astype(bool)
        if "point_eligible" in metrics.columns
        else pd.Series(True, index=metrics.index)
    )
    subset = metrics.loc[metrics["model"].isin(candidates) & point_eligible].copy()
    if subset.empty:
        return {}
    inv = 1.0 / subset[metric_name].clip(lower=1e-8)
    total = float(inv.sum())
    if total <= 0:
        return {}
    return {
        str(model): float(weight / total)
        for model, weight in zip(
            subset["model"].astype(str), inv.to_numpy(dtype=float), strict=False
        )
    }


def append_weighted_point_ensemble(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    weights = compute_point_ensemble_weights(metrics, config)
    model_name = str((config.get("ensemble", {}) or {}).get("model_name", "WeightedEnsemble"))
    if predictions.empty or not weights:
        return predictions, {}
    subset = predictions.loc[predictions["model"].isin(weights)].copy()
    if subset.empty:
        return predictions, {}
    wide = subset.pivot_table(
        index=["cutoff", "ds", "horizon_step", "unique_id", "y_true"],
        columns="model",
        values="y_pred",
        aggfunc="mean",
    ).reset_index()
    required = [model for model in weights if model in wide.columns]
    if not required:
        return predictions, {}
    numer = sum(wide[model].astype(float) * weights[model] for model in required)
    denom = sum(weights[model] for model in required)
    wide["y_pred"] = numer / max(denom, 1e-12)
    out = wide[["cutoff", "ds", "horizon_step", "unique_id", "y_true", "y_pred"]].copy()
    out["model"] = model_name
    enriched = pd.concat([predictions, out], ignore_index=True, sort=False)
    return enriched, {model: weights[model] for model in required}


def run_portfolio_backtest(
    history: pd.DataFrame,
    config: dict[str, Any],
    *,
    future_covariates: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = _require_ts_columns(history)
    horizon = int(config["horizon"])
    season_length = int(config.get("season_length", 12))
    roll_cfg = config.get("rolling_origin", {})
    cutoffs = build_backtest_cutoffs(
        ts["ds"],
        horizon=horizon,
        min_train_periods=int(roll_cfg.get("min_train_periods", 72)),
        step_months=int(roll_cfg.get("step_months", 1)),
        max_windows=roll_cfg.get("max_windows"),
    )

    prediction_rows: list[dict[str, Any]] = []
    cutoff_metric_rows: list[dict[str, Any]] = []
    model_errors: defaultdict[str, list[float]] = defaultdict(list)

    for cutoff in cutoffs:
        train = ts.loc[ts["ds"] <= cutoff, ["unique_id", "ds", "y"]].copy()
        actual = ts.loc[ts["ds"] > cutoff].head(horizon).copy()
        if len(actual) < horizon:
            continue
        season_mae_scale, season_rmse_scale = _seasonal_error_scales(
            train["y"].to_numpy(), season_length
        )

        forecasts_by_model: dict[str, pd.DataFrame] = {}
        try:
            _, stats_fc = train_baseline_forecasters(
                train,
                horizon=horizon,
                freq=config.get("freq", "MS"),
                model_names=list(config.get("models", {}).get("statistical", [])),
            )
            for model_name in config.get("models", {}).get("statistical", []):
                cols = ["unique_id", "ds", model_name]
                extra = [
                    f"{model_name}-lo-90",
                    f"{model_name}-hi-90",
                    f"{model_name}-lo-95",
                    f"{model_name}-hi-95",
                ]
                cols.extend([col for col in extra if col in stats_fc.columns])
                if model_name in stats_fc.columns:
                    forecasts_by_model[model_name] = stats_fc[cols].copy()
        except Exception as exc:
            logger.warning("Statistical window forecast failed at {}: {}", cutoff, exc)

        if bool(
            (config.get("interval_policy", {}) or {}).get("include_conformal_statistical", False)
        ):
            try:
                _, stats_cf = train_baseline_forecasters(
                    train,
                    horizon=horizon,
                    freq=config.get("freq", "MS"),
                    model_names=list(config.get("models", {}).get("statistical", [])),
                    conformal_windows=int(
                        (config.get("interval_policy", {}) or {}).get("conformal_n_windows", 3)
                    ),
                    model_suffix="_CF",
                )
                for model_name in config.get("models", {}).get("statistical", []):
                    cf_model = f"{model_name}_CF"
                    cols = ["unique_id", "ds", cf_model]
                    extra = [
                        f"{cf_model}-lo-90",
                        f"{cf_model}-hi-90",
                        f"{cf_model}-lo-95",
                        f"{cf_model}-hi-95",
                    ]
                    cols.extend([col for col in extra if col in stats_cf.columns])
                    if cf_model in stats_cf.columns:
                        forecasts_by_model[cf_model] = stats_cf[cols].copy()
            except Exception as exc:
                logger.warning(
                    "Conformal statistical window forecast failed at {}: {}", cutoff, exc
                )

        exog_train = (
            _build_exogenous_feature_frame(train["ds"], future_covariates=future_covariates)
            .set_index("ds")
            .reindex(train["ds"])
        )
        exog_future = (
            _build_exogenous_feature_frame(actual["ds"], future_covariates=future_covariates)
            .set_index("ds")
            .reindex(actual["ds"])
        )

        try:
            sarimax_fc = fit_sarimax_forecaster(
                train,
                horizon=horizon,
                exog_train=exog_train if not exog_train.empty else None,
                exog_future=exog_future if not exog_future.empty else None,
                season_length=season_length,
            )
            forecasts_by_model["SARIMAX"] = sarimax_fc
        except Exception as exc:
            logger.warning("SARIMAX window forecast failed at {}: {}", cutoff, exc)

        try:
            stl_fc = fit_stl_catboost_forecaster(
                train,
                horizon=horizon,
                season_length=season_length,
                exog_train=exog_train if not exog_train.empty else None,
                exog_future=exog_future if not exog_future.empty else None,
            )
            forecasts_by_model["STL_CatBoost"] = stl_fc
        except Exception as exc:
            logger.warning("STL_CatBoost window forecast failed at {}: {}", cutoff, exc)

        for model_name, fc in forecasts_by_model.items():
            merged = actual[["ds", "y"]].merge(fc, on="ds", how="left")
            if model_name not in merged.columns or merged[model_name].isna().all():
                continue
            y_true = merged["y"].to_numpy(dtype=float)
            y_pred = np.asarray(merged[model_name], dtype=float)
            lo90, hi90, lo95, hi95 = _extract_interval_columns(merged, model_name)
            metrics = _window_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_lo_90=lo90,
                y_hi_90=hi90,
                y_lo_95=lo95,
                y_hi_95=hi95,
                season_mae_scale=season_mae_scale,
                season_rmse_scale=season_rmse_scale,
            )
            cutoff_metric_rows.append(
                {
                    "cutoff": pd.Timestamp(cutoff),
                    "model": model_name,
                    "n_obs": int(len(y_true)),
                    **metrics,
                }
            )
            model_errors[model_name].extend(np.abs(y_true - y_pred).tolist())

            for horizon_step, row in enumerate(merged.itertuples(index=False), start=1):
                prediction_rows.append(
                    {
                        "cutoff": pd.Timestamp(cutoff),
                        "ds": pd.Timestamp(row.ds),
                        "horizon_step": int(horizon_step),
                        "unique_id": "portfolio",
                        "model": model_name,
                        "y_true": float(row.y),
                        "y_pred": float(getattr(row, model_name)),
                        "lo_90": _safe_float(getattr(row, f"{model_name}-lo-90", np.nan)),
                        "hi_90": _safe_float(getattr(row, f"{model_name}-hi-90", np.nan)),
                        "lo_95": _safe_float(getattr(row, f"{model_name}-lo-95", np.nan)),
                        "hi_95": _safe_float(getattr(row, f"{model_name}-hi-95", np.nan)),
                    }
                )

    predictions = (
        pd.DataFrame(prediction_rows)
        .sort_values(["model", "cutoff", "horizon_step"])
        .reset_index(drop=True)
    )
    cutoff_metrics = (
        pd.DataFrame(cutoff_metric_rows).sort_values(["model", "cutoff"]).reset_index(drop=True)
    )
    if cutoff_metrics.empty:
        return predictions, cutoff_metrics

    summary = (
        cutoff_metrics.groupby("model", as_index=False)
        .agg(
            n_cutoffs=("cutoff", "nunique"),
            n_obs=("n_obs", "sum"),
            mae=("forecast_mae", "mean"),
            rmse=("forecast_rmse", "mean"),
            mase=("mase", "mean"),
            rmsse=("rmsse", "mean"),
            mean_bias=("mean_bias", "mean"),
            abs_bias=("abs_bias", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
            coverage_90=("coverage_90", "mean"),
            coverage_95=("coverage_95", "mean"),
            avg_interval_width_90=("avg_interval_width_90", "mean"),
            avg_interval_width_95=("avg_interval_width_95", "mean"),
            winkler_90=("winkler_90", "mean"),
            winkler_95=("winkler_95", "mean"),
            pinball_90=("pinball_90", "mean"),
            wis_90=("wis_90", "mean"),
        )
        .reset_index(drop=True)
    )

    baseline_row = summary.loc[summary["model"] == "SeasonalNaive"]
    baseline_mae = float(baseline_row["mae"].iloc[0]) if not baseline_row.empty else np.nan
    baseline_rmsse = float(baseline_row["rmsse"].iloc[0]) if not baseline_row.empty else np.nan

    revision_rows = compute_revision_metrics(predictions)
    summary = summary.merge(revision_rows, on="model", how="left")

    dm_rows = []
    baseline_preds = predictions.loc[
        predictions["model"] == "SeasonalNaive",
        ["cutoff", "ds", "horizon_step", "y_true", "y_pred"],
    ].copy()
    baseline_preds["baseline_loss"] = np.abs(baseline_preds["y_true"] - baseline_preds["y_pred"])
    for model_name in summary["model"]:
        model_preds = predictions.loc[
            predictions["model"] == model_name, ["cutoff", "ds", "horizon_step", "y_true", "y_pred"]
        ].copy()
        model_preds["model_loss"] = np.abs(model_preds["y_true"] - model_preds["y_pred"])
        merged = model_preds.merge(
            baseline_preds[["cutoff", "ds", "horizon_step", "baseline_loss"]],
            on=["cutoff", "ds", "horizon_step"],
            how="inner",
        )
        dm = diebold_mariano_test(
            merged["model_loss"].to_numpy(dtype=float),
            merged["baseline_loss"].to_numpy(dtype=float),
            lag=horizon,
        )
        dm_rows.append(
            {
                "model": model_name,
                "dm_stat_vs_seasonal_naive": float(dm["dm_stat"]),
                "dm_pvalue_vs_seasonal_naive": float(dm["p_value"]),
                "dm_reject_vs_seasonal_naive": bool(dm["reject"]),
            }
        )
    summary = summary.merge(pd.DataFrame(dm_rows), on="model", how="left")

    if math.isfinite(baseline_mae):
        summary["fva_mae_abs"] = baseline_mae - summary["mae"]
        summary["fva_mae_pct"] = 1.0 - (summary["mae"] / max(baseline_mae, 1e-8))
    else:
        summary["fva_mae_abs"] = np.nan
        summary["fva_mae_pct"] = np.nan
    if math.isfinite(baseline_rmsse):
        summary["rmsse_vs_seasonal_naive"] = summary["rmsse"] / max(baseline_rmsse, 1e-8)
    else:
        summary["rmsse_vs_seasonal_naive"] = np.nan
    summary["coverage_gap_90"] = (summary["coverage_90"] - 0.90).abs()
    summary["coverage_gap_95"] = (summary["coverage_95"] - 0.95).abs()
    summary["family"] = summary["model"].map(_resolve_interval_family)
    summary["interval_subfamily"] = summary["model"].map(_resolve_interval_subfamily)
    summary["point_eligible"] = ~summary["model"].astype(str).str.endswith("_CF")
    summary["interval_eligible"] = True
    if bool((config.get("ensemble", {}) or {}).get("enabled", False)):
        predictions, weights = append_weighted_point_ensemble(predictions, summary, config)
        if weights:
            ensemble_name = str(
                (config.get("ensemble", {}) or {}).get("model_name", "WeightedEnsemble")
            )
            ens_preds = predictions.loc[predictions["model"] == ensemble_name].copy()
            if not ens_preds.empty:
                ens_rows = []
                for cutoff, group in ens_preds.groupby("cutoff", sort=True):
                    ordered = group.sort_values("horizon_step")
                    y_true = ordered["y_true"].to_numpy(dtype=float)
                    y_pred = ordered["y_pred"].to_numpy(dtype=float)
                    metrics_row = _window_metrics(
                        y_true=y_true,
                        y_pred=y_pred,
                        y_lo_90=None,
                        y_hi_90=None,
                        y_lo_95=None,
                        y_hi_95=None,
                        season_mae_scale=_seasonal_error_scales(
                            ts.loc[ts["ds"] <= pd.Timestamp(cutoff), "y"].to_numpy(dtype=float),
                            season_length,
                        )[0],
                        season_rmse_scale=_seasonal_error_scales(
                            ts.loc[ts["ds"] <= pd.Timestamp(cutoff), "y"].to_numpy(dtype=float),
                            season_length,
                        )[1],
                    )
                    ens_rows.append(
                        {
                            "cutoff": pd.Timestamp(cutoff),
                            "model": ensemble_name,
                            "n_obs": int(len(y_true)),
                            **metrics_row,
                        }
                    )
                if ens_rows:
                    ens_cutoff = pd.DataFrame(ens_rows)
                    cutoff_metrics = pd.concat(
                        [cutoff_metrics, ens_cutoff], ignore_index=True, sort=False
                    )
                    ens_summary = (
                        ens_cutoff.groupby("model", as_index=False)
                        .agg(
                            n_cutoffs=("cutoff", "nunique"),
                            n_obs=("n_obs", "sum"),
                            mae=("forecast_mae", "mean"),
                            rmse=("forecast_rmse", "mean"),
                            mase=("mase", "mean"),
                            rmsse=("rmsse", "mean"),
                            mean_bias=("mean_bias", "mean"),
                            abs_bias=("abs_bias", "mean"),
                            directional_accuracy=("directional_accuracy", "mean"),
                        )
                        .reset_index(drop=True)
                    )
                    ens_summary["mean_abs_revision"] = np.nan
                    ens_summary["max_abs_revision"] = np.nan
                    ens_summary["n_revisions"] = 0
                    ens_summary["dm_stat_vs_seasonal_naive"] = np.nan
                    ens_summary["dm_pvalue_vs_seasonal_naive"] = np.nan
                    ens_summary["dm_reject_vs_seasonal_naive"] = False
                    ens_summary["fva_mae_abs"] = np.nan
                    ens_summary["fva_mae_pct"] = np.nan
                    ens_summary["rmsse_vs_seasonal_naive"] = np.nan
                    ens_summary["coverage_gap_90"] = np.nan
                    ens_summary["coverage_gap_95"] = np.nan
                    ens_summary["coverage_90"] = np.nan
                    ens_summary["coverage_95"] = np.nan
                    ens_summary["avg_interval_width_90"] = np.nan
                    ens_summary["avg_interval_width_95"] = np.nan
                    ens_summary["winkler_90"] = np.nan
                    ens_summary["winkler_95"] = np.nan
                    ens_summary["pinball_90"] = np.nan
                    ens_summary["wis_90"] = np.nan
                    ens_summary["family"] = "ensemble"
                    ens_summary["interval_subfamily"] = "point_ensemble"
                    ens_summary["point_eligible"] = True
                    ens_summary["interval_eligible"] = False
                    summary = pd.concat([summary, ens_summary], ignore_index=True, sort=False)
    return predictions, summary.sort_values(["mase", "rmsse", "abs_bias"]).reset_index(drop=True)


def compute_revision_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=["model", "mean_abs_revision", "max_abs_revision", "n_revisions"]
        )
    rows = []
    for model_name, model_df in predictions.groupby("model", sort=True):
        revision_values: list[float] = []
        grouped = model_df.sort_values(["ds", "cutoff"]).groupby("ds", sort=True)
        for _, target_df in grouped:
            vals = target_df["y_pred"].to_numpy(dtype=float)
            if len(vals) > 1:
                revision_values.extend(np.abs(np.diff(vals)).tolist())
        rows.append(
            {
                "model": model_name,
                "mean_abs_revision": float(np.mean(revision_values)) if revision_values else np.nan,
                "max_abs_revision": float(np.max(revision_values)) if revision_values else np.nan,
                "n_revisions": int(len(revision_values)),
            }
        )
    return pd.DataFrame(rows)


def select_time_series_champions(
    metrics: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    if metrics.empty:
        raise ValueError("Backtest metrics are empty; cannot select champions.")

    point_cfg = config.get("point_champion", {})
    interval_cfg = config.get("interval_policy", {})
    baseline = metrics.loc[metrics["model"] == "SeasonalNaive"]
    baseline_mae = float(baseline["mae"].iloc[0]) if not baseline.empty else np.nan
    baseline_rmsse = float(baseline["rmsse"].iloc[0]) if not baseline.empty else np.nan

    point_metrics = metrics.loc[
        metrics["point_eligible"].astype(bool)
        if "point_eligible" in metrics.columns
        else slice(None)
    ].copy()
    if point_metrics.empty:
        point_metrics = metrics.copy()
    point_sort = point_metrics.sort_values(["mase", "rmsse", "abs_bias", "mae"]).reset_index(
        drop=True
    )
    point_row = point_sort.iloc[0]
    point_promotable = True
    point_reasons: list[str] = []
    if (
        bool(point_cfg.get("must_beat_seasonal_naive", True))
        and math.isfinite(baseline_mae)
        and float(point_row["mae"]) >= baseline_mae
    ):
        point_promotable = False
        point_reasons.append("mae_not_better_than_seasonal_naive")
    max_rmsse = float(point_cfg.get("max_rmsse_vs_seasonal_naive", 1.0))
    if math.isfinite(baseline_rmsse):
        ratio = float(point_row["rmsse"]) / max(baseline_rmsse, 1e-8)
        if ratio > max_rmsse:
            point_promotable = False
            point_reasons.append("rmsse_worse_than_allowed_vs_seasonal_naive")

    interval_metrics = metrics.loc[
        metrics["interval_eligible"].astype(bool)
        if "interval_eligible" in metrics.columns
        else slice(None)
    ].copy()
    if interval_metrics.empty:
        interval_metrics = metrics.copy()
    interval_sort_cols = [
        column
        for column in ["coverage_gap_90", "wis_90", "winkler_90", "avg_interval_width_90"]
        if column in interval_metrics.columns
    ]
    if not interval_sort_cols:
        interval_sort_cols = ["model"]
    interval_candidates = interval_metrics.loc[
        interval_metrics["coverage_gap_90"] <= float(interval_cfg.get("max_coverage_gap", 0.03))
    ].copy()
    eligible_families = [
        str(family).strip().lower()
        for family in interval_cfg.get(
            "eligible_interval_families",
            [interval_cfg.get("official_family", "statistical")],
        )
        if str(family).strip()
    ]
    if eligible_families:
        preferred = interval_candidates.loc[
            interval_candidates["family"].astype(str).str.lower().isin(eligible_families)
        ]
        if not preferred.empty:
            interval_candidates = preferred
    max_winkler = interval_cfg.get("max_winkler_90")
    if max_winkler is not None:
        filtered = interval_candidates.loc[
            interval_candidates["winkler_90"].astype(float) <= float(max_winkler)
        ]
        if not filtered.empty:
            interval_candidates = filtered
    if interval_candidates.empty:
        fallback = interval_metrics.copy()
        if eligible_families:
            family_fallback = fallback.loc[
                fallback["family"].astype(str).str.lower().isin(eligible_families)
            ]
            if not family_fallback.empty:
                fallback = family_fallback
        interval_row = (
            fallback.sort_values(interval_sort_cols).iloc[0]
            if not fallback.empty
            else interval_metrics.sort_values(interval_sort_cols).iloc[0]
        )
        interval_promotable = False
        interval_reasons = ["no_model_within_coverage_gap_policy"]
    else:
        interval_row = interval_candidates.sort_values(interval_sort_cols).iloc[0]
        interval_promotable = True
        interval_reasons = []

    return {
        "point": {
            "model": str(point_row["model"]),
            "promotable": bool(point_promotable),
            "reasons": point_reasons,
            "mae": float(point_row["mae"]),
            "mase": float(point_row["mase"]),
            "rmsse": float(point_row["rmsse"]),
            "abs_bias": float(point_row["abs_bias"]),
        },
        "interval": {
            "model": str(interval_row["model"]),
            "promotable": bool(interval_promotable),
            "reasons": interval_reasons,
            "coverage_90": _safe_float(interval_row.get("coverage_90")),
            "coverage_gap_90": _safe_float(interval_row.get("coverage_gap_90")),
            "winkler_90": _safe_float(interval_row.get("winkler_90")),
            "wis_90": _safe_float(interval_row.get("wis_90")),
            "pinball_90": _safe_float(interval_row.get("pinball_90")),
            "avg_interval_width_90": _safe_float(interval_row.get("avg_interval_width_90")),
            "family": str(interval_row.get("family", "")),
            "interval_subfamily": str(interval_row.get("interval_subfamily", "")),
        },
        "baseline": {
            "model": "SeasonalNaive",
            "mae": baseline_mae,
            "rmsse": baseline_rmsse,
        },
    }


def forecast_portfolio_models(
    history: pd.DataFrame,
    config: dict[str, Any],
    *,
    future_covariates: pd.DataFrame | None = None,
    point_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    ts = _require_ts_columns(history)
    horizon = int(config["horizon"])
    season_length = int(config.get("season_length", 12))

    forecasts: list[pd.DataFrame] = []
    _, stats_fc = train_baseline_forecasters(
        ts,
        horizon=horizon,
        freq=config.get("freq", "MS"),
        model_names=list(config.get("models", {}).get("statistical", [])),
    )
    forecasts.append(stats_fc.copy())
    if bool((config.get("interval_policy", {}) or {}).get("include_conformal_statistical", False)):
        try:
            _, stats_cf = train_baseline_forecasters(
                ts,
                horizon=horizon,
                freq=config.get("freq", "MS"),
                model_names=list(config.get("models", {}).get("statistical", [])),
                conformal_windows=int(
                    (config.get("interval_policy", {}) or {}).get("conformal_n_windows", 3)
                ),
                model_suffix="_CF",
            )
            forecasts.append(stats_cf.copy())
        except Exception as exc:
            logger.warning("Final conformal statistical forecast failed: {}", exc)

    future_dates = pd.date_range(
        ts["ds"].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    exog_train = (
        _build_exogenous_feature_frame(ts["ds"], future_covariates=future_covariates)
        .set_index("ds")
        .reindex(ts["ds"])
    )
    exog_future = (
        _build_exogenous_feature_frame(future_dates, future_covariates=future_covariates)
        .set_index("ds")
        .reindex(future_dates)
    )

    try:
        forecasts.append(
            fit_sarimax_forecaster(
                ts,
                horizon=horizon,
                exog_train=exog_train if not exog_train.empty else None,
                exog_future=exog_future if not exog_future.empty else None,
                season_length=season_length,
            )
        )
    except Exception as exc:
        logger.warning("Final SARIMAX forecast failed: {}", exc)

    try:
        forecasts.append(
            fit_stl_catboost_forecaster(
                ts,
                horizon=horizon,
                season_length=season_length,
                exog_train=exog_train if not exog_train.empty else None,
                exog_future=exog_future if not exog_future.empty else None,
            )
        )
    except Exception as exc:
        logger.warning("Final STL_CatBoost forecast failed: {}", exc)

    merged = forecasts[0]
    for frame in forecasts[1:]:
        merged = merged.merge(frame, on=["unique_id", "ds"], how="outer")
    if point_weights:
        valid = [name for name in point_weights if name in merged.columns]
        if valid:
            weight_total = sum(point_weights[name] for name in valid)
            ensemble_name = str(
                (config.get("ensemble", {}) or {}).get("model_name", "WeightedEnsemble")
            )
            merged[ensemble_name] = sum(
                merged[name].astype(float) * point_weights[name] for name in valid
            ) / max(weight_total, 1e-12)
    return merged.sort_values("ds").reset_index(drop=True)


def _load_macro_context_events(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def build_minimal_macro_covariates(
    dates: Iterable[pd.Timestamp] | pd.Series,
    *,
    macro_context_path: str | Path = "data/processed/macro_context.json",
) -> pd.DataFrame:
    ds = pd.Index(pd.to_datetime(list(dates), errors="coerce")).dropna().sort_values().unique()
    frame = pd.DataFrame({"ds": ds})
    if frame.empty:
        return frame

    frame["month"] = frame["ds"].dt.month.astype(int)
    frame["quarter"] = frame["ds"].dt.quarter.astype(int)
    frame["is_q4"] = frame["quarter"].eq(4).astype(int)
    frame["month_sin"] = np.sin(2.0 * np.pi * frame["month"] / 12.0)
    frame["month_cos"] = np.cos(2.0 * np.pi * frame["month"] / 12.0)

    events = _load_macro_context_events(macro_context_path)
    if not events:
        for col in (
            "macro_event_count_12m",
            "macro_crisis_count_12m",
            "macro_recovery_count_12m",
            "macro_policy_count_12m",
            "macro_regime_crisis",
            "macro_regime_recovery",
            "macro_regime_policy",
        ):
            frame[col] = 0
        return frame

    events_df = pd.DataFrame(events)
    if "date" not in events_df.columns:
        return frame
    events_df = events_df.copy()
    events_df["ds"] = pd.to_datetime(events_df["date"].astype(str) + "-01", errors="coerce")
    events_df["type"] = events_df.get("type", "unknown").astype(str).str.lower()
    events_df = events_df.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    if events_df.empty:
        return frame

    event_types = ("crisis", "recovery", "policy")
    event_dates = events_df["ds"]
    for event_type in event_types:
        mask = events_df["type"].eq(event_type)
        counts: list[int] = []
        for current_ds in frame["ds"]:
            window_start = current_ds - pd.DateOffset(months=12)
            count = int(((event_dates > window_start) & (event_dates <= current_ds) & mask).sum())
            counts.append(count)
        frame[f"macro_{event_type}_count_12m"] = counts

    total_counts = []
    regime_cols = {event_type: [] for event_type in event_types}
    for current_ds in frame["ds"]:
        window_start = current_ds - pd.DateOffset(months=12)
        total_counts.append(int(((event_dates > window_start) & (event_dates <= current_ds)).sum()))
        historical = events_df.loc[events_df["ds"] <= current_ds]
        latest_type = str(historical.iloc[-1]["type"]) if not historical.empty else "unknown"
        for event_type in event_types:
            regime_cols[event_type].append(1 if latest_type == event_type else 0)

    frame["macro_event_count_12m"] = total_counts
    for event_type, values in regime_cols.items():
        frame[f"macro_regime_{event_type}"] = values
    return frame


def _build_exogenous_feature_frame(
    dates: Iterable[pd.Timestamp] | pd.Series,
    *,
    future_covariates: pd.DataFrame | None = None,
    macro_context_path: str | Path = "data/processed/macro_context.json",
) -> pd.DataFrame:
    frame = build_minimal_macro_covariates(dates, macro_context_path=macro_context_path)
    if frame.empty:
        return frame

    if future_covariates is not None and not future_covariates.empty:
        overlay = future_covariates.copy()
        overlay["ds"] = pd.to_datetime(overlay["ds"], errors="coerce")
        overlay = overlay.dropna(subset=["ds"]).sort_values("ds").drop_duplicates("ds")
        value_cols = [col for col in overlay.columns if col != "ds"]
        if value_cols:
            frame = frame.merge(
                overlay[["ds", *value_cols]],
                on="ds",
                how="left",
                suffixes=("", "__overlay"),
            )
            for col in value_cols:
                overlay_col = f"{col}__overlay"
                if overlay_col not in frame.columns:
                    continue
                if col in frame.columns:
                    frame[col] = frame[overlay_col].where(frame[overlay_col].notna(), frame[col])
                else:
                    frame[col] = frame[overlay_col]
            frame = frame[[col for col in frame.columns if not col.endswith("__overlay")]]

    feature_cols = [col for col in frame.columns if col != "ds"]
    if feature_cols:
        frame = frame.sort_values("ds").reset_index(drop=True)
        frame[feature_cols] = (
            frame[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        )
    return frame


def build_future_covariates_contract(
    history: pd.DataFrame,
    *,
    horizon: int,
    freq: str = "MS",
    macro_context_path: str | Path = "data/processed/macro_context.json",
) -> pd.DataFrame:
    ts = _require_ts_columns(history)
    start = pd.Timestamp(ts["ds"].max()) + pd.tseries.frequencies.to_offset(freq)
    future_dates = pd.date_range(start=start, periods=horizon, freq=freq)
    return build_minimal_macro_covariates(future_dates, macro_context_path=macro_context_path)


def benchmark_mapie_time_series_intervals(
    history: pd.DataFrame,
    *,
    confidence_level: float = 0.90,
    evaluation_size: int = 24,
    gamma: float = 0.01,
    macro_context_path: str | Path = "data/processed/macro_context.json",
    estimator_name: str = "linear",
) -> dict[str, Any]:
    from mapie.regression import TimeSeriesRegressor
    from mapie.subsample import BlockBootstrap

    ts = _require_ts_columns(history)
    series = ts.sort_values("ds").reset_index(drop=True)
    macro_cov = build_minimal_macro_covariates(series["ds"], macro_context_path=macro_context_path)
    frame = series.merge(macro_cov, on="ds", how="left")
    for lag in (1, 2, 3, 6, 12):
        frame[f"lag_{lag}"] = frame["y"].shift(lag)
    frame = frame.dropna().reset_index(drop=True)

    min_train = max(60, int(len(frame) * 0.60))
    eval_rows = min(max(int(evaluation_size), 12), max(len(frame) - min_train, 0))
    if eval_rows <= 0:
        return {
            "available": False,
            "reason": "insufficient_history_for_mapie_benchmark",
            "candidate_methods_tested": [],
        }

    feature_cols = [col for col in frame.columns if col not in {"unique_id", "ds", "y"}]
    train = frame.iloc[:-eval_rows].reset_index(drop=True)
    evaluation = frame.iloc[-eval_rows:].reset_index(drop=True)
    X_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train["y"].to_numpy(dtype=float)
    alpha = 1.0 - float(confidence_level)

    estimator_name = str(estimator_name).strip().lower()

    def _make_estimator() -> Any:
        if estimator_name == "catboost":
            return _build_catboost_regressor(
                iterations=200,
                learning_rate=0.05,
                depth=5,
            )
        from sklearn.linear_model import LinearRegression

        return LinearRegression()

    results: list[dict[str, Any]] = []
    for method in ("aci", "enbpi"):
        cv = (
            BlockBootstrap(
                n_resamplings=20,
                length=max(6, min(12, len(train) // 6)),
                overlapping=False,
                random_state=42,
            )
            if method == "enbpi"
            else 5
        )
        regressor = TimeSeriesRegressor(
            estimator=_make_estimator(),
            method=method,
            cv=cv,
            random_state=42,
        )
        regressor.fit(X_train, y_train)

        rows: list[dict[str, Any]] = []
        method_failure_reason: str | None = None
        for eval_row in evaluation.itertuples(index=False):
            X_row = np.asarray([[getattr(eval_row, col) for col in feature_cols]], dtype=float)
            try:
                y_pred, intervals_raw = regressor.predict(X_row, confidence_level=confidence_level)
            except ValueError as exc:
                method_failure_reason = str(exc)
                logger.warning("MAPIE {} benchmark skipped: {}", method, exc)
                break
            intervals = np.asarray(intervals_raw[:, :, 0], dtype=float)
            low = float(intervals[0, 0])
            high = float(intervals[0, 1])
            y_true = float(eval_row.y)
            rows.append(
                {
                    "ds": pd.Timestamp(eval_row.ds),
                    "y_true": y_true,
                    "y_pred": float(np.asarray(y_pred, dtype=float)[0]),
                    "lo_90": low,
                    "hi_90": high,
                    "covered": float(low <= y_true <= high),
                    "width": float(high - low),
                    "winkler_90": float(
                        winkler_interval_score(
                            np.asarray([y_true], dtype=float),
                            np.asarray([low], dtype=float),
                            np.asarray([high], dtype=float),
                            alpha=alpha,
                        )[0]
                    ),
                }
            )
            if method == "aci":
                regressor.adapt_conformal_inference(
                    X_row,
                    np.asarray([y_true], dtype=float),
                    gamma=gamma,
                    confidence_level=confidence_level,
                )
            else:
                regressor.update(
                    X_row,
                    np.asarray([y_true], dtype=float),
                    confidence_level=confidence_level,
                )

        if method_failure_reason is not None or not rows:
            continue

        result_df = pd.DataFrame(rows)
        rolling = (
            result_df.assign(
                rolling_coverage_6=result_df["covered"].rolling(6, min_periods=3).mean()
            )
            if not result_df.empty
            else pd.DataFrame()
        )
        results.append(
            {
                "method": method,
                "n_eval": int(len(result_df)),
                "coverage_90": float(result_df["covered"].mean()),
                "coverage_gap_90": float(abs(result_df["covered"].mean() - confidence_level)),
                "avg_interval_width_90": float(result_df["width"].mean()),
                "winkler_90": float(result_df["winkler_90"].mean()),
                "pinball_90": float(
                    (
                        _pinball_loss(
                            result_df["y_true"].to_numpy(dtype=float),
                            result_df["lo_90"].to_numpy(dtype=float),
                            0.05,
                        )
                        + _pinball_loss(
                            result_df["y_true"].to_numpy(dtype=float),
                            result_df["hi_90"].to_numpy(dtype=float),
                            0.95,
                        )
                    )
                    / 2.0
                ),
                "wis_90": float(
                    _weighted_interval_score(
                        result_df["y_true"].to_numpy(dtype=float),
                        result_df["y_pred"].to_numpy(dtype=float),
                        result_df["lo_90"].to_numpy(dtype=float),
                        result_df["hi_90"].to_numpy(dtype=float),
                        alpha=alpha,
                    )
                ),
                "rolling_coverage_summary": {
                    "min_rolling_coverage_6": float(rolling["rolling_coverage_6"].min())
                    if not rolling.empty
                    else None,
                    "last_rolling_coverage_6": float(rolling["rolling_coverage_6"].iloc[-1])
                    if not rolling.empty
                    else None,
                },
            }
        )

    if not results:
        return {
            "available": False,
            "reason": "mapie_benchmark_unavailable_for_effective_score_size",
            "estimator_name": estimator_name,
            "candidate_methods_tested": [],
            "results": [],
            "rolling_coverage_summary": {},
        }

    results_df = pd.DataFrame(results).sort_values(
        ["coverage_gap_90", "wis_90", "winkler_90", "avg_interval_width_90"]
    )
    best = results_df.iloc[0].to_dict()
    return {
        "available": True,
        "estimator_name": estimator_name,
        "best_method": str(best.get("method", "")),
        "candidate_methods_tested": results_df["method"].astype(str).tolist(),
        "results": results_df.to_dict(orient="records"),
        "rolling_coverage_summary": best.get("rolling_coverage_summary", {}),
    }


def build_canonical_forecast_frame(
    forecasts: pd.DataFrame,
    champions: dict[str, Any],
) -> pd.DataFrame:
    out = forecasts.copy()
    point_model = str(champions["point"]["model"])
    interval_model = str(champions["interval"]["model"])
    out["y"] = out[point_model].astype(float)
    out["point_model"] = point_model
    out["interval_model"] = interval_model
    for level in (90, 95):
        lo_name = f"{interval_model}-lo-{level}"
        hi_name = f"{interval_model}-hi-{level}"
        out[f"y_lo_{level}"] = out[lo_name].astype(float) if lo_name in out.columns else np.nan
        out[f"y_hi_{level}"] = out[hi_name].astype(float) if hi_name in out.columns else np.nan
    out["point_promotable"] = bool(champions["point"]["promotable"])
    out["interval_promotable"] = bool(champions["interval"]["promotable"])
    out["official_status"] = np.where(
        out["point_promotable"] & out["interval_promotable"],
        "official",
        "diagnostic",
    )
    return out


def build_ifrs9_temporal_scenarios(canonical_forecasts: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ds",
        "y",
        "y_lo_90",
        "y_hi_90",
        "y_lo_95",
        "y_hi_95",
        "point_model",
        "interval_model",
        "official_status",
    ]
    out = canonical_forecasts[cols].copy()
    out = out.rename(
        columns={
            "ds": "month",
            "y": "point_forecast",
            "y_lo_90": "optimistic_90",
            "y_hi_90": "adverse_90",
            "y_lo_95": "optimistic_95",
            "y_hi_95": "adverse_95",
        }
    )
    return out


def forecast_panel_bottom_up(
    panel_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    horizon = int(config["horizon"])
    bottom = panel_df.loc[
        panel_df["series_level"] == "grade_term",
        ["unique_id", "ds", "default_count", "loan_count", "grade", "term_months"],
    ].copy()
    if bottom.empty:
        return pd.DataFrame(), {
            "available": False,
            "reason": "missing_bottom_level_grade_term_series",
        }

    def _forecast_target(target: str) -> pd.DataFrame:
        try:
            from mlforecast import MLForecast
            from mlforecast.lag_transforms import RollingMean

            train_df = bottom.rename(columns={target: "y"})[["unique_id", "ds", "y"]].copy()
            train_df["y"] = train_df["y"].astype(float)
            model = MLForecast(
                models={
                    "panel_catboost": _build_catboost_regressor(
                        iterations=200,
                        learning_rate=0.05,
                        depth=6,
                    )
                },
                freq=config.get("freq", "MS"),
                lags=[1, 2, 3, 6, 12],
                lag_transforms={
                    1: [
                        RollingMean(window_size=3),
                        RollingMean(window_size=6),
                        RollingMean(window_size=12),
                    ]
                },
                date_features=["month"],
            )
            model.fit(train_df)
            pred = model.predict(h=horizon)
            return pred.rename(columns={"panel_catboost": target})
        except Exception as exc:
            logger.warning("Panel global forecast fallback for {}: {}", target, exc)
            fallback_rows = []
            for unique_id, group in bottom.groupby("unique_id", sort=True):
                tail = group[target].astype(float).tail(12).to_numpy()
                values = np.resize(tail if tail.size else np.array([0.0]), horizon)
                future_ds = pd.date_range(
                    group["ds"].max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
                )
                fallback_rows.append(
                    pd.DataFrame({"unique_id": unique_id, "ds": future_ds, target: values})
                )
            return pd.concat(fallback_rows, ignore_index=True)

    default_fc = _forecast_target("default_count")
    loan_fc = _forecast_target("loan_count")
    merged = default_fc.merge(loan_fc, on=["unique_id", "ds"], how="inner")
    metadata = bottom[["unique_id", "grade", "term_months"]].drop_duplicates()
    merged = merged.merge(metadata, on="unique_id", how="left")
    merged["series_level"] = "grade_term"

    grade = (
        merged.groupby(["ds", "grade"], observed=True)
        .agg(default_count=("default_count", "sum"), loan_count=("loan_count", "sum"))
        .reset_index()
    )
    grade["unique_id"] = grade["grade"].map(lambda g: f"grade::{g}")
    grade["term_months"] = np.nan
    grade["series_level"] = "grade"

    portfolio = (
        merged.groupby("ds", observed=True)
        .agg(default_count=("default_count", "sum"), loan_count=("loan_count", "sum"))
        .reset_index()
    )
    portfolio["unique_id"] = "portfolio"
    portfolio["grade"] = "ALL"
    portfolio["term_months"] = np.nan
    portfolio["series_level"] = "portfolio"

    out = pd.concat([merged, grade, portfolio], ignore_index=True, sort=False)
    out["default_count"] = out["default_count"].clip(lower=0.0)
    out["loan_count"] = out["loan_count"].clip(lower=1.0)
    out["default_rate"] = (out["default_count"] / out["loan_count"]).clip(0.0, 1.0)
    out = out.sort_values(["series_level", "unique_id", "ds"]).reset_index(drop=True)
    status = {
        "available": True,
        "method": "bottom_up_counts_with_global_catboost",
        "n_bottom_series": int(bottom["unique_id"].nunique()),
        "n_forecast_rows": int(len(out)),
    }
    return out, status


def _build_hierarchy_inputs(
    panel_df: pd.DataFrame,
    *,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    from hierarchicalforecast.utils import aggregate

    bottom = panel_df.loc[
        panel_df["series_level"] == "grade_term",
        ["ds", "grade", "term_months", target_col],
    ].rename(columns={target_col: "y"})
    if bottom.empty:
        raise ValueError("Missing bottom-level panel data for hierarchy evaluation.")
    Y_df, S_df, tags = aggregate(bottom, spec=[["grade"], ["grade", "term_months"]])
    portfolio = (
        bottom.groupby("ds", observed=True)["y"].sum().reset_index().assign(unique_id="portfolio")
    )
    Y_df = pd.concat([portfolio[["unique_id", "ds", "y"]], Y_df], ignore_index=True, sort=False)
    top_row = pd.DataFrame(
        [{"unique_id": "portfolio", **{col: 1.0 for col in S_df.columns if col != "unique_id"}}]
    )
    S_df = pd.concat([top_row, S_df], ignore_index=True, sort=False)
    tags = {"portfolio": np.asarray(["portfolio"]), **tags}
    return Y_df.sort_values(["unique_id", "ds"]).reset_index(drop=True), S_df, tags


def evaluate_hierarchical_reconciliation(
    panel_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from hierarchicalforecast.core import HierarchicalReconciliation
    from hierarchicalforecast.methods import BottomUp, MinTrace, TopDown
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    recon_cfg = config.get("hierarchy_reconciliation", {})
    if not bool(recon_cfg.get("enabled", False)):
        return pd.DataFrame(), {"available": False, "reason": "hierarchy_reconciliation_disabled"}

    horizon = int(recon_cfg.get("evaluation_horizon", config.get("horizon", 12)))
    methods = list(recon_cfg.get("methods", ["BottomUp", "TopDown", "MinTrace"]))
    target_columns = list(recon_cfg.get("target_columns", ["default_count", "loan_count"]))
    reconcilers_map = {
        "BottomUp": BottomUp(),
        "TopDown": TopDown(method="forecast_proportions"),
        "MinTrace": MinTrace(method="ols"),
    }
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"available": True, "targets": {}}

    for target_col in target_columns:
        Y_df, S_df, tags = _build_hierarchy_inputs(panel_df, target_col=target_col)
        if Y_df["ds"].nunique() <= horizon:
            summary["targets"][target_col] = {"available": False, "reason": "insufficient_history"}
            continue
        cutoff = pd.Timestamp(sorted(Y_df["ds"].unique())[-horizon - 1])
        Y_train = Y_df.loc[Y_df["ds"] <= cutoff].copy()
        Y_test = Y_df.loc[Y_df["ds"] > cutoff].copy()
        sf = StatsForecast(
            models=[AutoARIMA(season_length=12)], freq=str(config.get("freq", "MS")), n_jobs=1
        )
        base_fc = sf.forecast(h=horizon, df=Y_train)
        base_fc = base_fc.rename(columns={"AutoARIMA": "base_forecast"})
        base_eval = Y_test.merge(base_fc, on=["unique_id", "ds"], how="left")
        level_lookup: dict[str, str] = {"portfolio": "portfolio"}
        for level_name, ids in tags.items():
            for unique_id in ids.tolist():
                level_lookup[str(unique_id)] = "grade" if level_name == "grade" else "grade_term"
        for series_level, group in base_eval.groupby(
            base_eval["unique_id"].map(level_lookup), sort=True
        ):
            truth = group["y"].to_numpy(dtype=float)
            pred = group["base_forecast"].to_numpy(dtype=float)
            rows.append(
                {
                    "target": target_col,
                    "method": "base",
                    "series_level": str(series_level),
                    "mae": float(np.mean(np.abs(truth - pred))),
                    "abs_bias": float(abs(np.mean(pred - truth))),
                    "n_obs": int(len(group)),
                }
            )

        selected_reconcilers = [
            reconcilers_map[name] for name in methods if name in reconcilers_map
        ]
        if not selected_reconcilers:
            summary["targets"][target_col] = {"available": False, "reason": "no_reconcilers"}
            continue
        reconciler = HierarchicalReconciliation(reconcilers=selected_reconcilers)
        reconciled = reconciler.reconcile(
            Y_hat_df=base_fc.rename(columns={"base_forecast": "AutoARIMA"}),
            Y_df=Y_train,
            S_df=S_df,
            tags=tags,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
        )
        for method_name in methods:
            recon_col = f"AutoARIMA/{method_name}"
            if recon_col not in reconciled.columns:
                continue
            eval_df = Y_test.merge(
                reconciled[["unique_id", "ds", recon_col]],
                on=["unique_id", "ds"],
                how="left",
            )
            for series_level, group in eval_df.groupby(
                eval_df["unique_id"].map(level_lookup), sort=True
            ):
                truth = group["y"].to_numpy(dtype=float)
                pred = group[recon_col].to_numpy(dtype=float)
                rows.append(
                    {
                        "target": target_col,
                        "method": method_name,
                        "series_level": str(series_level),
                        "mae": float(np.mean(np.abs(truth - pred))),
                        "abs_bias": float(abs(np.mean(pred - truth))),
                        "n_obs": int(len(group)),
                    }
                )
        target_rows = [row for row in rows if row["target"] == target_col]
        target_df = pd.DataFrame(target_rows)
        summary["targets"][target_col] = (
            target_df.sort_values(["series_level", "mae"])
            .groupby("series_level")
            .first()
            .reset_index()
            .to_dict(orient="records")
            if not target_df.empty
            else []
        )

    report = pd.DataFrame(rows)
    if report.empty:
        return report, {"available": False, "reason": "no_hierarchy_rows"}
    best = (
        report.groupby(["target", "series_level", "method"], as_index=False)
        .agg(mae=("mae", "mean"), abs_bias=("abs_bias", "mean"))
        .sort_values(["target", "series_level", "mae", "abs_bias"])
    )
    summary["best_by_level"] = (
        best.groupby(["target", "series_level"]).first().reset_index().to_dict(orient="records")
    )
    return report.sort_values(["target", "series_level", "method"]).reset_index(drop=True), summary


def build_status_payload(
    *,
    config: dict[str, Any],
    metrics: pd.DataFrame,
    champions: dict[str, Any],
    diagnostics: dict[str, Any],
    panel_status: dict[str, Any],
    future_covariates: pd.DataFrame | None,
    residual_predictions: pd.DataFrame,
    artifacts: dict[str, str],
    run_tag: str = "untracked",
) -> dict[str, Any]:
    warnings: list[str] = []
    if not bool(champions["point"]["promotable"]):
        warnings.append("point_champion_not_promotable")
    if not bool(champions["interval"]["promotable"]):
        warnings.append("interval_champion_not_promotable")
    exogenous_enabled = bool(config.get("exogenous", {}).get("enabled", False))
    if future_covariates is None or future_covariates.empty:
        warnings.append(
            "future_exogenous_contract_missing"
            if exogenous_enabled
            else "exogenous_available_but_disabled_by_policy"
        )

    status = "pass"
    if warnings:
        status = "warn" if "point_champion_not_promotable" not in warnings else "fail"

    residual_drift = {}
    if not residual_predictions.empty:
        champion_model = str(champions["point"]["model"])
        champ = residual_predictions.loc[residual_predictions["model"] == champion_model].copy()
        if len(champ) >= 24:
            champ["residual"] = champ["y_true"] - champ["y_pred"]
            midpoint = len(champ) // 2
            early = champ["residual"].iloc[:midpoint].to_numpy(dtype=float)
            late = champ["residual"].iloc[midpoint:].to_numpy(dtype=float)
            drift = ks_two_sample_test(early, late)
            residual_drift = {
                "ks_statistic": float(drift["ks_statistic"]),
                "p_value": float(drift["ks_pvalue"]),
                "early_mean": float(np.mean(early)),
                "late_mean": float(np.mean(late)),
            }

    return {
        "schema_version": "2026-03-07.1",
        "run_tag": run_tag,
        "status": status,
        "warnings": warnings,
        "config": {
            "horizon": int(config["horizon"]),
            "freq": config.get("freq", "MS"),
            "season_length": int(config.get("season_length", 12)),
            "rolling_origin": config.get("rolling_origin", {}),
            "exogenous_enabled": exogenous_enabled,
        },
        "summary": {
            "n_models_evaluated": int(metrics["model"].nunique()) if not metrics.empty else 0,
            "n_backtest_rows": int(len(residual_predictions)),
            "point_model": champions["point"]["model"],
            "interval_model": champions["interval"]["model"],
            "point_promotable": bool(champions["point"]["promotable"]),
            "interval_promotable": bool(champions["interval"]["promotable"]),
            "recent_actual_mean_12m": diagnostics.get("recent_actual_mean_12m"),
        },
        "point_champion": champions["point"],
        "interval_champion": champions["interval"],
        "panel_global_model": panel_status,
        "diagnostics": {
            "forecastability": diagnostics,
            "residual_drift": residual_drift,
        },
        "research_backlog": list(config.get("research_backlog", [])),
        "artifacts": artifacts,
    }


def infer_run_tag(default: str = "untracked") -> str:
    candidates: list[str | None] = [str(os.environ.get("PIPELINE_RUN_TAG", "")).strip() or None]
    pipeline_summary = Path("data/processed/pipeline_summary.json")
    if pipeline_summary.exists():
        try:
            payload = json.loads(pipeline_summary.read_text(encoding="utf-8"))
            candidates.append(str(payload.get("run_tag", "")).strip() or None)
        except Exception:
            pass
    return resolve_run_tag(
        fallback_candidates=candidates,
        require_explicit=not bool(str(default).strip()),
        allow_untracked=bool(str(default).strip()),
    )


__all__ = [
    "append_weighted_point_ensemble",
    "benchmark_mapie_time_series_intervals",
    "build_backtest_cutoffs",
    "build_canonical_forecast_frame",
    "build_future_covariates_contract",
    "build_ifrs9_temporal_scenarios",
    "build_minimal_macro_covariates",
    "build_status_payload",
    "compute_forecastability_diagnostics",
    "compute_forecastability_report",
    "compute_point_ensemble_weights",
    "compute_revision_metrics",
    "diebold_mariano_test",
    "evaluate_hierarchical_reconciliation",
    "fit_sarimax_forecaster",
    "fit_stl_catboost_forecaster",
    "forecast_panel_bottom_up",
    "forecast_portfolio_models",
    "infer_run_tag",
    "load_future_covariates",
    "load_time_series_config",
    "run_portfolio_backtest",
    "select_time_series_champions",
    "train_baseline_forecasters",
    "train_ml_forecaster",
]
