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
    payload["interval_policy"].setdefault("ml_intervals_diagnostic_only_if_unpromoted", True)
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


def train_baseline_forecasters(
    df: pd.DataFrame,
    horizon: int = 12,
    freq: str = "MS",
    levels: list[int] | None = None,
) -> tuple[Any, pd.DataFrame]:
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
        mlf.fit(resid_df)
        resid_pred = mlf.predict(h=horizon)
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
                train, horizon=horizon, freq=config.get("freq", "MS")
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

        exog_train = pd.DataFrame()
        exog_future = pd.DataFrame()
        if future_covariates is not None and not future_covariates.empty:
            exog_cols = [col for col in future_covariates.columns if col != "ds"]
            exog_train = (
                future_covariates.loc[future_covariates["ds"] <= cutoff, ["ds", *exog_cols]]
                .set_index("ds")
                .reindex(train["ds"])
                .fillna(method="ffill")
                .fillna(method="bfill")
            )
            exog_future = (
                future_covariates.loc[
                    future_covariates["ds"].isin(actual["ds"]), ["ds", *exog_cols]
                ]
                .set_index("ds")
                .reindex(actual["ds"])
                .fillna(method="ffill")
                .fillna(method="bfill")
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
                train, horizon=horizon, season_length=season_length
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
    summary["family"] = np.where(
        summary["model"].isin(config.get("models", {}).get("statistical", [])),
        "statistical",
        "challenger",
    )
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

    point_sort = metrics.sort_values(["mase", "rmsse", "abs_bias", "mae"]).reset_index(drop=True)
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

    interval_candidates = metrics.loc[
        metrics["coverage_gap_90"] <= float(interval_cfg.get("max_coverage_gap", 0.03))
    ].copy()
    official_family = str(interval_cfg.get("official_family", "statistical")).strip().lower()
    if official_family:
        preferred = interval_candidates.loc[
            interval_candidates["family"].str.lower() == official_family
        ]
        if not preferred.empty:
            interval_candidates = preferred
    if interval_candidates.empty:
        fallback = metrics.loc[metrics["family"].str.lower() == official_family]
        interval_row = (
            fallback.sort_values(["coverage_gap_90", "winkler_90", "avg_interval_width_90"]).iloc[0]
            if not fallback.empty
            else metrics.sort_values(
                ["coverage_gap_90", "winkler_90", "avg_interval_width_90"]
            ).iloc[0]
        )
        interval_promotable = False
        interval_reasons = ["no_model_within_coverage_gap_policy"]
    else:
        interval_row = interval_candidates.sort_values(
            ["winkler_90", "avg_interval_width_90", "coverage_gap_90"]
        ).iloc[0]
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
            "avg_interval_width_90": _safe_float(interval_row.get("avg_interval_width_90")),
            "family": str(interval_row.get("family", "")),
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
) -> pd.DataFrame:
    ts = _require_ts_columns(history)
    horizon = int(config["horizon"])
    season_length = int(config.get("season_length", 12))

    forecasts: list[pd.DataFrame] = []
    _, stats_fc = train_baseline_forecasters(ts, horizon=horizon, freq=config.get("freq", "MS"))
    forecasts.append(stats_fc.copy())

    exog_train = pd.DataFrame()
    exog_future = pd.DataFrame()
    if future_covariates is not None and not future_covariates.empty:
        exog_cols = [col for col in future_covariates.columns if col != "ds"]
        exog_train = (
            future_covariates.loc[future_covariates["ds"].isin(ts["ds"]), ["ds", *exog_cols]]
            .set_index("ds")
            .reindex(ts["ds"])
            .fillna(method="ffill")
            .fillna(method="bfill")
        )
        future_dates = pd.date_range(
            ts["ds"].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
        )
        exog_future = (
            future_covariates.loc[future_covariates["ds"].isin(future_dates), ["ds", *exog_cols]]
            .set_index("ds")
            .reindex(future_dates)
            .fillna(method="ffill")
            .fillna(method="bfill")
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
            fit_stl_catboost_forecaster(ts, horizon=horizon, season_length=season_length)
        )
    except Exception as exc:
        logger.warning("Final STL_CatBoost forecast failed: {}", exc)

    merged = forecasts[0]
    for frame in forecasts[1:]:
        merged = merged.merge(frame, on=["unique_id", "ds"], how="outer")
    return merged.sort_values("ds").reset_index(drop=True)


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
    if future_covariates is None or future_covariates.empty:
        warnings.append("exogenous_disabled_or_missing_future_contract")

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
            "exogenous_enabled": bool(config.get("exogenous", {}).get("enabled", False)),
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
    env_value = str(os.environ.get("PIPELINE_RUN_TAG", "")).strip()
    if env_value:
        return env_value
    pipeline_summary = Path("data/processed/pipeline_summary.json")
    if pipeline_summary.exists():
        try:
            payload = json.loads(pipeline_summary.read_text(encoding="utf-8"))
            value = str(payload.get("run_tag", "")).strip()
            if value:
                return value
        except Exception:
            pass
    return default


__all__ = [
    "build_backtest_cutoffs",
    "build_canonical_forecast_frame",
    "build_ifrs9_temporal_scenarios",
    "build_status_payload",
    "compute_forecastability_diagnostics",
    "compute_revision_metrics",
    "diebold_mariano_test",
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
