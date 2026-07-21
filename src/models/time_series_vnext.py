"""Research-only time-series vNext helpers for enriched temporal benchmarking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from src.evaluation.backtesting import winkler_interval_score
from src.evaluation.metrics import forecast_backtest_metrics
from src.models.time_series import (
    _build_exogenous_feature_frame,
    _extract_interval_columns,
    _pinball_loss,
    _require_ts_columns,
    _safe_float,
    _seasonal_error_scales,
    _weighted_interval_score,
    benchmark_mapie_time_series_intervals,
    build_backtest_cutoffs,
    build_canonical_forecast_frame,
    build_ifrs9_temporal_scenarios,
    compute_revision_metrics,
    diebold_mariano_test,
    fit_sarimax_forecaster,
    fit_stl_catboost_forecaster,
    forecast_panel_bottom_up,
    train_baseline_forecasters,
)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    column: str
    transform: str = "identity"
    lower_bound: float = 1e-6
    upper_bound: float = 1.0 - 1e-6


def target_specs_from_config(config: dict[str, Any]) -> list[TargetSpec]:
    candidates = list((config.get("targets", {}) or {}).get("candidates", []))
    if not candidates:
        return [
            TargetSpec(name="raw_rate", column="y", transform="identity"),
            TargetSpec(name="logit_rate", column="y_logit", transform="logit"),
        ]
    specs: list[TargetSpec] = []
    for row in candidates:
        specs.append(
            TargetSpec(
                name=str(row.get("name", row.get("column", "target"))),
                column=str(row.get("column", "y")),
                transform=str(row.get("transform", "identity")),
                lower_bound=float(row.get("lower_bound", 1e-6)),
                upper_bound=float(row.get("upper_bound", 1.0 - 1e-6)),
            )
        )
    return specs


def _clip_probability(values: np.ndarray, spec: TargetSpec) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), spec.lower_bound, spec.upper_bound)


def inverse_transform_array(values: np.ndarray | pd.Series, spec: TargetSpec) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if spec.transform == "logit":
        return 1.0 / (1.0 + np.exp(-arr))
    return arr


def _target_history(
    portfolio_history: pd.DataFrame, spec: TargetSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = portfolio_history.loc[:, ["unique_id", "ds", "y"]].copy()
    raw["ds"] = pd.to_datetime(raw["ds"], errors="coerce")
    raw["y"] = pd.to_numeric(raw["y"], errors="coerce")
    model = portfolio_history.loc[:, ["unique_id", "ds", spec.column]].copy()
    model = model.rename(columns={spec.column: "y"})
    model["ds"] = pd.to_datetime(model["ds"], errors="coerce")
    model["y"] = pd.to_numeric(model["y"], errors="coerce")
    return _require_ts_columns(model), _require_ts_columns(raw)


def _window_metrics_raw(
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


def _inverse_forecast_columns(
    frame: pd.DataFrame,
    base_model_names: list[str],
    spec: TargetSpec,
) -> pd.DataFrame:
    if spec.transform == "identity":
        out = frame.copy()
        numeric_cols = [col for col in out.columns if col not in {"unique_id", "ds"}]
        if numeric_cols:
            out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
        return out

    out = frame.copy()
    for model_name in base_model_names:
        cols = [
            model_name,
            f"{model_name}-lo-90",
            f"{model_name}-hi-90",
            f"{model_name}-lo-95",
            f"{model_name}-hi-95",
        ]
        for col in cols:
            if col in out.columns:
                out[col] = inverse_transform_array(out[col], spec)
    return out


def _model_family(model_name: str) -> tuple[str, str]:
    lower = model_name.lower()
    if lower.startswith("mapie_"):
        return "adaptive", "adaptive"
    if lower in {"sarimax", "stl_catboost"}:
        return "challenger", "challenger"
    return "statistical", "native_statistical"


def run_portfolio_backtest_vnext(
    portfolio_history: pd.DataFrame,
    config: dict[str, Any],
    target_spec: TargetSpec,
    *,
    future_covariates: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_history, raw_history = _target_history(portfolio_history, target_spec)
    horizon = int(config["horizon"])
    season_length = int(config.get("season_length", 12))
    roll_cfg = config.get("rolling_origin", {})
    cutoffs = build_backtest_cutoffs(
        model_history["ds"],
        horizon=horizon,
        min_train_periods=int(roll_cfg.get("min_train_periods", 72)),
        step_months=int(roll_cfg.get("step_months", 1)),
        max_windows=roll_cfg.get("max_windows"),
    )

    prediction_rows: list[dict[str, Any]] = []
    cutoff_metric_rows: list[dict[str, Any]] = []

    for cutoff in cutoffs:
        train_model = model_history.loc[
            model_history["ds"] <= cutoff, ["unique_id", "ds", "y"]
        ].copy()
        train_raw = raw_history.loc[raw_history["ds"] <= cutoff, ["unique_id", "ds", "y"]].copy()
        actual_raw = raw_history.loc[raw_history["ds"] > cutoff].head(horizon).copy()
        if len(actual_raw) < horizon:
            continue

        season_mae_scale, season_rmse_scale = _seasonal_error_scales(
            train_raw["y"].to_numpy(dtype=float),
            season_length,
        )

        forecasts_by_model: dict[str, pd.DataFrame] = {}
        statistical_models = list(config.get("models", {}).get("statistical", []))
        try:
            _, stats_fc = train_baseline_forecasters(
                train_model,
                horizon=horizon,
                freq=config.get("freq", "MS"),
                model_names=statistical_models,
            )
            stats_fc = _inverse_forecast_columns(stats_fc, statistical_models, target_spec)
            for model_name in statistical_models:
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
            logger.warning(
                "vNext statistical window forecast failed at {} [{}]: {}",
                cutoff,
                target_spec.name,
                exc,
            )

        exog_train = (
            _build_exogenous_feature_frame(train_model["ds"], future_covariates=future_covariates)
            .set_index("ds")
            .reindex(train_model["ds"])
        )
        exog_future = (
            _build_exogenous_feature_frame(actual_raw["ds"], future_covariates=future_covariates)
            .set_index("ds")
            .reindex(actual_raw["ds"])
        )
        try:
            sarimax_fc = fit_sarimax_forecaster(
                train_model,
                horizon=horizon,
                exog_train=exog_train if not exog_train.empty else None,
                exog_future=exog_future if not exog_future.empty else None,
                season_length=season_length,
            )
            forecasts_by_model["SARIMAX"] = _inverse_forecast_columns(
                sarimax_fc,
                ["SARIMAX"],
                target_spec,
            )
        except Exception as exc:
            logger.warning(
                "vNext SARIMAX window forecast failed at {} [{}]: {}", cutoff, target_spec.name, exc
            )
        try:
            stl_fc = fit_stl_catboost_forecaster(
                train_model,
                horizon=horizon,
                season_length=season_length,
                exog_train=exog_train if not exog_train.empty else None,
                exog_future=exog_future if not exog_future.empty else None,
            )
            forecasts_by_model["STL_CatBoost"] = _inverse_forecast_columns(
                stl_fc,
                ["STL_CatBoost"],
                target_spec,
            )
        except Exception as exc:
            logger.warning(
                "vNext STL_CatBoost window forecast failed at {} [{}]: {}",
                cutoff,
                target_spec.name,
                exc,
            )

        for model_name, fc in forecasts_by_model.items():
            merged = actual_raw[["ds", "y"]].merge(fc, on="ds", how="left")
            if model_name not in merged.columns or merged[model_name].isna().all():
                continue
            y_true = merged["y"].to_numpy(dtype=float)
            y_pred = np.asarray(merged[model_name], dtype=float)
            lo90, hi90, lo95, hi95 = _extract_interval_columns(merged, model_name)
            metrics = _window_metrics_raw(
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
                    "target_variant": target_spec.name,
                    "n_obs": int(len(y_true)),
                    **metrics,
                }
            )
            family, subfamily = _model_family(model_name)
            for horizon_step, row in enumerate(merged.itertuples(index=False), start=1):
                prediction_rows.append(
                    {
                        "cutoff": pd.Timestamp(cutoff),
                        "ds": pd.Timestamp(row.ds),
                        "horizon_step": int(horizon_step),
                        "unique_id": "portfolio",
                        "model": model_name,
                        "target_variant": target_spec.name,
                        "family": family,
                        "interval_subfamily": subfamily,
                        "y_true": float(row.y),
                        "y_pred": float(getattr(row, model_name)),
                        "lo_90": _safe_float(getattr(row, f"{model_name}-lo-90", np.nan)),
                        "hi_90": _safe_float(getattr(row, f"{model_name}-hi-90", np.nan)),
                        "lo_95": _safe_float(getattr(row, f"{model_name}-lo-95", np.nan)),
                        "hi_95": _safe_float(getattr(row, f"{model_name}-hi-95", np.nan)),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        return predictions, pd.DataFrame()
    predictions = predictions.sort_values(
        ["target_variant", "model", "cutoff", "horizon_step"]
    ).reset_index(drop=True)

    cutoff_metrics = pd.DataFrame(cutoff_metric_rows)
    cutoff_metrics = cutoff_metrics.sort_values(["target_variant", "model", "cutoff"]).reset_index(
        drop=True
    )
    summary = (
        cutoff_metrics.groupby(["target_variant", "model"], as_index=False)
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
    revision_rows: list[dict[str, Any]] = []
    for (target_name, model_name), group in predictions.groupby(
        ["target_variant", "model"], sort=True
    ):
        model_revision = compute_revision_metrics(group.loc[:, ["model", "cutoff", "ds", "y_pred"]])
        if model_revision.empty:
            continue
        row = model_revision.iloc[0].to_dict()
        row["target_variant"] = target_name
        row["model"] = model_name
        revision_rows.append(row)
    if revision_rows:
        summary = summary.merge(
            pd.DataFrame(revision_rows), on=["target_variant", "model"], how="left"
        )
    for target_name, target_df in summary.groupby("target_variant", sort=False):
        baseline_row = target_df.loc[target_df["model"] == "SeasonalNaive"]
        if baseline_row.empty:
            continue
        baseline_mae = float(baseline_row["mae"].iloc[0])
        baseline_rmsse = float(baseline_row["rmsse"].iloc[0])
        mask = summary["target_variant"] == target_name
        summary.loc[mask, "fva_mae_abs"] = baseline_mae - summary.loc[mask, "mae"]
        summary.loc[mask, "fva_mae_pct"] = 1.0 - (
            summary.loc[mask, "mae"] / max(baseline_mae, 1e-8)
        )
        summary.loc[mask, "rmsse_vs_seasonal_naive"] = summary.loc[mask, "rmsse"] / max(
            baseline_rmsse, 1e-8
        )
        baseline_preds = predictions.loc[
            (predictions["target_variant"] == target_name)
            & (predictions["model"] == "SeasonalNaive"),
            ["cutoff", "ds", "horizon_step", "y_true", "y_pred"],
        ].copy()
        baseline_preds["baseline_loss"] = np.abs(
            baseline_preds["y_true"].to_numpy(dtype=float)
            - baseline_preds["y_pred"].to_numpy(dtype=float)
        )
        dm_rows: list[dict[str, Any]] = []
        for model_name in summary.loc[mask, "model"]:
            model_preds = predictions.loc[
                (predictions["target_variant"] == target_name)
                & (predictions["model"] == model_name),
                ["cutoff", "ds", "horizon_step", "y_true", "y_pred"],
            ].copy()
            model_preds["model_loss"] = np.abs(
                model_preds["y_true"].to_numpy(dtype=float)
                - model_preds["y_pred"].to_numpy(dtype=float)
            )
            merged = model_preds.merge(
                baseline_preds[["cutoff", "ds", "horizon_step", "baseline_loss"]],
                on=["cutoff", "ds", "horizon_step"],
                how="inner",
            )
            if merged.empty:
                continue
            dm = diebold_mariano_test(
                merged["model_loss"].to_numpy(dtype=float),
                merged["baseline_loss"].to_numpy(dtype=float),
                lag=horizon,
            )
            dm_rows.append(
                {
                    "target_variant": target_name,
                    "model": model_name,
                    "dm_stat_vs_seasonal_naive": float(dm["dm_stat"]),
                    "dm_pvalue_vs_seasonal_naive": float(dm["p_value"]),
                    "dm_reject_vs_seasonal_naive": bool(dm["reject"]),
                }
            )
        if dm_rows:
            summary = summary.merge(
                pd.DataFrame(dm_rows), on=["target_variant", "model"], how="left"
            )

    summary["coverage_gap_90"] = (summary["coverage_90"] - 0.90).abs()
    summary["coverage_gap_95"] = (summary["coverage_95"] - 0.95).abs()
    summary["family"] = summary["model"].map(lambda name: _model_family(name)[0])
    summary["interval_subfamily"] = summary["model"].map(lambda name: _model_family(name)[1])
    summary["point_eligible"] = True
    summary["interval_eligible"] = True
    return predictions, summary.sort_values(
        ["target_variant", "mase", "rmsse", "abs_bias"]
    ).reset_index(drop=True)


def forecast_portfolio_models_vnext(
    portfolio_history: pd.DataFrame,
    config: dict[str, Any],
    target_spec: TargetSpec,
    *,
    future_covariates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    model_history, _ = _target_history(portfolio_history, target_spec)
    horizon = int(config["horizon"])
    season_length = int(config.get("season_length", 12))

    statistical_models = list(config.get("models", {}).get("statistical", []))
    forecasts: list[pd.DataFrame] = []
    _, stats_fc = train_baseline_forecasters(
        model_history,
        horizon=horizon,
        freq=config.get("freq", "MS"),
        model_names=statistical_models,
    )
    forecasts.append(_inverse_forecast_columns(stats_fc, statistical_models, target_spec))

    future_dates = pd.date_range(
        model_history["ds"].iloc[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    exog_train = (
        _build_exogenous_feature_frame(model_history["ds"], future_covariates=future_covariates)
        .set_index("ds")
        .reindex(model_history["ds"])
    )
    exog_future = (
        _build_exogenous_feature_frame(future_dates, future_covariates=future_covariates)
        .set_index("ds")
        .reindex(future_dates)
    )
    try:
        sarimax_fc = fit_sarimax_forecaster(
            model_history,
            horizon=horizon,
            exog_train=exog_train if not exog_train.empty else None,
            exog_future=exog_future if not exog_future.empty else None,
            season_length=season_length,
        )
        forecasts.append(_inverse_forecast_columns(sarimax_fc, ["SARIMAX"], target_spec))
    except Exception as exc:
        logger.warning("vNext final SARIMAX forecast failed [{}]: {}", target_spec.name, exc)
    try:
        stl_fc = fit_stl_catboost_forecaster(
            model_history,
            horizon=horizon,
            season_length=season_length,
            exog_train=exog_train if not exog_train.empty else None,
            exog_future=exog_future if not exog_future.empty else None,
        )
        forecasts.append(_inverse_forecast_columns(stl_fc, ["STL_CatBoost"], target_spec))
    except Exception as exc:
        logger.warning("vNext final STL_CatBoost forecast failed [{}]: {}", target_spec.name, exc)

    merged = forecasts[0]
    for frame in forecasts[1:]:
        merged = merged.merge(frame, on=["unique_id", "ds"], how="outer")
    return merged.sort_values("ds").reset_index(drop=True)


def benchmark_mapie_time_series_intervals_vnext(
    portfolio_history: pd.DataFrame,
    target_spec: TargetSpec,
    config: dict[str, Any],
) -> dict[str, Any]:
    if target_spec.transform == "identity":
        return benchmark_mapie_time_series_intervals(
            portfolio_history.loc[:, ["unique_id", "ds", "y"]],
            confidence_level=0.90,
            evaluation_size=int(
                (config.get("interval_policy", {}) or {}).get(
                    "mapie_evaluation_size",
                    max(int(config.get("horizon", 12)), 24),
                )
            ),
            estimator_name=str(
                (config.get("interval_policy", {}) or {}).get("mapie_estimator", "linear")
            ),
        )

    from mapie.regression import TimeSeriesRegressor
    from mapie.subsample import BlockBootstrap

    model_history, raw_history = _target_history(portfolio_history, target_spec)
    frame = model_history.sort_values("ds").reset_index(drop=True)
    raw = raw_history.sort_values("ds").reset_index(drop=True)
    for lag in (1, 2, 3, 6, 12):
        frame[f"lag_{lag}"] = frame["y"].shift(lag)
    frame = frame.dropna().reset_index(drop=True)
    raw = raw.loc[raw["ds"].isin(frame["ds"])].reset_index(drop=True)
    min_train = max(60, int(len(frame) * 0.60))
    eval_rows = min(
        max(int((config.get("interval_policy", {}) or {}).get("mapie_evaluation_size", 24)), 12),
        max(len(frame) - min_train, 0),
    )
    if eval_rows <= 0:
        return {
            "available": False,
            "reason": "insufficient_history_for_mapie_benchmark",
            "candidate_methods_tested": [],
        }
    feature_cols = [col for col in frame.columns if col not in {"unique_id", "ds", "y"}]
    train = frame.iloc[:-eval_rows].reset_index(drop=True)
    evaluation = frame.iloc[-eval_rows:].reset_index(drop=True)
    evaluation_raw = raw.iloc[-eval_rows:].reset_index(drop=True)
    from sklearn.linear_model import LinearRegression

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
            estimator=LinearRegression(),
            method=method,
            cv=cv,
            random_state=42,
        )
        X_train = train[feature_cols].to_numpy(dtype=float)
        y_train = train["y"].to_numpy(dtype=float)
        regressor.fit(X_train, y_train)
        rows: list[dict[str, Any]] = []
        for eval_row, eval_raw in zip(
            evaluation.itertuples(index=False), evaluation_raw.itertuples(index=False), strict=False
        ):
            X_row = np.asarray([[getattr(eval_row, col) for col in feature_cols]], dtype=float)
            y_pred_t, intervals_raw = regressor.predict(X_row, confidence_level=0.90)
            bounds = np.asarray(intervals_raw[:, :, 0], dtype=float)
            low = inverse_transform_array(bounds[:, 0], target_spec)[0]
            high = inverse_transform_array(bounds[:, 1], target_spec)[0]
            y_pred = inverse_transform_array(np.asarray(y_pred_t, dtype=float), target_spec)[0]
            y_true = float(eval_raw.y)
            rows.append(
                {
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "lo_90": low,
                    "hi_90": high,
                    "covered": float(low <= y_true <= high),
                    "width": float(high - low),
                    "winkler_90": float(
                        winkler_interval_score(
                            np.asarray([y_true]),
                            np.asarray([low]),
                            np.asarray([high]),
                            alpha=0.10,
                        )[0]
                    ),
                }
            )
            if method == "aci":
                regressor.adapt_conformal_inference(
                    X_row,
                    np.asarray([eval_row.y], dtype=float),
                    gamma=0.01,
                    confidence_level=0.90,
                )
            else:
                regressor.update(
                    X_row,
                    np.asarray([eval_row.y], dtype=float),
                    confidence_level=0.90,
                )
        if not rows:
            continue
        result_df = pd.DataFrame(rows)
        results.append(
            {
                "method": method,
                "n_eval": int(len(result_df)),
                "coverage_90": float(result_df["covered"].mean()),
                "coverage_gap_90": float(abs(result_df["covered"].mean() - 0.90)),
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
                        alpha=0.10,
                    )
                ),
            }
        )
    if not results:
        return {
            "available": False,
            "reason": "mapie_benchmark_unavailable",
            "candidate_methods_tested": [],
        }
    results_df = pd.DataFrame(results).sort_values(
        ["coverage_gap_90", "wis_90", "winkler_90", "avg_interval_width_90"]
    )
    best = results_df.iloc[0].to_dict()
    return {
        "available": True,
        "best_method": str(best.get("method", "")),
        "candidate_methods_tested": results_df["method"].astype(str).tolist(),
        "results": results_df.to_dict(orient="records"),
    }


def evaluate_target_champions(
    backtest_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    if backtest_metrics.empty:
        raise ValueError("Backtest metrics are empty for vNext evaluation.")
    point_cfg = config.get("point_champion", {})
    interval_cfg = config.get("interval_policy", {})
    by_target: dict[str, Any] = {}
    for target_name, target_df in backtest_metrics.groupby("target_variant", sort=True):
        point_df = target_df.loc[
            target_df["point_eligible"].astype(bool)
            if "point_eligible" in target_df.columns
            else slice(None)
        ].copy()
        if point_df.empty:
            point_df = target_df.copy()
        point_sort = point_df.sort_values(["mase", "rmsse", "abs_bias", "mae"]).reset_index(
            drop=True
        )
        point_row = point_sort.iloc[0]
        baseline_row = target_df.loc[target_df["model"] == "SeasonalNaive"]
        baseline_mae = float(baseline_row["mae"].iloc[0]) if not baseline_row.empty else math.inf
        baseline_rmsse = (
            float(baseline_row["rmsse"].iloc[0]) if not baseline_row.empty else math.inf
        )
        point_promotable = True
        point_reasons: list[str] = []
        if (
            bool(point_cfg.get("must_beat_seasonal_naive", True))
            and float(point_row["mae"]) >= baseline_mae
        ):
            point_promotable = False
            point_reasons.append("mae_not_better_than_seasonal_naive")
        max_rmsse = float(point_cfg.get("max_rmsse_vs_seasonal_naive", 1.0))
        if (
            baseline_rmsse < math.inf
            and float(point_row["rmsse"]) / max(baseline_rmsse, 1e-8) > max_rmsse
        ):
            point_promotable = False
            point_reasons.append("rmsse_worse_than_allowed_vs_seasonal_naive")

        interval_pool = target_df.loc[
            target_df["interval_eligible"].astype(bool)
            if "interval_eligible" in target_df.columns
            else slice(None)
        ].copy()
        if interval_pool.empty:
            interval_pool = target_df.copy()
        interval_candidates = interval_pool.loc[
            interval_pool["coverage_gap_90"] <= float(interval_cfg.get("max_coverage_gap", 0.03))
        ].copy()
        eligible_families = [
            str(family).strip().lower()
            for family in interval_cfg.get(
                "eligible_interval_families",
                [interval_cfg.get("official_family", "statistical")],
            )
            if str(family).strip()
        ]
        if eligible_families and not interval_candidates.empty:
            preferred = interval_candidates.loc[
                interval_candidates["family"].astype(str).str.lower().isin(eligible_families)
            ]
            if not preferred.empty:
                interval_candidates = preferred
        if interval_candidates.empty:
            fallback = interval_pool.copy()
            if eligible_families:
                preferred = fallback.loc[
                    fallback["family"].astype(str).str.lower().isin(eligible_families)
                ]
                if not preferred.empty:
                    fallback = preferred
            interval_row = fallback.sort_values(
                ["coverage_gap_90", "wis_90", "winkler_90", "avg_interval_width_90"]
            ).iloc[0]
            interval_promotable = False
            interval_reasons = ["no_model_within_coverage_gap_policy"]
        else:
            interval_row = interval_candidates.sort_values(
                ["coverage_gap_90", "wis_90", "winkler_90", "avg_interval_width_90"]
            ).iloc[0]
            interval_promotable = True
            interval_reasons = []
        by_target[target_name] = {
            "point": {
                "model": str(point_row["model"]),
                "target_variant": target_name,
                "promotable": bool(point_promotable),
                "mae": float(point_row["mae"]),
                "mase": float(point_row["mase"]),
                "rmsse": float(point_row["rmsse"]),
                "abs_bias": float(point_row["abs_bias"]),
                "reasons": point_reasons,
            },
            "interval": {
                "model": str(interval_row["model"]),
                "target_variant": target_name,
                "promotable": bool(interval_promotable),
                "coverage_90": _safe_float(interval_row.get("coverage_90")),
                "coverage_gap_90": _safe_float(interval_row.get("coverage_gap_90")),
                "avg_interval_width_90": _safe_float(interval_row.get("avg_interval_width_90")),
                "winkler_90": _safe_float(interval_row.get("winkler_90")),
                "wis_90": _safe_float(interval_row.get("wis_90")),
                "pinball_90": _safe_float(interval_row.get("pinball_90")),
                "family": str(interval_row.get("family", "")),
                "interval_subfamily": str(interval_row.get("interval_subfamily", "")),
                "reasons": interval_reasons,
            },
            "baseline": {
                "model": "SeasonalNaive",
                "mae": baseline_mae,
                "rmsse": baseline_rmsse,
            },
        }

    point_choices = pd.DataFrame([payload["point"] for payload in by_target.values()])
    point_choices["_promotable_rank"] = (~point_choices["promotable"].astype(bool)).astype(int)
    best_point = (
        point_choices.sort_values(["_promotable_rank", "mase", "rmsse", "abs_bias", "mae"])
        .iloc[0]
        .drop(labels="_promotable_rank")
        .to_dict()
    )
    chosen_target = str(best_point["target_variant"])
    best_interval = by_target[chosen_target]["interval"]
    return {
        "by_target": by_target,
        "selected_target_variant": chosen_target,
        "point": best_point,
        "interval": best_interval,
    }


def _build_residual_matrix(backtest_predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    subset = backtest_predictions.loc[backtest_predictions["model"] == model_name].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["residual"] = subset["y_true"].astype(float) - subset["y_pred"].astype(float)
    return subset.pivot_table(
        index="cutoff", columns="horizon_step", values="residual", aggfunc="mean"
    ).sort_index()


def _build_truth_matrix(backtest_predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    subset = backtest_predictions.loc[backtest_predictions["model"] == model_name].copy()
    if subset.empty:
        return pd.DataFrame()
    return subset.pivot_table(
        index="cutoff", columns="horizon_step", values="y_true", aggfunc="mean"
    ).sort_index()


def generate_joint_sample_paths(
    canonical_forecasts: pd.DataFrame,
    backtest_predictions: pd.DataFrame,
    *,
    point_model: str,
    n_samples: int = 512,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    point = canonical_forecasts["y"].to_numpy(dtype=float)
    sigma = (
        canonical_forecasts["y_hi_90"].to_numpy(dtype=float)
        - canonical_forecasts["y_lo_90"].to_numpy(dtype=float)
    ) / (2.0 * 1.6448536269514722)
    sigma = np.clip(sigma, 1e-4, None)
    horizon = len(point)
    residual_matrix = _build_residual_matrix(backtest_predictions, point_model)
    truth_matrix = _build_truth_matrix(backtest_predictions, point_model)
    if residual_matrix.empty:
        residual_matrix = pd.DataFrame(np.eye(horizon))
    residual_matrix = residual_matrix.reindex(columns=range(1, horizon + 1), fill_value=0.0)
    truth_matrix = truth_matrix.reindex(columns=range(1, horizon + 1), fill_value=np.nan)
    history = (
        backtest_predictions.loc[backtest_predictions["model"] == point_model, ["ds", "y_true"]]
        .drop_duplicates(subset=["ds"])
        .sort_values("ds")
    )
    history_values = history["y_true"].to_numpy(dtype=float)
    if len(history_values) >= 3:
        diff_series = np.diff(history_values)
        rho = float(pd.Series(diff_series).autocorr(lag=1))
        if not math.isfinite(rho):
            rho = 0.0
    else:
        rho = 0.0
    rho = float(np.clip(rho, -0.95, 0.95))
    distances = np.abs(np.subtract.outer(np.arange(horizon), np.arange(horizon)))
    corr = np.power(rho, distances, dtype=float)
    corr += np.eye(horizon) * 1e-6
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal(size=(n_samples, horizon)) @ chol.T
    gaussian_u = stats.norm.cdf(z)
    gaussian_paths = np.clip(stats.norm.ppf(gaussian_u, loc=point, scale=sigma), 0.0, 1.0)

    indep_z = rng.standard_normal(size=(n_samples, horizon))
    indep_u = stats.norm.cdf(indep_z)
    indep = np.clip(stats.norm.ppf(indep_u, loc=point, scale=sigma), 0.0, 1.0)
    hist = truth_matrix.dropna(how="any")
    if hist.empty:
        hist = pd.DataFrame(
            gaussian_paths[: min(len(gaussian_paths), 32)], columns=range(1, horizon + 1)
        )
    selected_hist = hist.sample(n=n_samples, replace=True, random_state=random_seed).to_numpy(
        dtype=float
    )
    schaake_paths = np.zeros_like(indep)
    for idx in range(horizon):
        sorted_samples = np.sort(indep[:, idx])
        ranks = np.argsort(np.argsort(selected_hist[:, idx]))
        schaake_paths[:, idx] = sorted_samples[ranks]
    methods = {
        "gaussian_copula": gaussian_paths,
        "schaake_shuffle": np.clip(schaake_paths, 0.0, 1.0),
    }
    rows: list[dict[str, Any]] = []
    samples: list[pd.DataFrame] = []
    recent_actual_mean = float(
        backtest_predictions.loc[backtest_predictions["model"] == point_model, "y_true"]
        .tail(12)
        .mean()
    )
    recent_actual_mean = (
        recent_actual_mean if math.isfinite(recent_actual_mean) else float(np.mean(point))
    )
    recent_threshold = recent_actual_mean * 1.10
    for method_name, paths in methods.items():
        path_sum = paths.sum(axis=1)
        path_mean = paths.mean(axis=1)
        consecutive = (paths > recent_threshold).astype(int)
        has_three = bool(
            np.any(
                np.apply_along_axis(
                    lambda arr: np.max(np.convolve(arr, np.ones(3, dtype=int), mode="valid")) >= 3,
                    1,
                    consecutive,
                )
            )
        )
        rows.append(
            {
                "method": method_name,
                "n_samples": int(paths.shape[0]),
                "estimated_ar1_rho": rho,
                "mean_path_sum": float(np.mean(path_sum)),
                "p05_path_sum": float(np.percentile(path_sum, 5)),
                "p50_path_sum": float(np.percentile(path_sum, 50)),
                "p95_path_sum": float(np.percentile(path_sum, 95)),
                "mean_path_avg": float(np.mean(path_mean)),
                "p95_path_avg": float(np.percentile(path_mean, 95)),
                "prob_avg_above_recent_x110": float(np.mean(path_mean > recent_threshold)),
                "prob_any_month_above_recent_x110": float(
                    np.mean(np.any(paths > recent_threshold, axis=1))
                ),
                "prob_three_consecutive_above_recent_x110": float(
                    np.mean(
                        np.apply_along_axis(
                            lambda arr: (
                                np.max(np.convolve(arr, np.ones(3, dtype=int), mode="valid")) >= 3
                            ),
                            1,
                            consecutive,
                        )
                    )
                ),
                "three_consecutive_signal_present": bool(has_three),
            }
        )
        sample_frame = pd.DataFrame(paths, columns=canonical_forecasts["ds"].astype(str).tolist())
        sample_frame.insert(0, "sample_id", np.arange(len(sample_frame), dtype=int))
        sample_frame.insert(0, "method", method_name)
        samples.append(sample_frame)
    return pd.DataFrame(rows), pd.concat(samples, ignore_index=True)


def build_policy_review(
    *,
    canonical_status: dict[str, Any],
    vnext_status: dict[str, Any],
    joint_path_eval: pd.DataFrame,
    ecl_eval: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    canonical_point = canonical_status.get("point_champion", {}) or {}
    canonical_interval = canonical_status.get("interval_champion", {}) or {}
    vnext_point = vnext_status.get("point_champion", {}) or {}
    vnext_interval = vnext_status.get("interval_champion", {}) or {}
    point_improved = (
        float(vnext_point.get("mae", math.inf)) + 1e-12
        < float(canonical_point.get("mae", math.inf)) * 0.99
    )
    interval_gap_improved = float(vnext_interval.get("coverage_gap_90", math.inf)) < float(
        canonical_interval.get("coverage_gap_90", math.inf)
    )
    sample_path_available = not joint_path_eval.empty
    ecl_range = (
        pd.to_numeric(ecl_eval["ecl_range_90"], errors="coerce").fillna(0.0)
        if not ecl_eval.empty and "ecl_range_90" in ecl_eval.columns
        else pd.Series(dtype=float)
    )
    ecl_artifact_stable = bool(not ecl_range.empty and ecl_range.ge(0).all())
    decision_rows = [
        {
            "component": "data_contract",
            "canonical_state": "baseline_portfolio_and_panel",
            "vnext_state": "enriched_internal_only",
            "recommendation": "research_only" if not point_improved else "promote",
            "rationale": "Promote only if the enriched contract improves or stabilizes the point layer.",
        },
        {
            "component": "point_forecast",
            "canonical_state": str(canonical_point.get("model", "unknown")),
            "vnext_state": str(vnext_point.get("model", "unknown")),
            "recommendation": (
                "promote"
                if bool(vnext_point.get("promotable", False)) and point_improved
                else "maintain"
            ),
            "rationale": (
                "vNext point champion clears the existing gate and improves MAE materially."
                if bool(vnext_point.get("promotable", False)) and point_improved
                else "Canonical point champion remains the operational reference."
            ),
        },
        {
            "component": "interval_layer",
            "canonical_state": str(canonical_interval.get("model", "unknown")),
            "vnext_state": str(vnext_interval.get("model", "unknown")),
            "recommendation": (
                "promote"
                if bool(vnext_interval.get("promotable", False))
                else ("research_only" if interval_gap_improved else "maintain")
            ),
            "rationale": (
                "Promote only if the interval layer passes governed coverage and width policy."
                if bool(vnext_interval.get("promotable", False))
                else (
                    "Coverage gap improves but still does not justify operational promotion."
                    if interval_gap_improved
                    else "vNext interval layer does not yet beat the canonical diagnostic baseline."
                )
            ),
        },
        {
            "component": "sample_paths",
            "canonical_state": "not_available",
            "vnext_state": "gaussian_copula_and_schaake_shuffle"
            if sample_path_available
            else "not_available",
            "recommendation": "research_only" if sample_path_available else "deprecate",
            "rationale": (
                "Joint paths add prudential insight but remain research until they prove operational value."
                if sample_path_available
                else "No stable sample-path evidence was produced."
            ),
        },
        {
            "component": "ts_ecl_artifact",
            "canonical_state": "legacy_ts_ecl_intervals",
            "vnext_state": "vnext_ts_ecl_intervals" if not ecl_eval.empty else "missing",
            "recommendation": "maintain" if ecl_artifact_stable else "deprecate",
            "rationale": (
                "The vNext ECL translation is numerically stable and can support further review."
                if ecl_artifact_stable
                else "The TS→ECL artifact remains too unstable or incomplete to retain."
            ),
        },
    ]
    decision_matrix = pd.DataFrame(decision_rows)
    overall_recommendation = (
        "promote_vnext"
        if "promote" in decision_matrix["recommendation"].tolist()
        and bool(vnext_interval.get("promotable", False))
        else "maintain_canonical_keep_vnext_research"
    )
    payload = {
        "overall_recommendation": overall_recommendation,
        "selected_target_variant": vnext_status.get("selected_target_variant"),
        "canonical_point_model": canonical_point.get("model"),
        "canonical_interval_model": canonical_interval.get("model"),
        "vnext_point_model": vnext_point.get("model"),
        "vnext_interval_model": vnext_interval.get("model"),
        "point_material_improvement": bool(point_improved),
        "interval_gap_improvement": bool(interval_gap_improved),
        "sample_path_available": bool(sample_path_available),
        "ecl_artifact_stable": bool(ecl_artifact_stable),
        "decision_matrix": decision_matrix.to_dict(orient="records"),
    }
    return payload, decision_matrix


__all__ = [
    "TargetSpec",
    "benchmark_mapie_time_series_intervals_vnext",
    "build_canonical_forecast_frame",
    "build_ifrs9_temporal_scenarios",
    "build_policy_review",
    "evaluate_target_champions",
    "forecast_panel_bottom_up",
    "forecast_portfolio_models_vnext",
    "generate_joint_sample_paths",
    "inverse_transform_array",
    "run_portfolio_backtest_vnext",
    "target_specs_from_config",
]
