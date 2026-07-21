"""Research-only TS/IFRS9 vNext lane.

Builds enriched internal-only vintage series, benchmarks point and interval models,
evaluates joint uncertainty via sample paths, translates the result into IFRS9-style
temporal scenarios, and writes a policy review without touching canonical artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.data.build_datasets import (
    build_time_series_panel_vnext,
    build_time_series_vnext,
    clean_raw_columns,
    load_historical_time_series_source,
)
from src.models.time_series import (
    build_ifrs9_temporal_scenarios,
    build_status_payload,
    compute_forecastability_diagnostics,
    compute_forecastability_report,
    evaluate_hierarchical_reconciliation,
    forecast_panel_bottom_up,
    infer_run_tag,
    load_future_covariates,
    load_time_series_config,
)
from src.models.time_series_vnext import (
    benchmark_mapie_time_series_intervals_vnext,
    build_canonical_forecast_frame,
    build_policy_review,
    evaluate_target_champions,
    forecast_portfolio_models_vnext,
    generate_joint_sample_paths,
    run_portfolio_backtest_vnext,
    target_specs_from_config,
)
from src.utils.pipeline_runtime import atomic_write_json, atomic_write_parquet

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")


def _resolve_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    outputs = config.get("outputs", {})
    return {key: Path(str(value)) for key, value in outputs.items()}


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_optional_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _load_history_vnext() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = clean_raw_columns(load_historical_time_series_source(DATA_DIR / "train.parquet"))
    portfolio = build_time_series_vnext(source)
    panel = build_time_series_panel_vnext(source)
    return portfolio, panel


def _build_adaptive_interval_rows(
    benchmark: dict[str, Any],
    *,
    target_variant: str,
) -> pd.DataFrame:
    rows = benchmark.get("results", []) if isinstance(benchmark, dict) else []
    if not rows:
        return pd.DataFrame()
    adaptive = pd.DataFrame(rows).rename(columns={"method": "model"})
    adaptive["model"] = adaptive["model"].astype(str).map(lambda value: f"MAPIE_{value.upper()}")
    adaptive["target_variant"] = target_variant
    adaptive["family"] = "adaptive"
    adaptive["interval_subfamily"] = "adaptive"
    adaptive["point_eligible"] = False
    adaptive["interval_eligible"] = True
    return adaptive


def _select_operational_interval(
    future_forecasts: pd.DataFrame,
    champions: dict[str, Any],
    interval_eval: pd.DataFrame,
) -> tuple[dict[str, Any], str | None]:
    preferred = dict(champions["interval"])
    model_name = str(preferred.get("model", ""))
    lower_col = f"{model_name}-lo-90"
    upper_col = f"{model_name}-hi-90"
    if lower_col in future_forecasts.columns and upper_col in future_forecasts.columns:
        return preferred, None

    selected_target = str(champions.get("selected_target_variant", ""))
    eligible = interval_eval.copy()
    if "target_variant" in eligible.columns:
        eligible = eligible.loc[eligible["target_variant"] == selected_target].copy()
    eligible = eligible.loc[
        eligible["model"]
        .astype(str)
        .map(
            lambda candidate: (
                f"{candidate}-lo-90" in future_forecasts.columns
                and f"{candidate}-hi-90" in future_forecasts.columns
            )
        )
    ]
    if eligible.empty:
        fallback = dict(preferred)
        fallback["promotable"] = False
        return (
            fallback,
            "selected interval model has no forward intervals and no fallback was found",
        )

    fallback_row = eligible.sort_values(
        ["coverage_gap_90", "wis_90", "winkler_90", "avg_interval_width_90"]
    ).iloc[0]
    operational = {
        "model": str(fallback_row["model"]),
        "target_variant": selected_target,
        "promotable": bool(fallback_row.get("interval_eligible", False))
        and bool(fallback_row.get("coverage_gap_90", math.inf) <= 0.03),
        "coverage_90": float(pd.to_numeric(fallback_row.get("coverage_90"), errors="coerce")),
        "coverage_gap_90": float(
            pd.to_numeric(fallback_row.get("coverage_gap_90"), errors="coerce")
        ),
        "avg_interval_width_90": float(
            pd.to_numeric(fallback_row.get("avg_interval_width_90"), errors="coerce")
        ),
        "winkler_90": float(pd.to_numeric(fallback_row.get("winkler_90"), errors="coerce")),
        "wis_90": float(pd.to_numeric(fallback_row.get("wis_90"), errors="coerce")),
        "pinball_90": float(pd.to_numeric(fallback_row.get("pinball_90"), errors="coerce")),
        "family": str(fallback_row.get("family", "")),
        "interval_subfamily": str(fallback_row.get("interval_subfamily", "")),
        "reasons": ["forward_projection_not_available_for_selected_interval_model"],
    }
    return operational, (
        f"selected interval model {model_name} is backtest-only; "
        f"using {operational['model']} for forward interval generation"
    )


def _load_grade_ecl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing grade-level IFRS9 artifact: {path}")
    return pd.read_parquet(path)


def _load_conformal_intervals() -> pd.DataFrame:
    path = DATA_DIR / "conformal_intervals_mondrian.parquet"
    cols = ["y_true", "y_pred", "width_90", "grade", "loan_amnt"]
    if not path.exists():
        raise FileNotFoundError(f"Missing conformal artifact: {path}")
    return pd.read_parquet(path, columns=cols)


def _ecl_portfolio(
    conf_df: pd.DataFrame,
    grade_ecl: pd.DataFrame,
    *,
    pd_mult: float,
    lgd: float,
    discount_rate: float,
) -> dict[str, float]:
    pd_s1 = grade_ecl.set_index("Grade")["PD_12m"].to_dict()
    pd_lt = grade_ecl.set_index("Grade")["PD_lifetime"].to_dict()
    sub = conf_df.copy()
    sub["pd_12m_scaled"] = sub["grade"].map(pd_s1).fillna(0.15) * float(pd_mult)
    sub["pd_lifetime"] = sub["grade"].map(pd_lt).fillna(0.35)
    is_stage2 = sub["pd_12m_scaled"] > 0.20
    discount = 1.0 + float(discount_rate)
    ecl_stage1 = (
        (~is_stage2).astype(float) * sub["pd_12m_scaled"] * float(lgd) * sub["loan_amnt"] / discount
    )
    ecl_stage2 = (
        is_stage2.astype(float) * sub["pd_lifetime"] * float(lgd) * sub["loan_amnt"] / discount
    )
    return {
        "total_ecl": float((ecl_stage1 + ecl_stage2).sum()),
        "ecl_stage1": float(ecl_stage1.sum()),
        "ecl_stage2": float(ecl_stage2.sum()),
        "n_stage1": int((~is_stage2).sum()),
        "n_stage2": int(is_stage2.sum()),
    }


def _build_vnext_ecl_intervals(
    scenarios: pd.DataFrame,
    portfolio_history: pd.DataFrame,
    grade_ecl: pd.DataFrame,
    conf_df: pd.DataFrame,
    joint_path_eval: pd.DataFrame,
    *,
    lgd: float = 0.40,
    discount_rate: float = 0.05,
) -> pd.DataFrame:
    recent_actual_mean = float(
        pd.to_numeric(portfolio_history["y"], errors="coerce").tail(12).mean()
    )
    point_mean = float(
        pd.to_numeric(scenarios["point_forecast"], errors="coerce").clip(lower=0.0).mean()
    )
    if not math.isfinite(recent_actual_mean) or recent_actual_mean <= 0:
        recent_actual_mean = max(point_mean, 1e-3)
    if not math.isfinite(point_mean) or point_mean <= 0:
        point_mean = recent_actual_mean
    baseline_pd_mult = float(np.clip(point_mean / max(recent_actual_mean, 1e-6), 0.85, 1.35))

    joint_lookup = (
        joint_path_eval.set_index("method").to_dict(orient="index")
        if not joint_path_eval.empty
        else {}
    )
    rows: list[dict[str, Any]] = []
    for row in scenarios.itertuples(index=False):
        point_forecast = float(max(row.point_forecast, 0.0))
        optimistic_90 = float(max(row.optimistic_90, 0.0))
        adverse_90 = float(max(row.adverse_90, 0.0))
        pd_mult_point = float(
            np.clip(baseline_pd_mult * point_forecast / max(point_mean, 1e-6), 0.50, 3.0)
        )
        pd_mult_optimistic = float(
            np.clip(baseline_pd_mult * optimistic_90 / max(point_mean, 1e-6), 0.50, 3.0)
        )
        pd_mult_adverse = float(
            np.clip(baseline_pd_mult * adverse_90 / max(point_mean, 1e-6), 0.50, 3.0)
        )
        ecl_point = _ecl_portfolio(
            conf_df,
            grade_ecl,
            pd_mult=pd_mult_point,
            lgd=lgd,
            discount_rate=discount_rate,
        )
        ecl_opt = _ecl_portfolio(
            conf_df,
            grade_ecl,
            pd_mult=pd_mult_optimistic,
            lgd=lgd,
            discount_rate=discount_rate,
        )
        ecl_adv = _ecl_portfolio(
            conf_df,
            grade_ecl,
            pd_mult=pd_mult_adverse,
            lgd=lgd,
            discount_rate=discount_rate,
        )
        month_row = {
            "month": row.month,
            "point_forecast": point_forecast,
            "optimistic_90": optimistic_90,
            "adverse_90": adverse_90,
            "recent_actual_mean_12m": recent_actual_mean,
            "baseline_pd_mult": baseline_pd_mult,
            "pd_mult_point": pd_mult_point,
            "pd_mult_optimistic": pd_mult_optimistic,
            "pd_mult_adverse": pd_mult_adverse,
            "ecl_point": ecl_point["total_ecl"],
            "ecl_optimistic_90": ecl_opt["total_ecl"],
            "ecl_adverse_90": ecl_adv["total_ecl"],
            "ecl_range_90": ecl_adv["total_ecl"] - ecl_opt["total_ecl"],
            "n_stage2_point": ecl_point["n_stage2"],
            "n_stage2_optimistic": ecl_opt["n_stage2"],
            "n_stage2_adverse": ecl_adv["n_stage2"],
        }
        for method_name, stats in joint_lookup.items():
            path_mean = float(stats.get("mean_path_avg", point_mean))
            pd_mult_path = float(
                np.clip(baseline_pd_mult * path_mean / max(point_mean, 1e-6), 0.50, 3.0)
            )
            path_low = float(stats.get("p05_path_avg", path_mean))
            path_high = float(stats.get("p95_path_avg", path_mean))
            ecl_path = _ecl_portfolio(
                conf_df,
                grade_ecl,
                pd_mult=pd_mult_path,
                lgd=lgd,
                discount_rate=discount_rate,
            )
            ecl_path_low = _ecl_portfolio(
                conf_df,
                grade_ecl,
                pd_mult=float(
                    np.clip(baseline_pd_mult * path_low / max(point_mean, 1e-6), 0.50, 3.0)
                ),
                lgd=lgd,
                discount_rate=discount_rate,
            )
            ecl_path_high = _ecl_portfolio(
                conf_df,
                grade_ecl,
                pd_mult=float(
                    np.clip(baseline_pd_mult * path_high / max(point_mean, 1e-6), 0.50, 3.0)
                ),
                lgd=lgd,
                discount_rate=discount_rate,
            )
            month_row[f"{method_name}_ecl_point"] = ecl_path["total_ecl"]
            month_row[f"{method_name}_ecl_p05"] = ecl_path_low["total_ecl"]
            month_row[f"{method_name}_ecl_p95"] = ecl_path_high["total_ecl"]
        rows.append(month_row)
    return pd.DataFrame(rows)


def _build_baseline_audit(
    canonical_status: dict[str, Any],
    canonical_interval_eval: pd.DataFrame,
    canonical_ts_ecl: pd.DataFrame,
) -> dict[str, Any]:
    canonical_point = canonical_status.get("point_champion", {}) or {}
    canonical_interval = canonical_status.get("interval_champion", {}) or {}
    return {
        "series_definition": (
            "monthly vintages grouped by issue_d using final default_flag outcomes; "
            "not realized calendar-time defaults"
        ),
        "point_layer_operational": bool(canonical_point.get("promotable", False)),
        "interval_layer_operational": bool(canonical_interval.get("promotable", False)),
        "interval_layer_current_role": (
            "official" if bool(canonical_interval.get("promotable", False)) else "diagnostic"
        ),
        "canonical_point_model": canonical_point.get("model"),
        "canonical_interval_model": canonical_interval.get("model"),
        "canonical_interval_eval_available": bool(not canonical_interval_eval.empty),
        "canonical_ts_ecl_available": bool(not canonical_ts_ecl.empty),
    }


def main(config_path: str = "configs/time_series_vnext.yaml", horizon: int | None = None) -> None:
    config = load_time_series_config(config_path)
    if horizon is not None:
        config["horizon"] = int(horizon)
    output_paths = _resolve_output_paths(config)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    portfolio_history, panel_history = _load_history_vnext()
    atomic_write_parquet(portfolio_history, output_paths["time_series_path"], index=False)
    atomic_write_parquet(panel_history, output_paths["time_series_panel_path"], index=False)

    canonical_status = _load_optional_json(
        Path(config.get("inputs", {}).get("canonical_status_path", ""))
    )
    canonical_interval_eval = _load_optional_parquet(
        Path(config.get("inputs", {}).get("canonical_interval_eval_path", ""))
    )
    canonical_ts_ecl = _load_optional_parquet(
        Path(config.get("inputs", {}).get("canonical_ts_ecl_path", ""))
    )
    baseline_audit = _build_baseline_audit(
        canonical_status, canonical_interval_eval, canonical_ts_ecl
    )

    forecastability_report, forecastability_status = compute_forecastability_report(
        panel_history,
        season_length=int(config.get("season_length", 12)),
        forecastability_cfg=config.get("forecastability", {}),
    )
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
    future_covariates = load_future_covariates(config)

    target_specs = target_specs_from_config(config)
    target_lookup = {spec.name: spec for spec in target_specs}
    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    all_interval_eval: list[pd.DataFrame] = []
    benchmark_by_target: dict[str, Any] = {}

    for spec in target_specs:
        logger.info("Running TS vNext backtest for target variant={}", spec.name)
        predictions, metrics = run_portfolio_backtest_vnext(
            portfolio_history,
            config,
            spec,
            future_covariates=future_covariates,
        )
        if not predictions.empty:
            all_predictions.append(predictions)
        if not metrics.empty:
            all_metrics.append(metrics)
            all_interval_eval.append(metrics.copy())
        benchmark = benchmark_mapie_time_series_intervals_vnext(portfolio_history, spec, config)
        benchmark_by_target[spec.name] = benchmark
        adaptive_rows = _build_adaptive_interval_rows(benchmark, target_variant=spec.name)
        if not adaptive_rows.empty:
            all_interval_eval.append(adaptive_rows)

    backtest_predictions = (
        pd.concat(all_predictions, ignore_index=True, sort=False)
        if all_predictions
        else pd.DataFrame()
    )
    backtest_metrics = (
        pd.concat(all_metrics, ignore_index=True, sort=False) if all_metrics else pd.DataFrame()
    )
    interval_eval = (
        pd.concat(all_interval_eval, ignore_index=True, sort=False)
        if all_interval_eval
        else pd.DataFrame()
    )
    if backtest_metrics.empty or backtest_predictions.empty:
        raise RuntimeError("Time-series vNext backtest produced no usable metrics.")

    champions = evaluate_target_champions(interval_eval, config)
    selected_target = str(champions["selected_target_variant"])
    selected_spec = target_lookup[selected_target]
    future_forecasts = forecast_portfolio_models_vnext(
        portfolio_history,
        config,
        selected_spec,
        future_covariates=future_covariates,
    )
    operational_interval, operational_interval_note = _select_operational_interval(
        future_forecasts,
        champions,
        interval_eval,
    )
    canonical_forecasts = build_canonical_forecast_frame(
        future_forecasts,
        {
            "point": champions["point"],
            "interval": operational_interval,
        },
    )
    canonical_forecasts["target_variant"] = selected_target
    canonical_forecasts["research_interval_champion"] = str(champions["interval"]["model"])
    scenarios = build_ifrs9_temporal_scenarios(canonical_forecasts)
    scenarios["target_variant"] = selected_target

    panel_forecasts, panel_status = forecast_panel_bottom_up(panel_history, config)
    joint_path_cfg = config.get("joint_paths", {}) or {}
    if bool(joint_path_cfg.get("enabled", True)):
        joint_path_eval, joint_path_samples = generate_joint_sample_paths(
            canonical_forecasts,
            backtest_predictions.loc[
                backtest_predictions["target_variant"] == selected_target
            ].copy(),
            point_model=str(champions["point"]["model"]),
            n_samples=int(joint_path_cfg.get("n_samples", 512)),
            random_seed=int(joint_path_cfg.get("random_seed", 42)),
        )
    else:
        joint_path_eval = pd.DataFrame()
        joint_path_samples = pd.DataFrame()

    grade_ecl = _load_grade_ecl(Path(config.get("inputs", {}).get("grade_ecl_path", "")))
    conformal = _load_conformal_intervals()
    ecl_intervals = _build_vnext_ecl_intervals(
        scenarios,
        portfolio_history,
        grade_ecl,
        conformal,
        joint_path_eval,
    )

    generated_at = datetime.now(tz=UTC).isoformat()
    run_tag = infer_run_tag()
    artifacts = {
        "config_path": str(Path(config_path)),
        "time_series_path": str(output_paths["time_series_path"]),
        "time_series_panel_path": str(output_paths["time_series_panel_path"]),
        "backtest_predictions_path": str(output_paths["backtest_predictions_path"]),
        "backtest_metrics_path": str(output_paths["backtest_metrics_path"]),
        "interval_eval_path": str(output_paths["interval_eval_path"]),
        "forecasts_path": str(output_paths["forecasts_path"]),
        "scenarios_path": str(output_paths["scenarios_path"]),
        "panel_forecasts_path": str(output_paths["panel_forecasts_path"]),
        "joint_path_eval_path": str(output_paths["joint_path_eval_path"]),
        "joint_path_samples_path": str(output_paths["joint_path_samples_path"]),
        "ecl_intervals_path": str(output_paths["ecl_intervals_path"]),
        "policy_review_matrix_path": str(output_paths["policy_review_matrix_path"]),
        "status_path": str(output_paths["status_path"]),
        "policy_review_path": str(output_paths["policy_review_path"]),
    }
    status_payload = build_status_payload(
        config=config,
        metrics=interval_eval,
        champions={
            "point": champions["point"],
            "interval": champions["interval"],
        },
        diagnostics={
            "generated_at_utc": generated_at,
            **diagnostics,
        },
        panel_status=panel_status,
        future_covariates=future_covariates,
        residual_predictions=backtest_predictions.loc[
            backtest_predictions["target_variant"] == selected_target
        ].copy(),
        artifacts=artifacts,
        run_tag=run_tag,
    )
    status_payload["schema_version"] = str(config.get("schema_version", "2026-04-02.1"))
    status_payload["generated_at_utc"] = generated_at
    status_payload["lane"] = "time_series_vnext_research"
    status_payload["status"] = "research_only"
    status_payload["selected_target_variant"] = selected_target
    status_payload["selected_target_transform"] = selected_spec.transform
    status_payload["baseline_audit"] = baseline_audit
    status_payload["forecastability_summary"] = forecastability_status
    status_payload["hierarchy_reconciliation"] = hierarchy_status
    status_payload["joint_path_summary"] = (
        joint_path_eval.to_dict(orient="records") if not joint_path_eval.empty else []
    )
    status_payload["operational_interval_model"] = operational_interval.get("model")
    status_payload["operational_interval_note"] = operational_interval_note
    status_payload["interval_benchmark_by_target"] = benchmark_by_target
    status_payload["summary"]["n_models_evaluated"] = int(interval_eval["model"].nunique())
    status_payload["summary"]["point_model"] = str(champions["point"]["model"])
    status_payload["summary"]["interval_model"] = str(champions["interval"]["model"])
    status_payload["summary"]["operational_interval_model"] = str(
        operational_interval.get("model", "unknown")
    )
    status_payload["summary"]["recent_actual_mean_12m"] = diagnostics.get("recent_actual_mean_12m")

    policy_review_payload, policy_review_matrix = build_policy_review(
        canonical_status=canonical_status,
        vnext_status=status_payload,
        joint_path_eval=joint_path_eval,
        ecl_eval=ecl_intervals,
    )
    policy_review_payload["generated_at_utc"] = generated_at
    policy_review_payload["baseline_audit"] = baseline_audit
    policy_review_payload["selected_target_variant"] = selected_target

    atomic_write_parquet(
        backtest_predictions, output_paths["backtest_predictions_path"], index=False
    )
    atomic_write_parquet(backtest_metrics, output_paths["backtest_metrics_path"], index=False)
    atomic_write_parquet(interval_eval, output_paths["interval_eval_path"], index=False)
    atomic_write_parquet(canonical_forecasts, output_paths["forecasts_path"], index=False)
    atomic_write_parquet(scenarios, output_paths["scenarios_path"], index=False)
    if not panel_forecasts.empty:
        atomic_write_parquet(panel_forecasts, output_paths["panel_forecasts_path"], index=False)
    if not joint_path_eval.empty:
        atomic_write_parquet(joint_path_eval, output_paths["joint_path_eval_path"], index=False)
    if not joint_path_samples.empty:
        atomic_write_parquet(
            joint_path_samples, output_paths["joint_path_samples_path"], index=False
        )
    atomic_write_parquet(ecl_intervals, output_paths["ecl_intervals_path"], index=False)
    atomic_write_parquet(
        policy_review_matrix, output_paths["policy_review_matrix_path"], index=False
    )
    atomic_write_json(output_paths["status_path"], status_payload)
    atomic_write_json(output_paths["policy_review_path"], policy_review_payload)

    logger.info(
        "TS vNext complete: target={}, point={}, interval={} (operational={})",
        selected_target,
        champions["point"]["model"],
        champions["interval"]["model"],
        operational_interval.get("model"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/time_series_vnext.yaml")
    parser.add_argument("--horizon", type=int, default=None)
    args = parser.parse_args()
    main(config_path=args.config, horizon=args.horizon)
