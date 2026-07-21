"""Benchmark conformal variants for coverage/efficiency trade-offs.

Usage:
    uv run python scripts/benchmark_conformal_variants.py
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from scripts.generate_conformal_intervals import (
    _build_feature_matrix,
    _load_calibrator,
    _load_model,
    _resolve_features,
    _subset_calibration_frame,
)
from src.evaluation.backtesting import winkler_interval_score
from src.models.conformal import (
    apply_probability_calibrator,
    build_mondrian_partition_labels,
    conditional_coverage_by_group,
    create_cross_conformal_score_intervals,
    create_pd_intervals,
    create_pd_intervals_mondrian,
    validate_coverage,
)
from src.utils.io_utils import read_with_fallback

TARGET_COL = "default_flag"
GROUP_COL = "grade"


def _summarize_variant(
    name: str,
    y_true: np.ndarray,
    y_intervals: np.ndarray,
    groups: pd.Series,
    alpha: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics = validate_coverage(y_true, y_intervals, alpha=alpha)
    by_group = conditional_coverage_by_group(y_true, y_intervals, groups)
    widths = y_intervals[:, 1] - y_intervals[:, 0]
    winkler_90 = float(
        np.mean(winkler_interval_score(y_true, y_intervals[:, 0], y_intervals[:, 1], alpha=alpha))
    )
    row = {
        "variant": name,
        "alpha": float(alpha),
        "target_coverage": float(1.0 - alpha),
        "coverage": float(metrics["empirical_coverage"]),
        "coverage_gap": float(metrics["coverage_gap"]),
        "avg_width": float(metrics["avg_interval_width"]),
        "median_width": float(metrics["median_interval_width"]),
        "winkler_90": winkler_90,
        "p90_width": float(np.quantile(widths, 0.90)),
        "p95_width": float(np.quantile(widths, 0.95)),
        "min_group_coverage": float(by_group["coverage"].min()),
        "max_group_coverage": float(by_group["coverage"].max()),
        "std_group_coverage": float(by_group["coverage"].std(ddof=0)),
    }
    by_group = by_group.copy()
    by_group["variant"] = name
    by_group["alpha"] = float(alpha)
    return row, by_group


def _summarize_temporal_stability(
    name: str,
    y_true: np.ndarray,
    y_intervals: np.ndarray,
    issue_dates: pd.Series,
) -> tuple[dict[str, float], pd.DataFrame]:
    dates = pd.to_datetime(issue_dates, errors="coerce")
    frame = pd.DataFrame(
        {
            "variant": name,
            "month": dates.dt.to_period("M").dt.to_timestamp(),
            "y_true": np.asarray(y_true, dtype=float),
            "low": np.asarray(y_intervals[:, 0], dtype=float),
            "high": np.asarray(y_intervals[:, 1], dtype=float),
        }
    ).dropna(subset=["month"])
    if frame.empty:
        return (
            {
                "min_monthly_coverage": float("nan"),
                "last_monthly_coverage": float("nan"),
                "max_monthly_gap": float("nan"),
                "stability_over_time": float("inf"),
            },
            pd.DataFrame(
                columns=[
                    "variant",
                    "month",
                    "n",
                    "coverage_90",
                    "avg_width_90",
                    "coverage_gap_90",
                ]
            ),
        )

    frame["covered"] = (
        (frame["y_true"] >= frame["low"]) & (frame["y_true"] <= frame["high"])
    ).astype(float)
    frame["width"] = frame["high"] - frame["low"]
    monthly = (
        frame.groupby("month", observed=True)
        .agg(
            n=("covered", "size"),
            coverage_90=("covered", "mean"),
            avg_width_90=("width", "mean"),
        )
        .reset_index()
        .sort_values("month")
    )
    monthly["coverage_gap_90"] = (monthly["coverage_90"] - 0.90).abs()
    stability = {
        "min_monthly_coverage": float(monthly["coverage_90"].min()),
        "last_monthly_coverage": float(monthly["coverage_90"].iloc[-1]),
        "max_monthly_gap": float(monthly["coverage_gap_90"].max()),
        "stability_over_time": float(monthly["coverage_gap_90"].mean()),
    }
    monthly.insert(0, "variant", name)
    return stability, monthly


def _load_policy_config(path: str = "configs/conformal_policy.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _promotion_pass(row: pd.Series, policy: dict[str, Any]) -> bool:
    return bool(
        float(row.get("coverage", 0.0)) >= float(policy.get("target_coverage_90_min", 0.90))
        and float(row.get("min_group_coverage", 0.0))
        >= float(policy.get("min_group_coverage_90_min", 0.88))
        and float(row.get("winkler_90", float("inf"))) <= float(policy.get("max_winkler_90", 1.20))
        and float(row.get("avg_width", float("inf"))) <= float(policy.get("max_avg_width_90", 0.80))
    )


def _build_output_paths(namespace: str | None = None) -> dict[str, Path]:
    if namespace:
        ns = str(namespace).strip().replace("/", "_")
        data_dir = Path("data/processed/conformal_gap") / ns
        models_dir = Path("models/conformal_gap") / ns
    else:
        data_dir = Path("data/processed")
        models_dir = Path("models")
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    return {
        "data_dir": data_dir,
        "models_dir": models_dir,
        "benchmark": data_dir / "conformal_variant_benchmark.parquet",
        "benchmark_by_group": data_dir / "conformal_variant_benchmark_by_group.parquet",
        "selection_report": data_dir / "conformal_variant_selection_report.parquet",
        "temporal_diagnostics": data_dir / "conformal_temporal_diagnostics.parquet",
        "local_diagnostics": data_dir / "conformal_local_diagnostics.parquet",
        "selection_status": models_dir / "conformal_variant_selection_status.json",
        "selected_intervals": data_dir / "conformal_intervals_mondrian.parquet",
    }


def _coerce_csv_tuple(raw: str | None, *, cast=str) -> tuple[Any, ...]:
    if raw is None:
        return ()
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(cast(token))
    return tuple(values)


def _variant_name(
    *,
    partition: str,
    partition_probability_source: str,
    n_score_bins: int,
    fallback_mode: str,
    score_scale_family: str,
    min_group_size: int,
    calibration_fraction: float | None = None,
) -> str:
    parts = [
        str(partition),
        f"prob={partition_probability_source}",
        f"bins={int(n_score_bins)}",
        f"fallback={fallback_mode}",
        f"scale={score_scale_family}",
        f"mgs={int(min_group_size)}",
    ]
    if calibration_fraction is not None:
        parts.append(f"calfrac={float(calibration_fraction):.2f}")
    return "::".join(parts)


def main(
    alpha: float = 0.10,
    selected_config_path: str = "models/conformal_results_mondrian.pkl",
    min_group_size_default: int = 500,
    cross_cal_sample_size: int = 5000,
    cross_test_sample_size: int = 5000,
    calibration_size_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0),
    partition_candidates: tuple[str, ...] = (
        "grade",
        "score_decile_mondrian",
        "grade_x_scoreband_mondrian",
    ),
    partition_probability_sources: tuple[str, ...] = ("raw",),
    n_score_bins_candidates: tuple[int, ...] = (10,),
    fallback_modes: tuple[str, ...] = ("grade_then_global",),
    score_scale_families: tuple[str, ...] = ("none", "bernoulli_sqrt"),
    min_group_sizes: tuple[int, ...] | None = None,
    artifact_namespace: str | None = None,
    calibrator_override_path: str | None = None,
) -> None:
    policy = _load_policy_config().get("policy", {}) or {}
    model, _ = _load_model()
    calibrator = _load_calibrator(calibrator_override_path)
    cal_df = read_with_fallback(
        "data/processed/calibration_fe.parquet", "data/processed/calibration.parquet"
    )
    test_df = read_with_fallback("data/processed/test_fe.parquet", "data/processed/test.parquet")

    features, categorical = _resolve_features(model, cal_df, test_df)
    X_cal = _build_feature_matrix(cal_df, features, categorical)
    y_cal = cal_df[TARGET_COL].astype(float)
    X_test = _build_feature_matrix(test_df, features, categorical)
    y_test = test_df[TARGET_COL].astype(float).to_numpy(dtype=float)
    group_cal = cal_df[GROUP_COL].fillna("UNKNOWN").astype(str).reset_index(drop=True)
    group_test = test_df[GROUP_COL].fillna("UNKNOWN").astype(str).reset_index(drop=True)
    issue_test = test_df.get("issue_d", pd.Series([pd.NaT] * len(test_df))).reset_index(drop=True)
    y_prob_cal_raw = model.predict_proba(X_cal)[:, 1]
    y_prob_test_raw = model.predict_proba(X_test)[:, 1]
    y_prob_calibrated = (
        apply_probability_calibrator(calibrator, y_prob_cal_raw)
        if calibrator is not None
        else np.asarray(y_prob_cal_raw, dtype=float)
    )
    y_prob_test_calibrated = (
        apply_probability_calibrator(calibrator, y_prob_test_raw)
        if calibrator is not None
        else np.asarray(y_prob_test_raw, dtype=float)
    )
    partition_candidates = tuple(
        dict.fromkeys(str(x).strip() for x in partition_candidates if str(x).strip())
    ) or ("grade",)
    partition_probability_sources = tuple(
        dict.fromkeys(
            str(x).strip().lower() for x in partition_probability_sources if str(x).strip()
        )
    ) or ("raw",)
    n_score_bins_candidates = tuple(int(x) for x in n_score_bins_candidates if int(x) > 0) or (10,)
    fallback_modes = tuple(
        dict.fromkeys(str(x).strip().lower() for x in fallback_modes if str(x).strip())
    ) or ("grade_then_global",)
    score_scale_families = tuple(
        dict.fromkeys(str(x).strip().lower() for x in score_scale_families if str(x).strip())
    ) or ("none",)
    min_group_sizes = tuple(
        int(x) for x in (min_group_sizes or (min_group_size_default,)) if int(x) > 0
    ) or (int(min_group_size_default),)
    prob_cal_lookup = {"raw": y_prob_cal_raw, "calibrated": y_prob_calibrated}
    prob_test_lookup = {"raw": y_prob_test_raw, "calibrated": y_prob_test_calibrated}

    rows: list[dict[str, Any]] = []
    by_group_rows: list[pd.DataFrame] = []
    temporal_rows: list[pd.DataFrame] = []
    local_rows: list[pd.DataFrame] = []

    def _append_mondrian_variant(
        name: str,
        *,
        partition: str,
        partition_probability_source: str,
        n_score_bins: int,
        fallback_mode: str,
        score_scale_family: str,
        alpha_used: float = alpha,
        min_group_size: int = min_group_size_default,
        X_cal_variant: pd.DataFrame | None = None,
        y_cal_variant: pd.Series | None = None,
        y_prob_cal_variant: np.ndarray | None = None,
        base_groups_cal_variant: pd.Series | None = None,
        calibration_fraction: float | None = None,
    ) -> None:
        X_cal_use = X_cal if X_cal_variant is None else X_cal_variant
        y_cal_use = y_cal if y_cal_variant is None else y_cal_variant
        y_prob_cal_use = (
            prob_cal_lookup[partition_probability_source]
            if y_prob_cal_variant is None
            else y_prob_cal_variant
        )
        base_groups_cal_use = (
            group_cal if base_groups_cal_variant is None else base_groups_cal_variant
        )
        group_cal_part, group_test_part, partition_meta = build_mondrian_partition_labels(
            y_prob_cal=y_prob_cal_use,
            y_prob_eval=prob_test_lookup[partition_probability_source],
            partition=partition,
            base_groups_cal=base_groups_cal_use.iloc[: len(y_cal_use)].reset_index(drop=True),
            base_groups_eval=group_test,
            n_score_bins=n_score_bins,
            min_group_size=min_group_size,
            fallback_mode=fallback_mode,
        )
        _y_pred, y_int, _ = create_pd_intervals_mondrian(
            classifier=model,
            X_cal=X_cal_use,
            y_cal=y_cal_use,
            X_test=X_test,
            group_cal=group_cal_part,
            group_test=group_test_part,
            alpha=alpha_used,
            min_group_size=min_group_size,
            calibrator=calibrator,
            score_scale_family=score_scale_family,
        )
        row, by_group = _summarize_variant(name, y_test, y_int, group_test_part, alpha)
        temporal_meta, temporal_monthly = _summarize_temporal_stability(
            name, y_test, y_int, issue_test
        )
        row.update(temporal_meta)
        row["partition"] = partition_meta.get("partition", partition)
        row["partition_probability_source"] = partition_probability_source
        row["n_score_bins"] = int(n_score_bins)
        row["fallback_mode"] = str(partition_meta.get("fallback_mode", fallback_mode))
        row["scaled_scores"] = bool(score_scale_family != "none")
        row["score_scale_family"] = score_scale_family
        row["min_group_size"] = int(min_group_size)
        row["selected_alpha_used"] = float(alpha_used)
        row["fallback_groups_n"] = int(len(partition_meta.get("fallback_groups", [])))
        row["calibration_fraction"] = float(calibration_fraction or 1.0)
        rows.append(row)
        by_group_rows.append(by_group)
        temporal_rows.append(temporal_monthly)

        local_diag = pd.DataFrame(
            {
                "record_type": "local_partition_summary",
                "variant": name,
                "partition": row["partition"],
                "group": pd.Series(group_test_part).astype(str),
                "y_true": y_test,
                "low": y_int[:, 0],
                "high": y_int[:, 1],
            }
        )
        local_diag["covered"] = (
            (local_diag["y_true"] >= local_diag["low"])
            & (local_diag["y_true"] <= local_diag["high"])
        ).astype(float)
        local_diag["width"] = local_diag["high"] - local_diag["low"]
        local_rows.append(local_diag)

    _y_pred_global, y_int_global = create_pd_intervals(
        classifier=model,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        alpha=alpha,
        calibrator=calibrator,
    )
    row, by_group = _summarize_variant("global_split", y_test, y_int_global, group_test, alpha)
    temporal_meta, temporal_monthly = _summarize_temporal_stability(
        "global_split", y_test, y_int_global, issue_test
    )
    row.update(temporal_meta)
    rows.append(row)
    by_group_rows.append(by_group)
    temporal_rows.append(temporal_monthly)

    for partition in partition_candidates:
        for partition_probability_source in partition_probability_sources:
            for n_score_bins in n_score_bins_candidates:
                for fallback_mode in fallback_modes:
                    for score_scale_family in score_scale_families:
                        for min_group_size in min_group_sizes:
                            _append_mondrian_variant(
                                _variant_name(
                                    partition=partition,
                                    partition_probability_source=partition_probability_source,
                                    n_score_bins=n_score_bins,
                                    fallback_mode=fallback_mode,
                                    score_scale_family=score_scale_family,
                                    min_group_size=min_group_size,
                                ),
                                partition=partition,
                                partition_probability_source=partition_probability_source,
                                n_score_bins=n_score_bins,
                                fallback_mode=fallback_mode,
                                score_scale_family=score_scale_family,
                                min_group_size=min_group_size,
                            )

    rng = np.random.RandomState(42)
    cal_idx = (
        rng.choice(len(y_cal), size=min(int(cross_cal_sample_size), len(y_cal)), replace=False)
        if len(y_cal) > int(cross_cal_sample_size)
        else np.arange(len(y_cal))
    )
    test_idx = (
        rng.choice(len(y_test), size=min(int(cross_test_sample_size), len(y_test)), replace=False)
        if len(y_test) > int(cross_test_sample_size)
        else np.arange(len(y_test))
    )
    _y_pred_cross, y_int_cross = create_cross_conformal_score_intervals(
        y_cal=y_cal.iloc[cal_idx].reset_index(drop=True),
        y_prob_cal=y_prob_calibrated[cal_idx],
        y_prob_test=y_prob_test_calibrated[test_idx],
        alpha=alpha,
        method="plus",
        cv=5,
    )
    row, by_group = _summarize_variant(
        "cross_conformal_score_space",
        y_test[test_idx],
        y_int_cross,
        group_test.iloc[test_idx].reset_index(drop=True),
        alpha,
    )
    temporal_meta, temporal_monthly = _summarize_temporal_stability(
        "cross_conformal_score_space",
        y_test[test_idx],
        y_int_cross,
        issue_test.iloc[test_idx].reset_index(drop=True),
    )
    row.update(temporal_meta)
    row["implementation_note"] = (
        "Cross conformal executed on calibrated score space with a lightweight linear regressor."
    )
    row["evaluation_sample_n_cal"] = int(len(cal_idx))
    row["evaluation_sample_n_test"] = int(len(test_idx))
    rows.append(row)
    by_group_rows.append(by_group)
    temporal_rows.append(temporal_monthly)

    cfg_path = Path(selected_config_path)
    if cfg_path.exists():
        with open(cfg_path, "rb") as f:
            payload = pickle.load(f)
        best = payload.get("tuning_90_best", {}) if isinstance(payload, dict) else {}
        alpha_used = float(best.get("alpha_used_90", alpha))
        min_group_size = int(best.get("min_group_size", min_group_size_default))
        partition_probability_source = str(best.get("partition_probability_source", "raw"))
        n_score_bins = int(best.get("n_score_bins", 10))
        fallback_mode = str(best.get("fallback_mode", "grade_then_global"))
        score_scale_family = str(best.get("score_scale_family", "none"))
        _append_mondrian_variant(
            "mondrian_selected_cfg",
            partition=str(best.get("partition", "grade")),
            partition_probability_source=partition_probability_source,
            n_score_bins=n_score_bins,
            fallback_mode=fallback_mode,
            score_scale_family=score_scale_family,
            alpha_used=alpha_used,
            min_group_size=min_group_size,
        )

    sensitivity_rows: list[dict[str, Any]] = []
    for frac in calibration_size_fractions:
        frac_float = float(frac)
        if frac_float <= 0 or frac_float > 1:
            continue
        cal_df_sub = _subset_calibration_frame(cal_df, calibration_fraction=frac_float)
        X_cal_sub = _build_feature_matrix(cal_df_sub, features, categorical)
        y_cal_sub = cal_df_sub[TARGET_COL].astype(float).reset_index(drop=True)
        group_cal_sub = cal_df_sub[GROUP_COL].fillna("UNKNOWN").astype(str).reset_index(drop=True)
        y_prob_cal_sub_raw = model.predict_proba(X_cal_sub)[:, 1]
        y_prob_cal_sub_calibrated = (
            apply_probability_calibrator(calibrator, y_prob_cal_sub_raw)
            if calibrator is not None
            else np.asarray(y_prob_cal_sub_raw, dtype=float)
        )
        prob_cal_sub_lookup = {"raw": y_prob_cal_sub_raw, "calibrated": y_prob_cal_sub_calibrated}
        for partition in ("score_decile_mondrian", "grade_x_scoreband_mondrian"):
            for partition_probability_source in partition_probability_sources:
                group_cal_part, group_test_part, partition_meta = build_mondrian_partition_labels(
                    y_prob_cal=prob_cal_sub_lookup[partition_probability_source],
                    y_prob_eval=prob_test_lookup[partition_probability_source],
                    partition=partition,
                    base_groups_cal=group_cal_sub,
                    base_groups_eval=group_test,
                    n_score_bins=n_score_bins_candidates[0],
                    min_group_size=min_group_sizes[0],
                    fallback_mode=fallback_modes[0],
                )
                _y_pred_sub, y_int_sub, _ = create_pd_intervals_mondrian(
                    classifier=model,
                    X_cal=X_cal_sub,
                    y_cal=y_cal_sub,
                    X_test=X_test,
                    group_cal=group_cal_part,
                    group_test=group_test_part,
                    alpha=alpha,
                    min_group_size=min_group_sizes[0],
                    calibrator=calibrator,
                    score_scale_family=score_scale_families[0],
                )
                metrics = validate_coverage(y_test, y_int_sub, alpha=alpha)
                by_group_sub = conditional_coverage_by_group(y_test, y_int_sub, group_test_part)
                stability_sub, _ = _summarize_temporal_stability(
                    partition, y_test, y_int_sub, issue_test
                )
                sensitivity_rows.append(
                    {
                        "record_type": "calibration_size_sensitivity",
                        "variant": _variant_name(
                            partition=partition,
                            partition_probability_source=partition_probability_source,
                            n_score_bins=n_score_bins_candidates[0],
                            fallback_mode=fallback_modes[0],
                            score_scale_family=score_scale_families[0],
                            min_group_size=min_group_sizes[0],
                            calibration_fraction=frac_float,
                        ),
                        "partition": partition_meta.get("partition", partition),
                        "partition_probability_source": partition_probability_source,
                        "calibration_fraction": frac_float,
                        "n_calibration_rows": int(len(X_cal_sub)),
                        "coverage": float(metrics["empirical_coverage"]),
                        "coverage_gap": float(metrics["coverage_gap"]),
                        "avg_width": float(metrics["avg_interval_width"]),
                        "min_group_coverage": float(by_group_sub["coverage"].min()),
                        "winkler_90": float(
                            np.mean(
                                winkler_interval_score(
                                    y_test, y_int_sub[:, 0], y_int_sub[:, 1], alpha=alpha
                                )
                            )
                        ),
                        "stability_over_time": float(stability_sub["stability_over_time"]),
                    }
                )

    bench = pd.DataFrame(rows)
    bench["promotion_pass"] = bench.apply(lambda row: _promotion_pass(row, policy), axis=1)
    bench = bench.sort_values(
        [
            "promotion_pass",
            "coverage_gap",
            "min_group_coverage",
            "winkler_90",
            "avg_width",
            "stability_over_time",
        ],
        ascending=[False, True, False, True, True, True],
    ).reset_index(drop=True)
    bench["selection_rank"] = np.arange(1, len(bench) + 1, dtype=int)
    bench_by_group = (
        pd.concat(by_group_rows, ignore_index=True)
        .sort_values(["variant", "group"])
        .reset_index(drop=True)
    )
    temporal_diagnostics = (
        pd.concat(temporal_rows, ignore_index=True)
        .sort_values(["variant", "month"])
        .reset_index(drop=True)
    )
    local_diagnostics = pd.concat(local_rows, ignore_index=True) if local_rows else pd.DataFrame()
    if sensitivity_rows:
        local_diagnostics = pd.concat(
            [local_diagnostics, pd.DataFrame(sensitivity_rows)],
            ignore_index=True,
            sort=False,
        )

    output_paths = _build_output_paths(artifact_namespace)
    selected_cfg_path = Path(selected_config_path)
    selected_intervals_path = output_paths["selected_intervals"]
    if selected_cfg_path.exists() and selected_intervals_path.exists():
        with open(selected_cfg_path, "rb") as f:
            selected_payload = pickle.load(f)
        selected_intervals = pd.read_parquet(selected_intervals_path)
        if {"y_true", "pd_low_90", "pd_high_90", GROUP_COL}.issubset(selected_intervals.columns):
            selected_local = pd.DataFrame(
                {
                    "record_type": "local_partition_summary",
                    "variant": "mondrian_selected_cfg",
                    "partition": str(selected_payload.get("partition", "grade")),
                    "group": selected_intervals[GROUP_COL].fillna("UNKNOWN").astype(str),
                    "y_true": pd.to_numeric(selected_intervals["y_true"], errors="coerce"),
                    "low": pd.to_numeric(selected_intervals["pd_low_90"], errors="coerce"),
                    "high": pd.to_numeric(selected_intervals["pd_high_90"], errors="coerce"),
                }
            )
            selected_local["covered"] = (
                (selected_local["y_true"] >= selected_local["low"])
                & (selected_local["y_true"] <= selected_local["high"])
            ).astype(float)
            selected_local["width"] = selected_local["high"] - selected_local["low"]
            local_diagnostics = pd.concat(
                [local_diagnostics, selected_local],
                ignore_index=True,
                sort=False,
            )

    bench_path = output_paths["benchmark"]
    bench_group_path = output_paths["benchmark_by_group"]
    selection_path = output_paths["selection_report"]
    temporal_path = output_paths["temporal_diagnostics"]
    local_path = output_paths["local_diagnostics"]
    bench.to_parquet(bench_path, index=False)
    bench_by_group.to_parquet(bench_group_path, index=False)
    bench.to_parquet(selection_path, index=False)
    temporal_diagnostics.to_parquet(temporal_path, index=False)
    if not local_diagnostics.empty:
        local_diagnostics.to_parquet(local_path, index=False)

    selected = bench.iloc[0].to_dict()
    status_path = output_paths["selection_status"]
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "schema_version": "2026-04-03.1",
                "generated_at_utc": datetime.now(tz=UTC).isoformat(),
                "artifact_namespace": artifact_namespace or "",
                "calibrator_override_path": str(calibrator_override_path or ""),
                "selected_variant": str(selected.get("variant", "")),
                "selection_rank": int(selected.get("selection_rank", 1)),
                "promotion_pass": bool(selected.get("promotion_pass", False)),
                "selection_criteria": [
                    "promotion_pass",
                    "coverage_gap",
                    "min_group_coverage",
                    "winkler_90",
                    "avg_width",
                    "stability_over_time",
                ],
                "strict_diagnostics_role": (
                    "Kupiec/Christoffersen remain strict diagnostics in "
                    "validate_conformal_policy.py; they are not the primary selector here."
                ),
                "variants_tested": bench["variant"].astype(str).tolist(),
                "report_path": str(selection_path),
                "summary_path": str(bench_path),
                "temporal_diagnostics_path": str(temporal_path),
                "local_diagnostics_path": str(local_path),
                "selected_metrics": {
                    "coverage": float(selected.get("coverage", 0.0)),
                    "coverage_gap": float(selected.get("coverage_gap", 0.0)),
                    "avg_width": float(selected.get("avg_width", 0.0)),
                    "min_group_coverage": float(selected.get("min_group_coverage", 0.0)),
                    "winkler_90": float(selected.get("winkler_90", 0.0)),
                    "stability_over_time": float(selected.get("stability_over_time", 0.0)),
                },
                "search_space": {
                    "partition_candidates": list(partition_candidates),
                    "partition_probability_sources": list(partition_probability_sources),
                    "n_score_bins_candidates": [int(x) for x in n_score_bins_candidates],
                    "fallback_modes": list(fallback_modes),
                    "score_scale_families": list(score_scale_families),
                    "min_group_sizes": [int(x) for x in min_group_sizes],
                    "calibration_size_fractions": [float(x) for x in calibration_size_fractions],
                },
                "top_variants": bench.head(5).to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    logger.info("Saved conformal benchmark summary: {} ({})", bench_path, bench.shape)
    logger.info(
        "Saved conformal benchmark by-group: {} ({})", bench_group_path, bench_by_group.shape
    )
    logger.info(
        "Saved conformal temporal diagnostics: {} ({})", temporal_path, temporal_diagnostics.shape
    )
    if not local_diagnostics.empty:
        logger.info(
            "Saved conformal local diagnostics: {} ({})", local_path, local_diagnostics.shape
        )
    logger.info("Saved conformal variant selection report: {}", selection_path)
    logger.info("Saved conformal variant selection status: {}", status_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--selected_config_path", default="models/conformal_results_mondrian.pkl")
    parser.add_argument("--min_group_size_default", type=int, default=500)
    parser.add_argument("--cross_cal_sample_size", type=int, default=5000)
    parser.add_argument("--cross_test_sample_size", type=int, default=5000)
    parser.add_argument("--calibration_size_fractions", default="0.25,0.50,0.75,1.0")
    parser.add_argument(
        "--partition_candidates",
        default="grade,score_decile_mondrian,grade_x_scoreband_mondrian",
    )
    parser.add_argument("--partition_probability_sources", default="raw")
    parser.add_argument("--n_score_bins_candidates", default="10")
    parser.add_argument("--fallback_modes", default="grade_then_global")
    parser.add_argument("--score_scale_families", default="none,bernoulli_sqrt")
    parser.add_argument("--min_group_sizes", default=None)
    parser.add_argument("--artifact_namespace", default=None)
    parser.add_argument("--calibrator_override_path", default=None)
    args = parser.parse_args()
    calibration_size_fractions = tuple(
        float(x.strip()) for x in str(args.calibration_size_fractions).split(",") if x.strip()
    )
    main(
        alpha=args.alpha,
        selected_config_path=args.selected_config_path,
        min_group_size_default=args.min_group_size_default,
        cross_cal_sample_size=args.cross_cal_sample_size,
        cross_test_sample_size=args.cross_test_sample_size,
        calibration_size_fractions=calibration_size_fractions,
        partition_candidates=_coerce_csv_tuple(args.partition_candidates, cast=str),
        partition_probability_sources=_coerce_csv_tuple(
            args.partition_probability_sources, cast=str
        ),
        n_score_bins_candidates=_coerce_csv_tuple(args.n_score_bins_candidates, cast=int),
        fallback_modes=_coerce_csv_tuple(args.fallback_modes, cast=str),
        score_scale_families=_coerce_csv_tuple(args.score_scale_families, cast=str),
        min_group_sizes=(
            _coerce_csv_tuple(args.min_group_sizes, cast=int)
            if args.min_group_sizes is not None
            else None
        ),
        artifact_namespace=args.artifact_namespace,
        calibrator_override_path=args.calibrator_override_path,
    )
