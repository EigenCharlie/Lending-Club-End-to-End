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
)
from src.evaluation.backtesting import winkler_interval_score
from src.models.conformal import (
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


def main(
    alpha: float = 0.10,
    selected_config_path: str = "models/conformal_results_mondrian.pkl",
    min_group_size_default: int = 500,
    cross_cal_sample_size: int = 5000,
    cross_test_sample_size: int = 5000,
    calibration_size_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0),
) -> None:
    policy = _load_policy_config().get("policy", {}) or {}
    model, _ = _load_model()
    calibrator = _load_calibrator()
    cal_df = read_with_fallback(
        "data/processed/calibration_fe.parquet", "data/processed/calibration.parquet"
    )
    test_df = read_with_fallback("data/processed/test_fe.parquet", "data/processed/test.parquet")

    features, categorical = _resolve_features(model, cal_df, test_df)
    X_cal = _build_feature_matrix(cal_df, features, categorical)
    y_cal = cal_df[TARGET_COL].astype(float)
    X_test = _build_feature_matrix(test_df, features, categorical)
    y_test = test_df[TARGET_COL].astype(float).to_numpy(dtype=float)
    group_cal = cal_df[GROUP_COL].fillna("UNKNOWN").astype(str)
    group_test = test_df[GROUP_COL].fillna("UNKNOWN").astype(str)
    issue_test = test_df.get("issue_d", pd.Series([pd.NaT] * len(test_df)))
    y_prob_cal = model.predict_proba(X_cal)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    rows: list[dict[str, Any]] = []
    by_group_rows: list[pd.DataFrame] = []
    temporal_rows: list[pd.DataFrame] = []
    local_rows: list[pd.DataFrame] = []

    def _append_mondrian_variant(
        name: str,
        *,
        partition: str,
        scaled_scores: bool,
        alpha_used: float = alpha,
        min_group_size: int = min_group_size_default,
        X_cal_variant: pd.DataFrame | None = None,
        y_cal_variant: pd.Series | None = None,
        y_prob_cal_variant: np.ndarray | None = None,
    ) -> None:
        X_cal_use = X_cal if X_cal_variant is None else X_cal_variant
        y_cal_use = y_cal if y_cal_variant is None else y_cal_variant
        y_prob_cal_use = y_prob_cal if y_prob_cal_variant is None else y_prob_cal_variant
        group_cal_part, group_test_part, partition_meta = build_mondrian_partition_labels(
            y_prob_cal=y_prob_cal_use,
            y_prob_eval=y_prob_test,
            partition=partition,
            base_groups_cal=group_cal.iloc[: len(y_cal_use)].reset_index(drop=True),
            base_groups_eval=group_test,
            n_score_bins=10,
            min_group_size=min_group_size,
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
            scaled_scores=scaled_scores,
        )
        row, by_group = _summarize_variant(name, y_test, y_int, group_test_part, alpha)
        temporal_meta, temporal_monthly = _summarize_temporal_stability(
            name, y_test, y_int, issue_test
        )
        row.update(temporal_meta)
        row["partition"] = partition_meta.get("partition", partition)
        row["scaled_scores"] = bool(scaled_scores)
        row["min_group_size"] = int(min_group_size)
        row["selected_alpha_used"] = float(alpha_used)
        row["fallback_groups_n"] = int(len(partition_meta.get("fallback_groups", [])))
        rows.append(row)
        by_group_rows.append(by_group)
        temporal_rows.append(temporal_monthly)

        local_diag = pd.DataFrame(
            {
                "record_type": "local_partition_summary",
                "variant": name,
                "partition": partition_meta.get("partition", partition),
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

    # 1) Global split conformal (reference)
    _y_pred_global, y_int_global = create_pd_intervals(
        classifier=model,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        alpha=alpha,
    )
    row, by_group = _summarize_variant("global_split", y_test, y_int_global, group_test, alpha)
    temporal_meta, temporal_monthly = _summarize_temporal_stability(
        "global_split", y_test, y_int_global, issue_test
    )
    row.update(temporal_meta)
    rows.append(row)
    by_group_rows.append(by_group)
    temporal_rows.append(temporal_monthly)

    # 2) Mondrian unscaled scores
    _append_mondrian_variant(
        "mondrian_unscaled",
        partition="grade",
        scaled_scores=False,
        min_group_size=min_group_size_default,
    )

    # 3) Mondrian scaled scores
    _append_mondrian_variant(
        "mondrian_scaled",
        partition="grade",
        scaled_scores=True,
        min_group_size=min_group_size_default,
    )

    # 3b) Score-band Mondrian
    _append_mondrian_variant(
        "score_decile_mondrian",
        partition="score_decile_mondrian",
        scaled_scores=True,
        min_group_size=min_group_size_default,
    )

    # 3c) Hybrid grade x score-band Mondrian
    _append_mondrian_variant(
        "grade_x_scoreband_mondrian",
        partition="grade_x_scoreband_mondrian",
        scaled_scores=True,
        min_group_size=min_group_size_default,
    )

    # 4) Score-space cross conformal (research sidecar, cheap to recompute)
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
        y_prob_cal=y_prob_cal[cal_idx],
        y_prob_test=y_prob_test[test_idx],
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
        "Cross conformal executed on raw-score space with a lightweight linear regressor "
        "to avoid retraining the frozen upstream CatBoost classifier."
    )
    row["evaluation_sample_n_cal"] = int(len(cal_idx))
    row["evaluation_sample_n_test"] = int(len(test_idx))
    rows.append(row)
    by_group_rows.append(by_group)
    temporal_rows.append(temporal_monthly)

    # 5) Canonical selected config (if available)
    cfg_path = Path(selected_config_path)
    if cfg_path.exists():
        with open(cfg_path, "rb") as f:
            payload = pickle.load(f)
        best = payload.get("tuning_90_best", {}) if isinstance(payload, dict) else {}
        alpha_used = float(best.get("alpha_used_90", alpha))
        min_group_size = int(best.get("min_group_size", min_group_size_default))
        scaled_scores = bool(best.get("scaled_scores", False))
        _y_pred_selected, y_int_selected, _ = create_pd_intervals_mondrian(
            classifier=model,
            X_cal=X_cal,
            y_cal=y_cal,
            X_test=X_test,
            group_cal=group_cal,
            group_test=group_test,
            alpha=alpha_used,
            min_group_size=min_group_size,
            calibrator=calibrator,
            scaled_scores=scaled_scores,
        )
        row, by_group = _summarize_variant(
            "mondrian_selected_cfg",
            y_test,
            y_int_selected,
            group_test,
            alpha,
        )
        temporal_meta, temporal_monthly = _summarize_temporal_stability(
            "mondrian_selected_cfg", y_test, y_int_selected, issue_test
        )
        row.update(temporal_meta)
        row["selected_alpha_used"] = alpha_used
        row["selected_min_group_size"] = min_group_size
        row["selected_scaled_scores"] = scaled_scores
        rows.append(row)
        by_group_rows.append(by_group)
        temporal_rows.append(temporal_monthly)

    sensitivity_rows: list[dict[str, Any]] = []
    rng_sensitivity = np.random.RandomState(123)
    for frac in calibration_size_fractions:
        frac_float = float(frac)
        if frac_float <= 0 or frac_float > 1:
            continue
        sample_n = max(1000, int(round(len(X_cal) * frac_float)))
        sample_n = min(sample_n, len(X_cal))
        sample_idx = (
            rng_sensitivity.choice(len(X_cal), size=sample_n, replace=False)
            if sample_n < len(X_cal)
            else np.arange(len(X_cal))
        )
        sample_idx = np.sort(sample_idx)
        X_cal_sub = X_cal.iloc[sample_idx].reset_index(drop=True)
        y_cal_sub = y_cal.iloc[sample_idx].reset_index(drop=True)
        y_prob_cal_sub = y_prob_cal[sample_idx]
        for variant_name, partition in (
            ("score_decile_mondrian", "score_decile_mondrian"),
            ("grade_x_scoreband_mondrian", "grade_x_scoreband_mondrian"),
        ):
            group_cal_part, group_test_part, partition_meta = build_mondrian_partition_labels(
                y_prob_cal=y_prob_cal_sub,
                y_prob_eval=y_prob_test,
                partition=partition,
                base_groups_cal=group_cal.iloc[sample_idx].reset_index(drop=True),
                base_groups_eval=group_test,
                n_score_bins=10,
                min_group_size=min_group_size_default,
            )
            _y_pred_sub, y_int_sub, _ = create_pd_intervals_mondrian(
                classifier=model,
                X_cal=X_cal_sub,
                y_cal=y_cal_sub,
                X_test=X_test,
                group_cal=group_cal_part,
                group_test=group_test_part,
                alpha=alpha,
                min_group_size=min_group_size_default,
                calibrator=calibrator,
                scaled_scores=True,
            )
            metrics = validate_coverage(y_test, y_int_sub, alpha=alpha)
            by_group_sub = conditional_coverage_by_group(y_test, y_int_sub, group_test_part)
            stability_sub, _ = _summarize_temporal_stability(
                variant_name, y_test, y_int_sub, issue_test
            )
            sensitivity_rows.append(
                {
                    "record_type": "calibration_size_sensitivity",
                    "variant": variant_name,
                    "partition": partition_meta.get("partition", partition),
                    "calibration_fraction": frac_float,
                    "n_calibration_rows": int(sample_n),
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
    selected_cfg_path = Path(selected_config_path)
    selected_intervals_path = Path("data/processed/conformal_intervals_mondrian.parquet")
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

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_path = out_dir / "conformal_variant_benchmark.parquet"
    bench_group_path = out_dir / "conformal_variant_benchmark_by_group.parquet"
    selection_path = out_dir / "conformal_variant_selection_report.parquet"
    temporal_path = out_dir / "conformal_temporal_diagnostics.parquet"
    local_path = out_dir / "conformal_local_diagnostics.parquet"
    bench.to_parquet(bench_path, index=False)
    bench_by_group.to_parquet(bench_group_path, index=False)
    bench.to_parquet(selection_path, index=False)
    temporal_diagnostics.to_parquet(temporal_path, index=False)
    if not local_diagnostics.empty:
        local_diagnostics.to_parquet(local_path, index=False)

    selected = bench.iloc[0].to_dict()
    status_path = Path("models/conformal_variant_selection_status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "schema_version": "2026-03-13.1",
                "generated_at_utc": datetime.now(tz=UTC).isoformat(),
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
                "top_variants": bench.head(5).to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    logger.info(f"Saved conformal benchmark summary: {bench_path} ({bench.shape})")
    logger.info(f"Saved conformal benchmark by-group: {bench_group_path} ({bench_by_group.shape})")
    logger.info(
        f"Saved conformal temporal diagnostics: {temporal_path} ({temporal_diagnostics.shape})"
    )
    if not local_diagnostics.empty:
        logger.info(
            "Saved conformal local diagnostics: {} ({})", local_path, local_diagnostics.shape
        )
    logger.info("Saved conformal variant selection report: {}", selection_path)
    logger.info("Saved conformal variant selection status: {}", status_path)
    logger.info(f"\n{bench}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--selected_config_path", default="models/conformal_results_mondrian.pkl")
    parser.add_argument("--min_group_size_default", type=int, default=500)
    parser.add_argument("--cross_cal_sample_size", type=int, default=5000)
    parser.add_argument("--cross_test_sample_size", type=int, default=5000)
    parser.add_argument("--calibration_size_fractions", default="0.25,0.50,0.75,1.0")
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
    )
