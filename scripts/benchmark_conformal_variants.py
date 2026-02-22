"""Benchmark conformal variants for coverage/efficiency trade-offs.

Usage:
    uv run python scripts/benchmark_conformal_variants.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from scripts.generate_conformal_intervals import (
    _build_feature_matrix,
    _load_calibrator,
    _load_model,
    _resolve_features,
)
from src.models.conformal import (
    conditional_coverage_by_group,
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
    row = {
        "variant": name,
        "alpha": float(alpha),
        "target_coverage": float(1.0 - alpha),
        "coverage": float(metrics["empirical_coverage"]),
        "coverage_gap": float(metrics["coverage_gap"]),
        "avg_width": float(metrics["avg_interval_width"]),
        "median_width": float(metrics["median_interval_width"]),
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


def main(
    alpha: float = 0.10,
    selected_config_path: str = "models/conformal_results_mondrian.pkl",
    min_group_size_default: int = 500,
) -> None:
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

    rows: list[dict[str, Any]] = []
    by_group_rows: list[pd.DataFrame] = []

    # 1) Global split conformal (reference)
    _y_pred_global, y_int_global = create_pd_intervals(
        classifier=model,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        alpha=alpha,
    )
    row, by_group = _summarize_variant("global_split", y_test, y_int_global, group_test, alpha)
    rows.append(row)
    by_group_rows.append(by_group)

    # 2) Mondrian unscaled scores
    _y_pred_m_unscaled, y_int_m_unscaled, _ = create_pd_intervals_mondrian(
        classifier=model,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        group_cal=group_cal,
        group_test=group_test,
        alpha=alpha,
        min_group_size=min_group_size_default,
        calibrator=calibrator,
        scaled_scores=False,
    )
    row, by_group = _summarize_variant(
        "mondrian_unscaled",
        y_test,
        y_int_m_unscaled,
        group_test,
        alpha,
    )
    rows.append(row)
    by_group_rows.append(by_group)

    # 3) Mondrian scaled scores
    _y_pred_m_scaled, y_int_m_scaled, _ = create_pd_intervals_mondrian(
        classifier=model,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        group_cal=group_cal,
        group_test=group_test,
        alpha=alpha,
        min_group_size=min_group_size_default,
        calibrator=calibrator,
        scaled_scores=True,
    )
    row, by_group = _summarize_variant(
        "mondrian_scaled",
        y_test,
        y_int_m_scaled,
        group_test,
        alpha,
    )
    rows.append(row)
    by_group_rows.append(by_group)

    # 4) Canonical selected config (if available)
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
        row["selected_alpha_used"] = alpha_used
        row["selected_min_group_size"] = min_group_size
        row["selected_scaled_scores"] = scaled_scores
        rows.append(row)
        by_group_rows.append(by_group)

    bench = pd.DataFrame(rows).sort_values(["coverage_gap", "avg_width"]).reset_index(drop=True)
    bench_by_group = (
        pd.concat(by_group_rows, ignore_index=True)
        .sort_values(["variant", "group"])
        .reset_index(drop=True)
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_path = out_dir / "conformal_variant_benchmark.parquet"
    bench_group_path = out_dir / "conformal_variant_benchmark_by_group.parquet"
    bench.to_parquet(bench_path, index=False)
    bench_by_group.to_parquet(bench_group_path, index=False)

    logger.info(f"Saved conformal benchmark summary: {bench_path} ({bench.shape})")
    logger.info(f"Saved conformal benchmark by-group: {bench_group_path} ({bench_by_group.shape})")
    logger.info(f"\n{bench}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--selected_config_path", default="models/conformal_results_mondrian.pkl")
    parser.add_argument("--min_group_size_default", type=int, default=500)
    args = parser.parse_args()
    main(
        alpha=args.alpha,
        selected_config_path=args.selected_config_path,
        min_group_size_default=args.min_group_size_default,
    )
