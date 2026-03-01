"""Generate Mondrian conformal PD intervals with automatic 90% tuning.

Usage:
    uv run python scripts/generate_conformal_intervals.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger

from src.models.conformal import (
    conditional_coverage_by_group,
    create_pd_intervals_mondrian,
    validate_coverage,
)
from src.models.conformal_tuning import (
    apply_group_multipliers,
    build_group_temporal_segments,
    choose_best_tuning_row,
    enforce_group_coverage_floor,
    enforce_segment_coverage_floor,
    mark_pareto_front,
    split_calibration_for_tuning,
    to_python_scalar,
)
from src.models.pd_contract import (
    CONTRACT_PATH,
    load_contract,
    resolve_calibrator_path,
    resolve_model_path,
)
from src.utils.io_utils import read_with_fallback

TARGET_COL = "default_flag"
GROUP_COL = "grade"


def _load_model() -> tuple[CatBoostClassifier, Path]:
    """Load canonical PD model (with fallback candidates)."""
    model_path = resolve_model_path()

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    logger.info(f"Loaded PD model: {model_path}")
    return model, model_path


def _load_calibrator() -> Any | None:
    """Load canonical calibrator (with fallback candidates)."""
    cal_path = resolve_calibrator_path()
    if cal_path is None:
        logger.warning("No calibrator found. Using raw probabilities.")
        return None
    with open(cal_path, "rb") as f:
        calibrator = pickle.load(f)
    logger.info(f"Loaded calibrator: {type(calibrator).__name__}")
    return calibrator


def _resolve_features(
    model: CatBoostClassifier,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Resolve feature list, preferring explicit contract then model metadata."""
    contract = load_contract(CONTRACT_PATH)
    if isinstance(contract, dict):
        contract_features = contract.get("feature_names", [])
        contract_categorical = contract.get("categorical_features", [])
        if contract_features:
            categorical = [c for c in contract_categorical if c in contract_features]
            logger.info(
                f"Using {len(contract_features)} contract features ({len(categorical)} categorical) "
                f"from {CONTRACT_PATH}"
            )
            return list(contract_features), categorical

    model_features = list(getattr(model, "feature_names_", []) or [])
    if model_features:
        cat_idxs = set(model.get_cat_feature_indices())
        categorical = [f for i, f in enumerate(model_features) if i in cat_idxs]
        logger.info(
            f"Using {len(model_features)} model-native features ({len(categorical)} categorical)"
        )
        return model_features, categorical

    # Fallback path if model metadata is unavailable.
    feature_cfg_path = Path("data/processed/feature_config.pkl")
    feature_cfg: dict[str, Any] = {}
    if feature_cfg_path.exists():
        with open(feature_cfg_path, "rb") as f:
            feature_cfg = pickle.load(f)

    if isinstance(feature_cfg, dict):
        catboost_features = feature_cfg.get("CATBOOST_FEATURES", [])
        categorical = feature_cfg.get("CATEGORICAL_FEATURES", [])
    else:
        catboost_features = []
        categorical = []

    features = [c for c in catboost_features if c in cal_df.columns and c in test_df.columns]
    if not features:
        from src.models.pd_model import get_available_features

        features = [c for c in get_available_features(cal_df) if c in test_df.columns]

    if not features:
        raise ValueError("Unable to resolve feature list for conformal generation.")

    categorical = [c for c in categorical if c in features]
    logger.info(f"Using {len(features)} features ({len(categorical)} categorical)")
    return features, categorical


def _build_feature_matrix(
    df: pd.DataFrame,
    features: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    """Build model matrix with stable order and consistent dtypes."""
    X = df.copy()
    categorical_set = set(categorical)
    for col in features:
        if col not in X.columns:
            X[col] = "UNKNOWN" if col in categorical_set else np.nan

    X = X[features].copy()
    for col in features:
        if col in categorical_set:
            X[col] = X[col].astype("string").fillna("UNKNOWN").astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def _scale_intervals_around_prediction(
    y_pred: np.ndarray,
    y_intervals: np.ndarray,
    factor: float,
) -> np.ndarray:
    """Scale interval radius around point predictions with clipping."""
    f = max(0.01, float(factor))
    low = y_intervals[:, 0].astype(float)
    high = y_intervals[:, 1].astype(float)
    radius = np.maximum(y_pred - low, high - y_pred)
    out_low = np.clip(y_pred - radius * f, 0.0, 1.0)
    out_high = np.clip(y_pred + radius * f, 0.0, 1.0)
    return np.column_stack([out_low, out_high])


def _empirical_interval_coverage(y_true: np.ndarray, y_intervals: np.ndarray) -> float:
    inside = (y_true >= y_intervals[:, 0]) & (y_true <= y_intervals[:, 1])
    return float(inside.mean()) if len(inside) else float("nan")


def _min_group_interval_coverage(
    y_true: np.ndarray,
    y_intervals: np.ndarray,
    groups: pd.Series | np.ndarray,
) -> float:
    group_series = pd.Series(groups).fillna("UNKNOWN").astype(str).reset_index(drop=True)
    y_arr = np.asarray(y_true, dtype=float)
    covs: list[float] = []
    for group_name in sorted(group_series.unique()):
        mask = group_series == group_name
        if int(mask.sum()) == 0:
            continue
        grp_cov = (
            (y_arr[mask] >= y_intervals[mask, 0]) & (y_arr[mask] <= y_intervals[mask, 1])
        ).mean()
        covs.append(float(grp_cov))
    return float(min(covs)) if covs else float("nan")


def main(
    alpha_target_90: float = 0.10,
    alpha_95: float = 0.05,
    alpha_candidates_90: tuple[float, ...] = (0.10, 0.095, 0.09, 0.085, 0.08),
    min_group_sizes: tuple[int, ...] = (200, 500, 1000, 2000),
    min_group_coverage_target: float = 0.88,
    group_coverage_floor_target_90: float = 0.90,
    max_width_budget_90: float | None = 0.80,
    coverage_guardband_90: float = 0.015,
    min_group_guardband_90: float = 0.0,
    tuning_holdout_ratio: float = 0.20,
    tuning_random_state: int = 42,
    temporal_segment_floor_enabled: bool = True,
    temporal_segment_freq: str = "Q",
    temporal_segment_min_size: int = 250,
    global_rebalance_enabled: bool = False,
    global_rebalance_min_factor: float = 0.75,
    global_rebalance_max_factor: float = 1.05,
    global_rebalance_step: float = 0.01,
):
    logger.info("Starting Mondrian conformal interval generation with 90% auto-tuning")

    # Load artifacts and data.
    model, model_path = _load_model()
    calibrator = _load_calibrator()
    cal_df = read_with_fallback(
        "data/processed/calibration_fe.parquet", "data/processed/calibration.parquet"
    )
    test_df = read_with_fallback("data/processed/test_fe.parquet", "data/processed/test.parquet")
    if TARGET_COL not in cal_df.columns or TARGET_COL not in test_df.columns:
        raise KeyError(f"Missing target column '{TARGET_COL}' in calibration/test data.")
    if GROUP_COL not in cal_df.columns or GROUP_COL not in test_df.columns:
        raise KeyError(f"Missing group column '{GROUP_COL}' in calibration/test data.")

    features, categorical = _resolve_features(model, cal_df, test_df)
    X_cal = _build_feature_matrix(cal_df, features, categorical)
    y_cal = cal_df[TARGET_COL].astype(float)
    X_test = _build_feature_matrix(test_df, features, categorical)
    y_test = test_df[TARGET_COL].astype(float)
    group_cal = cal_df[GROUP_COL].fillna("UNKNOWN").astype(str)
    group_test = test_df[GROUP_COL].fillna("UNKNOWN").astype(str)
    idx_cal_fit, idx_cal_tune = split_calibration_for_tuning(
        y_cal=y_cal,
        group_cal=group_cal,
        issue_dates=cal_df.get("issue_d"),
        holdout_ratio=tuning_holdout_ratio,
        random_state=tuning_random_state,
    )
    if len(idx_cal_tune) == 0:
        raise ValueError("Calibration holdout split is empty; cannot run leakage-free tuning.")

    X_cal_fit = X_cal.iloc[idx_cal_fit].reset_index(drop=True)
    y_cal_fit = y_cal.iloc[idx_cal_fit].reset_index(drop=True)
    group_cal_fit = group_cal.iloc[idx_cal_fit].reset_index(drop=True)
    X_tune = X_cal.iloc[idx_cal_tune].reset_index(drop=True)
    y_tune = y_cal.iloc[idx_cal_tune].reset_index(drop=True)
    group_tune = group_cal.iloc[idx_cal_tune].reset_index(drop=True)
    issue_cal = (
        pd.to_datetime(cal_df.get("issue_d"), errors="coerce")
        if "issue_d" in cal_df.columns
        else pd.Series(pd.NaT, index=cal_df.index, dtype="datetime64[ns]")
    )
    issue_tune = issue_cal.iloc[idx_cal_tune].reset_index(drop=True)
    issue_test = (
        pd.to_datetime(test_df.get("issue_d"), errors="coerce")
        if "issue_d" in test_df.columns
        else pd.Series(pd.NaT, index=test_df.index, dtype="datetime64[ns]")
    ).reset_index(drop=True)
    logger.info(
        "Calibration split for conformal tuning: "
        f"fit={len(X_cal_fit):,}, holdout={len(X_tune):,}, "
        f"holdout_ratio={len(X_tune) / max(len(X_cal), 1):.2%}"
    )
    if "issue_d" in cal_df.columns:
        fit_issue = issue_cal.iloc[idx_cal_fit]
        tune_issue = issue_tune
        if fit_issue.notna().any() and tune_issue.notna().any():
            logger.info(
                "Calibration split date ranges: "
                f"fit_max={fit_issue.max():%Y-%m}, "
                f"holdout_min={tune_issue.min():%Y-%m}, "
                f"holdout_max={tune_issue.max():%Y-%m}"
            )

    target_coverage_90 = 1.0 - alpha_target_90
    group_coverage_floor_target_90 = max(
        float(min_group_coverage_target),
        float(group_coverage_floor_target_90),
    )
    tuning_rows: list[dict[str, Any]] = []

    # Tune 90% interval config.
    for alpha_used in alpha_candidates_90:
        for scaled_scores in (True, False):
            for min_group_size in min_group_sizes:
                y_pred, y_int, _diag = create_pd_intervals_mondrian(
                    classifier=model,
                    X_cal=X_cal_fit,
                    y_cal=y_cal_fit,
                    X_test=X_tune,
                    group_cal=group_cal_fit,
                    group_test=group_tune,
                    alpha=alpha_used,
                    min_group_size=min_group_size,
                    calibrator=calibrator,
                    scaled_scores=scaled_scores,
                )

                metrics = validate_coverage(y_tune.to_numpy(dtype=float), y_int, alpha_target_90)
                g_metrics = conditional_coverage_by_group(
                    y_tune.to_numpy(dtype=float), y_int, group_tune
                )

                tuning_rows.append(
                    {
                        "alpha_target_90": alpha_target_90,
                        "alpha_used_90": alpha_used,
                        "scaled_scores": bool(scaled_scores),
                        "min_group_size": int(min_group_size),
                        "empirical_coverage": float(metrics["empirical_coverage"]),
                        "target_coverage": float(metrics["target_coverage"]),
                        "coverage_gap": float(metrics["coverage_gap"]),
                        "avg_interval_width": float(metrics["avg_interval_width"]),
                        "median_interval_width": float(metrics["median_interval_width"]),
                        "min_group_coverage": float(g_metrics["coverage"].min()),
                        "max_group_coverage": float(g_metrics["coverage"].max()),
                        "std_group_coverage": float(g_metrics["coverage"].std(ddof=0)),
                    }
                )

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df["is_pareto"] = mark_pareto_front(tuning_df)
    tuning_df["global_ok"] = tuning_df["empirical_coverage"] >= target_coverage_90
    tuning_df["group_ok"] = tuning_df["min_group_coverage"] >= min_group_coverage_target
    if max_width_budget_90 is None:
        tuning_df["width_ok"] = True
    else:
        tuning_df["width_ok"] = tuning_df["avg_interval_width"] <= max_width_budget_90
    tuning_df = tuning_df.sort_values(
        by=["empirical_coverage", "min_group_coverage", "avg_interval_width"],
        ascending=[False, False, True],
    )
    best_row, selection_tier = choose_best_tuning_row(
        tuning_df,
        target_coverage=target_coverage_90,
        min_group_coverage_target=min_group_coverage_target,
        max_width_budget=max_width_budget_90,
        coverage_guardband=coverage_guardband_90,
        min_group_guardband=min_group_guardband_90,
    )
    best_cfg = {
        "alpha_target_90": float(alpha_target_90),
        "alpha_used_90": float(best_row["alpha_used_90"]),
        "scaled_scores": bool(best_row["scaled_scores"]),
        "min_group_size": int(best_row["min_group_size"]),
        "min_group_coverage_target": float(min_group_coverage_target),
        "group_coverage_floor_target_90": float(group_coverage_floor_target_90),
        "coverage_guardband_90": float(coverage_guardband_90),
        "min_group_guardband_90": float(min_group_guardband_90),
        "max_width_budget_90": None if max_width_budget_90 is None else float(max_width_budget_90),
        "selection_tier": selection_tier,
    }
    logger.info(
        "Best 90% tuning config: "
        f"alpha_used={best_cfg['alpha_used_90']}, scaled_scores={best_cfg['scaled_scores']}, "
        f"min_group_size={best_cfg['min_group_size']}, "
        f"coverage={best_row['empirical_coverage']:.4f}, "
        f"min_group_coverage={best_row['min_group_coverage']:.4f}, "
        f"width={best_row['avg_interval_width']:.4f}, "
        f"tier={selection_tier}"
    )

    # Final 90% intervals with tuned config.
    y_pred_90, y_int_90, diag_90 = create_pd_intervals_mondrian(
        classifier=model,
        X_cal=X_cal_fit,
        y_cal=y_cal_fit,
        X_test=X_test,
        group_cal=group_cal_fit,
        group_test=group_test,
        alpha=best_cfg["alpha_used_90"],
        min_group_size=best_cfg["min_group_size"],
        calibrator=calibrator,
        scaled_scores=best_cfg["scaled_scores"],
    )
    metrics_90 = validate_coverage(y_test.to_numpy(dtype=float), y_int_90, alpha_target_90)
    group_metrics_90 = conditional_coverage_by_group(
        y_test.to_numpy(dtype=float), y_int_90, group_test
    )
    # Learn group multipliers on calibration holdout only (no test-label adaptation).
    y_pred_tune, y_int_tune, _diag_tune = create_pd_intervals_mondrian(
        classifier=model,
        X_cal=X_cal_fit,
        y_cal=y_cal_fit,
        X_test=X_tune,
        group_cal=group_cal_fit,
        group_test=group_tune,
        alpha=best_cfg["alpha_used_90"],
        min_group_size=best_cfg["min_group_size"],
        calibrator=calibrator,
        scaled_scores=best_cfg["scaled_scores"],
    )
    tune_metrics_90_before = validate_coverage(
        y_tune.to_numpy(dtype=float), y_int_tune, alpha_target_90
    )
    y_int_90_adjusted, group_multipliers, coverage_floor_report = enforce_group_coverage_floor(
        y_true=y_tune.to_numpy(dtype=float),
        y_pred=y_pred_tune,
        y_intervals=y_int_tune,
        groups=group_tune,
        target_coverage=group_coverage_floor_target_90,
    )
    tune_metrics_90_after = validate_coverage(
        y_tune.to_numpy(dtype=float), y_int_90_adjusted, alpha_target_90
    )
    test_temporal_segments: pd.Series | None = None
    temporal_segment_multipliers: dict[str, float] = {}
    temporal_segment_report = pd.DataFrame(
        columns=[
            "segment",
            "support_n",
            "coverage_before",
            "coverage_after",
            "target_coverage",
            "min_segment_size",
            "multiplier",
            "adjusted",
        ]
    )
    y_int_90_tune_working = y_int_90_adjusted
    tune_metrics_90_after_temporal = tune_metrics_90_after.copy()
    if group_multipliers:
        logger.info(
            "Applying group coverage floor multipliers learned on calibration holdout: "
            f"{group_multipliers}"
        )
        y_int_90 = apply_group_multipliers(y_pred_90, y_int_90, group_test, group_multipliers)
        metrics_90 = validate_coverage(y_test.to_numpy(dtype=float), y_int_90, alpha_target_90)
        group_metrics_90 = conditional_coverage_by_group(
            y_test.to_numpy(dtype=float), y_int_90, group_test
        )
    else:
        logger.info("No group coverage floor adjustments were required.")
    if (
        temporal_segment_floor_enabled
        and issue_tune.notna().any()
        and issue_test.notna().any()
        and len(issue_tune) == len(group_tune)
        and len(issue_test) == len(group_test)
    ):
        tune_temporal_segments = build_group_temporal_segments(
            groups=group_tune,
            issue_dates=issue_tune,
            freq=temporal_segment_freq,
        )
        test_temporal_segments = build_group_temporal_segments(
            groups=group_test,
            issue_dates=issue_test,
            freq=temporal_segment_freq,
        )
        y_int_90_tune_temporal, temporal_segment_multipliers, temporal_segment_report = (
            enforce_segment_coverage_floor(
                y_true=y_tune.to_numpy(dtype=float),
                y_pred=y_pred_tune,
                y_intervals=y_int_90_tune_working,
                segments=tune_temporal_segments,
                target_coverage=group_coverage_floor_target_90,
                min_segment_size=temporal_segment_min_size,
            )
        )
        y_int_90_tune_working = y_int_90_tune_temporal
        tune_metrics_90_after_temporal = validate_coverage(
            y_tune.to_numpy(dtype=float), y_int_90_tune_temporal, alpha_target_90
        )
        if temporal_segment_multipliers:
            logger.info(
                "Applying temporal coverage floor multipliers learned on holdout "
                f"(freq={temporal_segment_freq}): {temporal_segment_multipliers}"
            )
            y_int_90 = apply_group_multipliers(
                y_pred_90,
                y_int_90,
                test_temporal_segments,
                temporal_segment_multipliers,
            )
            metrics_90 = validate_coverage(y_test.to_numpy(dtype=float), y_int_90, alpha_target_90)
            group_metrics_90 = conditional_coverage_by_group(
                y_test.to_numpy(dtype=float), y_int_90, group_test
            )
        else:
            logger.info("No temporal segment coverage adjustments were required.")
    elif temporal_segment_floor_enabled:
        logger.info("Temporal segment coverage adjustments skipped (missing issue_d coverage).")

    # Optional global rebalance: tune one uniform radius factor on calibration holdout
    # to get closer to nominal global coverage while preserving minimum group floor.
    global_rebalance_factor = 1.0
    global_rebalance_diagnostics: dict[str, float | bool] = {
        "enabled": bool(global_rebalance_enabled),
        "applied": False,
    }
    if global_rebalance_enabled and len(y_int_90_tune_working) > 0:
        min_factor = max(0.05, float(global_rebalance_min_factor))
        max_factor = max(min_factor, float(global_rebalance_max_factor))
        step = max(0.001, float(global_rebalance_step))
        n_steps = int(round((max_factor - min_factor) / step)) + 1
        candidate_factors = np.linspace(min_factor, max_factor, max(2, n_steps))

        tune_y_true = y_tune.to_numpy(dtype=float)
        tune_target_cov = target_coverage_90
        tune_group_floor = float(min_group_coverage_target)

        best_trial: dict[str, float] | None = None
        for factor in candidate_factors:
            tune_trial = _scale_intervals_around_prediction(
                y_pred_tune, y_int_90_tune_working, factor
            )
            cov_trial = _empirical_interval_coverage(tune_y_true, tune_trial)
            min_group_cov_trial = _min_group_interval_coverage(tune_y_true, tune_trial, group_tune)
            floor_shortfall = max(0.0, tune_group_floor - min_group_cov_trial)
            score = abs(cov_trial - tune_target_cov) + 100.0 * floor_shortfall
            trial = {
                "factor": float(factor),
                "coverage": float(cov_trial),
                "min_group_coverage": float(min_group_cov_trial),
                "score": float(score),
            }
            if best_trial is None or trial["score"] < best_trial["score"]:
                best_trial = trial

        if best_trial is not None:
            global_rebalance_factor = float(best_trial["factor"])
            global_rebalance_diagnostics.update(
                {
                    "factor": global_rebalance_factor,
                    "tune_coverage_after_rebalance": float(best_trial["coverage"]),
                    "tune_min_group_coverage_after_rebalance": float(
                        best_trial["min_group_coverage"]
                    ),
                    "target_coverage_90": float(tune_target_cov),
                    "min_group_floor_target": float(tune_group_floor),
                    "applied": abs(global_rebalance_factor - 1.0) > 1e-9,
                }
            )
            if abs(global_rebalance_factor - 1.0) > 1e-9:
                logger.info(
                    "Applying global interval rebalance factor learned on holdout: "
                    f"factor={global_rebalance_factor:.4f}, "
                    f"tune_cov={best_trial['coverage']:.4f}, "
                    f"tune_min_group_cov={best_trial['min_group_coverage']:.4f}"
                )
                y_int_90 = _scale_intervals_around_prediction(
                    y_pred_90, y_int_90, global_rebalance_factor
                )
                metrics_90 = validate_coverage(
                    y_test.to_numpy(dtype=float), y_int_90, alpha_target_90
                )
                group_metrics_90 = conditional_coverage_by_group(
                    y_test.to_numpy(dtype=float), y_int_90, group_test
                )

    # 95% intervals using same structure settings for consistency.
    y_pred_95, y_int_95, diag_95 = create_pd_intervals_mondrian(
        classifier=model,
        X_cal=X_cal_fit,
        y_cal=y_cal_fit,
        X_test=X_test,
        group_cal=group_cal_fit,
        group_test=group_test,
        alpha=alpha_95,
        min_group_size=best_cfg["min_group_size"],
        calibrator=calibrator,
        scaled_scores=best_cfg["scaled_scores"],
    )
    if group_multipliers:
        y_int_95 = apply_group_multipliers(y_pred_95, y_int_95, group_test, group_multipliers)
    if temporal_segment_multipliers and test_temporal_segments is not None:
        y_int_95 = apply_group_multipliers(
            y_pred_95, y_int_95, test_temporal_segments, temporal_segment_multipliers
        )
    if abs(global_rebalance_factor - 1.0) > 1e-9:
        y_int_95 = _scale_intervals_around_prediction(y_pred_95, y_int_95, global_rebalance_factor)
    metrics_95 = validate_coverage(y_test.to_numpy(dtype=float), y_int_95, alpha_95)
    group_metrics_95 = conditional_coverage_by_group(
        y_test.to_numpy(dtype=float), y_int_95, group_test
    )

    # Compose output tables.
    intervals_payload = {
        "y_true": y_test.to_numpy(dtype=float),
        "y_pred": y_pred_90,
        "pd_low_90": y_int_90[:, 0],
        "pd_high_90": y_int_90[:, 1],
        "pd_low_95": y_int_95[:, 0],
        "pd_high_95": y_int_95[:, 1],
        "width_90": y_int_90[:, 1] - y_int_90[:, 0],
        "width_95": y_int_95[:, 1] - y_int_95[:, 0],
        GROUP_COL: group_test.to_numpy(dtype=str),
        "loan_amnt": test_df["loan_amnt"].to_numpy(dtype=float)
        if "loan_amnt" in test_df.columns
        else np.nan,
    }
    if "id" in test_df.columns:
        intervals_payload["id"] = test_df["id"].astype(str).to_numpy()
    if test_temporal_segments is not None:
        intervals_payload["temporal_segment"] = test_temporal_segments.to_numpy(dtype=str)
    intervals_df = pd.DataFrame(intervals_payload)

    gm90 = group_metrics_90.rename(
        columns={
            "coverage": "coverage_90",
            "avg_width": "avg_width_90",
            "median_width": "median_width_90",
        }
    )
    gm95 = group_metrics_95.rename(
        columns={
            "coverage": "coverage_95",
            "avg_width": "avg_width_95",
            "median_width": "median_width_95",
        }
    )
    group_metrics_df = gm90.merge(
        gm95[["group", "coverage_95", "avg_width_95", "median_width_95"]],
        on="group",
        how="outer",
    ).sort_values("group")
    group_metrics_df = group_metrics_df.merge(
        coverage_floor_report[
            ["group", "coverage_before", "coverage_after", "multiplier", "adjusted"]
        ],
        on="group",
        how="left",
    )

    # Persist artifacts.
    data_dir = Path("data/processed")
    models_dir = Path("models")
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    intervals_mondrian_path = data_dir / "conformal_intervals_mondrian.parquet"
    intervals_default_path = data_dir / "conformal_intervals.parquet"  # legacy compatibility copy
    group_metrics_path = data_dir / "conformal_group_metrics_mondrian.parquet"
    tuning_path = data_dir / "conformal_mondrian_tuning_90.parquet"
    pareto_path = data_dir / "conformal_mondrian_tuning_90_pareto.parquet"
    coverage_floor_path = data_dir / "conformal_group_coverage_floor_report.parquet"
    temporal_coverage_floor_path = data_dir / "conformal_temporal_coverage_floor_report.parquet"
    results_path = models_dir / "conformal_results_mondrian.pkl"

    intervals_df.to_parquet(intervals_mondrian_path, index=False)
    intervals_df.to_parquet(intervals_default_path, index=False)
    group_metrics_df.to_parquet(group_metrics_path, index=False)
    tuning_df.to_parquet(tuning_path, index=False)
    tuning_df[tuning_df["is_pareto"]].copy().to_parquet(pareto_path, index=False)
    coverage_floor_report.to_parquet(coverage_floor_path, index=False)
    temporal_segment_report.to_parquet(temporal_coverage_floor_path, index=False)

    payload = {
        "model_path": str(model_path),
        "metrics_90": {k: to_python_scalar(v) for k, v in metrics_90.items()},
        "metrics_95": {k: to_python_scalar(v) for k, v in metrics_95.items()},
        "diag_90": diag_90,
        "diag_95": diag_95,
        "group_metrics_90": group_metrics_90.to_dict(orient="records"),
        "group_metrics_95": group_metrics_95.to_dict(orient="records"),
        "tuning_90_best": best_cfg,
        "tuning_90_table_path": str(tuning_path),
        "tuning_90_pareto_path": str(pareto_path),
        "group_coverage_floor_path": str(coverage_floor_path),
        "group_coverage_multipliers": {k: float(v) for k, v in group_multipliers.items()},
        "temporal_segment_floor_enabled": bool(temporal_segment_floor_enabled),
        "temporal_segment_freq": str(temporal_segment_freq),
        "temporal_segment_min_size": int(temporal_segment_min_size),
        "temporal_segment_coverage_floor_path": str(temporal_coverage_floor_path),
        "temporal_segment_multipliers": {
            k: float(v) for k, v in temporal_segment_multipliers.items()
        },
        "global_rebalance": global_rebalance_diagnostics,
        "group_coverage_floor_target_90": float(group_coverage_floor_target_90),
        "calibration_split": {
            "fit_n": int(len(X_cal_fit)),
            "holdout_n": int(len(X_tune)),
            "holdout_ratio": float(tuning_holdout_ratio),
            "random_state": int(tuning_random_state),
            "preferred_mode": "temporal_if_issue_d_available",
        },
        "tune_metrics_90_before_floor": {
            k: to_python_scalar(v) for k, v in tune_metrics_90_before.items()
        },
        "tune_metrics_90_after_floor": {
            k: to_python_scalar(v) for k, v in tune_metrics_90_after.items()
        },
        "tune_metrics_90_after_temporal_floor": {
            k: to_python_scalar(v) for k, v in tune_metrics_90_after_temporal.items()
        },
    }
    with open(results_path, "wb") as f:
        pickle.dump(payload, f)

    logger.info("Conformal artifacts saved:")
    logger.info(f"  - {intervals_mondrian_path}")
    logger.info(f"  - {intervals_default_path} (legacy compatibility copy)")
    logger.info(f"  - {group_metrics_path}")
    logger.info(f"  - {tuning_path}")
    logger.info(f"  - {pareto_path}")
    logger.info(f"  - {coverage_floor_path}")
    logger.info(f"  - {temporal_coverage_floor_path}")
    logger.info(f"  - {results_path}")
    logger.info(
        "Final metrics: "
        f"90% coverage={metrics_90['empirical_coverage']:.4f} "
        f"(target={metrics_90['target_coverage']:.4f}, width={metrics_90['avg_interval_width']:.4f}) | "
        f"95% coverage={metrics_95['empirical_coverage']:.4f} "
        f"(target={metrics_95['target_coverage']:.4f}, width={metrics_95['avg_interval_width']:.4f})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha_target_90", type=float, default=0.10)
    parser.add_argument("--alpha_95", type=float, default=0.05)
    parser.add_argument("--min_group_coverage_target", type=float, default=0.88)
    parser.add_argument("--group_coverage_floor_target_90", type=float, default=0.90)
    parser.add_argument("--max_width_budget_90", type=float, default=0.80)
    parser.add_argument("--coverage_guardband_90", type=float, default=0.015)
    parser.add_argument("--min_group_guardband_90", type=float, default=0.0)
    parser.add_argument("--tuning_holdout_ratio", type=float, default=0.20)
    parser.add_argument("--tuning_random_state", type=int, default=42)
    parser.add_argument("--temporal_segment_floor_enabled", type=int, default=1)
    parser.add_argument("--temporal_segment_freq", default="Q")
    parser.add_argument("--temporal_segment_min_size", type=int, default=250)
    parser.add_argument("--global_rebalance_enabled", type=int, default=0)
    parser.add_argument("--global_rebalance_min_factor", type=float, default=0.75)
    parser.add_argument("--global_rebalance_max_factor", type=float, default=1.05)
    parser.add_argument("--global_rebalance_step", type=float, default=0.01)
    args = parser.parse_args()
    main(
        alpha_target_90=args.alpha_target_90,
        alpha_95=args.alpha_95,
        min_group_coverage_target=args.min_group_coverage_target,
        group_coverage_floor_target_90=args.group_coverage_floor_target_90,
        max_width_budget_90=args.max_width_budget_90,
        coverage_guardband_90=args.coverage_guardband_90,
        min_group_guardband_90=args.min_group_guardband_90,
        tuning_holdout_ratio=args.tuning_holdout_ratio,
        tuning_random_state=args.tuning_random_state,
        temporal_segment_floor_enabled=bool(args.temporal_segment_floor_enabled),
        temporal_segment_freq=args.temporal_segment_freq,
        temporal_segment_min_size=args.temporal_segment_min_size,
        global_rebalance_enabled=bool(args.global_rebalance_enabled),
        global_rebalance_min_factor=args.global_rebalance_min_factor,
        global_rebalance_max_factor=args.global_rebalance_max_factor,
        global_rebalance_step=args.global_rebalance_step,
    )
