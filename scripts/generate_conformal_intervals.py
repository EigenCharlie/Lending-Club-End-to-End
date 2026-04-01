"""Generate Mondrian conformal PD intervals with automatic 90% tuning.

Usage:
    uv run python scripts/generate_conformal_intervals.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger

from src.models.calibration import load_probability_calibrator
from src.models.conformal import (
    build_mondrian_partition_labels,
    conditional_coverage_by_group,
    create_pd_intervals_mondrian,
    validate_coverage,
)
from src.models.conformal_tuning import (
    apply_group_multipliers,
    build_group_temporal_segments,
    choose_best_tuning_row,
    empirical_interval_coverage,
    enforce_group_coverage_floor,
    enforce_segment_coverage_floor,
    mark_pareto_front,
    mean_winkler_score,
    min_group_interval_coverage,
    shrink_group_multipliers,
    split_calibration_for_tuning,
    temporal_stability_summary,
    to_python_scalar,
)
from src.models.pd_contract import (
    CONTRACT_PATH,
    load_contract,
    resolve_calibrator_path,
    resolve_model_path,
)
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag
from src.utils.io_utils import read_with_fallback
from src.utils.replay_manifest import load_replay_manifest, manifest_section

TARGET_COL = "default_flag"
GROUP_COL = "grade"


def _resolve_artifact_paths(namespace: str | None = None) -> dict[str, Path]:
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
        "intervals": data_dir / "conformal_intervals_mondrian.parquet",
        "group_metrics": data_dir / "conformal_group_metrics_mondrian.parquet",
        "tuning": data_dir / "conformal_mondrian_tuning_90.parquet",
        "pareto": data_dir / "conformal_mondrian_tuning_90_pareto.parquet",
        "group_floor": data_dir / "conformal_group_coverage_floor_report.parquet",
        "temporal_floor": data_dir / "conformal_temporal_coverage_floor_report.parquet",
        "shrinkback": data_dir / "conformal_shrinkback_report.parquet",
        "width_attr": data_dir / "pd_conformal_width_attribution.parquet",
        "results": models_dir / "conformal_results_mondrian.pkl",
        "width_attr_status": models_dir / "pd_conformal_width_attribution_status.json",
    }


def _copy_replay_artifact(source: Path, target: Path, *, run_tag: str) -> None:
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload["run_tag"] = run_tag
            payload["generated_at_utc"] = build_artifact_metadata(
                schema_version=str(payload.get("schema_version", "2026-03-26.1")),
                run_tag=run_tag,
            )["generated_at_utc"]
            target.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            return
    shutil.copy2(source, target)


def _restore_replay_namespace(source_namespace: str) -> None:
    source_paths = _resolve_artifact_paths(source_namespace)
    target_paths = _resolve_artifact_paths(None)
    run_tag = resolve_run_tag(require_explicit=True)
    for key, source in source_paths.items():
        if key in {"data_dir", "models_dir"}:
            continue
        target = target_paths[key]
        if not source.exists():
            raise FileNotFoundError(f"Missing conformal replay source artifact: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_replay_artifact(source, target, run_tag=run_tag)
    logger.info("Restored blessed conformal artifacts from namespace {}", source_namespace)


def _stage_metrics(
    *,
    dataset_scope: str,
    stage: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_intervals: np.ndarray,
    groups: pd.Series | np.ndarray,
    issue_dates: pd.Series | np.ndarray | None,
    alpha: float,
    target_coverage: float,
) -> dict[str, Any]:
    temporal = temporal_stability_summary(
        y_true,
        y_intervals,
        issue_dates,
        target_coverage=target_coverage,
        freq="M",
    )
    return {
        "dataset_scope": dataset_scope,
        "stage": stage,
        "coverage_90": float(empirical_interval_coverage(y_true, y_intervals)),
        "min_group_coverage_90": float(min_group_interval_coverage(y_true, y_intervals, groups)),
        "avg_width_90": float(
            np.mean(
                np.asarray(y_intervals, dtype=float)[:, 1]
                - np.asarray(y_intervals, dtype=float)[:, 0]
            )
        ),
        "winkler_90": float(mean_winkler_score(y_true, y_intervals, alpha=alpha)),
        "max_monthly_gap": float(temporal["max_monthly_gap"]),
        "stability_over_time": float(temporal["stability_over_time"]),
        "min_monthly_coverage": float(temporal["min_monthly_coverage"]),
        "last_monthly_coverage": float(temporal["last_monthly_coverage"]),
        "target_coverage": float(target_coverage),
        "n_obs": int(len(y_true)),
    }


def _load_model() -> tuple[CatBoostClassifier, Path]:
    """Load canonical PD model (with fallback candidates)."""
    model_path = resolve_model_path()

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    logger.info(f"Loaded PD model: {model_path}")
    return model, model_path


def _load_calibrator(calibrator_override_path: str | None = None) -> Any | None:
    """Load canonical or shadow calibrator."""
    cal_path = (
        Path(calibrator_override_path) if calibrator_override_path else resolve_calibrator_path()
    )
    if cal_path is None:
        logger.warning("No calibrator found. Using raw probabilities.")
        return None
    calibrator = load_probability_calibrator(str(cal_path))
    logger.info(f"Loaded calibrator: {type(calibrator).__name__} ({cal_path})")
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


def main(
    alpha_target_90: float = 0.10,
    alpha_95: float = 0.05,
    alpha_candidates_90: tuple[float, ...] = (0.10, 0.095, 0.09, 0.085, 0.08),
    min_group_sizes: tuple[int, ...] = (200, 500, 1000, 2000),
    min_group_coverage_target: float = 0.88,
    group_coverage_floor_target_90: float = 0.92,
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
    partition: str = "grade",
    partition_candidates: tuple[str, ...] | None = None,
    group_coverage_floor_enabled: bool = True,
    shrinkback_enabled: bool = False,
    group_multiplier_grid: tuple[float, ...] = (1.0, 1.02, 1.05, 1.08, 1.12, 1.16, 1.20),
    temporal_multiplier_grid: tuple[float, ...] = (1.0, 1.02, 1.05, 1.08, 1.12, 1.16, 1.20),
    artifact_namespace: str | None = None,
    scaled_scores_options: tuple[bool, ...] = (True, False),
    mode: str = "search",
    replay_manifest_path: str | None = None,
    calibrator_override_path: str | None = None,
):
    logger.info("Starting Mondrian conformal interval generation with 90% auto-tuning")
    run_mode = str(mode or "search").strip().lower() or "search"
    replay_cfg = manifest_section(load_replay_manifest(replay_manifest_path), "conformal")
    if run_mode == "replay":
        source_namespace = str(replay_cfg.get("source_namespace", "")).strip()
        replay_mode = str(replay_cfg.get("replay_mode", "")).strip().lower()
        if replay_mode != "restore_blessed_namespace" or not source_namespace:
            raise ValueError(
                "Conformal replay requires source_namespace + restore_blessed_namespace."
            )
        _restore_replay_namespace(source_namespace)
        return

    # Load artifacts and data.
    model, model_path = _load_model()
    calibrator = _load_calibrator(calibrator_override_path)
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
    group_cal_base = cal_df[GROUP_COL].fillna("UNKNOWN").astype(str)
    group_test_base = test_df[GROUP_COL].fillna("UNKNOWN").astype(str)
    y_prob_cal_raw = model.predict_proba(X_cal)[:, 1]
    y_prob_test_raw = model.predict_proba(X_test)[:, 1]
    idx_cal_fit, idx_cal_tune = split_calibration_for_tuning(
        y_cal=y_cal,
        group_cal=group_cal_base,
        issue_dates=cal_df.get("issue_d"),
        holdout_ratio=tuning_holdout_ratio,
        random_state=tuning_random_state,
    )
    if len(idx_cal_tune) == 0:
        raise ValueError("Calibration holdout split is empty; cannot run leakage-free tuning.")

    X_cal_fit = X_cal.iloc[idx_cal_fit].reset_index(drop=True)
    y_cal_fit = y_cal.iloc[idx_cal_fit].reset_index(drop=True)
    X_tune = X_cal.iloc[idx_cal_tune].reset_index(drop=True)
    y_tune = y_cal.iloc[idx_cal_tune].reset_index(drop=True)
    y_prob_cal_fit = y_prob_cal_raw[idx_cal_fit]
    y_prob_cal_tune = y_prob_cal_raw[idx_cal_tune]
    group_cal_fit_base = group_cal_base.iloc[idx_cal_fit].reset_index(drop=True)
    group_tune_base = group_cal_base.iloc[idx_cal_tune].reset_index(drop=True)
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
    partition_candidates = tuple(
        dict.fromkeys(
            [
                str(token).strip()
                for token in (partition_candidates or (partition,))
                if str(token).strip()
            ]
        )
    ) or (str(partition).strip() or "grade",)
    tuning_rows: list[dict[str, Any]] = []

    # Tune 90% interval config across candidate Mondrian partitions.
    for partition_candidate in partition_candidates:
        for alpha_used in alpha_candidates_90:
            for scaled_scores in tuple(bool(x) for x in scaled_scores_options):
                for min_group_size in min_group_sizes:
                    group_cal_fit, group_tune, partition_meta_candidate = (
                        build_mondrian_partition_labels(
                            y_prob_cal=y_prob_cal_fit,
                            y_prob_eval=y_prob_cal_tune,
                            partition=partition_candidate,
                            base_groups_cal=group_cal_fit_base,
                            base_groups_eval=group_tune_base,
                            n_score_bins=10,
                            min_group_size=min_group_size,
                        )
                    )
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

                    metrics = validate_coverage(
                        y_tune.to_numpy(dtype=float), y_int, alpha_target_90
                    )
                    g_metrics = conditional_coverage_by_group(
                        y_tune.to_numpy(dtype=float), y_int, group_tune
                    )
                    temporal_metrics = temporal_stability_summary(
                        y_tune.to_numpy(dtype=float),
                        y_int,
                        issue_tune,
                        target_coverage=target_coverage_90,
                        freq="M",
                    )

                    tuning_rows.append(
                        {
                            "partition": str(
                                partition_meta_candidate.get("partition", partition_candidate)
                            ),
                            "fallback_groups_n": int(
                                len(partition_meta_candidate.get("fallback_groups", []))
                            ),
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
                            "winkler_90": float(
                                mean_winkler_score(
                                    y_tune.to_numpy(dtype=float),
                                    y_int,
                                    alpha=alpha_target_90,
                                )
                            ),
                            "max_monthly_gap": float(temporal_metrics["max_monthly_gap"]),
                            "stability_over_time": float(temporal_metrics["stability_over_time"]),
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
        by=["empirical_coverage", "min_group_coverage", "winkler_90", "avg_interval_width"],
        ascending=[False, False, True, True],
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
        "partition": str(best_row.get("partition", partition_candidates[0])),
        "partition_candidates": list(partition_candidates),
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
        f"partition={best_cfg['partition']}, "
        f"alpha_used={best_cfg['alpha_used_90']}, scaled_scores={best_cfg['scaled_scores']}, "
        f"min_group_size={best_cfg['min_group_size']}, "
        f"coverage={best_row['empirical_coverage']:.4f}, "
        f"min_group_coverage={best_row['min_group_coverage']:.4f}, "
        f"width={best_row['avg_interval_width']:.4f}, "
        f"tier={selection_tier}"
    )

    group_cal_fit, group_test, partition_meta = build_mondrian_partition_labels(
        y_prob_cal=y_prob_cal_fit,
        y_prob_eval=y_prob_test_raw,
        partition=best_cfg["partition"],
        base_groups_cal=group_cal_fit_base,
        base_groups_eval=group_test_base,
        n_score_bins=10,
        min_group_size=best_cfg["min_group_size"],
    )
    group_cal_fit_holdout, group_tune, _ = build_mondrian_partition_labels(
        y_prob_cal=y_prob_cal_fit,
        y_prob_eval=y_prob_cal_tune,
        partition=best_cfg["partition"],
        base_groups_cal=group_cal_fit_base,
        base_groups_eval=group_tune_base,
        n_score_bins=10,
        min_group_size=best_cfg["min_group_size"],
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
    width_attr_rows: list[dict[str, Any]] = [
        _stage_metrics(
            dataset_scope="test",
            stage="base_interval",
            y_true=y_test.to_numpy(dtype=float),
            y_pred=y_pred_90,
            y_intervals=y_int_90,
            groups=group_test,
            issue_dates=issue_test,
            alpha=alpha_target_90,
            target_coverage=target_coverage_90,
        )
    ]
    # Learn group multipliers on calibration holdout only (no test-label adaptation).
    y_pred_tune, y_int_tune, _diag_tune = create_pd_intervals_mondrian(
        classifier=model,
        X_cal=X_cal_fit,
        y_cal=y_cal_fit,
        X_test=X_tune,
        group_cal=group_cal_fit_holdout,
        group_test=group_tune,
        alpha=best_cfg["alpha_used_90"],
        min_group_size=best_cfg["min_group_size"],
        calibrator=calibrator,
        scaled_scores=best_cfg["scaled_scores"],
    )
    tune_metrics_90_before = validate_coverage(
        y_tune.to_numpy(dtype=float), y_int_tune, alpha_target_90
    )
    width_attr_rows.append(
        _stage_metrics(
            dataset_scope="tune_holdout",
            stage="base_interval",
            y_true=y_tune.to_numpy(dtype=float),
            y_pred=y_pred_tune,
            y_intervals=y_int_tune,
            groups=group_tune,
            issue_dates=issue_tune,
            alpha=alpha_target_90,
            target_coverage=target_coverage_90,
        )
    )
    if group_coverage_floor_enabled:
        y_int_90_adjusted, group_multipliers, coverage_floor_report = enforce_group_coverage_floor(
            y_true=y_tune.to_numpy(dtype=float),
            y_pred=y_pred_tune,
            y_intervals=y_int_tune,
            groups=group_tune,
            target_coverage=group_coverage_floor_target_90,
            multiplier_grid=group_multiplier_grid,
        )
    else:
        y_int_90_adjusted = np.asarray(y_int_tune, dtype=float).copy()
        group_multipliers = {}
        coverage_floor_report = pd.DataFrame(
            columns=[
                "group",
                "coverage_before",
                "coverage_after",
                "target_coverage",
                "multiplier",
                "adjusted",
            ]
        )
    tune_metrics_90_after = validate_coverage(
        y_tune.to_numpy(dtype=float), y_int_90_adjusted, alpha_target_90
    )
    width_attr_rows.append(
        _stage_metrics(
            dataset_scope="tune_holdout",
            stage="after_group_floor",
            y_true=y_tune.to_numpy(dtype=float),
            y_pred=y_pred_tune,
            y_intervals=y_int_90_adjusted,
            groups=group_tune,
            issue_dates=issue_tune,
            alpha=alpha_target_90,
            target_coverage=target_coverage_90,
        )
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
    y_int_90_base_test = np.asarray(y_int_90, dtype=float).copy()
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
    width_attr_rows.append(
        _stage_metrics(
            dataset_scope="test",
            stage="after_group_floor",
            y_true=y_test.to_numpy(dtype=float),
            y_pred=y_pred_90,
            y_intervals=y_int_90,
            groups=group_test,
            issue_dates=issue_test,
            alpha=alpha_target_90,
            target_coverage=target_coverage_90,
        )
    )
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
                multiplier_grid=temporal_multiplier_grid,
            )
        )
        y_int_90_tune_working = y_int_90_tune_temporal
        tune_metrics_90_after_temporal = validate_coverage(
            y_tune.to_numpy(dtype=float), y_int_90_tune_temporal, alpha_target_90
        )
        width_attr_rows.append(
            _stage_metrics(
                dataset_scope="tune_holdout",
                stage="after_temporal_floor",
                y_true=y_tune.to_numpy(dtype=float),
                y_pred=y_pred_tune,
                y_intervals=y_int_90_tune_temporal,
                groups=group_tune,
                issue_dates=issue_tune,
                alpha=alpha_target_90,
                target_coverage=target_coverage_90,
            )
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
    width_attr_rows.append(
        _stage_metrics(
            dataset_scope="test",
            stage="after_temporal_floor",
            y_true=y_test.to_numpy(dtype=float),
            y_pred=y_pred_90,
            y_intervals=y_int_90,
            groups=group_test,
            issue_dates=issue_test,
            alpha=alpha_target_90,
            target_coverage=target_coverage_90,
        )
    )

    shrinkback_report = pd.DataFrame(
        columns=[
            "stage",
            "factor_scope",
            "factor_key",
            "candidate_factor",
            "accepted",
            "coverage",
            "min_group_coverage",
            "avg_width",
            "winkler_90",
            "max_monthly_gap",
            "stability_over_time",
        ]
    )
    if shrinkback_enabled and (group_multipliers or temporal_segment_multipliers):
        shrink_max_monthly_gap = temporal_stability_summary(
            y_tune.to_numpy(dtype=float),
            y_int_90_tune_working,
            issue_tune,
            target_coverage=target_coverage_90,
            freq="M",
        )["max_monthly_gap"]
        (
            y_int_90_tune_working,
            group_multipliers,
            temporal_segment_multipliers,
            shrinkback_report,
        ) = shrink_group_multipliers(
            y_true=y_tune.to_numpy(dtype=float),
            y_pred=y_pred_tune,
            base_intervals=y_int_tune,
            groups=group_tune,
            issue_dates=issue_tune,
            group_factors=group_multipliers,
            temporal_segments=tune_temporal_segments if temporal_segment_floor_enabled else None,
            temporal_factors=temporal_segment_multipliers,
            target_coverage=target_coverage_90,
            min_group_coverage_target=min_group_coverage_target,
            max_monthly_gap_target=float(shrink_max_monthly_gap)
            if np.isfinite(shrink_max_monthly_gap)
            else None,
            alpha=alpha_target_90,
            group_multiplier_grid=group_multiplier_grid,
            temporal_multiplier_grid=temporal_multiplier_grid,
        )
        y_int_90 = np.asarray(y_int_90_base_test, dtype=float).copy()
        if group_multipliers:
            y_int_90 = apply_group_multipliers(y_pred_90, y_int_90, group_test, group_multipliers)
        if temporal_segment_multipliers and test_temporal_segments is not None:
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
    width_attr_rows.append(
        _stage_metrics(
            dataset_scope="tune_holdout",
            stage="after_shrinkback",
            y_true=y_tune.to_numpy(dtype=float),
            y_pred=y_pred_tune,
            y_intervals=y_int_90_tune_working,
            groups=group_tune,
            issue_dates=issue_tune,
            alpha=alpha_target_90,
            target_coverage=target_coverage_90,
        )
    )
    width_attr_rows.append(
        _stage_metrics(
            dataset_scope="test",
            stage="after_shrinkback",
            y_true=y_test.to_numpy(dtype=float),
            y_pred=y_pred_90,
            y_intervals=y_int_90,
            groups=group_test,
            issue_dates=issue_test,
            alpha=alpha_target_90,
            target_coverage=target_coverage_90,
        )
    )

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
            cov_trial = empirical_interval_coverage(tune_y_true, tune_trial)
            min_group_cov_trial = min_group_interval_coverage(tune_y_true, tune_trial, group_tune)
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
    intervals_df.insert(0, "_row_number", range(len(intervals_df)))

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
    paths = _resolve_artifact_paths(artifact_namespace)
    intervals_mondrian_path = paths["intervals"]
    group_metrics_path = paths["group_metrics"]
    tuning_path = paths["tuning"]
    pareto_path = paths["pareto"]
    coverage_floor_path = paths["group_floor"]
    temporal_coverage_floor_path = paths["temporal_floor"]
    shrinkback_path = paths["shrinkback"]
    width_attr_path = paths["width_attr"]
    results_path = paths["results"]
    width_attr_status_path = paths["width_attr_status"]
    resolved_run_tag = resolve_run_tag(require_explicit=True)

    intervals_df.to_parquet(intervals_mondrian_path, index=False)
    group_metrics_df.to_parquet(group_metrics_path, index=False)
    tuning_df.to_parquet(tuning_path, index=False)
    tuning_df[tuning_df["is_pareto"]].copy().to_parquet(pareto_path, index=False)
    coverage_floor_report.to_parquet(coverage_floor_path, index=False)
    temporal_segment_report.to_parquet(temporal_coverage_floor_path, index=False)
    shrinkback_report.to_parquet(shrinkback_path, index=False)
    width_attr_df = pd.DataFrame(width_attr_rows)
    width_attr_df.to_parquet(width_attr_path, index=False)

    payload = {
        "model_path": str(model_path),
        "calibrator_override_path": str(calibrator_override_path or ""),
        "metrics_90": {k: to_python_scalar(v) for k, v in metrics_90.items()},
        "metrics_95": {k: to_python_scalar(v) for k, v in metrics_95.items()},
        "diag_90": diag_90,
        "diag_95": diag_95,
        "partition": str(partition_meta.get("partition", partition)),
        "partition_meta": partition_meta,
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
        "group_coverage_floor_enabled": bool(group_coverage_floor_enabled),
        "shrinkback_enabled": bool(shrinkback_enabled),
        "shrinkback_path": str(shrinkback_path),
        "width_attribution_path": str(width_attr_path),
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
    width_attr_status_path.write_text(
        json.dumps(
            {
                "artifact_namespace": artifact_namespace or "",
                "selected_partition": str(partition_meta.get("partition", partition)),
                "selected_alpha_used_90": float(best_cfg["alpha_used_90"]),
                "selected_min_group_size": int(best_cfg["min_group_size"]),
                "selected_scaled_scores": bool(best_cfg["scaled_scores"]),
                "group_factors_after_shrinkback": group_multipliers,
                "temporal_factors_after_shrinkback": temporal_segment_multipliers,
                "width_attribution_path": str(width_attr_path),
                "shrinkback_path": str(shrinkback_path),
                **build_artifact_metadata(
                    schema_version="2026-03-13.1",
                    run_tag=resolved_run_tag,
                    require_explicit=True,
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    logger.info("Conformal artifacts saved:")
    logger.info(f"  - {intervals_mondrian_path}")
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

    def _parse_float_tuple(raw: str) -> tuple[float, ...]:
        values = [float(token.strip()) for token in str(raw).split(",") if token.strip()]
        if not values:
            raise ValueError("Expected at least one float value.")
        return tuple(values)

    def _parse_int_tuple(raw: str) -> tuple[int, ...]:
        values = [int(token.strip()) for token in str(raw).split(",") if token.strip()]
        if not values:
            raise ValueError("Expected at least one integer value.")
        return tuple(values)

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha_target_90", type=float, default=0.10)
    parser.add_argument("--alpha_95", type=float, default=0.05)
    parser.add_argument("--alpha_candidates_90", default="0.10,0.095,0.09,0.085,0.08")
    parser.add_argument("--min_group_sizes", default="200,500,1000,2000")
    parser.add_argument("--min_group_coverage_target", type=float, default=0.88)
    parser.add_argument("--group_coverage_floor_target_90", type=float, default=0.92)
    parser.add_argument("--max_width_budget_90", type=float, default=0.80)
    parser.add_argument("--coverage_guardband_90", type=float, default=0.015)
    parser.add_argument("--min_group_guardband_90", type=float, default=0.0)
    parser.add_argument("--tuning_holdout_ratio", type=float, default=0.20)
    parser.add_argument("--tuning_random_state", type=int, default=42)
    parser.add_argument("--temporal_segment_floor_enabled", type=int, default=1)
    parser.add_argument("--temporal_segment_freq", default="Q")
    parser.add_argument("--temporal_segment_min_size", type=int, default=250)
    parser.add_argument("--group_coverage_floor_enabled", type=int, default=1)
    parser.add_argument("--shrinkback_enabled", type=int, default=0)
    parser.add_argument("--group_multiplier_grid", default="1.0,1.02,1.05,1.08,1.12,1.16,1.20")
    parser.add_argument("--temporal_multiplier_grid", default="1.0,1.02,1.05,1.08,1.12,1.16,1.20")
    parser.add_argument("--scaled_scores_options", default="true,false")
    parser.add_argument("--global_rebalance_enabled", type=int, default=0)
    parser.add_argument("--global_rebalance_min_factor", type=float, default=0.75)
    parser.add_argument("--global_rebalance_max_factor", type=float, default=1.05)
    parser.add_argument("--global_rebalance_step", type=float, default=0.01)
    parser.add_argument("--partition", default="grade")
    parser.add_argument("--partition_candidates", default=None)
    parser.add_argument("--artifact_namespace", default=None)
    parser.add_argument("--calibrator_override_path", default=None)
    parser.add_argument("--mode", choices=["search", "replay"], default="search")
    parser.add_argument("--replay_manifest", default=None)
    args = parser.parse_args()
    main(
        alpha_target_90=args.alpha_target_90,
        alpha_95=args.alpha_95,
        alpha_candidates_90=_parse_float_tuple(args.alpha_candidates_90),
        min_group_sizes=_parse_int_tuple(args.min_group_sizes),
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
        group_coverage_floor_enabled=bool(args.group_coverage_floor_enabled),
        shrinkback_enabled=bool(args.shrinkback_enabled),
        group_multiplier_grid=_parse_float_tuple(args.group_multiplier_grid),
        temporal_multiplier_grid=_parse_float_tuple(args.temporal_multiplier_grid),
        global_rebalance_enabled=bool(args.global_rebalance_enabled),
        global_rebalance_min_factor=args.global_rebalance_min_factor,
        global_rebalance_max_factor=args.global_rebalance_max_factor,
        global_rebalance_step=args.global_rebalance_step,
        partition=args.partition,
        partition_candidates=(
            tuple(
                token.strip()
                for token in str(args.partition_candidates).split(",")
                if token.strip()
            )
            if args.partition_candidates is not None
            else None
        ),
        artifact_namespace=args.artifact_namespace,
        calibrator_override_path=args.calibrator_override_path,
        scaled_scores_options=tuple(
            token.strip().lower() in {"1", "true", "yes", "y"}
            for token in str(args.scaled_scores_options).split(",")
            if token.strip()
        ),
        mode=str(args.mode),
        replay_manifest_path=args.replay_manifest,
    )
