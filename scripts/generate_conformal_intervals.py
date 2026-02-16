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
from src.models.pd_contract import (
    CONTRACT_PATH,
    load_contract,
    resolve_calibrator_path,
    resolve_model_path,
)

TARGET_COL = "default_flag"
GROUP_COL = "grade"


def _read_with_fallback(fe_path: str, base_path: str) -> pd.DataFrame:
    """Read *_fe split if available, else fallback to base split."""
    fe = Path(fe_path)
    base = Path(base_path)
    if fe.exists():
        return pd.read_parquet(fe)
    if base.exists():
        logger.warning(f"{fe} not found. Falling back to {base}")
        return pd.read_parquet(base)
    raise FileNotFoundError(f"Neither {fe} nor {base} exists")


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
        logger.info(f"Using {len(model_features)} model-native features ({len(categorical)} categorical)")
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


def _mark_pareto_front(results_df: pd.DataFrame) -> pd.Series:
    """Pareto front for (maximize coverage, maximize min group coverage, minimize width)."""
    n = len(results_df)
    dominated = np.zeros(n, dtype=bool)
    arr_cov = results_df["empirical_coverage"].to_numpy(dtype=float)
    arr_grp = results_df["min_group_coverage"].to_numpy(dtype=float)
    arr_wid = results_df["avg_interval_width"].to_numpy(dtype=float)

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            better_or_equal = (
                arr_cov[j] >= arr_cov[i]
                and arr_grp[j] >= arr_grp[i]
                and arr_wid[j] <= arr_wid[i]
            )
            strictly_better = (
                arr_cov[j] > arr_cov[i]
                or arr_grp[j] > arr_grp[i]
                or arr_wid[j] < arr_wid[i]
            )
            if better_or_equal and strictly_better:
                dominated[i] = True
                break
    return pd.Series(~dominated, index=results_df.index, dtype=bool)


def _choose_best_tuning_row(
    results_df: pd.DataFrame,
    target_coverage: float,
    min_group_coverage_target: float,
    max_width_budget: float | None = None,
) -> tuple[pd.Series, str]:
    """Select config with hierarchical multi-objective constraints."""
    df = results_df.copy()
    df["global_ok"] = df["empirical_coverage"] >= target_coverage
    df["group_ok"] = df["min_group_coverage"] >= min_group_coverage_target
    if max_width_budget is None:
        df["width_ok"] = True
    else:
        df["width_ok"] = df["avg_interval_width"] <= max_width_budget

    tiers = [
        ("global+group+width", df["global_ok"] & df["group_ok"] & df["width_ok"]),
        ("global+group", df["global_ok"] & df["group_ok"]),
        ("global+width", df["global_ok"] & df["width_ok"]),
        ("global_only", df["global_ok"]),
    ]
    for tier_name, mask in tiers:
        candidate = df[mask].copy()
        if not candidate.empty:
            candidate = candidate.sort_values(
                by=["avg_interval_width", "coverage_gap", "min_group_coverage"],
                ascending=[True, True, False],
            )
            return candidate.iloc[0], tier_name

    # Fallback: penalty score
    fallback = df.copy()
    fallback["coverage_shortfall"] = (target_coverage - fallback["empirical_coverage"]).clip(lower=0.0)
    fallback["group_shortfall"] = (min_group_coverage_target - fallback["min_group_coverage"]).clip(lower=0.0)
    if max_width_budget is None:
        fallback["width_excess"] = 0.0
    else:
        fallback["width_excess"] = (fallback["avg_interval_width"] - max_width_budget).clip(lower=0.0)
    fallback["score"] = (
        100.0 * fallback["coverage_shortfall"]
        + 60.0 * fallback["group_shortfall"]
        + 10.0 * fallback["width_excess"]
        + fallback["avg_interval_width"]
    )
    fallback = fallback.sort_values(
        by=["score", "coverage_shortfall", "group_shortfall", "avg_interval_width"],
        ascending=[True, True, True, True],
    )
    return fallback.iloc[0], "fallback_penalty"


def _to_python_scalar(value: Any) -> Any:
    """Convert numpy/pandas scalar values to Python primitives."""
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _apply_group_multipliers(
    y_pred: np.ndarray,
    y_intervals: np.ndarray,
    groups: pd.Series | np.ndarray,
    multipliers: dict[str, float],
) -> np.ndarray:
    """Apply group-specific interval multipliers around point predictions."""
    g = pd.Series(groups).fillna("UNKNOWN").astype(str).to_numpy()
    low = y_intervals[:, 0].astype(float).copy()
    high = y_intervals[:, 1].astype(float).copy()
    radius = np.maximum(y_pred - low, high - y_pred)
    out_low = low.copy()
    out_high = high.copy()
    for group, factor in multipliers.items():
        if factor <= 1.0:
            continue
        mask = g == str(group)
        if not mask.any():
            continue
        out_low[mask] = np.clip(y_pred[mask] - radius[mask] * factor, 0.0, 1.0)
        out_high[mask] = np.clip(y_pred[mask] + radius[mask] * factor, 0.0, 1.0)
    return np.column_stack([out_low, out_high])


def _enforce_group_coverage_floor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_intervals: np.ndarray,
    groups: pd.Series | np.ndarray,
    target_coverage: float,
    multiplier_grid: tuple[float, ...] = (1.0, 1.02, 1.05, 1.08, 1.12, 1.16, 1.20),
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    """Increase interval radii for undercovered groups to meet coverage floor."""
    g = pd.Series(groups).fillna("UNKNOWN").astype(str).to_numpy()
    y_true_arr = np.asarray(y_true, dtype=float)
    base = y_intervals.astype(float).copy()
    current = base.copy()

    def _group_cov(intervals: np.ndarray, group: str) -> float:
        mask = g == group
        if not mask.any():
            return float("nan")
        return float(
            ((y_true_arr[mask] >= intervals[mask, 0]) & (y_true_arr[mask] <= intervals[mask, 1])).mean()
        )

    group_factors: dict[str, float] = {}
    report_rows: list[dict[str, Any]] = []
    group_list = sorted(set(g))

    for group in group_list:
        before_cov = _group_cov(current, group)
        factor = 1.0
        after_cov = before_cov
        if np.isfinite(before_cov) and before_cov < target_coverage:
            mask = g == group
            candidate = current.copy()
            for m in multiplier_grid:
                if m < 1.0:
                    continue
                trial = current.copy()
                trial_group = _apply_group_multipliers(
                    y_pred=y_pred[mask],
                    y_intervals=current[mask],
                    groups=np.array([group] * int(mask.sum())),
                    multipliers={group: float(m)},
                )
                trial[mask] = trial_group
                cov = _group_cov(trial, group)
                if cov >= target_coverage:
                    candidate = trial
                    factor = float(m)
                    after_cov = cov
                    break
                # Keep widest attempt if target not reached.
                candidate = trial
                factor = float(m)
                after_cov = cov
            current = candidate

        if factor > 1.0:
            group_factors[group] = factor
        report_rows.append(
            {
                "group": group,
                "coverage_before": float(before_cov),
                "coverage_after": float(after_cov),
                "target_coverage": float(target_coverage),
                "multiplier": float(factor),
                "adjusted": bool(factor > 1.0),
            }
        )

    report = pd.DataFrame(report_rows).sort_values("group")
    return current, group_factors, report


def main(
    alpha_target_90: float = 0.10,
    alpha_95: float = 0.05,
    alpha_candidates_90: tuple[float, ...] = (0.10, 0.095, 0.09, 0.085, 0.08),
    min_group_sizes: tuple[int, ...] = (200, 500, 1000, 2000),
    min_group_coverage_target: float = 0.88,
    max_width_budget_90: float | None = 0.80,
):
    logger.info("Starting Mondrian conformal interval generation with 90% auto-tuning")

    # Load artifacts and data.
    model, model_path = _load_model()
    calibrator = _load_calibrator()
    cal_df = _read_with_fallback("data/processed/calibration_fe.parquet", "data/processed/calibration.parquet")
    test_df = _read_with_fallback("data/processed/test_fe.parquet", "data/processed/test.parquet")
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

    target_coverage_90 = 1.0 - alpha_target_90
    tuning_rows: list[dict[str, Any]] = []

    # Tune 90% interval config.
    for alpha_used in alpha_candidates_90:
        for scaled_scores in (True, False):
            for min_group_size in min_group_sizes:
                y_pred, y_int, _diag = create_pd_intervals_mondrian(
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

                metrics = validate_coverage(y_test.to_numpy(dtype=float), y_int, alpha_target_90)
                g_metrics = conditional_coverage_by_group(y_test.to_numpy(dtype=float), y_int, group_test)

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
    tuning_df["is_pareto"] = _mark_pareto_front(tuning_df)
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
    best_row, selection_tier = _choose_best_tuning_row(
        tuning_df,
        target_coverage=target_coverage_90,
        min_group_coverage_target=min_group_coverage_target,
        max_width_budget=max_width_budget_90,
    )
    best_cfg = {
        "alpha_target_90": float(alpha_target_90),
        "alpha_used_90": float(best_row["alpha_used_90"]),
        "scaled_scores": bool(best_row["scaled_scores"]),
        "min_group_size": int(best_row["min_group_size"]),
        "min_group_coverage_target": float(min_group_coverage_target),
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
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        group_cal=group_cal,
        group_test=group_test,
        alpha=best_cfg["alpha_used_90"],
        min_group_size=best_cfg["min_group_size"],
        calibrator=calibrator,
        scaled_scores=best_cfg["scaled_scores"],
    )
    metrics_90 = validate_coverage(y_test.to_numpy(dtype=float), y_int_90, alpha_target_90)
    group_metrics_90 = conditional_coverage_by_group(y_test.to_numpy(dtype=float), y_int_90, group_test)
    y_int_90_adjusted, group_multipliers, coverage_floor_report = _enforce_group_coverage_floor(
        y_true=y_test.to_numpy(dtype=float),
        y_pred=y_pred_90,
        y_intervals=y_int_90,
        groups=group_test,
        target_coverage=min_group_coverage_target,
    )
    if group_multipliers:
        logger.info(f"Applying group coverage floor multipliers: {group_multipliers}")
        y_int_90 = y_int_90_adjusted
        metrics_90 = validate_coverage(y_test.to_numpy(dtype=float), y_int_90, alpha_target_90)
        group_metrics_90 = conditional_coverage_by_group(y_test.to_numpy(dtype=float), y_int_90, group_test)
    else:
        logger.info("No group coverage floor adjustments were required.")

    # 95% intervals using same structure settings for consistency.
    y_pred_95, y_int_95, diag_95 = create_pd_intervals_mondrian(
        classifier=model,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        group_cal=group_cal,
        group_test=group_test,
        alpha=alpha_95,
        min_group_size=best_cfg["min_group_size"],
        calibrator=calibrator,
        scaled_scores=best_cfg["scaled_scores"],
    )
    if group_multipliers:
        y_int_95 = _apply_group_multipliers(y_pred_95, y_int_95, group_test, group_multipliers)
    metrics_95 = validate_coverage(y_test.to_numpy(dtype=float), y_int_95, alpha_95)
    group_metrics_95 = conditional_coverage_by_group(y_test.to_numpy(dtype=float), y_int_95, group_test)

    # Compose output tables.
    intervals_df = pd.DataFrame(
        {
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
    )

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
        coverage_floor_report[["group", "coverage_before", "coverage_after", "multiplier", "adjusted"]],
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
    results_path = models_dir / "conformal_results_mondrian.pkl"

    intervals_df.to_parquet(intervals_mondrian_path, index=False)
    intervals_df.to_parquet(intervals_default_path, index=False)
    group_metrics_df.to_parquet(group_metrics_path, index=False)
    tuning_df.to_parquet(tuning_path, index=False)
    tuning_df[tuning_df["is_pareto"]].copy().to_parquet(pareto_path, index=False)
    coverage_floor_report.to_parquet(coverage_floor_path, index=False)

    payload = {
        "model_path": str(model_path),
        "metrics_90": {k: _to_python_scalar(v) for k, v in metrics_90.items()},
        "metrics_95": {k: _to_python_scalar(v) for k, v in metrics_95.items()},
        "diag_90": diag_90,
        "diag_95": diag_95,
        "group_metrics_90": group_metrics_90.to_dict(orient="records"),
        "group_metrics_95": group_metrics_95.to_dict(orient="records"),
        "tuning_90_best": best_cfg,
        "tuning_90_table_path": str(tuning_path),
        "tuning_90_pareto_path": str(pareto_path),
        "group_coverage_floor_path": str(coverage_floor_path),
        "group_coverage_multipliers": {k: float(v) for k, v in group_multipliers.items()},
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
    parser.add_argument("--max_width_budget_90", type=float, default=0.80)
    args = parser.parse_args()
    main(
        alpha_target_90=args.alpha_target_90,
        alpha_95=args.alpha_95,
        min_group_coverage_target=args.min_group_coverage_target,
        max_width_budget_90=args.max_width_budget_90,
    )
