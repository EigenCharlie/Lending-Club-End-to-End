"""Conformal interval tuning utilities.

Extracted from scripts/generate_conformal_intervals.py to keep the script
under the 400-line guideline. Contains:
- Calibration split logic for leakage-free hyperparameter tuning.
- Pareto front identification for multi-objective config selection.
- Hierarchical config selection with guardbands.
- Group coverage floor enforcement via interval widening.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


def split_calibration_for_tuning(
    y_cal: pd.Series,
    group_cal: pd.Series,
    issue_dates: pd.Series | None = None,
    holdout_ratio: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split calibration rows into fit/tuning partitions without touching test labels.

    Prefers a temporal split using ``issue_dates`` (latest tail as tuning holdout).
    Falls back to stratified random split when temporal metadata is unavailable or
    would create degenerate class partitions.
    """
    n = int(len(y_cal))
    if n <= 1:
        idx = np.arange(n, dtype=int)
        return idx, np.array([], dtype=int)

    holdout_ratio = float(np.clip(holdout_ratio, 0.05, 0.50))
    idx = np.arange(n, dtype=int)

    n_tune = max(1, int(round(n * holdout_ratio)))
    n_tune = min(n - 1, n_tune)
    y_arr = np.asarray(y_cal, dtype=float)

    if issue_dates is not None:
        issue_dt = pd.to_datetime(issue_dates, errors="coerce")
        valid_dates = int(issue_dt.notna().sum())
        if valid_dates >= max(100, int(0.70 * n)):
            ordered = pd.DataFrame({"idx": idx, "issue_d": issue_dt})
            ordered["issue_d_filled"] = ordered["issue_d"].fillna(pd.Timestamp("1900-01-01"))
            ordered = ordered.sort_values(["issue_d_filled", "idx"]).reset_index(drop=True)

            idx_sorted = ordered["idx"].to_numpy(dtype=int)
            idx_fit = idx_sorted[:-n_tune]
            idx_tune = idx_sorted[-n_tune:]

            fit_classes = np.unique(y_arr[idx_fit].astype(int))
            tune_classes = np.unique(y_arr[idx_tune].astype(int))
            if len(fit_classes) >= 2 and len(tune_classes) >= 2:
                logger.info(
                    "Using temporal calibration holdout by issue_d: "
                    f"valid_dates={valid_dates:,}/{n:,}, holdout_ratio={holdout_ratio:.2%}"
                )
                return np.sort(idx_fit), np.sort(idx_tune)

            logger.warning(
                "Temporal calibration split produced single-class partition; "
                "falling back to stratified random split."
            )

    stratify = (
        pd.Series(group_cal).fillna("UNKNOWN").astype(str)
        + "|"
        + pd.Series(y_cal).astype(int).astype(str)
    )
    try:
        idx_fit, idx_tune = train_test_split(
            idx,
            test_size=holdout_ratio,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        logger.warning(
            "Stratified split failed for calibration holdout; using deterministic random split."
        )
        rng = np.random.default_rng(random_state)
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        idx_tune = shuffled[:n_tune]
        idx_fit = shuffled[n_tune:]

    return np.sort(np.asarray(idx_fit, dtype=int)), np.sort(np.asarray(idx_tune, dtype=int))


def mark_pareto_front(results_df: pd.DataFrame) -> pd.Series:
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
                arr_cov[j] >= arr_cov[i] and arr_grp[j] >= arr_grp[i] and arr_wid[j] <= arr_wid[i]
            )
            strictly_better = (
                arr_cov[j] > arr_cov[i] or arr_grp[j] > arr_grp[i] or arr_wid[j] < arr_wid[i]
            )
            if better_or_equal and strictly_better:
                dominated[i] = True
                break
    return pd.Series(~dominated, index=results_df.index, dtype=bool)


def choose_best_tuning_row(
    results_df: pd.DataFrame,
    target_coverage: float,
    min_group_coverage_target: float,
    max_width_budget: float | None = None,
    coverage_guardband: float = 0.015,
    min_group_guardband: float = 0.0,
) -> tuple[pd.Series, str]:
    """Select config with hierarchical multi-objective constraints."""
    df = results_df.copy()
    df["global_ok"] = df["empirical_coverage"] >= target_coverage
    df["group_ok"] = df["min_group_coverage"] >= min_group_coverage_target
    strong_cov_target = target_coverage + max(0.0, float(coverage_guardband))
    strong_group_target = min_group_coverage_target + max(0.0, float(min_group_guardband))
    df["global_strong"] = df["empirical_coverage"] >= strong_cov_target
    df["group_strong"] = df["min_group_coverage"] >= strong_group_target
    df["coverage_guard_shortfall"] = (strong_cov_target - df["empirical_coverage"]).clip(lower=0.0)
    df["group_guard_shortfall"] = (strong_group_target - df["min_group_coverage"]).clip(lower=0.0)

    if max_width_budget is None:
        df["width_ok"] = True
    else:
        df["width_ok"] = df["avg_interval_width"] <= max_width_budget

    tiers = [
        (
            "strong_global+strong_group+width",
            df["global_strong"] & df["group_strong"] & df["width_ok"],
        ),
        ("strong_global+strong_group", df["global_strong"] & df["group_strong"]),
        ("strong_global+width", df["global_strong"] & df["width_ok"]),
        ("strong_global_only", df["global_strong"]),
        ("global+group+width", df["global_ok"] & df["group_ok"] & df["width_ok"]),
        ("global+group", df["global_ok"] & df["group_ok"]),
        ("global+width", df["global_ok"] & df["width_ok"]),
        ("global_only", df["global_ok"]),
    ]
    for tier_name, mask in tiers:
        candidate = df[mask].copy()
        if not candidate.empty:
            candidate = candidate.sort_values(
                by=[
                    "avg_interval_width",
                    "coverage_guard_shortfall",
                    "group_guard_shortfall",
                    "coverage_gap",
                    "min_group_coverage",
                ],
                ascending=[True, True, True, True, False],
            )
            return candidate.iloc[0], tier_name

    # Fallback: penalty score
    fallback = df.copy()
    fallback["coverage_shortfall"] = (target_coverage - fallback["empirical_coverage"]).clip(
        lower=0.0
    )
    fallback["group_shortfall"] = (min_group_coverage_target - fallback["min_group_coverage"]).clip(
        lower=0.0
    )
    if max_width_budget is None:
        fallback["width_excess"] = 0.0
    else:
        fallback["width_excess"] = (fallback["avg_interval_width"] - max_width_budget).clip(
            lower=0.0
        )
    fallback["score"] = (
        120.0 * fallback["coverage_guard_shortfall"]
        + 80.0 * fallback["group_guard_shortfall"]
        + 40.0 * fallback["coverage_shortfall"]
        + 20.0 * fallback["group_shortfall"]
        + 10.0 * fallback["width_excess"]
        + fallback["avg_interval_width"]
    )
    fallback = fallback.sort_values(
        by=["score", "coverage_shortfall", "group_shortfall", "avg_interval_width"],
        ascending=[True, True, True, True],
    )
    return fallback.iloc[0], "fallback_penalty"


def apply_group_multipliers(
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


def enforce_group_coverage_floor(
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
            (
                (y_true_arr[mask] >= intervals[mask, 0]) & (y_true_arr[mask] <= intervals[mask, 1])
            ).mean()
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
                trial_group = apply_group_multipliers(
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


def to_python_scalar(value: Any) -> Any:
    """Convert numpy/pandas scalar values to Python primitives."""
    if isinstance(value, np.floating | np.integer | np.bool_):
        return value.item()
    return value
