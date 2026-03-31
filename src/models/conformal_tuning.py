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
            sort_cols = [
                col
                for col in [
                    "coverage_guard_shortfall",
                    "group_guard_shortfall",
                    "coverage_gap",
                    "avg_interval_width",
                    "winkler_90",
                    "max_monthly_gap",
                    "stability_over_time",
                    "min_group_coverage",
                ]
                if col in candidate.columns
            ]
            ascending = [col not in {"min_group_coverage"} for col in sort_cols]
            candidate = candidate.sort_values(
                by=sort_cols,
                ascending=ascending,
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
    fallback["winkler_penalty"] = fallback.get("winkler_90", pd.Series(0.0, index=fallback.index))
    fallback["monthly_gap_penalty"] = fallback.get(
        "max_monthly_gap", pd.Series(0.0, index=fallback.index)
    )
    fallback["stability_penalty"] = fallback.get(
        "stability_over_time", pd.Series(0.0, index=fallback.index)
    )
    fallback["score"] = (
        120.0 * fallback["coverage_guard_shortfall"]
        + 80.0 * fallback["group_guard_shortfall"]
        + 40.0 * fallback["coverage_shortfall"]
        + 20.0 * fallback["group_shortfall"]
        + 10.0 * fallback["width_excess"]
        + 8.0 * fallback["winkler_penalty"]
        + 6.0 * fallback["monthly_gap_penalty"]
        + 4.0 * fallback["stability_penalty"]
        + fallback["avg_interval_width"]
    )
    fallback = fallback.sort_values(
        by=[
            "score",
            "coverage_shortfall",
            "group_shortfall",
            "winkler_penalty",
            "avg_interval_width",
        ],
        ascending=[True, True, True, True, True],
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


def build_group_temporal_segments(
    groups: pd.Series | np.ndarray,
    issue_dates: pd.Series | np.ndarray,
    *,
    freq: str = "Q",
    missing_bucket: str = "UNKNOWN",
) -> pd.Series:
    """Build stable segment keys for group-vintage adjustments."""
    group_series = pd.Series(groups).fillna("UNKNOWN").astype(str).reset_index(drop=True)
    issue_dt = pd.to_datetime(pd.Series(issue_dates), errors="coerce").reset_index(drop=True)
    vintage = issue_dt.dt.to_period(freq).astype(str)
    vintage = vintage.where(issue_dt.notna(), missing_bucket)
    return (group_series + "|vintage=" + vintage.astype(str)).astype(str)


def enforce_segment_coverage_floor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_intervals: np.ndarray,
    segments: pd.Series | np.ndarray,
    target_coverage: float,
    min_segment_size: int = 250,
    multiplier_grid: tuple[float, ...] = (1.0, 1.02, 1.05, 1.08, 1.12, 1.16, 1.20),
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    """Increase interval radii for undercovered temporal segments.

    Segment adjustments are learned only for segments with enough support to avoid
    overfitting tiny slices. Output report includes per-segment support.
    """
    seg = pd.Series(segments).fillna("UNKNOWN").astype(str).to_numpy()
    y_true_arr = np.asarray(y_true, dtype=float)
    current = np.asarray(y_intervals, dtype=float).copy()
    min_segment_size = max(1, int(min_segment_size))

    def _mask_for(segment: str) -> np.ndarray:
        return seg == segment

    def _segment_cov(intervals: np.ndarray, segment: str) -> float:
        mask = _mask_for(segment)
        if not mask.any():
            return float("nan")
        return float(
            (
                (y_true_arr[mask] >= intervals[mask, 0]) & (y_true_arr[mask] <= intervals[mask, 1])
            ).mean()
        )

    segment_factors: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    segment_list = sorted(set(seg))

    for segment in segment_list:
        mask = _mask_for(segment)
        support = int(mask.sum())
        before_cov = _segment_cov(current, segment)
        factor = 1.0
        after_cov = before_cov
        adjusted = False

        if support >= min_segment_size and np.isfinite(before_cov) and before_cov < target_coverage:
            candidate = current.copy()
            for m in multiplier_grid:
                if m < 1.0:
                    continue
                trial = current.copy()
                trial_segment = apply_group_multipliers(
                    y_pred=y_pred[mask],
                    y_intervals=current[mask],
                    groups=np.array([segment] * support),
                    multipliers={segment: float(m)},
                )
                trial[mask] = trial_segment
                cov = _segment_cov(trial, segment)
                candidate = trial
                factor = float(m)
                after_cov = cov
                if cov >= target_coverage:
                    break
            current = candidate
            adjusted = factor > 1.0

        if factor > 1.0:
            segment_factors[str(segment)] = factor
        rows.append(
            {
                "segment": str(segment),
                "support_n": support,
                "coverage_before": float(before_cov),
                "coverage_after": float(after_cov),
                "target_coverage": float(target_coverage),
                "min_segment_size": int(min_segment_size),
                "multiplier": float(factor),
                "adjusted": bool(adjusted),
            }
        )

    report = pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)
    return current, segment_factors, report


def to_python_scalar(value: Any) -> Any:
    """Convert numpy/pandas scalar values to Python primitives."""
    if isinstance(value, np.floating | np.integer | np.bool_):
        return value.item()
    return value


def empirical_interval_coverage(y_true: np.ndarray, y_intervals: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    intervals = np.asarray(y_intervals, dtype=float)
    if len(y_true_arr) == 0 or len(intervals) == 0:
        return float("nan")
    inside = (y_true_arr >= intervals[:, 0]) & (y_true_arr <= intervals[:, 1])
    return float(np.mean(inside))


def min_group_interval_coverage(
    y_true: np.ndarray,
    y_intervals: np.ndarray,
    groups: pd.Series | np.ndarray,
) -> float:
    g = pd.Series(groups).fillna("UNKNOWN").astype(str).reset_index(drop=True)
    y_true_arr = np.asarray(y_true, dtype=float)
    intervals = np.asarray(y_intervals, dtype=float)
    covs: list[float] = []
    for group in sorted(g.unique()):
        mask = g == group
        if not mask.any():
            continue
        inside = (y_true_arr[mask] >= intervals[mask, 0]) & (y_true_arr[mask] <= intervals[mask, 1])
        covs.append(float(np.mean(inside)))
    return float(min(covs)) if covs else float("nan")


def average_interval_width(y_intervals: np.ndarray) -> float:
    intervals = np.asarray(y_intervals, dtype=float)
    if len(intervals) == 0:
        return float("nan")
    return float(np.mean(intervals[:, 1] - intervals[:, 0]))


def mean_winkler_score(
    y_true: np.ndarray,
    y_intervals: np.ndarray,
    *,
    alpha: float,
) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    intervals = np.asarray(y_intervals, dtype=float)
    if len(y_true_arr) == 0 or len(intervals) == 0:
        return float("inf")
    low = intervals[:, 0]
    high = intervals[:, 1]
    width = np.maximum(high - low, 0.0)
    below = np.maximum(low - y_true_arr, 0.0)
    above = np.maximum(y_true_arr - high, 0.0)
    penalty = (2.0 / max(float(alpha), 1e-12)) * (below + above)
    return float(np.mean(width + penalty))


def temporal_stability_summary(
    y_true: np.ndarray,
    y_intervals: np.ndarray,
    issue_dates: pd.Series | np.ndarray | None,
    *,
    target_coverage: float,
    freq: str = "M",
) -> dict[str, float]:
    if issue_dates is None:
        return {
            "min_monthly_coverage": float("nan"),
            "last_monthly_coverage": float("nan"),
            "max_monthly_gap": float("nan"),
            "stability_over_time": float("nan"),
        }
    dates = pd.to_datetime(pd.Series(issue_dates), errors="coerce")
    y_true_arr = np.asarray(y_true, dtype=float)
    intervals = np.asarray(y_intervals, dtype=float)
    if len(dates) == 0 or len(y_true_arr) == 0 or len(intervals) == 0:
        return {
            "min_monthly_coverage": float("nan"),
            "last_monthly_coverage": float("nan"),
            "max_monthly_gap": float("nan"),
            "stability_over_time": float("nan"),
        }
    frame = pd.DataFrame(
        {
            "month": dates.dt.to_period(freq).dt.to_timestamp(),
            "y_true": y_true_arr,
            "low": intervals[:, 0],
            "high": intervals[:, 1],
        }
    ).dropna(subset=["month"])
    if frame.empty:
        return {
            "min_monthly_coverage": float("nan"),
            "last_monthly_coverage": float("nan"),
            "max_monthly_gap": float("nan"),
            "stability_over_time": float("nan"),
        }
    frame["covered"] = (
        (frame["y_true"] >= frame["low"]) & (frame["y_true"] <= frame["high"])
    ).astype(float)
    monthly = (
        frame.groupby("month", observed=True)
        .agg(coverage=("covered", "mean"))
        .reset_index()
        .sort_values("month")
    )
    monthly["gap"] = (monthly["coverage"] - float(target_coverage)).abs()
    return {
        "min_monthly_coverage": float(monthly["coverage"].min()),
        "last_monthly_coverage": float(monthly["coverage"].iloc[-1]),
        "max_monthly_gap": float(monthly["gap"].max()),
        "stability_over_time": float(monthly["gap"].mean()),
    }


def shrink_group_multipliers(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    base_intervals: np.ndarray,
    groups: pd.Series | np.ndarray,
    issue_dates: pd.Series | np.ndarray | None,
    group_factors: dict[str, float] | None = None,
    temporal_segments: pd.Series | np.ndarray | None = None,
    temporal_factors: dict[str, float] | None = None,
    target_coverage: float = 0.90,
    min_group_coverage_target: float = 0.88,
    max_monthly_gap_target: float | None = None,
    alpha: float = 0.10,
    group_multiplier_grid: tuple[float, ...] = (1.0, 1.02, 1.05, 1.08, 1.12, 1.16, 1.20),
    temporal_multiplier_grid: tuple[float, ...] = (1.0, 1.02, 1.05, 1.08, 1.12, 1.16, 1.20),
) -> tuple[np.ndarray, dict[str, float], dict[str, float], pd.DataFrame]:
    """Greedily shrink learned widening factors while preserving constraints."""
    group_factors_cur = {
        str(k): float(v) for k, v in (group_factors or {}).items() if float(v) > 1.0
    }
    temporal_factors_cur = {
        str(k): float(v) for k, v in (temporal_factors or {}).items() if float(v) > 1.0
    }
    base = np.asarray(base_intervals, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_true_arr = np.asarray(y_true, dtype=float)
    group_series = pd.Series(groups).fillna("UNKNOWN").astype(str).reset_index(drop=True)
    temporal_series = (
        pd.Series(temporal_segments).fillna("UNKNOWN").astype(str).reset_index(drop=True)
        if temporal_segments is not None
        else None
    )

    def _apply_all(
        gf: dict[str, float],
        tf: dict[str, float],
    ) -> np.ndarray:
        intervals = base.copy()
        if gf:
            intervals = apply_group_multipliers(y_pred_arr, intervals, group_series, gf)
        if tf and temporal_series is not None:
            intervals = apply_group_multipliers(y_pred_arr, intervals, temporal_series, tf)
        return intervals

    def _metrics(intervals: np.ndarray) -> dict[str, float]:
        temporal = temporal_stability_summary(
            y_true_arr,
            intervals,
            issue_dates,
            target_coverage=float(target_coverage),
            freq="M",
        )
        return {
            "coverage": empirical_interval_coverage(y_true_arr, intervals),
            "min_group_coverage": min_group_interval_coverage(y_true_arr, intervals, group_series),
            "avg_width": average_interval_width(intervals),
            "winkler_90": mean_winkler_score(y_true_arr, intervals, alpha=alpha),
            "max_monthly_gap": float(temporal["max_monthly_gap"]),
            "stability_over_time": float(temporal["stability_over_time"]),
        }

    def _constraints_ok(metrics: dict[str, float]) -> bool:
        if float(metrics["coverage"]) < float(target_coverage):
            return False
        if float(metrics["min_group_coverage"]) < float(min_group_coverage_target):
            return False
        return not (
            max_monthly_gap_target is not None
            and np.isfinite(max_monthly_gap_target)
            and float(metrics["max_monthly_gap"]) > float(max_monthly_gap_target)
        )

    current_intervals = _apply_all(group_factors_cur, temporal_factors_cur)
    current_metrics = _metrics(current_intervals)
    report_rows: list[dict[str, Any]] = [
        {
            "stage": "initial",
            "factor_scope": "all",
            "factor_key": "all",
            "candidate_factor": np.nan,
            "accepted": True,
            **current_metrics,
        }
    ]

    if not _constraints_ok(current_metrics):
        report_rows.append(
            {
                "stage": "initial_infeasible",
                "factor_scope": "all",
                "factor_key": "all",
                "candidate_factor": np.nan,
                "accepted": False,
                **current_metrics,
            }
        )
        return current_intervals, group_factors_cur, temporal_factors_cur, pd.DataFrame(report_rows)

    def _next_lower(value: float, grid: tuple[float, ...]) -> float | None:
        ordered = sorted({round(float(x), 6) for x in grid if float(x) <= float(value) + 1e-9})
        current = round(float(value), 6)
        if current not in ordered:
            ordered.append(current)
            ordered = sorted(set(ordered))
        idx = ordered.index(current)
        if idx == 0:
            return None
        return float(ordered[idx - 1])

    while True:
        best_candidate: dict[str, Any] | None = None

        for key, value in list(group_factors_cur.items()):
            next_factor = _next_lower(value, group_multiplier_grid)
            if next_factor is None:
                continue
            trial_group = dict(group_factors_cur)
            if next_factor <= 1.0:
                trial_group.pop(key, None)
            else:
                trial_group[key] = next_factor
            trial_intervals = _apply_all(trial_group, temporal_factors_cur)
            trial_metrics = _metrics(trial_intervals)
            accepted = _constraints_ok(trial_metrics)
            candidate = {
                "scope": "group",
                "key": key,
                "factor": next_factor,
                "accepted": accepted,
                "intervals": trial_intervals,
                "group_factors": trial_group,
                "temporal_factors": dict(temporal_factors_cur),
                "metrics": trial_metrics,
            }
            if accepted and (
                best_candidate is None
                or float(candidate["metrics"]["avg_width"])
                < float(best_candidate["metrics"]["avg_width"])
                or (
                    np.isclose(
                        float(candidate["metrics"]["avg_width"]),
                        float(best_candidate["metrics"]["avg_width"]),
                    )
                    and float(candidate["metrics"]["winkler_90"])
                    < float(best_candidate["metrics"]["winkler_90"])
                )
            ):
                best_candidate = candidate
            report_rows.append(
                {
                    "stage": "attempt",
                    "factor_scope": "group",
                    "factor_key": key,
                    "candidate_factor": float(next_factor),
                    "accepted": bool(accepted),
                    **trial_metrics,
                }
            )

        for key, value in list(temporal_factors_cur.items()):
            next_factor = _next_lower(value, temporal_multiplier_grid)
            if next_factor is None:
                continue
            trial_temporal = dict(temporal_factors_cur)
            if next_factor <= 1.0:
                trial_temporal.pop(key, None)
            else:
                trial_temporal[key] = next_factor
            trial_intervals = _apply_all(group_factors_cur, trial_temporal)
            trial_metrics = _metrics(trial_intervals)
            accepted = _constraints_ok(trial_metrics)
            candidate = {
                "scope": "temporal",
                "key": key,
                "factor": next_factor,
                "accepted": accepted,
                "intervals": trial_intervals,
                "group_factors": dict(group_factors_cur),
                "temporal_factors": trial_temporal,
                "metrics": trial_metrics,
            }
            if accepted and (
                best_candidate is None
                or float(candidate["metrics"]["avg_width"])
                < float(best_candidate["metrics"]["avg_width"])
                or (
                    np.isclose(
                        float(candidate["metrics"]["avg_width"]),
                        float(best_candidate["metrics"]["avg_width"]),
                    )
                    and float(candidate["metrics"]["winkler_90"])
                    < float(best_candidate["metrics"]["winkler_90"])
                )
            ):
                best_candidate = candidate
            report_rows.append(
                {
                    "stage": "attempt",
                    "factor_scope": "temporal",
                    "factor_key": key,
                    "candidate_factor": float(next_factor),
                    "accepted": bool(accepted),
                    **trial_metrics,
                }
            )

        if best_candidate is None:
            break

        current_intervals = np.asarray(best_candidate["intervals"], dtype=float)
        group_factors_cur = dict(best_candidate["group_factors"])
        temporal_factors_cur = dict(best_candidate["temporal_factors"])
        current_metrics = dict(best_candidate["metrics"])
        report_rows.append(
            {
                "stage": "accepted",
                "factor_scope": str(best_candidate["scope"]),
                "factor_key": str(best_candidate["key"]),
                "candidate_factor": float(best_candidate["factor"]),
                "accepted": True,
                **current_metrics,
            }
        )

    report_rows.append(
        {
            "stage": "final",
            "factor_scope": "all",
            "factor_key": "all",
            "candidate_factor": np.nan,
            "accepted": True,
            **current_metrics,
        }
    )
    report = pd.DataFrame(report_rows)
    return current_intervals, group_factors_cur, temporal_factors_cur, report
