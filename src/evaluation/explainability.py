"""Reusable explainability utilities for operational governance artifacts."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


def rank_overlap_ratio(reference: list[str], comparison: list[str], top_k: int = 10) -> float:
    """Return normalized overlap between two ranked feature lists."""
    ref = [str(x) for x in reference[:top_k]]
    cmp = [str(x) for x in comparison[:top_k]]
    if not ref or not cmp:
        return 0.0
    denom = max(min(len(ref), len(cmp)), 1)
    return float(len(set(ref).intersection(cmp)) / denom)


def effective_driver_count(
    importances: pd.Series | dict[str, float],
    *,
    coverage: float = 0.80,
) -> int:
    """Count features needed to explain the target importance mass."""
    if isinstance(importances, dict):
        series = pd.Series(importances, dtype=float)
    else:
        series = pd.Series(importances, dtype=float)
    series = series.replace([np.inf, -np.inf], np.nan).dropna().abs().sort_values(ascending=False)
    total = float(series.sum())
    if total <= 0 or series.empty:
        return 0
    cumulative = series.cumsum() / total
    return int((cumulative <= float(coverage)).sum() + 1)


def compute_ale_curve(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute first-order ALE for a numeric feature using model probabilities."""
    if feature not in X.columns:
        return pd.DataFrame()

    values = pd.to_numeric(X[feature], errors="coerce")
    valid_mask = values.notna()
    if int(valid_mask.sum()) < max(n_bins * 5, 50):
        return pd.DataFrame()

    valid_values = values.loc[valid_mask]
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(valid_values.to_numpy(dtype=float), quantiles)
    edges = np.unique(np.asarray(edges, dtype=float))
    if edges.size < 3:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    diffs: list[float] = []

    work = X.loc[valid_mask].copy()
    work_values = valid_values.to_numpy(dtype=float)
    bin_ids = np.digitize(work_values, edges[1:-1], right=True)

    for bin_id in range(len(edges) - 1):
        lower = float(edges[bin_id])
        upper = float(edges[bin_id + 1])
        mask = bin_ids == bin_id
        n_obs = int(mask.sum())
        if n_obs == 0:
            diffs.append(0.0)
            rows.append(
                {
                    "feature": feature,
                    "bin_id": int(bin_id),
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "midpoint": float((lower + upper) / 2.0),
                    "ale_value": 0.0,
                    "n_obs": 0,
                }
            )
            continue

        x_low = work.loc[mask].copy()
        x_high = work.loc[mask].copy()
        x_low[feature] = lower
        x_high[feature] = upper
        pred_low = np.asarray(model.predict_proba(x_low)[:, 1], dtype=float)
        pred_high = np.asarray(model.predict_proba(x_high)[:, 1], dtype=float)
        delta = float(np.mean(pred_high - pred_low))
        diffs.append(delta)
        rows.append(
            {
                "feature": feature,
                "bin_id": int(bin_id),
                "lower_bound": lower,
                "upper_bound": upper,
                "midpoint": float((lower + upper) / 2.0),
                "ale_value": delta,
                "n_obs": n_obs,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["ale_value"] = out["ale_value"].cumsum()
    mean_ale = float(np.average(out["ale_value"], weights=np.maximum(out["n_obs"], 1)))
    out["ale_value"] = out["ale_value"] - mean_ale
    return out


def pairwise_shap_redundancy(
    shap_df: pd.DataFrame,
    features: list[str],
    *,
    max_features: int = 10,
) -> pd.DataFrame:
    """Approximate SHAP interactions using pairwise dependence between contributions."""
    top_features = [str(f) for f in features[:max_features]]
    rows: list[dict[str, float | str | bool]] = []
    for feature_a, feature_b in combinations(top_features, 2):
        shap_a = f"shap_{feature_a}"
        shap_b = f"shap_{feature_b}"
        val_a = f"val_{feature_a}"
        val_b = f"val_{feature_b}"
        if shap_a not in shap_df.columns or shap_b not in shap_df.columns:
            continue
        subset = shap_df[[shap_a, shap_b]].copy()
        shap_corr = subset.corr(method="spearman").iloc[0, 1]
        value_corr = np.nan
        if val_a in shap_df.columns and val_b in shap_df.columns:
            values = shap_df[[val_a, val_b]].apply(pd.to_numeric, errors="coerce")
            if values.notna().sum().min() > 10:
                value_corr = values.corr(method="spearman").iloc[0, 1]
        rows.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "shap_spearman": float(np.nan_to_num(shap_corr)),
                "value_spearman": float(np.nan_to_num(value_corr)),
                "redundancy_flag": bool(
                    abs(float(np.nan_to_num(shap_corr))) >= 0.35
                    or abs(float(np.nan_to_num(value_corr))) >= 0.60
                ),
                "relation_type": (
                    "synergy" if float(np.nan_to_num(shap_corr)) >= 0 else "tradeoff"
                ),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "shap_spearman",
                "value_spearman",
                "redundancy_flag",
                "relation_type",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["redundancy_flag", "shap_spearman"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def extract_top_reason_feature(
    row: pd.Series,
    features: list[str],
    *,
    direction: str,
) -> str:
    """Return the dominant positive or negative reason feature for a row."""
    pairs: list[tuple[str, float]] = []
    for feature in features:
        col = f"shap_{feature}"
        if col not in row.index:
            continue
        raw_value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        value = 0.0 if pd.isna(raw_value) else float(raw_value)
        if direction == "positive" and value > 0 or direction == "negative" and value < 0:
            pairs.append((feature, value))

    if not pairs:
        fallback = []
        for feature in features:
            col = f"shap_{feature}"
            if col not in row.index:
                continue
            raw_value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            value = 0.0 if pd.isna(raw_value) else float(raw_value)
            fallback.append((feature, abs(value)))
        if not fallback:
            return ""
        return max(fallback, key=lambda item: item[1])[0]

    if direction == "positive":
        return max(pairs, key=lambda item: item[1])[0]
    return min(pairs, key=lambda item: item[1])[0]


def dominant_reason_match_rate(
    reference_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    features: list[str],
    *,
    pd_col: str = "pd_calibrated",
    threshold: float = 0.5,
    min_rows_per_band: int = 25,
) -> tuple[float, list[dict[str, str | float | bool]]]:
    """Compare dominant reason codes across PD bands between two cohorts."""
    if reference_df.empty or comparison_df.empty:
        return 0.0, []

    bands = [
        ("low", lambda s: s <= threshold * 0.7),
        ("near_threshold", lambda s: (s > threshold * 0.7) & (s <= threshold * 1.1)),
        ("high", lambda s: s > threshold * 1.1),
    ]

    details: list[dict[str, str | float | bool]] = []
    matches = 0
    valid_bands = 0

    for band_name, selector in bands:
        ref_band = reference_df.loc[selector(pd.to_numeric(reference_df[pd_col], errors="coerce"))]
        cmp_band = comparison_df.loc[
            selector(pd.to_numeric(comparison_df[pd_col], errors="coerce"))
        ]
        if len(ref_band) < min_rows_per_band or len(cmp_band) < min_rows_per_band:
            continue

        ref_reason = ref_band.apply(
            lambda row: extract_top_reason_feature(row, features, direction="positive"),
            axis=1,
        )
        cmp_reason = cmp_band.apply(
            lambda row: extract_top_reason_feature(row, features, direction="positive"),
            axis=1,
        )
        ref_mode = str(ref_reason.mode().iloc[0]) if not ref_reason.mode().empty else ""
        cmp_mode = str(cmp_reason.mode().iloc[0]) if not cmp_reason.mode().empty else ""
        matched = ref_mode != "" and ref_mode == cmp_mode
        valid_bands += 1
        matches += int(matched)
        details.append(
            {
                "band": band_name,
                "reference_reason": ref_mode,
                "comparison_reason": cmp_mode,
                "matched": bool(matched),
            }
        )

    if valid_bands == 0:
        return 0.0, details
    return float(matches / valid_bands), details


def monotonic_violation_rate(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    direction: int,
    *,
    grid_size: int = 7,
    sample_size: int = 128,
    random_state: int = 42,
) -> float:
    """Estimate monotonic violations by varying one feature over a grid."""
    if feature not in X.columns or direction == 0 or X.empty:
        return 0.0

    values = pd.to_numeric(X[feature], errors="coerce").dropna()
    if len(values) < max(grid_size * 5, 50):
        return 0.0

    sample = X.copy()
    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=random_state)

    grid = np.quantile(values.to_numpy(dtype=float), np.linspace(0.05, 0.95, grid_size))
    grid = np.unique(grid)
    if grid.size < 3:
        return 0.0

    preds: list[np.ndarray] = []
    for value in grid:
        x_tmp = sample.copy()
        x_tmp[feature] = value
        preds.append(np.asarray(model.predict_proba(x_tmp)[:, 1], dtype=float))

    stacked = np.vstack(preds)
    diffs = np.diff(stacked, axis=0)
    violations = diffs < -1e-6 if direction > 0 else diffs > 1e-6
    return float(np.mean(violations))


__all__ = [
    "compute_ale_curve",
    "dominant_reason_match_rate",
    "effective_driver_count",
    "extract_top_reason_feature",
    "monotonic_violation_rate",
    "pairwise_shap_redundancy",
    "rank_overlap_ratio",
]
