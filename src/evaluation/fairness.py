"""Fairness metrics for credit risk models.

Computes demographic parity, equalized odds, and disparate impact
across protected attribute groups. Designed for proxy fairness analysis
(Lending Club has no race/gender data).

Metrics:
    - Demographic Parity Difference (DPD): max gap in positive prediction rate
    - Equalized Odds Gap (EO): max gap in TPR or FPR across groups
    - Disparate Impact Ratio (DIR): min(rate_i / rate_j) across group pairs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

_EPS = np.finfo(float).eps


def demographic_parity_difference(
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | str | dict]:
    """Compute max gap in positive prediction rate across groups.

    Args:
        y_pred: Binary predictions (0/1).
        groups: Group labels per observation.

    Returns:
        Dict with dpd, max_rate_group, min_rate_group, group_rates.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    group_rates: dict[str, float] = {}
    for g in unique_groups:
        mask = groups == g
        group_rates[str(g)] = float(y_pred[mask].mean()) if mask.sum() > 0 else 0.0

    rates = list(group_rates.values())
    dpd = max(rates) - min(rates)
    max_group = max(group_rates, key=group_rates.get)  # type: ignore[arg-type]
    min_group = min(group_rates, key=group_rates.get)  # type: ignore[arg-type]

    return {
        "dpd": dpd,
        "max_rate_group": max_group,
        "min_rate_group": min_group,
        "group_rates": group_rates,
    }


def equalized_odds_gap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | dict]:
    """Compute max gap in TPR and FPR across groups.

    Args:
        y_true: Binary ground truth (0/1).
        y_pred: Binary predictions (0/1).
        groups: Group labels per observation.

    Returns:
        Dict with tpr_gap, fpr_gap, eo_gap (max of both), group_tpr, group_fpr.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    group_tpr: dict[str, float] = {}
    group_fpr: dict[str, float] = {}

    for g in unique_groups:
        mask = groups == g
        yt, yp = y_true[mask], y_pred[mask]

        positives = yt == 1
        negatives = yt == 0

        tpr = float(yp[positives].mean()) if positives.sum() > 0 else 0.0
        fpr = float(yp[negatives].mean()) if negatives.sum() > 0 else 0.0

        group_tpr[str(g)] = tpr
        group_fpr[str(g)] = fpr

    tpr_values = list(group_tpr.values())
    fpr_values = list(group_fpr.values())

    tpr_gap = max(tpr_values) - min(tpr_values) if tpr_values else 0.0
    fpr_gap = max(fpr_values) - min(fpr_values) if fpr_values else 0.0
    eo_gap = max(tpr_gap, fpr_gap)

    return {
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "eo_gap": eo_gap,
        "group_tpr": group_tpr,
        "group_fpr": group_fpr,
    }


def disparate_impact_ratio(
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | str]:
    """Compute min(rate_i / rate_j) for all ordered group pairs.

    The 4/5ths rule (DIR >= 0.80) is a common regulatory threshold.

    Args:
        y_pred: Binary predictions (0/1).
        groups: Group labels per observation.

    Returns:
        Dict with dir, numerator_group, denominator_group.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    group_rates: dict[str, float] = {}
    for g in unique_groups:
        mask = groups == g
        group_rates[str(g)] = float(y_pred[mask].mean()) if mask.sum() > 0 else 0.0

    min_ratio = float("inf")
    num_group, den_group = "", ""

    for gi, ri in group_rates.items():
        for gj, rj in group_rates.items():
            if gi == gj:
                continue
            ratio = ri / (rj + _EPS)
            if ratio < min_ratio:
                min_ratio = ratio
                num_group, den_group = gi, gj

    if min_ratio == float("inf"):
        min_ratio = 1.0

    return {
        "dir": min_ratio,
        "numerator_group": num_group,
        "denominator_group": den_group,
    }


def fairness_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    groups_dict: dict[str, np.ndarray],
    threshold: float = 0.5,
    dpd_threshold: float = 0.10,
    eo_gap_threshold: float = 0.10,
    dir_threshold: float = 0.80,
) -> pd.DataFrame:
    """Run all fairness metrics for multiple group attribute definitions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred_binary = (np.asarray(y_pred_proba, dtype=float) >= threshold).astype(float)
    return fairness_report_from_binary(
        y_true=y_true,
        y_pred_binary=y_pred_binary,
        groups_dict=groups_dict,
        dpd_threshold=dpd_threshold,
        eo_gap_threshold=eo_gap_threshold,
        dir_threshold=dir_threshold,
    )


def fairness_report_from_binary(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    groups_dict: dict[str, np.ndarray],
    dpd_threshold: float = 0.10,
    eo_gap_threshold: float = 0.10,
    dir_threshold: float = 0.80,
) -> pd.DataFrame:
    """Run all fairness metrics for multiple group attribute definitions.

    Args:
        y_true: Binary ground truth (0/1).
        y_pred_proba: Predicted probabilities.
        groups_dict: Mapping of attribute name to group labels array.
        threshold: Probability cutoff for binarization.
        dpd_threshold: Maximum acceptable DPD.
        eo_gap_threshold: Maximum acceptable EO gap.
        dir_threshold: Minimum acceptable DIR (4/5ths rule).

    Returns:
        DataFrame with one row per attribute: attribute, dpd, eo_gap, dir,
        passed_dpd, passed_eo, passed_dir, passed_all.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_binary = np.asarray(y_pred_binary, dtype=float)

    rows: list[dict] = []
    for attr_name, groups in groups_dict.items():
        groups = np.asarray(groups)

        dpd_result = demographic_parity_difference(y_pred_binary, groups)
        eo_result = equalized_odds_gap(y_true, y_pred_binary, groups)
        dir_result = disparate_impact_ratio(y_pred_binary, groups)

        passed_dpd = dpd_result["dpd"] < dpd_threshold
        passed_eo = eo_result["eo_gap"] < eo_gap_threshold
        passed_dir = dir_result["dir"] > dir_threshold

        rows.append(
            {
                "attribute": attr_name,
                "dpd": dpd_result["dpd"],
                "eo_gap": eo_result["eo_gap"],
                "dir": dir_result["dir"],
                "tpr_gap": eo_result["tpr_gap"],
                "fpr_gap": eo_result["fpr_gap"],
                "passed_dpd": passed_dpd,
                "passed_eo": passed_eo,
                "passed_dir": passed_dir,
                "passed_all": passed_dpd and passed_eo and passed_dir,
            }
        )

    logger.info(f"Fairness report: {len(rows)} attributes evaluated")
    return pd.DataFrame(rows)


def build_intersectional_groups(
    groups_dict: dict[str, np.ndarray],
    *,
    max_order: int = 2,
    min_group_size: int = 200,
) -> dict[str, np.ndarray]:
    """Build pairwise intersectional groups from already-resolved attribute arrays."""
    items = [(str(name), np.asarray(values).astype(str)) for name, values in groups_dict.items()]
    if not items or max_order < 2:
        return {}

    n_rows = len(items[0][1])
    out: dict[str, np.ndarray] = {}
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            name_i, values_i = items[i]
            name_j, values_j = items[j]
            if len(values_i) != n_rows or len(values_j) != n_rows:
                continue
            labels = np.array(
                [f"{left} | {right}" for left, right in zip(values_i, values_j, strict=False)],
                dtype=object,
            )
            counts = pd.Series(labels).value_counts()
            allowed = set(counts[counts >= int(min_group_size)].index.astype(str).tolist())
            collapsed = np.array(
                [label if label in allowed else "OTHER_SMALL_GROUP" for label in labels],
                dtype=object,
            )
            out[f"{name_i}__x__{name_j}"] = collapsed
    return out


def fairness_threshold_frontier(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    groups_dict: dict[str, np.ndarray],
    *,
    thresholds: list[float],
    primary_threshold: float,
    dpd_threshold: float = 0.10,
    eo_gap_threshold: float = 0.10,
    dir_threshold: float = 0.80,
) -> pd.DataFrame:
    """Evaluate fairness metrics across multiple thresholds."""
    rows: list[dict[str, float | str | bool]] = []
    for threshold in thresholds:
        report = fairness_report(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            groups_dict=groups_dict,
            threshold=float(threshold),
            dpd_threshold=dpd_threshold,
            eo_gap_threshold=eo_gap_threshold,
            dir_threshold=dir_threshold,
        )
        if report.empty:
            continue
        for _, row in report.iterrows():
            rows.append(
                {
                    "attribute": str(row["attribute"]),
                    "threshold": float(threshold),
                    "is_primary_threshold": bool(
                        abs(float(threshold) - float(primary_threshold)) < 1e-9
                    ),
                    "dpd": float(row["dpd"]),
                    "eo_gap": float(row["eo_gap"]),
                    "dir": float(row["dir"]),
                    "passed_dpd": bool(row["passed_dpd"]),
                    "passed_eo": bool(row["passed_eo"]),
                    "passed_dir": bool(row["passed_dir"]),
                    "passed_all": bool(row["passed_all"]),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "attribute",
                "threshold",
                "is_primary_threshold",
                "dpd",
                "eo_gap",
                "dir",
                "passed_dpd",
                "passed_eo",
                "passed_dir",
                "passed_all",
            ]
        )
    return pd.DataFrame(rows).sort_values(["attribute", "threshold"]).reset_index(drop=True)


def conformal_fairness_report(
    y_true: np.ndarray,
    y_intervals: np.ndarray,
    groups_dict: dict[str, np.ndarray],
    alpha: float = 0.10,
    coverage_disparity_threshold: float = 0.05,
    width_ratio_threshold: float = 2.0,
) -> pd.DataFrame:
    """Evaluate conformal interval fairness across protected groups.

    Checks whether conformal intervals exhibit disparate coverage or width
    across groups. Coverage disparity indicates that some groups receive
    weaker uncertainty guarantees. Width disparity indicates that some
    groups receive less informative (wider) intervals.

    Args:
        y_true: Ground truth values (float).
        y_intervals: Prediction intervals array of shape (n, 2) — [low, high].
        groups_dict: Mapping of attribute name to group labels array.
        alpha: Nominal significance level (for reference).
        coverage_disparity_threshold: Max acceptable coverage gap across groups.
        width_ratio_threshold: Max acceptable max_width/min_width ratio.

    Returns:
        DataFrame with one row per attribute: attribute, n_groups,
        min_coverage, max_coverage, coverage_disparity, min_avg_width,
        max_avg_width, width_ratio, passed_coverage, passed_width, passed_all,
        group_details (dict).
    """
    y_true = np.asarray(y_true, dtype=float)
    low = y_intervals[:, 0]
    high = y_intervals[:, 1]
    covered = (y_true >= low) & (y_true <= high)
    widths = high - low

    rows: list[dict] = []
    for attr_name, groups in groups_dict.items():
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)

        group_details: dict[str, dict[str, float]] = {}
        coverages: list[float] = []
        avg_widths: list[float] = []

        for g in unique_groups:
            mask = groups == g
            n_g = int(mask.sum())
            if n_g == 0:
                continue
            cov_g = float(covered[mask].mean())
            w_g = float(widths[mask].mean())
            group_details[str(g)] = {
                "n": n_g,
                "coverage": cov_g,
                "avg_width": w_g,
            }
            coverages.append(cov_g)
            avg_widths.append(w_g)

        if not coverages:
            continue

        min_cov = min(coverages)
        max_cov = max(coverages)
        cov_disp = max_cov - min_cov
        min_w = min(avg_widths)
        max_w = max(avg_widths)
        w_ratio = max_w / max(min_w, _EPS)

        passed_cov = cov_disp <= coverage_disparity_threshold
        passed_w = w_ratio <= width_ratio_threshold

        rows.append(
            {
                "attribute": attr_name,
                "n_groups": len(unique_groups),
                "min_coverage": min_cov,
                "max_coverage": max_cov,
                "coverage_disparity": cov_disp,
                "min_avg_width": min_w,
                "max_avg_width": max_w,
                "width_ratio": w_ratio,
                "passed_coverage": passed_cov,
                "passed_width": passed_w,
                "passed_all": passed_cov and passed_w,
                "group_details": group_details,
            }
        )

    logger.info(f"Conformal fairness report: {len(rows)} attributes evaluated")
    return pd.DataFrame(rows)
