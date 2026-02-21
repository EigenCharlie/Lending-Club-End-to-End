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
    y_pred_binary = (np.asarray(y_pred_proba, dtype=float) >= threshold).astype(float)

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
