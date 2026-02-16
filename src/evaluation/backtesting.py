"""Backtesting and out-of-time validation for credit risk models."""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger
from src.evaluation.metrics import classification_metrics


def cohort_analysis(
    df: pd.DataFrame,
    y_true_col: str = "default_flag",
    y_prob_col: str = "pd_predicted",
    cohort_col: str = "issue_quarter",
) -> pd.DataFrame:
    """Evaluate model performance across vintage cohorts."""
    results = []
    for cohort, group in df.groupby(cohort_col):
        if len(group) < 50:
            continue
        metrics = classification_metrics(
            group[y_true_col].values,
            group[y_prob_col].values,
        )
        metrics["cohort"] = cohort
        metrics["n_loans"] = len(group)
        metrics["default_rate"] = group[y_true_col].mean()
        results.append(metrics)

    result = pd.DataFrame(results)
    logger.info(f"Cohort analysis: {len(result)} cohorts evaluated")
    return result


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute PSI to detect distribution drift between train and test."""
    bin_edges = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-6

    expected_pct = np.histogram(expected, bins=bin_edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=bin_edges)[0] / len(actual)

    # Avoid log(0)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    logger.info(f"PSI = {psi:.4f} ({'stable' if psi < 0.1 else 'drift detected' if psi < 0.25 else 'significant drift'})")
    return psi
