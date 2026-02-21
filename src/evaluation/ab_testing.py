"""A/B testing simulation framework for portfolio strategy comparison.

Provides tools for power analysis, stratified treatment assignment,
and statistical comparison of two competing portfolio strategies
applied retroactively to the OOT test set.

This is a SIMULATION framework — no live randomization is implied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


def power_analysis(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    n_groups: int = 2,
) -> dict[str, float]:
    """Estimate minimum sample size per group for a two-sample proportion test.

    Uses normal approximation for the difference of two proportions.

    Args:
        effect_size: Expected difference in means between groups.
        alpha: Significance level (Type I error rate).
        power: Statistical power (1 - Type II error rate).
        n_groups: Number of groups (default 2 for A/B).

    Returns:
        Dict with n_per_group, total_n, effect_size, alpha, power.
    """
    z_alpha = stats.norm.ppf(1 - alpha / n_groups)
    z_beta = stats.norm.ppf(power)
    # Standard formula for two-sample mean comparison
    # n = ((z_alpha + z_beta) / effect_size)^2 * 2 * sigma^2
    # For proportion test, sigma^2 ≈ p*(1-p) ≈ 0.25 (conservative)
    variance = 0.25  # conservative for proportions
    n_per_group = int(np.ceil(2 * variance * ((z_alpha + z_beta) / (effect_size + 1e-12)) ** 2))

    result = {
        "n_per_group": float(n_per_group),
        "total_n": float(n_per_group * n_groups),
        "effect_size": effect_size,
        "alpha": alpha,
        "power": power,
    }
    logger.info(f"Power analysis: n_per_group={n_per_group} for effect_size={effect_size:.4f}")
    return result


def stratified_split(
    df: pd.DataFrame,
    strata_col: str,
    treatment_ratio: float = 0.50,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Balanced treatment/control assignment within strata.

    Args:
        df: DataFrame to split.
        strata_col: Column to stratify on (e.g., 'grade').
        treatment_ratio: Proportion assigned to treatment group.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (control_df, treatment_df) with no index overlap.
    """
    rng = np.random.RandomState(seed)
    control_idx: list[int] = []
    treatment_idx: list[int] = []

    for _, group_df in df.groupby(strata_col):
        indices = group_df.index.tolist()
        rng.shuffle(indices)
        n_treatment = int(np.floor(len(indices) * treatment_ratio))
        treatment_idx.extend(indices[:n_treatment])
        control_idx.extend(indices[n_treatment:])

    control = df.loc[control_idx].copy()
    treatment = df.loc[treatment_idx].copy()

    logger.info(
        f"Stratified split on '{strata_col}': control={len(control)}, treatment={len(treatment)}"
    )
    return control, treatment


def compare_strategies(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    method: str = "bootstrap",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float | bool]:
    """Statistical comparison of two return distributions.

    Args:
        returns_a: Per-loan realized returns for strategy A (control).
        returns_b: Per-loan realized returns for strategy B (treatment).
        method: One of 'bootstrap', 'permutation', 'ttest'.
        n_boot: Number of bootstrap/permutation resamples.
        alpha: Significance level.
        seed: Random seed.

    Returns:
        Dict with mean_a, mean_b, diff, ci_low, ci_high, p_value, significant.
    """
    returns_a = np.asarray(returns_a, dtype=float)
    returns_b = np.asarray(returns_b, dtype=float)
    rng = np.random.RandomState(seed)

    mean_a = float(np.mean(returns_a))
    mean_b = float(np.mean(returns_b))
    observed_diff = mean_b - mean_a

    if method == "bootstrap":
        diffs = np.empty(n_boot)
        for i in range(n_boot):
            idx_a = rng.randint(0, len(returns_a), len(returns_a))
            idx_b = rng.randint(0, len(returns_b), len(returns_b))
            diffs[i] = np.mean(returns_b[idx_b]) - np.mean(returns_a[idx_a])
        ci_low = float(np.percentile(diffs, 100 * alpha / 2))
        ci_high = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
        p_value = float(np.mean(diffs <= 0)) if observed_diff > 0 else float(np.mean(diffs >= 0))

    elif method == "permutation":
        combined = np.concatenate([returns_a, returns_b])
        n_a = len(returns_a)
        perm_diffs = np.empty(n_boot)
        for i in range(n_boot):
            rng.shuffle(combined)
            perm_diffs[i] = np.mean(combined[n_a:]) - np.mean(combined[:n_a])
        ci_low = float(np.percentile(perm_diffs, 100 * alpha / 2))
        ci_high = float(np.percentile(perm_diffs, 100 * (1 - alpha / 2)))
        p_value = float(np.mean(np.abs(perm_diffs) >= abs(observed_diff)))

    elif method == "ttest":
        t_stat, p_value = stats.ttest_ind(returns_b, returns_a, equal_var=False)
        se = np.sqrt(np.var(returns_b) / len(returns_b) + np.var(returns_a) / len(returns_a))
        z = stats.norm.ppf(1 - alpha / 2)
        ci_low = float(observed_diff - z * se)
        ci_high = float(observed_diff + z * se)
        p_value = float(p_value)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bootstrap', 'permutation', or 'ttest'.")

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "diff": observed_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "significant": p_value < alpha,
    }


def ab_summary(
    control_metrics: dict[str, float],
    treatment_metrics: dict[str, float],
) -> pd.DataFrame:
    """Format control vs treatment metrics as a tidy comparison table.

    Args:
        control_metrics: Dict of metric_name → value for control.
        treatment_metrics: Dict of metric_name → value for treatment.

    Returns:
        DataFrame with columns: metric, control, treatment, lift_pct.
    """
    all_keys = sorted(set(control_metrics) | set(treatment_metrics))
    rows = []
    for key in all_keys:
        c = control_metrics.get(key, 0.0)
        t = treatment_metrics.get(key, 0.0)
        lift = ((t - c) / (abs(c) + 1e-12)) * 100
        rows.append({"metric": key, "control": c, "treatment": t, "lift_pct": lift})
    return pd.DataFrame(rows)
