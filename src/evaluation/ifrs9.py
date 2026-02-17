"""IFRS9 Expected Credit Loss (ECL) calculation and staging logic.

ECL = PD × LGD × EAD × Discount Factor

Stage 1: No SICR → 12-month PD
Stage 2: SICR detected → Lifetime PD
Stage 3: Credit-impaired → PD ≈ 1
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

# SICR thresholds
SICR_PD_INCREASE_THRESHOLD = 0.02  # Absolute PD increase triggering SICR
SICR_DPD_THRESHOLD = 30  # Days past due for rebuttable presumption
DEFAULT_DPD_THRESHOLD = 90  # Days past due for Stage 3


def assign_stage(
    pd_origination: np.ndarray,
    pd_current: np.ndarray,
    dpd: np.ndarray | None = None,
    pd_high: np.ndarray | None = None,
) -> np.ndarray:
    """Assign IFRS9 stages based on SICR criteria.

    Enhanced with conformal prediction: uses PD_high - PD_hat as SICR signal.

    Args:
        pd_origination: PD at loan origination.
        pd_current: Current PD estimate.
        dpd: Days past due (optional).
        pd_high: Upper conformal bound (optional, for enhanced SICR).

    Returns:
        Array of stages (1, 2, or 3).
    """
    stages = np.ones(len(pd_current), dtype=int)  # Default: Stage 1

    # Stage 2: Significant Increase in Credit Risk
    pd_increase = pd_current - pd_origination
    sicr_mask = pd_increase > SICR_PD_INCREASE_THRESHOLD
    stages[sicr_mask] = 2

    # Enhanced SICR with conformal prediction interval width
    if pd_high is not None:
        uncertainty = pd_high - pd_current
        high_uncertainty = uncertainty > np.percentile(uncertainty, 90)
        stages[sicr_mask & high_uncertainty] = 2  # Confirm Stage 2 with high uncertainty

    # DPD-based staging
    if dpd is not None:
        stages[dpd >= SICR_DPD_THRESHOLD] = 2
        stages[dpd >= DEFAULT_DPD_THRESHOLD] = 3

    logger.info(
        f"Staging: S1={np.sum(stages == 1)}, S2={np.sum(stages == 2)}, S3={np.sum(stages == 3)}"
    )
    return stages


def compute_ecl(
    pd_values: np.ndarray,
    lgd_values: np.ndarray,
    ead_values: np.ndarray,
    stages: np.ndarray,
    lifetime_pd: np.ndarray | None = None,
    discount_rate: float = 0.05,
    horizon_12m: int = 12,
) -> pd.DataFrame:
    """Compute Expected Credit Loss by stage.

    Stage 1: ECL = PD_12m × LGD × EAD
    Stage 2: ECL = PD_lifetime × LGD × EAD
    Stage 3: ECL = 1.0 × LGD × EAD (credit-impaired)

    Args:
        pd_values: 12-month PD estimates.
        lgd_values: LGD estimates.
        ead_values: EAD estimates.
        stages: IFRS9 stages (1, 2, 3).
        lifetime_pd: Lifetime PD for Stage 2 (if None, uses pd * scaling).
        discount_rate: Annual discount rate (EIR).

    Returns:
        DataFrame with ECL calculations per loan.
    """
    n = len(pd_values)
    discount_factor = 1 / (1 + discount_rate)

    # Effective PD by stage
    effective_pd = np.zeros(n)
    effective_pd[stages == 1] = pd_values[stages == 1]  # 12-month PD
    if lifetime_pd is not None:
        effective_pd[stages == 2] = lifetime_pd[stages == 2]
    else:
        effective_pd[stages == 2] = np.minimum(pd_values[stages == 2] * 3, 1.0)  # Scaled proxy
    effective_pd[stages == 3] = 1.0  # Credit-impaired

    ecl = effective_pd * lgd_values * ead_values * discount_factor

    result = pd.DataFrame(
        {
            "stage": stages,
            "pd_12m": pd_values,
            "effective_pd": effective_pd,
            "lgd": lgd_values,
            "ead": ead_values,
            "ecl": ecl,
        }
    )

    # Summary
    for s in [1, 2, 3]:
        mask = result["stage"] == s
        if mask.sum() > 0:
            logger.info(
                f"Stage {s}: n={mask.sum():,}, "
                f"total_ECL={result.loc[mask, 'ecl'].sum():,.0f}, "
                f"avg_ECL={result.loc[mask, 'ecl'].mean():,.2f}"
            )

    return result


def ecl_with_conformal_range(
    pd_low: np.ndarray,
    pd_point: np.ndarray,
    pd_high: np.ndarray,
    lgd: np.ndarray,
    ead: np.ndarray,
    stages: np.ndarray,
) -> pd.DataFrame:
    """Compute ECL range using conformal prediction intervals.

    Provides [ECL_low, ECL_point, ECL_high] for provision planning.
    """
    ecl_low = pd_low * lgd * ead
    ecl_point = pd_point * lgd * ead
    ecl_high = pd_high * lgd * ead

    result = pd.DataFrame(
        {
            "stage": stages,
            "ecl_low": ecl_low,
            "ecl_point": ecl_point,
            "ecl_high": ecl_high,
            "ecl_range": ecl_high - ecl_low,
        }
    )

    logger.info(
        f"ECL range: total_low={ecl_low.sum():,.0f}, "
        f"total_point={ecl_point.sum():,.0f}, "
        f"total_high={ecl_high.sum():,.0f}"
    )
    return result
