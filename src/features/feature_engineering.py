"""Feature engineering: WOE encoding, ratios, buckets, interactions.

Uses OptBinning for mathematically optimal WOE binning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


def create_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Create financial ratio features."""
    df = df.copy()
    df["loan_to_income"] = np.where(
        df["annual_inc"] > 0, df["loan_amnt"] / df["annual_inc"], np.nan
    )
    if "rev_utilization" not in df.columns and "revol_util" in df.columns:
        # revol_util is already percentage (0-100), convert to fraction
        df["rev_utilization"] = pd.to_numeric(df["revol_util"], errors="coerce") / 100.0
    elif "rev_utilization" not in df.columns and "revol_bal" in df.columns:
        if "total_rev_hi_lim" in df.columns:
            df["rev_utilization"] = np.where(
                df["total_rev_hi_lim"] > 0,
                df["revol_bal"] / df["total_rev_hi_lim"],
                np.nan,
            )
    logger.info("Created ratio features")
    return df


def create_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Create bucketed features for interpretability."""
    df = df.copy()
    if "int_rate" in df.columns:
        df["int_rate_bucket"] = pd.cut(
            df["int_rate"],
            bins=[0, 8, 12, 16, 20, 100],
            labels=["very_low", "low", "medium", "high", "very_high"],
        )
    if "dti" in df.columns:
        df["dti_bucket"] = pd.cut(
            df["dti"],
            bins=[-1, 10, 20, 30, 40, 100],
            labels=["low", "moderate", "high", "very_high", "extreme"],
        )
    logger.info("Created bucket features")
    return df


def create_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features."""
    df = df.copy()
    if "int_rate_bucket" in df.columns:
        grade_col = "grade_woe" if "grade_woe" in df.columns else "grade"
        if grade_col in df.columns:
            df["int_rate_bucket__grade"] = (
                df["int_rate_bucket"].astype(str) + "__" + df[grade_col].astype(str)
            )
    if "loan_to_income" in df.columns:
        df["loan_to_income_sq"] = df["loan_to_income"] ** 2
    logger.info("Created interaction features")
    return df


def compute_woe(
    df: pd.DataFrame,
    feature: str,
    target: str = "default_flag",
    **optbinning_kwargs: Any,
) -> tuple[pd.Series, Any]:
    """Compute WOE encoding using OptBinning.

    Returns:
        Tuple of (woe_encoded_series, fitted_optbinning_object)
    """
    from optbinning import OptimalBinning

    optb = OptimalBinning(
        name=feature,
        dtype="numerical" if df[feature].dtype in ["float64", "int64"] else "categorical",
        solver="cp",  # constraint programming for optimal bins
        **optbinning_kwargs,
    )
    optb.fit(df[feature].values, df[target].values)
    woe_values = optb.transform(df[feature].values, metric="woe")

    iv = optb.binning_table.build()["IV"].sum() if hasattr(optb, "binning_table") else None
    logger.info(f"WOE for {feature}: IV={iv:.4f}" if iv else f"WOE computed for {feature}")

    return pd.Series(woe_values, index=df.index, name=f"{feature}_woe"), optb


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features for survival/temporal analysis."""
    df = df.copy()
    if "earliest_cr_line" in df.columns and "issue_d" in df.columns:
        df["credit_history_months"] = (
            (pd.to_datetime(df["issue_d"]) - pd.to_datetime(df["earliest_cr_line"])).dt.days / 30
        ).clip(lower=0)

    if "delinq_2yrs" in df.columns:
        df["early_delinq"] = (df["delinq_2yrs"] > 0).astype(int)

    logger.info("Created temporal features")
    return df


def run_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    df = create_ratios(df)
    df = create_buckets(df)
    df = create_interactions(df)
    df = create_temporal_features(df)
    logger.info(f"Feature pipeline complete: {df.shape}")
    return df
