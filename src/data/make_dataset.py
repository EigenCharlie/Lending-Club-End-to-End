"""Download raw data and perform initial cleaning.

Usage:
    uv run python -m src.data.make_dataset --input data/raw/lending_club.csv --output data/interim/
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

# Columns to drop immediately (known leakage or irrelevant)
LEAKAGE_COLS = [
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "last_credit_pull_d",
    "out_prncp",
    "out_prncp_inv",
    "funded_amnt",
    "funded_amnt_inv",
    "total_bal_il",
    "il_util",
    "max_bal_bc",
    "all_util",
    "total_rev_hi_lim",
    "debt_settlement_flag",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "hardship_amount",
    "hardship_start_date",
    "hardship_end_date",
    "hardship_length",
    "hardship_dpd",
    "hardship_loan_status",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",
    "payment_plan_start_date",
]

LGD_SNAPSHOT_DATE = pd.Timestamp("2020-09-30")

# Default-indicating statuses
DEFAULT_STATUSES = ["Charged Off", "Default"]
CURRENT_STATUSES = ["Fully Paid", "Current"]


def _to_numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _parse_issue_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%b-%Y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series.loc[missing], errors="coerce")
    return parsed


def _compute_lgd(df: pd.DataFrame) -> pd.Series:
    """Compute realized LGD in [0, 1] using principal recovery components.

    Formula:
        LGD = 1 - (total_rec_prncp + recoveries - collection_recovery_fee) / exposure
    where exposure uses `funded_amnt` if available, otherwise `loan_amnt`.

    Notes:
    - LGD is a default-conditional target, but we persist 0.0 for non-default rows.
    - Values are clipped to [0, 1] to avoid data-quality outliers.
    """
    exposure = _to_numeric_series(df, "funded_amnt", default=0.0)
    if "loan_amnt" in df.columns:
        exposure = exposure.where(exposure > 0.0, _to_numeric_series(df, "loan_amnt", default=0.0))
    exposure = exposure.where(exposure > 0.0, pd.NA)

    total_rec_prncp = _to_numeric_series(df, "total_rec_prncp", default=0.0)
    recoveries = _to_numeric_series(df, "recoveries", default=0.0)
    collection_fee = _to_numeric_series(df, "collection_recovery_fee", default=0.0)
    recovered_principal = total_rec_prncp + recoveries - collection_fee

    lgd = 1.0 - (recovered_principal / exposure)
    lgd = lgd.clip(lower=0.0, upper=1.0)

    if "default_flag" in df.columns:
        lgd = lgd.where(df["default_flag"].astype(int) == 1, 0.0)
    lgd = lgd.fillna(1.0).clip(lower=0.0, upper=1.0)
    return lgd.astype(float)


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load raw Lending Club CSV."""
    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def initial_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leakage columns and filter to resolved loans."""
    # Filter to resolved loans only (Fully Paid or Default/Charged Off)
    resolved_statuses = DEFAULT_STATUSES + ["Fully Paid"]
    mask = df["loan_status"].isin(resolved_statuses)
    df = df[mask].copy()
    logger.info(f"Filtered to {len(df):,} resolved loans")

    # Create binary target
    df["default_flag"] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    logger.info(f"Default rate: {df['default_flag'].mean():.2%}")

    # Build LGD target before leakage fields are dropped.
    df["lgd"] = _compute_lgd(df)
    if "issue_d" in df.columns:
        issue_dt = _parse_issue_dates(df["issue_d"])
        age_months = ((LGD_SNAPSHOT_DATE - issue_dt).dt.days.astype(float) / 30.4375).clip(
            lower=0.0
        )
        df["lgd_months_since_issue"] = age_months.fillna(0.0).astype(float)
        df["lgd_is_mature_24m"] = (age_months >= 24.0).fillna(False).astype(int)
    lgd_default = df.loc[df["default_flag"] == 1, "lgd"]
    if not lgd_default.empty:
        logger.info(
            "LGD target built: default_rows={}, mean={:.4f}, p50={:.4f}",
            len(lgd_default),
            float(lgd_default.mean()),
            float(lgd_default.median()),
        )

    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped {len(cols_to_drop)} leakage/irrelevant columns")

    return df


def save_interim(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Save cleaned data to interim directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "lending_club_cleaned.parquet"
    df.to_parquet(filepath, index=False)
    logger.info(f"Saved interim data to {filepath} ({len(df):,} rows)")
    return filepath


def main(
    input_path: str = "data/raw/Loan_status_2007-2020Q3.csv", output_dir: str = "data/interim/"
) -> None:
    """Run full make_dataset pipeline."""
    df = load_raw_data(input_path)
    df = initial_clean(df)
    save_interim(df, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Make dataset from raw Lending Club CSV")
    parser.add_argument("--input", default="data/raw/Loan_status_2007-2020Q3.csv")
    parser.add_argument("--output", default="data/interim/")
    args = parser.parse_args()
    main(args.input, args.output)
