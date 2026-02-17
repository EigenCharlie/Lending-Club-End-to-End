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

# Default-indicating statuses
DEFAULT_STATUSES = ["Charged Off", "Default"]
CURRENT_STATUSES = ["Fully Paid", "Current"]


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load raw Lending Club CSV."""
    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def initial_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leakage columns and filter to resolved loans."""
    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped {len(cols_to_drop)} leakage/irrelevant columns")

    # Filter to resolved loans only (Fully Paid or Default/Charged Off)
    resolved_statuses = DEFAULT_STATUSES + ["Fully Paid"]
    mask = df["loan_status"].isin(resolved_statuses)
    df = df[mask].copy()
    logger.info(f"Filtered to {len(df):,} resolved loans")

    # Create binary target
    df["default_flag"] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    logger.info(f"Default rate: {df['default_flag'].mean():.2%}")

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
