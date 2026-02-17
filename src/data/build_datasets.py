"""Build the three analytical datasets from processed data.

Pipeline: raw parquet -> clean columns -> feature engineering -> 3 datasets

1. loan_master.parquet  — loan-level for PD, LGD, survival analysis
2. time_series.parquet  — monthly aggregates for forecasting
3. ead_dataset.parquet  — defaults only for EAD modeling
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


def clean_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and clean raw Lending Club column formats.

    Handles: int_rate '%' suffix, term 'months' suffix, revol_util '%',
    emp_length parsing, delinquency features.
    """
    df = df.copy()

    # int_rate: ' 13.75%' -> 13.75
    if "int_rate" in df.columns and df["int_rate"].dtype == object:
        df["int_rate"] = df["int_rate"].astype(str).str.strip().str.rstrip("%").astype(float)
        logger.info(f"Parsed int_rate: mean={df['int_rate'].mean():.2f}%")

    # term: ' 36 months' -> 36
    if "term" in df.columns and df["term"].dtype == object:
        df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)

    # revol_util: '55.3%' -> 55.3
    if "revol_util" in df.columns and df["revol_util"].dtype == object:
        df["revol_util"] = df["revol_util"].astype(str).str.strip().str.rstrip("%")
        df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")

    # rev_utilization from revol_util (percentage to fraction)
    if "revol_util" in df.columns and "rev_utilization" not in df.columns:
        df["rev_utilization"] = df["revol_util"] / 100.0

    # delinquency features
    if "delinq_2yrs" in df.columns and "num_delinq_2yrs" not in df.columns:
        df["num_delinq_2yrs"] = df["delinq_2yrs"]

    if "mths_since_last_delinq" in df.columns and "days_since_last_delinq" not in df.columns:
        df["days_since_last_delinq"] = df["mths_since_last_delinq"] * 30

    # dti: ensure numeric
    if "dti" in df.columns:
        df["dti"] = pd.to_numeric(df["dti"], errors="coerce")

    # annual_inc: ensure numeric
    if "annual_inc" in df.columns:
        df["annual_inc"] = pd.to_numeric(df["annual_inc"], errors="coerce")

    logger.info(f"Cleaned raw columns: {df.shape}")
    return df


def build_loan_master(df: pd.DataFrame) -> pd.DataFrame:
    """Build loan-level dataset for PD, LGD, and survival models.

    One row per loan with features available at origination.
    """
    feature_cols = [
        "loan_amnt",
        "annual_inc",
        "loan_to_income",
        "dti",
        "rev_utilization",
        "num_delinq_2yrs",
        "days_since_last_delinq",
        "grade_woe",
        "purpose_woe",
        "home_ownership_woe",
        "int_rate_bucket",
        "int_rate_bucket__grade",
        "int_rate",
        "term",
        "installment",
        # Additional raw features useful for modeling
        "grade",
        "sub_grade",
        "home_ownership",
        "purpose",
        "emp_length",
        "verification_status",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "fico_range_low",
        "fico_range_high",
        "credit_history_months",
        "early_delinq",
        "loan_to_income_sq",
    ]
    target_cols = ["default_flag", "issue_d", "loan_status"]
    id_cols = ["id"] if "id" in df.columns else []

    available = [c for c in feature_cols + target_cols + id_cols if c in df.columns]
    loan_master = df[available].copy()

    logger.info(f"Built loan_master: {loan_master.shape}")
    return loan_master


def build_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Build monthly aggregated dataset for time series forecasting."""
    df = df.copy()
    df["issue_month"] = pd.to_datetime(df["issue_d"]).dt.to_period("M").dt.to_timestamp()

    # Only use numeric columns for aggregation
    agg_dict = {
        "loan_count": ("loan_amnt", "count"),
        "total_amt_funded": ("loan_amnt", "sum"),
        "avg_loan_amnt": ("loan_amnt", "mean"),
        "default_rate": ("default_flag", "mean"),
    }
    if "int_rate" in df.columns and pd.api.types.is_numeric_dtype(df["int_rate"]):
        agg_dict["avg_int_rate"] = ("int_rate", "mean")
    if "dti" in df.columns and pd.api.types.is_numeric_dtype(df["dti"]):
        agg_dict["avg_dti"] = ("dti", "mean")

    ts = df.groupby("issue_month").agg(**agg_dict).reset_index()
    ts = ts.rename(columns={"issue_month": "ds"})
    ts = ts.sort_values("ds").reset_index(drop=True)

    # Nixtla compatibility
    ts["unique_id"] = "portfolio"
    ts["y"] = ts["default_rate"]

    logger.info(f"Built time_series: {ts.shape} ({ts['ds'].min()} to {ts['ds'].max()})")
    return ts


def build_ead_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Build EAD dataset filtered to defaults only."""
    ead = df[df["default_flag"] == 1].copy()
    logger.info(f"Built ead_dataset: {ead.shape} (defaults only)")
    return ead


def save_datasets(
    loan_master: pd.DataFrame,
    time_series: pd.DataFrame,
    ead_dataset: pd.DataFrame,
    output_dir: str | Path = "data/processed/",
) -> None:
    """Save all three datasets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loan_master.to_parquet(output_dir / "loan_master.parquet", index=False)
    time_series.to_parquet(output_dir / "time_series.parquet", index=False)
    ead_dataset.to_parquet(output_dir / "ead_dataset.parquet", index=False)
    logger.info(f"Saved 3 datasets to {output_dir}")


def main(input_path: str = "data/processed/train.parquet", output_dir: str = "data/processed/"):
    """Build all datasets from processed train data."""
    from src.features.feature_engineering import run_feature_pipeline

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    # Step 1: Clean raw column formats
    df = clean_raw_columns(df)

    # Step 2: Run feature engineering pipeline (ratios, buckets, interactions, temporal)
    df = run_feature_pipeline(df)

    # Step 3: Build the 3 analytical datasets
    loan_master = build_loan_master(df)
    time_series = build_time_series(df)
    ead_dataset = build_ead_dataset(df)
    save_datasets(loan_master, time_series, ead_dataset, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/train.parquet")
    parser.add_argument("--output", default="data/processed/")
    args = parser.parse_args()
    main(args.input, args.output)
