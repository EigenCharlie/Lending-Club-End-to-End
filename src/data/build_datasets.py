"""Build analytical datasets used across the credit-risk pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def clean_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and clean raw Lending Club column formats."""
    df = df.copy()

    def _is_text_dtype(col: pd.Series) -> bool:
        return pd.api.types.is_object_dtype(col.dtype) or pd.api.types.is_string_dtype(col.dtype)

    if "int_rate" in df.columns and _is_text_dtype(df["int_rate"]):
        df["int_rate"] = df["int_rate"].astype(str).str.strip().str.rstrip("%").astype(float)
    if "term" in df.columns and _is_text_dtype(df["term"]):
        df["term"] = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    if "revol_util" in df.columns and _is_text_dtype(df["revol_util"]):
        df["revol_util"] = df["revol_util"].astype(str).str.strip().str.rstrip("%")
        df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")
    if "revol_util" in df.columns and "rev_utilization" not in df.columns:
        df["rev_utilization"] = df["revol_util"] / 100.0
    if "delinq_2yrs" in df.columns and "num_delinq_2yrs" not in df.columns:
        df["num_delinq_2yrs"] = df["delinq_2yrs"]
    if "mths_since_last_delinq" in df.columns and "days_since_last_delinq" not in df.columns:
        df["days_since_last_delinq"] = df["mths_since_last_delinq"] * 30
    if "dti" in df.columns:
        df["dti"] = pd.to_numeric(df["dti"], errors="coerce")
    if "annual_inc" in df.columns:
        df["annual_inc"] = pd.to_numeric(df["annual_inc"], errors="coerce")
    if "loan_amnt" in df.columns:
        df["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce")
    if "issue_d" in df.columns:
        df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    if "default_flag" in df.columns:
        df["default_flag"] = (
            pd.to_numeric(df["default_flag"], errors="coerce").fillna(0).astype(int)
        )
    return df


def build_loan_master(df: pd.DataFrame) -> pd.DataFrame:
    """Build loan-level dataset for PD, LGD, and survival models."""
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
    target_cols = [
        "default_flag",
        "lgd",
        "lgd_months_since_issue",
        "lgd_is_mature_24m",
        "issue_d",
        "loan_status",
    ]
    id_cols = ["id"] if "id" in df.columns else []
    available = [c for c in feature_cols + target_cols + id_cols if c in df.columns]
    loan_master = df[available].copy()
    logger.info("Built loan_master: {}", loan_master.shape)
    return loan_master


def _aggregate_time_series(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    frame = df.copy()
    if "default_flag" not in frame.columns:
        frame["default_flag"] = 0
    frame = frame.dropna(subset=["issue_d", "loan_amnt"])
    frame["issue_month"] = pd.to_datetime(frame["issue_d"]).dt.to_period("M").dt.to_timestamp()
    agg_spec: dict[str, tuple[str, str]] = {
        "loan_count": ("loan_amnt", "count"),
        "default_count": ("default_flag", "sum"),
        "total_amt_funded": ("loan_amnt", "sum"),
        "avg_loan_amnt": ("loan_amnt", "mean"),
    }
    if "int_rate" in frame.columns:
        agg_spec["avg_int_rate"] = ("int_rate", "mean")
    if "dti" in frame.columns:
        agg_spec["avg_dti"] = ("dti", "mean")
    grouped = (
        frame.groupby(["issue_month", *group_cols], dropna=False, observed=True)
        .agg(**agg_spec)
        .reset_index()
        .rename(columns={"issue_month": "ds"})
    )
    if "avg_int_rate" not in grouped.columns:
        grouped["avg_int_rate"] = np.nan
    if "avg_dti" not in grouped.columns:
        grouped["avg_dti"] = np.nan
    grouped["default_rate"] = (
        grouped["default_count"].astype(float) / grouped["loan_count"].replace(0, np.nan)
    ).fillna(0.0)
    return grouped.sort_values(["ds", *group_cols]).reset_index(drop=True)


def build_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical monthly portfolio series for forecasting."""
    ts = _aggregate_time_series(df, [])
    ts["unique_id"] = "portfolio"
    ts["y"] = ts["default_rate"].astype(float)
    logger.info("Built time_series: {} ({} -> {})", ts.shape, ts["ds"].min(), ts["ds"].max())
    return ts


def build_time_series_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Build coherent panel series for portfolio -> grade -> grade x term."""
    frame = df.copy()
    frame["grade"] = (
        frame.get("grade", pd.Series(["UNKNOWN"] * len(frame))).astype(str).fillna("UNKNOWN")
    )
    frame["term_months"] = pd.to_numeric(frame.get("term"), errors="coerce")
    if "term" in frame.columns and frame["term_months"].isna().all():
        frame["term_months"] = (
            frame["term"].astype(str).str.extract(r"(\d+)")[0].pipe(pd.to_numeric, errors="coerce")
        )
    frame["term_months"] = frame["term_months"].fillna(-1).astype(int)

    grade_term = _aggregate_time_series(frame, ["grade", "term_months"])
    grade_term["series_level"] = "grade_term"
    grade_term["unique_id"] = grade_term.apply(
        lambda row: f"grade_term::{row['grade']}__{int(row['term_months'])}", axis=1
    )

    grade = _aggregate_time_series(frame, ["grade"])
    grade["series_level"] = "grade"
    grade["term_months"] = np.nan
    grade["unique_id"] = grade["grade"].map(lambda grade_value: f"grade::{grade_value}")

    portfolio = build_time_series(frame)
    portfolio["series_level"] = "portfolio"
    portfolio["grade"] = "ALL"
    portfolio["term_months"] = np.nan

    panel = pd.concat([portfolio, grade, grade_term], ignore_index=True, sort=False)
    panel["grade"] = panel.get("grade", pd.Series(["ALL"] * len(panel))).fillna("ALL")
    panel["term_months"] = pd.to_numeric(panel.get("term_months"), errors="coerce")
    panel = panel.sort_values(["series_level", "unique_id", "ds"]).reset_index(drop=True)
    logger.info("Built time_series_panel: {}", panel.shape)
    return panel


def build_ead_dataset(df: pd.DataFrame) -> pd.DataFrame:
    ead = df[df["default_flag"] == 1].copy()
    logger.info("Built ead_dataset: {}", ead.shape)
    return ead


def load_historical_time_series_source(
    input_path: str | Path = "data/processed/train.parquet",
) -> pd.DataFrame:
    """Load full-history splits when available; fallback to the provided input frame."""
    input_frame = pd.read_parquet(input_path)
    split_paths = [
        Path("data/processed/train.parquet"),
        Path("data/processed/calibration.parquet"),
        Path("data/processed/test.parquet"),
    ]
    if all(path.exists() for path in split_paths):
        parts = [pd.read_parquet(path) for path in split_paths]
        full_history = pd.concat(parts, ignore_index=True, sort=False)
        logger.info("Loaded full historical splits for time series: {}", full_history.shape)
        return full_history
    logger.info("Falling back to input-only time series source: {}", input_frame.shape)
    return input_frame


def save_datasets(
    loan_master: pd.DataFrame,
    time_series: pd.DataFrame,
    ead_dataset: pd.DataFrame,
    output_dir: str | Path = "data/processed/",
    *,
    time_series_full: pd.DataFrame | None = None,
    time_series_panel: pd.DataFrame | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loan_master.to_parquet(output_dir / "loan_master.parquet", index=False)
    ead_dataset.to_parquet(output_dir / "ead_dataset.parquet", index=False)
    time_series.to_parquet(output_dir / "time_series.parquet", index=False)
    (time_series_full if time_series_full is not None else time_series).to_parquet(
        output_dir / "time_series_full.parquet", index=False
    )
    if time_series_panel is not None:
        time_series_panel.to_parquet(output_dir / "time_series_panel.parquet", index=False)
    logger.info("Saved analytical datasets to {}", output_dir)


def main(input_path: str = "data/processed/train.parquet", output_dir: str = "data/processed/"):
    from src.features.feature_engineering import run_feature_pipeline

    train_df = pd.read_parquet(input_path)
    logger.info("Loaded training split: {} rows from {}", len(train_df), input_path)

    train_clean = clean_raw_columns(train_df)
    train_featured = run_feature_pipeline(train_clean)
    loan_master = build_loan_master(train_featured)
    ead_dataset = build_ead_dataset(train_featured)

    history_df = clean_raw_columns(load_historical_time_series_source(input_path))
    time_series_full = build_time_series(history_df)
    time_series_panel = build_time_series_panel(history_df)

    save_datasets(
        loan_master,
        time_series_full,
        ead_dataset,
        output_dir,
        time_series_full=time_series_full,
        time_series_panel=time_series_panel,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/train.parquet")
    parser.add_argument("--output", default="data/processed/")
    args = parser.parse_args()
    main(args.input, args.output)
