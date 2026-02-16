"""Prepare train/test splits with out-of-time validation.

Ensures no temporal leakage by splitting on issue_d.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

# Out-of-time split: train on loans issued before cutoff, test on after
DEFAULT_CUTOFF_DATE = "2018-01-01"
CALIBRATION_FRACTION = 0.15  # fraction of train set for conformal calibration


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date columns to datetime."""
    date_cols = ["issue_d", "earliest_cr_line", "default_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def out_of_time_split(
    df: pd.DataFrame,
    cutoff_date: str = DEFAULT_CUTOFF_DATE,
    date_col: str = "issue_d",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by issue date for out-of-time validation."""
    cutoff = pd.Timestamp(cutoff_date)
    train = df[df[date_col] < cutoff].copy()
    test = df[df[date_col] >= cutoff].copy()

    logger.info(
        f"Out-of-time split at {cutoff_date}: "
        f"train={len(train):,} ({train['default_flag'].mean():.2%} default), "
        f"test={len(test):,} ({test['default_flag'].mean():.2%} default)"
    )
    return train, test


def create_calibration_set(
    train: pd.DataFrame,
    fraction: float = CALIBRATION_FRACTION,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train into proper_train + calibration set for conformal prediction.

    Calibration set is sampled from the end of the training period to maintain
    temporal ordering.
    """
    train_sorted = train.sort_values("issue_d")
    n_cal = int(len(train_sorted) * fraction)
    proper_train = train_sorted.iloc[:-n_cal].copy()
    calibration = train_sorted.iloc[-n_cal:].copy()

    logger.info(
        f"Calibration split: proper_train={len(proper_train):,}, "
        f"calibration={len(calibration):,}"
    )
    return proper_train, calibration


def save_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
    calibration: pd.DataFrame | None = None,
    output_dir: str | Path = "data/processed/",
) -> None:
    """Save train/test/calibration splits."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(output_dir / "train.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)
    if calibration is not None:
        calibration.to_parquet(output_dir / "calibration.parquet", index=False)

    logger.info(f"Saved splits to {output_dir}")


def main(
    input_path: str = "data/interim/lending_club_cleaned.parquet",
    output_dir: str = "data/processed/",
    cutoff_date: str = DEFAULT_CUTOFF_DATE,
) -> None:
    """Run full prepare pipeline."""
    df = pd.read_parquet(input_path)
    df = parse_dates(df)
    train, test = out_of_time_split(df, cutoff_date)
    proper_train, calibration = create_calibration_set(train)
    save_splits(proper_train, test, calibration, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/lending_club_cleaned.parquet")
    parser.add_argument("--output", default="data/processed/")
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF_DATE)
    args = parser.parse_args()
    main(args.input, args.output, args.cutoff)
