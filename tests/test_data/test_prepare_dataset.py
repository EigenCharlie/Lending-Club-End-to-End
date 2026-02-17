"""Tests for src/data/prepare_dataset.py — temporal splits."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.prepare_dataset import (
    create_calibration_set,
    out_of_time_split,
    parse_dates,
)


@pytest.fixture
def cleaned_df() -> pd.DataFrame:
    """Synthetic cleaned DataFrame with issue dates spanning 2016-2019."""
    dates = pd.date_range("2016-01-01", periods=100, freq="W")
    return pd.DataFrame(
        {
            "id": range(100),
            "issue_d": dates,
            "loan_amnt": [10000 + i * 100 for i in range(100)],
            "default_flag": [1 if i % 5 == 0 else 0 for i in range(100)],
            "int_rate": [10.0 + i * 0.1 for i in range(100)],
        }
    )


class TestOutOfTimeSplit:
    def test_train_before_cutoff(self, cleaned_df: pd.DataFrame) -> None:
        train, test = out_of_time_split(cleaned_df, cutoff_date="2017-06-01")
        assert (train["issue_d"] < pd.Timestamp("2017-06-01")).all()

    def test_test_at_or_after_cutoff(self, cleaned_df: pd.DataFrame) -> None:
        train, test = out_of_time_split(cleaned_df, cutoff_date="2017-06-01")
        assert (test["issue_d"] >= pd.Timestamp("2017-06-01")).all()

    def test_no_temporal_leakage(self, cleaned_df: pd.DataFrame) -> None:
        """Max train date must be strictly before min test date."""
        train, test = out_of_time_split(cleaned_df, cutoff_date="2017-06-01")
        assert train["issue_d"].max() < test["issue_d"].min()

    def test_union_equals_original(self, cleaned_df: pd.DataFrame) -> None:
        train, test = out_of_time_split(cleaned_df, cutoff_date="2017-06-01")
        assert len(train) + len(test) == len(cleaned_df)


class TestCreateCalibrationSet:
    def test_calibration_size_approximately_correct(self, cleaned_df: pd.DataFrame) -> None:
        proper_train, calibration = create_calibration_set(cleaned_df, fraction=0.15)
        expected_cal = int(len(cleaned_df) * 0.15)
        assert len(calibration) == expected_cal

    def test_calibration_is_latest_data(self, cleaned_df: pd.DataFrame) -> None:
        """Calibration should come from the end of the training period."""
        proper_train, calibration = create_calibration_set(cleaned_df, fraction=0.15)
        assert proper_train["issue_d"].max() <= calibration["issue_d"].min()


class TestParseDates:
    def test_parses_string_dates(self) -> None:
        df = pd.DataFrame({"issue_d": ["2017-01-15", "2018-06-01"], "loan_amnt": [1, 2]})
        result = parse_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(result["issue_d"])

    def test_handles_missing_date_columns(self) -> None:
        df = pd.DataFrame({"loan_amnt": [1, 2]})
        result = parse_dates(df)
        assert len(result) == 2  # should not crash
