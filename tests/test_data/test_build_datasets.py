"""Tests for src/data/build_datasets.py — analytical dataset builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.build_datasets import (
    build_ead_dataset,
    build_loan_master,
    build_time_series,
    clean_raw_columns,
)


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """Synthetic DataFrame with features expected by build functions."""
    n = 200
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "id": range(n),
            "loan_amnt": rng.integers(5000, 40000, n),
            "annual_inc": rng.integers(30000, 150000, n),
            "dti": rng.uniform(5, 35, n),
            "int_rate": rng.uniform(5, 25, n),
            "term": rng.choice([36, 60], n),
            "installment": rng.uniform(100, 1500, n),
            "grade": rng.choice(["A", "B", "C", "D", "E"], n),
            "default_flag": rng.choice([0, 1], n, p=[0.8, 0.2]),
            "loan_status": [
                "Charged Off" if d == 1 else "Fully Paid"
                for d in rng.choice([0, 1], n, p=[0.8, 0.2])
            ],
            "issue_d": pd.date_range("2015-01-01", periods=n, freq="W"),
        }
    )


class TestCleanRawColumns:
    def test_parses_int_rate_string(self) -> None:
        df = pd.DataFrame({"int_rate": [" 13.75%", "10.5%", " 7.2% "]})
        result = clean_raw_columns(df)
        assert result["int_rate"].tolist() == pytest.approx([13.75, 10.5, 7.2])

    def test_parses_term_string(self) -> None:
        df = pd.DataFrame({"term": [" 36 months", " 60 months"]})
        result = clean_raw_columns(df)
        assert result["term"].tolist() == pytest.approx([36.0, 60.0])

    def test_parses_revol_util_string(self) -> None:
        df = pd.DataFrame({"revol_util": ["55.3%", "80.1%"]})
        result = clean_raw_columns(df)
        assert result["revol_util"].tolist() == pytest.approx([55.3, 80.1])
        assert "rev_utilization" in result.columns
        assert result["rev_utilization"].tolist() == pytest.approx([0.553, 0.801])

    def test_already_numeric_int_rate_unchanged(self) -> None:
        df = pd.DataFrame({"int_rate": [13.75, 10.5]})
        result = clean_raw_columns(df)
        assert result["int_rate"].tolist() == pytest.approx([13.75, 10.5])


class TestBuildLoanMaster:
    def test_contains_target_columns(self, feature_df: pd.DataFrame) -> None:
        result = build_loan_master(feature_df)
        assert "default_flag" in result.columns
        assert "issue_d" in result.columns

    def test_contains_id_if_present(self, feature_df: pd.DataFrame) -> None:
        result = build_loan_master(feature_df)
        assert "id" in result.columns

    def test_row_count_preserved(self, feature_df: pd.DataFrame) -> None:
        result = build_loan_master(feature_df)
        assert len(result) == len(feature_df)


class TestBuildTimeSeries:
    def test_nixtla_columns(self, feature_df: pd.DataFrame) -> None:
        result = build_time_series(feature_df)
        assert "unique_id" in result.columns
        assert "ds" in result.columns
        assert "y" in result.columns

    def test_unique_id_is_portfolio(self, feature_df: pd.DataFrame) -> None:
        result = build_time_series(feature_df)
        assert (result["unique_id"] == "portfolio").all()

    def test_y_is_default_rate(self, feature_df: pd.DataFrame) -> None:
        result = build_time_series(feature_df)
        assert (result["y"] == result["default_rate"]).all()
        assert result["y"].between(0, 1).all()

    def test_sorted_by_date(self, feature_df: pd.DataFrame) -> None:
        result = build_time_series(feature_df)
        assert result["ds"].is_monotonic_increasing


class TestBuildEadDataset:
    def test_only_defaults(self, feature_df: pd.DataFrame) -> None:
        result = build_ead_dataset(feature_df)
        assert (result["default_flag"] == 1).all()

    def test_fewer_rows_than_original(self, feature_df: pd.DataFrame) -> None:
        result = build_ead_dataset(feature_df)
        assert len(result) < len(feature_df)
        assert len(result) == (feature_df["default_flag"] == 1).sum()
