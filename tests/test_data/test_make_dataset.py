"""Tests for src/data/make_dataset.py — initial cleaning and leakage removal."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.make_dataset import (
    DEFAULT_STATUSES,
    LEAKAGE_COLS,
    initial_clean,
)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Minimal raw DataFrame mimicking Lending Club CSV."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "loan_amnt": [10000, 15000, 20000, 5000, 12000],
            "loan_status": [
                "Fully Paid",
                "Charged Off",
                "Default",
                "Current",
                "Fully Paid",
            ],
            "int_rate": [10.5, 15.2, 18.0, 7.5, 12.0],
            # Leakage columns that should be removed
            "total_pymnt": [10500, 14000, 3000, 5200, 12500],
            "recoveries": [0, 500, 100, 0, 0],
            "funded_amnt": [10000, 15000, 20000, 5000, 12000],
            "last_pymnt_amnt": [350, 0, 0, 175, 400],
        }
    )


class TestInitialClean:
    def test_drops_leakage_columns(self, raw_df: pd.DataFrame) -> None:
        result = initial_clean(raw_df)
        for col in LEAKAGE_COLS:
            assert col not in result.columns, f"Leakage column '{col}' was not removed"

    def test_keeps_non_leakage_columns(self, raw_df: pd.DataFrame) -> None:
        result = initial_clean(raw_df)
        assert "loan_amnt" in result.columns
        assert "int_rate" in result.columns

    def test_filters_to_resolved_loans_only(self, raw_df: pd.DataFrame) -> None:
        result = initial_clean(raw_df)
        # "Current" loan should be filtered out
        assert len(result) == 4
        assert "Current" not in result["loan_status"].values

    def test_creates_default_flag(self, raw_df: pd.DataFrame) -> None:
        result = initial_clean(raw_df)
        assert "default_flag" in result.columns
        assert set(result["default_flag"].unique()).issubset({0, 1})

    def test_default_flag_matches_statuses(self, raw_df: pd.DataFrame) -> None:
        result = initial_clean(raw_df)
        for _, row in result.iterrows():
            if row["loan_status"] in DEFAULT_STATUSES:
                assert row["default_flag"] == 1
            else:
                assert row["default_flag"] == 0

    def test_default_rate_is_reasonable(self, raw_df: pd.DataFrame) -> None:
        result = initial_clean(raw_df)
        rate = result["default_flag"].mean()
        # 2 defaults out of 4 resolved = 50% for our synthetic data
        assert 0.0 < rate < 1.0

    def test_handles_missing_leakage_cols_gracefully(self) -> None:
        """If some leakage columns don't exist, should not crash."""
        df = pd.DataFrame(
            {
                "loan_amnt": [10000],
                "loan_status": ["Fully Paid"],
                "int_rate": [10.0],
            }
        )
        result = initial_clean(df)
        assert len(result) == 1
        assert result["default_flag"].iloc[0] == 0

    def test_no_rows_when_all_current(self) -> None:
        """If all loans are 'Current', output should be empty."""
        df = pd.DataFrame(
            {
                "loan_amnt": [10000, 15000],
                "loan_status": ["Current", "Current"],
                "int_rate": [10.0, 12.0],
            }
        )
        result = initial_clean(df)
        assert len(result) == 0
