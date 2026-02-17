"""Unit tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import create_buckets, create_ratios


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "loan_amnt": [10000, 20000, 5000],
            "annual_inc": [50000, 100000, 30000],
            "int_rate": [7.0, 15.0, 22.0],
            "dti": [8.0, 22.0, 35.0],
        }
    )


def test_create_ratios(sample_df):
    result = create_ratios(sample_df)
    assert "loan_to_income" in result.columns
    assert result["loan_to_income"].iloc[0] == pytest.approx(0.2, abs=0.01)


def test_create_ratios_zero_income():
    df = pd.DataFrame({"loan_amnt": [10000], "annual_inc": [0]})
    result = create_ratios(df)
    assert np.isnan(result["loan_to_income"].iloc[0])


def test_create_buckets(sample_df):
    result = create_buckets(sample_df)
    assert "int_rate_bucket" in result.columns
    assert "dti_bucket" in result.columns
