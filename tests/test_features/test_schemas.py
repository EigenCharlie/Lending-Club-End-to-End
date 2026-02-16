"""Tests for Pandera schemas."""
import pandas as pd
import pytest
from src.features.schemas import validate_loan_master


def test_loan_master_schema_valid():
    df = pd.DataFrame({
        "loan_amnt": [10000.0, 20000.0],
        "annual_inc": [50000.0, 80000.0],
        "loan_to_income": [0.2, 0.25],
        "dti": [15.0, 20.0],
        "default_flag": [0, 1],
        "int_rate": [7.5, 12.3],
    })
    result = validate_loan_master(df)
    assert len(result) == 2


def test_loan_master_schema_invalid_default_flag():
    df = pd.DataFrame({
        "loan_amnt": [10000.0],
        "annual_inc": [50000.0],
        "loan_to_income": [0.2],
        "dti": [15.0],
        "default_flag": [2],
        "int_rate": [7.5],
    })
    with pytest.raises(Exception):
        validate_loan_master(df)
