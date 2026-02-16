"""Pandera schemas for DataFrame validation at pipeline boundaries."""

from __future__ import annotations

import pandera.pandas as pa

# ── Loan Master Schema ──
loan_master_schema = pa.DataFrameSchema(
    columns={
        "loan_amnt": pa.Column(float, pa.Check.greater_than(0), nullable=False),
        "annual_inc": pa.Column(float, pa.Check.greater_than(0), nullable=True),
        "loan_to_income": pa.Column(float, pa.Check.in_range(0, 100), nullable=True),
        "dti": pa.Column(float, pa.Check.in_range(0, 999), nullable=True),
        "default_flag": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "int_rate": pa.Column(float, pa.Check.in_range(0, 100), nullable=True),
    },
    coerce=True,
    strict=False,  # Allow extra columns
)

# ── Time Series Schema (Nixtla-compatible) ──
time_series_schema = pa.DataFrameSchema(
    columns={
        "ds": pa.Column("datetime64[ns]", nullable=False),
        "unique_id": pa.Column(str, nullable=False),
        "y": pa.Column(float, pa.Check.in_range(0, 1), nullable=False),  # default_rate
        "loan_count": pa.Column(int, pa.Check.greater_than(0), nullable=False),
    },
    coerce=True,
    strict=False,
)

# ── EAD Dataset Schema ──
ead_schema = pa.DataFrameSchema(
    columns={
        "default_flag": pa.Column(int, pa.Check.equal_to(1), nullable=False),  # Only defaults
        "loan_amnt": pa.Column(float, pa.Check.greater_than(0), nullable=False),
    },
    coerce=True,
    strict=False,
)

# ── Prediction Output Schema ──
prediction_schema = pa.DataFrameSchema(
    columns={
        "pd_point": pa.Column(float, pa.Check.in_range(0, 1), nullable=False),
        "pd_low": pa.Column(float, pa.Check.in_range(0, 1), nullable=False),
        "pd_high": pa.Column(float, pa.Check.in_range(0, 1), nullable=False),
    },
    checks=[
        pa.Check(lambda df: (df["pd_low"] <= df["pd_point"]).all(), error="pd_low must be <= pd_point"),
        pa.Check(lambda df: (df["pd_point"] <= df["pd_high"]).all(), error="pd_point must be <= pd_high"),
    ],
    coerce=True,
    strict=False,
)


def validate_loan_master(df):
    """Validate loan_master DataFrame."""
    return loan_master_schema.validate(df)


def validate_time_series(df):
    """Validate time_series DataFrame."""
    return time_series_schema.validate(df)


def validate_predictions(df):
    """Validate prediction output DataFrame."""
    return prediction_schema.validate(df)
