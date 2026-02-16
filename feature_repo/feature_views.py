"""Feast feature view definitions for the lending club risk project."""

from datetime import timedelta

from feast import FeatureView
from feast.field import Field
from feast.types import Float64, String

from data_sources import loan_features_source
from entities import loan

loan_origination_fv = FeatureView(
    name="loan_origination",
    entities=[loan],
    ttl=timedelta(days=3650),  # 10 years (historical data)
    schema=[
        Field(name="loan_amnt", dtype=Float64),
        Field(name="annual_inc", dtype=Float64),
        Field(name="loan_to_income", dtype=Float64),
        Field(name="dti", dtype=Float64),
        Field(name="rev_utilization", dtype=Float64),
        Field(name="int_rate", dtype=Float64),
        Field(name="installment", dtype=Float64),
        Field(name="term", dtype=String),
        Field(name="grade", dtype=String),
        Field(name="sub_grade", dtype=String),
    ],
    source=loan_features_source,
    online=True,
    description="Core loan origination features used for PD model inference",
)

loan_credit_history_fv = FeatureView(
    name="loan_credit_history",
    entities=[loan],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="num_delinq_2yrs", dtype=Float64),
        Field(name="days_since_last_delinq", dtype=Float64),
        Field(name="open_acc", dtype=Float64),
        Field(name="pub_rec", dtype=Float64),
        Field(name="revol_bal", dtype=Float64),
        Field(name="revol_util", dtype=Float64),
        Field(name="total_acc", dtype=Float64),
        Field(name="fico_range_low", dtype=Float64),
        Field(name="fico_range_high", dtype=Float64),
        Field(name="credit_history_months", dtype=Float64),
    ],
    source=loan_features_source,
    online=True,
    description="Borrower credit history features for risk assessment",
)

loan_demographics_fv = FeatureView(
    name="loan_demographics",
    entities=[loan],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="home_ownership", dtype=String),
        Field(name="purpose", dtype=String),
        Field(name="emp_length", dtype=String),
        Field(name="verification_status", dtype=String),
    ],
    source=loan_features_source,
    online=True,
    description="Borrower demographic and employment features",
)
