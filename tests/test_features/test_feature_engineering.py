"""Unit tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    apply_woe_encoders,
    build_feature_config,
    create_buckets,
    create_ratios,
    fit_woe_encoders,
    run_feature_pipeline,
)


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


def test_run_feature_pipeline_emits_richer_core_features() -> None:
    df = pd.DataFrame(
        {
            "loan_amnt": [10000, 20000],
            "installment": [350.0, 620.0],
            "annual_inc": [50000, 80000],
            "dti": [12.0, 24.0],
            "int_rate": [8.5, 17.2],
            "revol_bal": [4000, 8000],
            "revol_util": [40.0, 72.0],
            "open_acc": [6, 12],
            "total_acc": [12, 18],
            "fico_range_low": [680, 720],
            "fico_range_high": [684, 724],
            "emp_length": ["3 years", "10+ years"],
            "issue_d": ["2017-01-01", "2017-06-01"],
            "earliest_cr_line": ["2010-01-01", "2008-06-01"],
            "delinq_2yrs": [0, 2],
            "mths_since_last_delinq": [np.nan, 4],
            "pub_rec": [0, 1],
            "pub_rec_bankruptcies": [0, 0],
            "inq_last_6mths": [1, 0],
            "mort_acc": [1, 0],
            "num_tl_op_past_12m": [1, 4],
            "chargeoff_within_12_mths": [0, 1],
            "percent_bc_gt_75": [20.0, 85.0],
            "grade": ["B", "D"],
            "default_flag": [0, 1],
        }
    )

    out = run_feature_pipeline(df)

    for feature in [
        "installment_burden",
        "revol_bal_to_income",
        "open_acc_ratio",
        "fico_score",
        "credit_age_years",
        "emp_length_num",
        "delinq_severity",
        "delinq_recency",
        "has_recent_inq",
        "many_recent_opens",
        "high_util_pct",
        "fico_bucket",
        "loan_to_income_sq",
        "fico_x_dti",
    ]:
        assert feature in out.columns


def test_feature_config_and_woe_encoders_are_built_from_train_only() -> None:
    train = pd.DataFrame(
        {
            "loan_amnt": [10000, 12000, 20000, 18000],
            "installment": [320.0, 340.0, 610.0, 580.0],
            "annual_inc": [50000, 52000, 90000, 85000],
            "dti": [12.0, 14.0, 26.0, 28.0],
            "int_rate": [8.0, 9.0, 18.0, 17.0],
            "revol_bal": [3000, 3500, 9000, 8700],
            "revol_util": [35.0, 40.0, 75.0, 78.0],
            "open_acc": [7, 8, 12, 11],
            "total_acc": [14, 15, 20, 18],
            "fico_range_low": [700, 705, 640, 650],
            "fico_range_high": [704, 709, 644, 654],
            "emp_length": ["2 years", "3 years", "8 years", "10+ years"],
            "issue_d": ["2017-01-01", "2017-03-01", "2017-05-01", "2017-07-01"],
            "earliest_cr_line": ["2009-01-01", "2010-01-01", "2007-01-01", "2008-01-01"],
            "delinq_2yrs": [0, 0, 1, 2],
            "mths_since_last_delinq": [np.nan, np.nan, 8.0, 4.0],
            "pub_rec": [0, 0, 1, 0],
            "pub_rec_bankruptcies": [0, 0, 0, 1],
            "inq_last_6mths": [0, 1, 2, 0],
            "mort_acc": [1, 0, 1, 0],
            "num_tl_op_past_12m": [1, 2, 4, 5],
            "chargeoff_within_12_mths": [0, 0, 1, 0],
            "percent_bc_gt_75": [10.0, 15.0, 80.0, 65.0],
            "grade": ["A", "A", "D", "C"],
            "sub_grade": ["A2", "A3", "D1", "C4"],
            "home_ownership": ["RENT", "MORTGAGE", "RENT", "OWN"],
            "purpose": ["debt_consolidation", "credit_card", "small_business", "vacation"],
            "verification_status": ["Verified", "Source Verified", "Verified", "Not Verified"],
            "term": [36, 36, 60, 60],
            "default_flag": [0, 0, 1, 1],
        }
    )

    train_fe = run_feature_pipeline(train)
    encoders, iv_scores = fit_woe_encoders(train_fe)
    train_fe = apply_woe_encoders(train_fe, encoders)
    cfg = build_feature_config(train_fe, iv_scores=iv_scores)

    assert "CATBOOST_FEATURES" in cfg
    assert "CHALLENGER_FEATURE_POOL_V2" in cfg
    assert "SURVIVAL_FEATURES" in cfg
    assert "grade_woe" not in cfg["CATBOOST_FEATURES"]
    assert "grade_woe" in cfg["LOGREG_FEATURES"]
