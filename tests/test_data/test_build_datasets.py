"""Tests for src/data/build_datasets.py — analytical dataset builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.build_datasets import (
    build_ead_dataset,
    build_loan_master,
    build_time_series,
    build_time_series_panel,
    build_time_series_panel_vnext,
    build_time_series_vnext,
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

    def test_parses_string_dtype_columns(self) -> None:
        df = pd.DataFrame(
            {
                "int_rate": pd.Series([" 13.75%", "10.5%"], dtype="string"),
                "term": pd.Series([" 36 months", " 60 months"], dtype="string"),
                "revol_util": pd.Series(["55.3%", "80.1%"], dtype="string"),
            }
        )
        result = clean_raw_columns(df)
        assert result["int_rate"].tolist() == pytest.approx([13.75, 10.5])
        assert result["term"].tolist() == pytest.approx([36.0, 60.0])
        assert result["revol_util"].tolist() == pytest.approx([55.3, 80.1])
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

    def test_contains_additive_counts(self, feature_df: pd.DataFrame) -> None:
        result = build_time_series(feature_df)
        assert {"loan_count", "default_count", "total_amt_funded"}.issubset(result.columns)

    def test_handles_missing_optional_numeric_columns(self, feature_df: pd.DataFrame) -> None:
        result = build_time_series(feature_df.drop(columns=["int_rate", "dti"]))
        assert "avg_int_rate" in result.columns
        assert "avg_dti" in result.columns
        assert result["avg_int_rate"].isna().all()
        assert result["avg_dti"].isna().all()


class TestBuildTimeSeriesPanel:
    def test_contains_expected_levels(self, feature_df: pd.DataFrame) -> None:
        result = build_time_series_panel(feature_df)
        assert {"portfolio", "grade", "grade_term"} == set(result["series_level"].unique())

    def test_grade_counts_reconcile_to_portfolio(self, feature_df: pd.DataFrame) -> None:
        result = build_time_series_panel(feature_df)
        portfolio = (
            result.loc[result["series_level"] == "portfolio", ["ds", "loan_count", "default_count"]]
            .sort_values("ds")
            .reset_index(drop=True)
        )
        grade = (
            result.loc[result["series_level"] == "grade"]
            .groupby("ds", as_index=False)[["loan_count", "default_count"]]
            .sum()
            .sort_values("ds")
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            portfolio[["loan_count", "default_count"]],
            grade[["loan_count", "default_count"]],
            check_dtype=False,
        )

    def test_fills_missing_months_with_zero_volume_rows(self) -> None:
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "loan_amnt": [10000, 12000],
                "default_flag": [0, 1],
                "issue_d": pd.to_datetime(["2018-01-01", "2018-03-01"]),
                "grade": ["A", "A"],
                "term": [36, 36],
                "int_rate": [10.0, 10.0],
                "dti": [15.0, 15.0],
            }
        )
        result = build_time_series_panel(df)
        bottom = result.loc[result["unique_id"] == "grade_term::A__36"].sort_values("ds")

        assert bottom["ds"].tolist() == [
            pd.Timestamp("2018-01-01"),
            pd.Timestamp("2018-02-01"),
            pd.Timestamp("2018-03-01"),
        ]
        feb = bottom.loc[bottom["ds"] == pd.Timestamp("2018-02-01")].iloc[0]
        assert float(feb["loan_count"]) == 0.0
        assert float(feb["default_count"]) == 0.0
        assert float(feb["default_rate"]) == 0.0


class TestBuildTimeSeriesVNext:
    def test_enriched_portfolio_contains_vnext_targets_and_mix_features(
        self,
        feature_df: pd.DataFrame,
    ) -> None:
        enriched = feature_df.assign(
            sub_grade=["A1"] * len(feature_df),
            purpose=["debt_consolidation"] * len(feature_df),
            verification_status=["Verified"] * len(feature_df),
            home_ownership=["RENT"] * len(feature_df),
            application_type=["Individual"] * len(feature_df),
            fico_range_low=[680] * len(feature_df),
            fico_range_high=[700] * len(feature_df),
            revol_util=[55.0] * len(feature_df),
            mort_acc=[1.0] * len(feature_df),
            inq_last_6mths=[1.0] * len(feature_df),
            delinq_2yrs=[0.0] * len(feature_df),
            acc_now_delinq=[0.0] * len(feature_df),
            num_tl_30dpd=[0.0] * len(feature_df),
            num_tl_90g_dpd_24m=[0.0] * len(feature_df),
        )
        result = build_time_series_vnext(enriched)

        expected = {
            "y",
            "y_logit",
            "smoothed_default_rate",
            "share_grade_A",
            "share_term_36",
            "share_verified",
            "avg_fico_score",
            "std_loan_amnt",
        }
        assert expected.issubset(result.columns)
        assert np.isfinite(result["y_logit"]).all()
        assert result["smoothed_default_rate"].between(0, 1).all()

    def test_vnext_panel_fills_missing_months_and_recomputes_targets(self) -> None:
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "loan_amnt": [10000, 12000],
                "annual_inc": [50000, 60000],
                "dti": [15.0, 18.0],
                "int_rate": [10.0, 11.0],
                "installment": [320.0, 360.0],
                "term": [36, 36],
                "grade": ["A", "A"],
                "sub_grade": ["A1", "A1"],
                "purpose": ["debt_consolidation", "debt_consolidation"],
                "verification_status": ["Verified", "Verified"],
                "home_ownership": ["RENT", "RENT"],
                "application_type": ["Individual", "Individual"],
                "fico_range_low": [680, 690],
                "fico_range_high": [700, 710],
                "revol_util": [50.0, 55.0],
                "mort_acc": [1.0, 1.0],
                "inq_last_6mths": [1.0, 2.0],
                "delinq_2yrs": [0.0, 0.0],
                "acc_now_delinq": [0.0, 0.0],
                "num_tl_30dpd": [0.0, 0.0],
                "num_tl_90g_dpd_24m": [0.0, 0.0],
                "default_flag": [0, 1],
                "issue_d": pd.to_datetime(["2018-01-01", "2018-03-01"]),
            }
        )
        result = build_time_series_panel_vnext(df)
        bottom = result.loc[result["unique_id"] == "grade_term::A__36"].sort_values("ds")
        feb = bottom.loc[bottom["ds"] == pd.Timestamp("2018-02-01")].iloc[0]

        assert bottom["ds"].tolist() == [
            pd.Timestamp("2018-01-01"),
            pd.Timestamp("2018-02-01"),
            pd.Timestamp("2018-03-01"),
        ]
        assert float(feb["loan_count"]) == 0.0
        assert float(feb["default_count"]) == 0.0
        assert float(feb["default_rate"]) == 0.0
        assert np.isfinite(float(feb["y_logit"]))


class TestBuildEadDataset:
    def test_only_defaults(self, feature_df: pd.DataFrame) -> None:
        result = build_ead_dataset(feature_df)
        assert (result["default_flag"] == 1).all()

    def test_fewer_rows_than_original(self, feature_df: pd.DataFrame) -> None:
        result = build_ead_dataset(feature_df)
        assert len(result) < len(feature_df)
        assert len(result) == (feature_df["default_flag"] == 1).sum()
