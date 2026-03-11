"""Tests for governed time-series helper utilities."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.models.time_series import (
    _conf_int_bounds,
    build_backtest_cutoffs,
    compute_revision_metrics,
    diebold_mariano_test,
    infer_run_tag,
    load_future_covariates,
    select_time_series_champions,
)


def test_build_backtest_cutoffs_respects_max_windows() -> None:
    ds = pd.date_range("2015-01-01", periods=96, freq="MS")
    result = build_backtest_cutoffs(
        ds,
        horizon=12,
        min_train_periods=72,
        step_months=1,
        max_windows=3,
    )
    assert len(result) == 3
    assert result[-1] == pd.Timestamp("2021-12-01")


def test_compute_revision_metrics_aggregates_same_target_revisions() -> None:
    predictions = pd.DataFrame(
        {
            "model": ["AutoARIMA", "AutoARIMA", "AutoARIMA", "AutoARIMA"],
            "cutoff": pd.to_datetime(["2020-12-01", "2021-01-01", "2020-12-01", "2021-01-01"]),
            "ds": pd.to_datetime(["2021-02-01", "2021-02-01", "2021-03-01", "2021-03-01"]),
            "y_pred": [0.10, 0.12, 0.11, 0.14],
        }
    )
    result = compute_revision_metrics(predictions)
    assert result["n_revisions"].iloc[0] == 2
    assert result["mean_abs_revision"].iloc[0] == pytest.approx(0.025)
    assert result["max_abs_revision"].iloc[0] == pytest.approx(0.03)


def test_diebold_mariano_test_returns_finite_statistics() -> None:
    model_loss = [0.10, 0.08, 0.07, 0.09, 0.06, 0.05]
    baseline_loss = [0.12, 0.11, 0.10, 0.11, 0.09, 0.08]
    result = diebold_mariano_test(model_loss, baseline_loss, lag=2)
    assert math.isfinite(result["dm_stat"])
    assert 0.0 <= result["p_value"] <= 1.0
    assert isinstance(result["reject"], bool)


def test_conf_int_bounds_accepts_ndarray() -> None:
    lower, upper = _conf_int_bounds([[0.1, 0.2], [0.3, 0.4]])
    assert lower.tolist() == [0.1, 0.3]
    assert upper.tolist() == [0.2, 0.4]


def test_select_time_series_champions_applies_point_and_interval_policy() -> None:
    metrics = pd.DataFrame(
        [
            {
                "model": "SeasonalNaive",
                "mae": 0.10,
                "mase": 1.00,
                "rmsse": 1.00,
                "abs_bias": 0.020,
                "coverage_90": 0.92,
                "coverage_gap_90": 0.02,
                "winkler_90": 0.10,
                "avg_interval_width_90": 0.10,
                "family": "statistical",
            },
            {
                "model": "AutoARIMA",
                "mae": 0.08,
                "mase": 0.80,
                "rmsse": 0.90,
                "abs_bias": 0.010,
                "coverage_90": 0.89,
                "coverage_gap_90": 0.01,
                "winkler_90": 0.08,
                "avg_interval_width_90": 0.09,
                "family": "statistical",
            },
            {
                "model": "STL_CatBoost",
                "mae": 0.07,
                "mase": 0.75,
                "rmsse": 1.25,
                "abs_bias": 0.015,
                "coverage_90": 0.91,
                "coverage_gap_90": 0.01,
                "winkler_90": 0.06,
                "avg_interval_width_90": 0.07,
                "family": "challenger",
            },
        ]
    )
    config = {
        "point_champion": {
            "must_beat_seasonal_naive": True,
            "max_rmsse_vs_seasonal_naive": 1.0,
        },
        "interval_policy": {
            "max_coverage_gap": 0.03,
            "official_family": "statistical",
        },
    }
    champions = select_time_series_champions(metrics, config)
    assert champions["point"]["model"] == "STL_CatBoost"
    assert champions["point"]["promotable"] is False
    assert "rmsse_worse_than_allowed_vs_seasonal_naive" in champions["point"]["reasons"]
    assert champions["interval"]["model"] == "AutoARIMA"
    assert champions["interval"]["promotable"] is True


def test_load_future_covariates_requires_contract_when_enabled(tmp_path) -> None:
    config = {
        "exogenous": {
            "enabled": True,
            "future_covariates_path": str(tmp_path / "missing.parquet"),
            "required_columns": ["ds", "unemployment_rate"],
        }
    }
    with pytest.raises(FileNotFoundError):
        load_future_covariates(config)


def test_infer_run_tag_prefers_pipeline_env(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIPELINE_RUN_TAG", "run-from-env")
    (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "processed" / "pipeline_summary.json").write_text(
        '{"run_tag": "run-from-pipeline-summary"}',
        encoding="utf-8",
    )

    assert infer_run_tag() == "run-from-env"
