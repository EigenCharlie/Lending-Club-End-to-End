"""Tests for scripts/backtest_conformal_coverage.py."""

from __future__ import annotations

import pandas as pd

from scripts.backtest_conformal_coverage import _load_intervals, _monthly_metrics


def test_monthly_metrics_handles_single_class_months() -> None:
    df = pd.DataFrame(
        {
            "month": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"]),
            "y_true": [0.0, 0.0, 1.0, 1.0],
            "y_pred": [0.1, 0.2, 0.8, 0.9],
            "pd_low_90": [0.0, 0.0, 0.7, 0.7],
            "pd_high_90": [0.3, 0.3, 1.0, 1.0],
            "pd_low_95": [0.0, 0.0, 0.6, 0.6],
            "pd_high_95": [0.4, 0.4, 1.0, 1.0],
        }
    )

    monthly = _monthly_metrics(df)

    assert len(monthly) == 2
    assert monthly["cal_log_loss"].notna().all()


def test_load_intervals_supports_explicit_path(tmp_path) -> None:
    path = tmp_path / "intervals.parquet"
    expected = pd.DataFrame({"y_true": [0.0, 1.0], "pd_low_90": [0.0, 0.2]})
    expected.to_parquet(path, index=False)

    loaded = _load_intervals(str(path))

    pd.testing.assert_frame_equal(loaded, expected)
