"""Tests for forecast_default_rates artifact loading fallbacks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import forecast_default_rates as forecast_mod


def test_load_history_rebuilds_missing_governed_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)

    def _frame(start: str, periods: int) -> pd.DataFrame:
        issue_d = pd.date_range(start, periods=periods, freq="MS")
        return pd.DataFrame(
            {
                "id": [f"{start}-{i}" for i in range(periods)],
                "issue_d": issue_d,
                "loan_amnt": [10_000 + 100 * i for i in range(periods)],
                "default_flag": [i % 5 == 0 for i in range(periods)],
                "grade": ["A" if i % 2 == 0 else "B" for i in range(periods)],
                "term": ["36 months" if i % 2 == 0 else "60 months" for i in range(periods)],
                "int_rate": ["12.5%"] * periods,
                "dti": [15.0] * periods,
            }
        )

    _frame("2015-01-01", 24).to_parquet(data_dir / "train.parquet", index=False)
    _frame("2017-01-01", 12).to_parquet(data_dir / "calibration.parquet", index=False)
    _frame("2018-01-01", 12).to_parquet(data_dir / "test.parquet", index=False)

    portfolio, panel = forecast_mod._load_history()

    assert not portfolio.empty
    assert not panel.empty
    assert (data_dir / "time_series_full.parquet").exists()
    assert (data_dir / "time_series_panel.parquet").exists()
