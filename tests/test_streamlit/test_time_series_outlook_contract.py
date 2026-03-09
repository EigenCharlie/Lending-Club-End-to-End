"""Contract checks for the governed time-series Streamlit page."""

from __future__ import annotations

from pathlib import Path

PAGE_PATH = (
    Path(__file__).resolve().parents[2] / "streamlit_app" / "pages" / "time_series_outlook.py"
)


def test_time_series_page_uses_governed_artifacts() -> None:
    text = PAGE_PATH.read_text(encoding="utf-8")
    assert 'try_load_parquet("time_series_full")' in text
    assert 'try_load_parquet("ts_backtest_predictions")' in text
    assert 'try_load_parquet("ts_backtest_metrics")' in text
    assert 'try_load_json("time_series_status", directory="models"' in text


def test_time_series_page_does_not_fabricate_backtest_actuals() -> None:
    text = PAGE_PATH.read_text(encoding="utf-8")
    forbidden = [
        "cv_stats = forecasts.copy()",
        "~80.6%",
        "if cv_stats.empty and not history.empty and not forecasts.empty:",
    ]
    for pattern in forbidden:
        assert pattern not in text
    assert "no disponible" in text.lower()
