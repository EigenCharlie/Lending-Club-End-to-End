"""Unit tests for DVC KPI Streamlit helpers."""

from __future__ import annotations

from streamlit_app.components.dvc_kpi_spine import (
    DVC_KPI_SPECS,
    _format_metric_value,
    build_metric_cards,
    get_context_metric_keys,
)


def test_context_metric_keys_known_context() -> None:
    keys = get_context_metric_keys("model")
    assert "pd.auc" in keys
    assert "pd.ece" in keys
    assert len(keys) >= 4


def test_build_metric_cards_formats_values() -> None:
    metrics = {
        "pd.auc": 0.7171838,
        "pd.gini": 0.43436,
        "pd.ks": 0.31999,
        "pd.brier": 0.15384,
        "pd.ece": 0.00939,
    }
    cards = build_metric_cards(metrics, "model")
    labels = [c["label"] for c in cards]
    values = {c["label"]: c["value"] for c in cards}
    assert "AUC" in labels
    assert values["AUC"] == "0.7172"
    assert values["ECE"] == "0.0094"


def test_build_metric_cards_handles_missing_metrics() -> None:
    cards = build_metric_cards({}, "uncertainty")
    assert all(card["value"] == "N/D" for card in cards)


def test_format_metric_value_money_short() -> None:
    spec = DVC_KPI_SPECS["optimization.price_of_robustness"]
    assert _format_metric_value(38639.9, spec).startswith("$")
