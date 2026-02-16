"""Reusable KPI metric card components."""

from __future__ import annotations

import streamlit as st


def kpi_row(metrics: list[dict], n_cols: int | None = None):
    """Display a row of KPI metric cards.

    Args:
        metrics: List of dicts with keys: label, value, delta (optional), help (optional).
        n_cols: Number of columns (defaults to len(metrics)).
    """
    n_cols = n_cols or len(metrics)
    cols = st.columns(n_cols)
    for i, m in enumerate(metrics):
        with cols[i % n_cols]:
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=m.get("delta"),
                help=m.get("help"),
            )
