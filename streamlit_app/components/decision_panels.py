"""Reusable decision-oriented panels for pages that end in operational choices."""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st


def decision_checklist(title: str, items: Iterable[str]) -> None:
    """Render an operational checklist to translate analysis into a decision."""
    items = [str(x).strip() for x in items if str(x).strip()]
    if not items:
        return
    st.markdown(f"### {title}")
    for item in items:
        st.markdown(f"- :material/check_circle: {item}")


def tradeoff_panel(
    decision_label: str,
    upside: str,
    downside: str,
    monitoring: str,
    *,
    color: str = "#F6F8FB",
) -> None:
    """Render a compact trade-off summary panel."""
    _ = color  # Kept for API compat; border styling via container
    with st.container(border=True):
        st.markdown(f"**{decision_label}**")
        st.markdown(f":material/trending_up: **Upside:** {upside}")
        st.markdown(f":material/trending_down: **Downside:** {downside}")
        st.markdown(f":material/monitoring: **Qué monitorear:** {monitoring}")
