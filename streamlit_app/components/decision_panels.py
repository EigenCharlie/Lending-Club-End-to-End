"""Reusable decision-oriented panels for pages that end in operational choices."""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st


def decision_checklist(title: str, items: Iterable[str]) -> None:
    """Render an operational checklist to translate analysis into a decision."""
    items = [str(x).strip() for x in items if str(x).strip()]
    if not items:
        return
    st.markdown(f"### ✅ {title}")
    for item in items:
        st.markdown(f"- {item}")


def tradeoff_panel(
    decision_label: str,
    upside: str,
    downside: str,
    monitoring: str,
    *,
    color: str = "#F6F8FB",
) -> None:
    """Render a compact trade-off summary panel."""
    st.markdown(
        f"""
<div style="border:1px solid #E5EAF0;border-radius:10px;padding:0.9rem 1rem;background:{color};">
<div style="font-weight:700;margin-bottom:0.35rem;">{decision_label}</div>
<div><b>Upside:</b> {upside}</div>
<div><b>Downside:</b> {downside}</div>
<div><b>Qué monitorear:</b> {monitoring}</div>
</div>
""",
        unsafe_allow_html=True,
    )
