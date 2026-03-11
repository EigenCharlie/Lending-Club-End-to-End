"""Reusable TOBoML concept capsules for mixed-depth Streamlit pages."""

from __future__ import annotations

import streamlit as st

from streamlit_app.content.concept_map import (
    get_page_anti_patterns,
    get_page_concepts,
    get_page_focus_note,
)

_RESEARCH_PAGE_TYPES = {"research", "paper_draft"}


def render_concept_stack(
    page_id: str,
    *,
    page_type: str | None = None,
    max_cards: int = 3,
) -> None:
    """Render a compact executive capsule plus technical assumptions block."""
    cards = list(get_page_concepts(page_id))
    if not cards:
        return
    cards = cards[: max(1, max_cards)]
    is_research = page_type in _RESEARCH_PAGE_TYPES

    title = (
        "Marco conceptual rápido"
        if is_research
        else "Cápsula conceptual (lectura ejecutiva + técnica)"
    )
    with st.container(border=True):
        st.markdown(f"#### {title}")
        focus_note = get_page_focus_note(page_id)
        if focus_note:
            st.caption(focus_note)
        for card in cards:
            st.markdown(f"**{card.label}**: {card.what_is}")
            st.markdown(f"- Por qué importa: {card.why_business}")
            if not is_research:
                st.markdown(f"- Decisión habilitada: {card.decision_enabled}")

    with st.expander("Supuestos y límites operativos", expanded=False):
        for card in cards:
            st.markdown(f"**{card.label}**")
            st.markdown(f"- Error común: {card.common_misread}")
            st.markdown(f"- PASS esperado: {card.pass_when}")
            st.markdown(f"- WARN: {card.warn_when}")
            st.markdown(f"- FAIL: {card.fail_when}")
        anti_patterns = list(get_page_anti_patterns(page_id))
        if anti_patterns:
            st.markdown("**Trucos y anti-patrones**")
            for rule in anti_patterns:
                st.markdown(f"- {rule}")
