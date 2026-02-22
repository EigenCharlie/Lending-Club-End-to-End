"""Contextual glossary and methodology helpers using modern Streamlit UI primitives."""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st

from streamlit_app.content.glossary_registry import get_glossary_term


def term_popover(term_key: str, *, label: str | None = None, icon: str = "❓") -> None:
    """Render a compact glossary popover for a term if registered."""
    term = get_glossary_term(term_key)
    if term is None:
        return
    title = label or term.label
    if hasattr(st, "popover"):
        with st.popover(f"{icon} {title}"):
            st.markdown(f"**{term.label}**")
            st.caption(term.short_definition)
            st.markdown(term.why_it_matters)
    else:
        with st.expander(f"{icon} {title}", expanded=False):
            st.markdown(f"**{term.label}**")
            st.caption(term.short_definition)
            st.markdown(term.why_it_matters)


def metric_help_popover(metric_key: str, *, label: str | None = None) -> None:
    """Alias for term_popover when metric names map to glossary entries."""
    term_popover(metric_key, label=label or "Qué significa esta métrica", icon="📎")


def chart_help_popover(chart_id: str, what_to_look_at: str, common_misread: str = "") -> None:
    """Explain how to read a chart without adding text clutter to the page body."""
    title = f"Cómo leer `{chart_id}`"
    if hasattr(st, "popover"):
        with st.popover(f"🧭 {title}"):
            st.markdown(f"**Qué mirar primero**: {what_to_look_at}")
            if common_misread:
                st.markdown(f"**Error común**: {common_misread}")
    else:
        with st.expander(f"🧭 {title}", expanded=False):
            st.markdown(f"**Qué mirar primero**: {what_to_look_at}")
            if common_misread:
                st.markdown(f"**Error común**: {common_misread}")


def methodology_dialog(
    title: str,
    body_markdown: str,
    *,
    button_label: str = "Ver detalles metodológicos",
) -> None:
    """Open methodology details in dialog when available, fallback to expander."""
    if not hasattr(st, "dialog"):
        with st.expander(button_label, expanded=False):
            st.markdown(body_markdown)
        return

    @st.dialog(title)
    def _show_dialog() -> None:
        st.markdown(body_markdown)

    if st.button(button_label, key=f"btn_dialog_{title}"):
        _show_dialog()


def inline_glossary_row(terms: Iterable[str], *, label: str = "Glosario contextual") -> None:
    """Render multiple glossary popovers in a single row-like block."""
    terms = [str(t) for t in terms if str(t).strip()]
    if not terms:
        return
    st.caption(label)
    cols = st.columns(max(1, min(4, len(terms))))
    for idx, term in enumerate(terms):
        with cols[idx % len(cols)]:
            term_popover(term, label=term)
