"""Reusable storytelling shell primitives for Streamlit pages."""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st

from streamlit_app.content.page_contracts import PageStoryContract

_TYPE_LABEL = {
    "executive": "Ejecutivo",
    "analytics": "Analítica",
    "decision": "Decisión",
    "governance": "Gobernanza",
    "research": "Investigación",
    "paper_draft": "Borrador de paper",
    "appendix": "Anexo",
    "journey": "Recorrido",
    "glossary": "Glosario",
    "exploration": "Exploración",
}


def render_page_header(contract: PageStoryContract, *, show_intro: bool = False) -> None:
    """Render a compact page metadata banner and optional intro summary."""
    label = _TYPE_LABEL.get(contract.page_type, contract.page_type)
    audience = contract.primary_audience
    st.caption(f"Tipo de página: {label} · Audiencia principal: {audience}")
    if show_intro and (contract.goal or contract.business_value or contract.key_decision):
        with st.container(border=True):
            if contract.goal:
                st.markdown(f"**Objetivo**: {contract.goal}")
            if contract.business_value:
                st.markdown(f"**Valor de negocio**: {contract.business_value}")
            if contract.key_decision:
                st.markdown(f"**Decisión habilitada**: {contract.key_decision}")


def render_key_takeaway(text: str) -> None:
    """Highlight the primary takeaway to reduce cognitive load."""
    if not text:
        return
    st.info(f"**Idea clave:** {text}")


def render_decision_box(text: str, owner: str | None = None, cadence: str | None = None) -> None:
    """Render a decision summary box with optional ownership metadata."""
    meta = []
    if owner:
        meta.append(f"owner: {owner}")
    if cadence:
        meta.append(f"cadencia: {cadence}")
    suffix = f" ({' · '.join(meta)})" if meta else ""
    st.success(f"**Decisión operativa sugerida**{suffix}: {text}")


def render_evidence_block(
    title: str,
    claim: str,
    chart_fn,
    implication: str,
    caveat: str | None = None,
) -> None:
    """Render a standardized evidence block around a chart callback."""
    st.markdown(f"### {title}")
    st.markdown(f"**Claim**: {claim}")
    chart_fn()
    st.markdown(f"**Implicación de decisión**: {implication}")
    if caveat:
        st.caption(f"Caveat: {caveat}")


def render_caveats(items: Iterable[str], *, title: str = "Límites y caveats") -> None:
    """Render caveats consistently across pages."""
    clean = [str(i).strip() for i in items if str(i).strip()]
    if not clean:
        return
    with st.expander(f"⚠️ {title}", expanded=False):
        for item in clean:
            st.markdown(f"- {item}")


def render_section_checkpoint(title: str, bullets: Iterable[str]) -> None:
    """Checkpoint summary for long pages."""
    bullets = [str(b).strip() for b in bullets if str(b).strip()]
    if not bullets:
        return
    st.markdown(f"#### {title}")
    st.markdown("\n".join(f"- {b}" for b in bullets))


def render_next_steps(page_links: Iterable[tuple[str, str, str]]) -> None:
    """Render next-step links as a compact narrative continuation block."""
    links = [(t, d, p) for t, d, p in page_links if t and p]
    if not links:
        return
    st.markdown("### Siguientes pasos")
    for title, desc, path in links:
        cols = st.columns([4, 1])
        with cols[0]:
            st.markdown(f"**{title}**")
            if desc:
                st.caption(desc)
        with cols[1]:
            try:
                st.page_link(path, label="Abrir", icon="➡️")
            except Exception:
                st.caption(f"➡️ {path}")


def render_page_feedback(page_id: str) -> None:
    """Optional lightweight feedback widget for non-research pages."""
    if not hasattr(st, "feedback"):
        return
    st.caption("¿Te fue útil esta página?")
    st.feedback("thumbs", key=f"feedback_{page_id}")
