"""Bloques narrativos adaptados por audiencia."""

from __future__ import annotations

import streamlit as st


def narrative_block(
    audience: str,
    general: str,
    business: str = "",
    technical: str = "",
):
    """Muestra texto narrativo según el nivel de detalle seleccionado.

    Args:
        audience: Nivel actual (General/Negocio/Técnico).
        general: Texto base visible para toda audiencia.
        business: Texto adicional para audiencia de negocio.
        technical: Texto adicional para audiencia técnica.
    """
    st.markdown(general)
    if audience == "Negocio" and business:
        st.markdown(business)
    elif audience == "Técnico":
        if business:
            st.markdown(business)
        if technical:
            st.markdown(technical)


def next_page_teaser(title: str, description: str, page_path: str):
    """Muestra una tarjeta de continuidad narrativa hacia la siguiente página."""
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Siguiente:** {title}")
            st.caption(description)
        with col2:
            try:
                st.page_link(page_path, label=f"Ir a {title}", icon=":material/arrow_forward:")
            except Exception:
                st.caption(f":material/arrow_forward: {page_path}")


def storytelling_intro(
    page_goal: str,
    business_value: str,
    key_decision: str,
    how_to_read: list[str] | None = None,
) -> None:
    """Intro narrativa canónica compacta.

    Core prompts preserved for tests and page contracts:
    - Qué resuelve esta técnica
    - Por qué importa en negocio
    - Decisión que habilita
    """
    with st.expander("Guía de lectura", expanded=False, icon=":material/info:"):
        st.markdown(f"**Objetivo:** {page_goal}")
        st.markdown(f"**Valor de negocio:** {business_value}")
        st.markdown(f"**Decisión que habilita:** {key_decision}")
        if how_to_read:
            st.markdown("**Cómo leer esta página:**")
            for step in how_to_read:
                st.markdown(f"- {step}")


def reading_path(steps: list[str]) -> None:
    """Render a compact numbered path for long pages."""
    if not steps:
        return
    st.markdown("**Ruta de lectura sugerida**")
    for idx, step in enumerate(steps, start=1):
        st.markdown(f"{idx}. {step}")


def claim_evidence_implication(claim: str, evidence: str, implication: str) -> None:
    """Small narrative triad to standardize chart/table interpretation."""
    st.markdown(f"**Claim**: {claim}")
    st.markdown(f"**Evidencia**: {evidence}")
    st.markdown(f"**Implicación**: {implication}")


def threats_to_validity_dialog(title: str, bullets: list[str]) -> None:
    """Show threats-to-validity in dialog (or expander fallback) to reduce page clutter."""
    if not bullets:
        return
    body = "\n".join(f"- {b}" for b in bullets if str(b).strip())
    if hasattr(st, "dialog"):

        @st.dialog(title)
        def _dialog() -> None:
            st.markdown(body)

        if st.button("Ver amenazas a validez", key=f"threats_{title}"):
            _dialog()
    else:
        with st.expander("Ver amenazas a validez", expanded=False):
            st.markdown(body)
