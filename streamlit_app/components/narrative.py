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
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Siguiente:** {title}")
        st.caption(description)
    with col2:
        st.page_link(page_path, label=f"Ir a {title}", icon="➡️")
