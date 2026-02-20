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


def storytelling_intro(
    page_goal: str,
    business_value: str,
    key_decision: str,
    how_to_read: list[str] | None = None,
) -> None:
    """Renderiza una introducción narrativa corta y accionable.

    Se usa para páginas con audiencia mixta (no técnica + negocio + técnica).
    """
    st.markdown("### Cómo leer esta página")
    st.markdown(
        f"""
- **Qué resuelve esta técnica**: {page_goal}
- **Por qué importa en negocio**: {business_value}
- **Decisión que habilita**: {key_decision}
"""
    )
    if how_to_read:
        st.markdown("**Ruta sugerida de lectura**")
        for idx, step in enumerate(how_to_read, start=1):
            st.markdown(f"{idx}. {step}")
