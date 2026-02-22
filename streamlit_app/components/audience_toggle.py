"""Selector de nivel de detalle narrativo."""

from __future__ import annotations

import streamlit as st

AUDIENCES = {
    "General": "Explicación accesible para cualquier lector",
    "Negocio": "Enfoque para gestión de riesgo, portafolio y toma de decisiones",
    "Técnico": "Detalle metodológico: supuestos, métricas y trazabilidad",
}


def audience_selector() -> str:
    """Renderiza el selector de audiencia y retorna la opción elegida."""
    options = list(AUDIENCES.keys())
    if "audience_mode" not in st.session_state:
        st.session_state["audience_mode"] = options[0]

    current = str(st.session_state.get("audience_mode", options[0]))
    if current not in options:
        current = options[0]

    # Streamlit moderno: segmented control reduce fricción visual. Fallback a radio.
    if hasattr(st, "segmented_control"):
        selected = st.segmented_control(
            "Nivel de detalle",
            options=options,
            default=current,
            selection_mode="single",
            help="Ajusta profundidad de explicación según audiencia",
            key="audience_level_segmented",
        )
    else:
        selected = st.radio(
            "Nivel de detalle",
            options=options,
            horizontal=True,
            help="Ajusta profundidad de explicación según audiencia",
            index=options.index(current),
            key="audience_level",
        )
    selected = str(selected or current)
    st.session_state["audience_mode"] = selected
    return selected
