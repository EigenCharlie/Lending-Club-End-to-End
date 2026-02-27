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
    value_key = "audience_mode"
    widget_key = "_audience_mode_widget"

    if value_key not in st.session_state:
        st.session_state[value_key] = options[0]

    current = str(st.session_state.get(value_key, options[0]))
    if current not in options:
        current = options[0]
        st.session_state[value_key] = current

    # Mantener el estado del widget separado evita el error de Streamlit al
    # modificar la misma key después de instanciar el control.
    widget_current = st.session_state.get(widget_key, current)
    if isinstance(widget_current, list):
        widget_current = widget_current[0] if widget_current else current
    widget_current = str(widget_current or current)
    if widget_current not in options:
        widget_current = current
    if st.session_state.get(widget_key) != widget_current:
        st.session_state[widget_key] = widget_current

    # Streamlit moderno: segmented control reduce fricción visual. Fallback a radio.
    if hasattr(st, "segmented_control"):
        selected = st.segmented_control(
            "Nivel de detalle",
            options=options,
            selection_mode="single",
            help="Ajusta profundidad de explicación según audiencia",
            key=widget_key,
            width="content",
        )
    else:
        selected = st.radio(
            "Nivel de detalle",
            options=options,
            horizontal=True,
            help="Ajusta profundidad de explicación según audiencia",
            key=widget_key,
        )
    if isinstance(selected, list):
        selected = selected[0] if selected else current
    selected = str(selected or current)
    if selected not in options:
        selected = current
    if st.session_state.get(value_key) != selected:
        st.session_state[value_key] = selected
    return selected
