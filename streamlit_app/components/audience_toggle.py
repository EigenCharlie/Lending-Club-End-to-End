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
    return st.radio(
        "Nivel de detalle",
        options=list(AUDIENCES.keys()),
        horizontal=True,
        help="Ajusta profundidad de explicación según audiencia",
        index=0,
        key="audience_level",
    )
