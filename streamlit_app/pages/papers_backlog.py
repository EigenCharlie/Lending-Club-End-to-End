"""Publication roadmap page for unified papers + Quarto backlog."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from streamlit_app.components.release_governance import render_release_governance
from streamlit_app.components.story_shell import render_key_takeaway, render_page_header
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.utils import PROJECT_ROOT

DOCS_DIR = PROJECT_ROOT / "docs"
BACKLOG_UNIFIED = DOCS_DIR / "backlog-papers-unified.md"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


st.title("📝 Backlog Papers y Quarto")
st.caption(
    "Roadmap editorial y de publicaciones que conecta el backlog técnico con papers, "
    "paper estrella y migración futura a Quarto."
)

page_contract = get_page_contract("papers_backlog")
render_page_header(page_contract)
render_key_takeaway(
    "Esta página no reemplaza el backlog operativo: lo reorganiza para responder qué bloquea "
    "cada paper, qué alimenta el paper estrella y qué puede avanzar en paralelo en Quarto."
)

with st.container(border=True):
    st.markdown("#### Diferencia frente a `Roadmap y Backlog`")
    st.markdown(
        """
- **`Roadmap y Backlog`** responde: qué debemos mejorar en los pipelines y artefactos oficiales.
- **`Backlog Papers y Quarto`** responde: qué bloquea cada paper, cómo se traduce cada mejora a
  resultados publicables y qué se puede avanzar en el libro/Quarto mientras tanto.
- Regla práctica:
  si la prioridad es **operación / champion / baseline**, usa `Roadmap y Backlog`;
  si la prioridad es **papers / Q1 / narrativa editorial**, usa esta página.
"""
    )

render_release_governance(show_title=True)

st.markdown("## Documento unificado")
unified_text = _read_text(BACKLOG_UNIFIED)
if unified_text:
    st.download_button(
        "Descargar backlog-papers-unified.md",
        data=unified_text,
        file_name="backlog-papers-unified.md",
        mime="text/markdown",
    )
    st.markdown(unified_text)
else:
    st.error("No se pudo cargar `docs/backlog-papers-unified.md`.")

with st.expander("Cómo usar esta página"):
    st.markdown(
        """
1. Confirmar primero el baseline oficial y el estado promovido del champion.
2. Leer la prioridad global para identificar qué bloquea cada paper.
3. Traducir cada bloque técnico a entregables:
   figuras, tablas, claims, apéndices y capítulos Quarto.
4. Volver a `Roadmap y Backlog` cuando necesites bajar del roadmap editorial al trabajo técnico concreto.
"""
    )
