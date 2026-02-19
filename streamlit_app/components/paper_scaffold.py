"""Reusable paper-writing scaffolds for Streamlit research pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

PAPER_SECTION_CHECKLIST = [
    "Title + Abstract (objetivo, gap, contribucion, resultados clave)",
    "Introduction (problema, contexto, aportes, estructura)",
    "Related Work (comparacion critica y posicionamiento)",
    "Data and Methods (dataset, pipeline, diseno experimental)",
    "Results (main tables/figures + analisis)",
    "Robustness / Sensitivity / Ablations",
    "Threats to Validity + Limitations",
    "Conclusion + Future Work",
    "Reproducibility Package (codigo, artefactos, comandos)",
]


def render_phase_tracker(phase_rows: list[dict[str, str]]) -> None:
    """Render the current paper development phases with status evidence."""
    st.subheader("Fases del Paper y Estado Actual")
    phases = pd.DataFrame(phase_rows)
    st.dataframe(phases, use_container_width=True, hide_index=True)


def render_paper_section_checklist() -> None:
    """Render canonical section checklist for an academic paper."""
    st.subheader("Checklist Canónico de Secciones")
    for item in PAPER_SECTION_CHECKLIST:
        st.markdown(f"- {item}")


def render_best_practices_and_tooling() -> None:
    """Render publication best practices and practical tooling references."""
    st.subheader("Buenas Prácticas y Herramientas (2026)")

    st.markdown(
        """
**Prácticas recomendadas para subir calidad de paper:**
- Definir desde el inicio el **claim principal** y una métrica primaria por claim.
- Mantener separación explícita entre resultados **in-sample** y **out-of-time**.
- Incluir sección formal de **amenazas a validez** (interna, externa, constructo).
- Publicar tablas/figuras directamente desde artefactos versionados para evitar desalineaciones narrativas.
- Preparar anexos con ablations, sensibilidad y detalles de implementación reproducible.
"""
    )

    st.markdown(
        """
**Checklist y guías de venues (fuentes):**
- ICMJE Recommendations: https://www.icmje.org/recommendations/
- NeurIPS (paper checklist): https://neurips.cc/Conferences/2025/CallForPapers
- ICLR Author Guide (reproducibility statement/checklist): https://iclr.cc/Conferences/2025/AuthorGuide
- ACM Artifact Review and Badging: https://www.acm.org/publications/policies/artifact-review-and-badging-current
"""
    )

    st.markdown(
        """
**Toolchain para generar material de paper (fuentes oficiales):**
- Quarto (manuscrito técnico reproducible): https://quarto.org/docs/guide/
- Jupyter Book (libro técnico/publicación): https://jupyterbook.org/en/stable/intro.html
- MyST markdown authoring: https://mystmd.org/guide
- Papermill (notebooks parametrizados): https://papermill.readthedocs.io/en/latest/
- nbclient (ejecución programática de notebooks): https://nbclient.readthedocs.io/en/latest/
- Jupytext (sync `.ipynb` <-> `.md/.py`): https://jupytext.readthedocs.io/en/latest/
- Plotly static export (figuras para paper): https://plotly.com/python/static-image-export/
- Pandas Styler `to_latex` (tablas publicables):
  https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_latex.html
"""
    )

    st.markdown(
        """
**Automatización bibliográfica / búsqueda programática:**
- Crossref REST API: https://api.crossref.org/
- Semantic Scholar API: https://www.semanticscholar.org/product/api
- OpenAlex client (`pyalex`): https://github.com/J535D165/pyalex
"""
    )

    st.markdown(
        """
**Reproducibilidad y citabilidad del repo:**
- DVC docs: https://dvc.org/doc
- Zenodo API/docs: https://developers.zenodo.org/
- CITATION en GitHub: https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content
"""
    )
    st.caption(
        "Playbook local del repo: `docs/PAPER_DEVELOPMENT_PLAYBOOK_2026.md` "
        "(workflow y comandos operativos para material publicable)."
    )
