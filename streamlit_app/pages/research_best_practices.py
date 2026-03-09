"""Shared best practices and tooling for paper drafting in the research section."""

from __future__ import annotations

import streamlit as st

from streamlit_app.components.paper_scaffold import (
    render_best_practices_and_tooling,
    render_paper_section_checklist,
)
from streamlit_app.components.story_shell import render_key_takeaway, render_page_header
from streamlit_app.content.page_contracts import get_page_contract

st.title("🧰 Buenas Prácticas y Herramientas")
st.caption("Guía central para preparar drafts de paper en las páginas de Investigación")
page_contract = get_page_contract("research_best_practices")
render_page_header(page_contract)
render_key_takeaway(
    "Esta guía es el playbook de redacción para la sección de Investigación: prioriza claridad para revisores expertos, trazabilidad y control de claims."
)
st.warning(
    "Documento de trabajo. Esta guía resume prácticas aplicables a los 3 drafts antes de "
    "la revisión con profesor experto."
)

st.markdown("## 1) Objetivo de esta página")
st.markdown(
    """
Esta página concentra recomendaciones transversales para evitar duplicación en los 3 drafts.
La meta es que cada página de paper sea defendible, reproducible y fácil de auditar por un
revisor académico.
"""
)

st.markdown("## 1.1) Storytelling para sección de Investigación (audiencia experta)")
st.markdown(
    """
Checklist mínimo recomendado para páginas de investigación / draft:

1. **Empieza con el claim técnico / pregunta de investigación**.
2. **Explicita método, datos y diseño experimental** antes de interpretar resultados.
3. **Usa una sola idea por figura** (evita figuras “todo en uno”).
4. **Muestra baseline/benchmark/objetivo explícito**.
5. **Cierra con límites y qué falta para publication-ready**.

Plantilla corta reusable (modo paper):
- Claim / novelty
- Método y protocolo
- Resultado principal
- Amenazas a validez
- Siguiente experimento / figura pendiente
"""
)

st.info(
    "Regla práctica en Investigación: si un revisor experto no puede responder "
    "“¿cuál es el claim, con qué evidencia y bajo qué límites?” en la primera pantalla, el draft está desordenado."
)

with st.expander("Fuentes externas usadas para estas prácticas"):
    st.markdown(
        """
- Microsoft Learn — principles for dashboard design:
  https://learn.microsoft.com/en-us/power-bi/guidance/dashboard-design
- Microsoft Power BI — data storytelling overview:
  https://learn.microsoft.com/en-us/training/modules/power-bi-effective-storytelling/
- Tableau Blueprint — data storytelling best practices:
  https://www.tableau.com/learn/blueprint/data-storytelling
- Adaptive dashboards (research perspective on audience-adaptive storytelling):
  https://arxiv.org/abs/2404.11131
"""
    )

st.markdown("## 2) Estructura mínima de cada draft")
render_paper_section_checklist()

st.markdown("## 2.1) Checklist obligatorio de validez metodológica")
st.markdown(
    """
Antes de marcar un draft como "listo para revisión", verificar explícitamente:

1. **Supuestos de identificación/evaluación**:
   - ¿Qué supuestos se requieren (exchangeability, overlap/positivity, PH, i.i.d., etc.)?
   - ¿Qué evidencia local muestra que son plausibles?
   - ¿Qué decisión queda bloqueada si el supuesto falla?
2. **Amenazas a validez**:
   - Sesgo de selección / leakage / drift temporal.
   - Sensibilidad a random_state, split o hiperparámetros.
   - Riesgo de extrapolación fuera de soporte observado.
3. **Negativos y ablation**:
   - ¿Se reporta un baseline fuerte y al menos un challenger razonable?
   - ¿Existe ablation de componentes (sin calibración, sin conformal, sin robustez)?
   - ¿Se incluye al menos un resultado negativo o limitación no resuelta?
"""
)
with st.expander("Plantilla reusable: supuesto -> evidencia -> límite -> decisión"):
    st.code(
        "\n".join(
            [
                "| Supuesto | Evidencia en este repo | Si falla | Decisión afectada |",
                "|---|---|---|---|",
                "| Exchangeability | backtest por cohortes + alertas mensuales | Cobertura inválida por régimen | Política robusta por intervalos |",
                "| Overlap/Positivity | distribución de tratamiento por segmento | CATE inestable | Reglas causales por segmento |",
                "| PH (Cox) | Schoenfeld tests/reportes | HR no interpretable globalmente | Proyección de riesgo lifetime |",
            ]
        ),
        language="markdown",
    )

st.markdown("## 3) Inserción de LaTeX en Streamlit (recomendado)")
st.markdown(
    """
Buenas prácticas:
- Usar `st.latex(...)` para ecuaciones display y `st.caption(...)` para numeración (`Equation 1`, `Equation 2`, ...).
- Mantener una notación consistente entre secciones (mismas letras para los mismos conceptos).
- Evitar ecuaciones huérfanas: toda ecuación debe tener 1-2 frases de interpretación inmediata.
"""
)
st.code(
    "\n".join(
        [
            'st.latex(r"\\mathrm{ECL}_i = PD_i \\times LGD_i \\times EAD_i \\times DF_i")',
            'st.caption("Equation 1. Definicion de perdida esperada por prestamo.")',
        ]
    ),
    language="python",
)

st.markdown("## 4) Inserción de Figuras (plotly/matplotlib)")
st.markdown(
    """
Buenas prácticas:
- Numerar y titular figuras en formato paper (`Figure 1`, `Figure 2`, ...).
- Mostrar ejes con unidades, y agregar líneas de referencia cuando existan targets nominales.
- Mantener una figura por mensaje principal; mover gráficos secundarios a apéndice.
"""
)
st.code(
    "\n".join(
        [
            'fig = px.line(df, x="month", y="coverage_90", title="Figure 1. Monthly Coverage")',
            'fig.add_hline(y=0.90, line_dash="dash", line_color="orange")',
            'st.plotly_chart(fig, width="stretch")',
            'st.caption("Figure 1. Cobertura mensual vs target nominal 90%.")',
        ]
    ),
    language="python",
)

st.markdown("## 5) Inserción de Tablas y exportables")
st.markdown(
    """
Buenas prácticas:
- Publicar tablas principales en la página y detalles en `Appendix Tables`.
- Mantener columnas estrictamente necesarias en tabla principal.
- Exportar CSV y LaTeX desde artefactos versionados para trazabilidad.
"""
)
st.code(
    "\n".join(
        [
            'st.dataframe(table_df, width="stretch", hide_index=True)',
            'download_table(table_df, "paper_table1_main.csv")',
            'table_df.to_latex("reports/paper_material/paperX/tables/paper_table1_main.tex", index=False)',
        ]
    ),
    language="python",
)

st.markdown("## 6) Flujo operativo recomendado en este repo")
st.code(
    "\n".join(
        [
            "uv run dvc repro build_pipeline_results export_streamlit_artifacts",
            "uv run python scripts/run_paper_notebook_suite.py",
            "uv run pytest -q",
        ]
    ),
    language="bash",
)
st.caption(
    "Artefactos de paper disponibles en `reports/paper_material/paper1`, "
    "`reports/paper_material/paper2` y `reports/paper_material/paper3`."
)

st.markdown("## 7) Lista de revisión para reunión con profesor")
st.markdown(
    """
- Confirmar si la pregunta de investigación de cada paper es publicable y diferenciada.
- Validar si los baselines incluidos son suficientes para sostener el claim.
- Identificar qué figuras/tablas deben pasar de draft a versión final.
- Definir qué parte debe moverse a apéndice para mantener foco narrativo.
- Priorizar riesgos metodológicos que requieren experimentos adicionales.
"""
)

st.markdown("## 8) Referencias de tooling y estándares")
render_best_practices_and_tooling()

st.markdown("## 9) Enlaces de trabajo (evitar duplicación)")
st.markdown(
    """
- `Panorama de Investigación` = registro maestro de posicionamiento y referencias.
- `Paper 1/2/3` = drafts enfocados en novelty claim específico.
- Esta página = playbook transversal para estructura, figuras, tablas y revisión.
"""
)
