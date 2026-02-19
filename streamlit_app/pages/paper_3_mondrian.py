"""Paper 3 draft: Mondrian conformal prediction for group-conditional coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.components.paper_scaffold import render_phase_tracker
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import (
    download_table,
    format_pct,
    try_load_json,
    try_load_parquet,
)

st.title("📐 Paper 3 — Working Draft")
st.caption("Mondrian Conformal Prediction for Group-Conditional Credit Risk Coverage")
st.warning(
    "Borrador de trabajo para revisión académica. Esta version prioriza trazabilidad de "
    "resultados y claridad metodologica para evaluacion de factibilidad científica."
)

pipeline_summary = try_load_json("pipeline_summary", directory="data", default={})
conformal_status = try_load_json("conformal_policy_status", directory="models", default={})

group_metrics = try_load_parquet("conformal_group_metrics_mondrian")
monthly = try_load_parquet("conformal_backtest_monthly")
monthly_grade = try_load_parquet("conformal_backtest_monthly_grade")
alerts = try_load_parquet("conformal_backtest_alerts")
benchmark = try_load_parquet("conformal_variant_benchmark")
benchmark_by_group = try_load_parquet("conformal_variant_benchmark_by_group")

conformal = pipeline_summary.get("conformal", {})

coverage_90 = float(conformal_status.get("coverage_90", conformal.get("coverage_90", np.nan)))
coverage_95 = float(conformal_status.get("coverage_95", conformal.get("coverage_95", np.nan)))
min_group_cov = float(conformal_status.get("min_group_coverage_90", np.nan))
checks_passed = int(conformal_status.get("checks_passed", 0) or 0)
checks_total = int(conformal_status.get("checks_total", 0) or 0)

st.markdown("## 0) Metadata de Propuesta")
meta_df = pd.DataFrame(
    [
        {"Campo": "Estado", "Valor": "Working Draft"},
        {"Campo": "Venue sugerido", "Valor": "COPA / UQ Workshop / ML Risk venue"},
        {"Campo": "Pregunta", "Valor": "Como garantizar cobertura por subgrupo en riesgo crediticio con drift temporal"},
        {"Campo": "Dataset", "Valor": "Lending Club (grades A-G, split OOT)"},
        {"Campo": "Cobertura global 90%", "Valor": format_pct(coverage_90, 2) if np.isfinite(coverage_90) else "N/D"},
        {"Campo": "Cobertura global 95%", "Valor": format_pct(coverage_95, 2) if np.isfinite(coverage_95) else "N/D"},
        {"Campo": "Cobertura minima por grupo", "Valor": format_pct(min_group_cov, 2) if np.isfinite(min_group_cov) else "N/D"},
    ]
)
st.dataframe(meta_df, use_container_width=True, hide_index=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Cobertura 90%", format_pct(coverage_90, 2) if np.isfinite(coverage_90) else "N/D")
k2.metric("Cobertura 95%", format_pct(coverage_95, 2) if np.isfinite(coverage_95) else "N/D")
k3.metric("Min Coverage Grupo", format_pct(min_group_cov, 2) if np.isfinite(min_group_cov) else "N/D")
k4.metric("Policy Checks", f"{checks_passed}/{checks_total}")

st.markdown("## 1) Abstract (Draft)")
st.markdown(
    f"""
Aplicamos Mondrian Conformal Prediction para construir intervalos de riesgo con garantia
condicional por grupo (loan grades A-G) en una ventana out-of-time. El sistema alcanza
cobertura global de {format_pct(coverage_90, 2) if np.isfinite(coverage_90) else 'N/D'} al
nivel nominal 90% y {format_pct(coverage_95, 2) if np.isfinite(coverage_95) else 'N/D'}
al nivel 95%, con cobertura minima por grupo de
{format_pct(min_group_cov, 2) if np.isfinite(min_group_cov) else 'N/D'}.

Ademas del promedio global, presentamos monitoreo mensual de cobertura y alertas operativas
para detectar desviaciones de exchangeability. Comparamos variantes conformales y cuantificamos
el trade-off entre eficiencia (ancho) y garantia por subgrupo.
"""
)

st.markdown("## 2) Introduction")
st.markdown(
    """
En riesgo de credito, la cobertura marginal global puede ocultar under-coverage severa en
subgrupos con distinto perfil de riesgo. Mondrian conformal corrige este problema al construir
calibraciones separadas por grupo, pero introduce trade-offs de varianza cuando algunos grupos
son pequenos.

Este draft se enfoca en: (i) cobertura condicional por grade, (ii) estabilidad temporal de
cobertura en OOT, y (iii) comparacion sistematica contra variantes split/global para tomar una
decision metodologica defendible frente a un revisor experto.
"""
)

st.markdown("## 3) Related Work (Resumen para borrador)")
related = pd.DataFrame(
    [
        ["Vovk, Gammerman & Shafer (2005)", "Conformal foundations", "Garantias de cobertura en muestra finita"],
        ["Ding et al. (2023)", "Class-conditional CP", "Base para garantia por subgrupo"],
        ["Gibbs & Candes (2021)", "Adaptive conformal", "Relevante para drift temporal"],
        ["Plassier et al. (2024)", "Approx conditional validity", "Puente entre validez marginal y condicional"],
    ],
    columns=["Referencia", "Eje", "Relevancia para este draft"],
)
st.dataframe(related, use_container_width=True, hide_index=True)

st.markdown("## 4) Data and Experimental Protocol")
st.markdown(
    """
- Split temporal OOT fijo y monitoreo mensual por cohorte.
- Calibracion Mondrian por grade con alphas operativos (90%/95%).
- Evaluacion global + group-conditional + backtest mensual + alertas.
- Benchmark contra variantes globales y Mondrian alternativo.
"""
)

st.markdown("## 5) Methods")
st.markdown("### 5.1 Mondrian Conformal Set by Group")
st.latex(
    r"\widehat{C}_{1-\alpha}(x,g) = [\max(0,\widehat{p}(x)-q_{1-\alpha,g}),\;\min(1,\widehat{p}(x)+q_{1-\alpha,g})]"
)
st.caption("Equation 1. Intervalo conformal por grupo Mondrian `g`.")

st.markdown("### 5.2 Group-Conditional Coverage Estimator")
st.latex(
    r"\widehat{\mathrm{Cov}}_g = \frac{1}{n_g}\sum_{i: G_i=g}\mathbf{1}\{Y_i \in \widehat{C}_{1-\alpha}(X_i,g)\}"
)
st.caption("Equation 2. Cobertura empirica por subgrupo.")

st.markdown("### 5.3 Monitoring Rule")
st.latex(
    r"\mathrm{Alert}_{m,g}=\mathbf{1}\left[\widehat{\mathrm{Cov}}_{m,g}<1-\alpha-\epsilon\;\land\;n_{m,g}\ge n_{\min}\right]"
)
st.caption("Equation 3. Regla de alerta mensual por mes `m` y grupo `g`.")

st.markdown("## 6) Results")

if not group_metrics.empty and {"group", "coverage_90", "coverage_95"}.issubset(group_metrics.columns):
    gm_plot = group_metrics[["group", "coverage_90", "coverage_95"]].copy().sort_values("group")
    fig1 = px.bar(
        gm_plot.melt(id_vars=["group"], value_vars=["coverage_90", "coverage_95"], var_name="metric", value_name="coverage"),
        x="group",
        y="coverage",
        color="metric",
        barmode="group",
        labels={"group": "Grade", "coverage": "Coverage"},
        title="Figure 1. Coverage by Grade (90% and 95%)",
        template=PLOTLY_TEMPLATE,
    )
    fig1.add_hline(y=0.90, line_dash="dash", line_color="orange")
    fig1.add_hline(y=0.95, line_dash="dot", line_color="green")
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Figure 1. Cobertura por grade con lineas de referencia nominales.")

if not group_metrics.empty and {"group", "avg_width_90", "n"}.issubset(group_metrics.columns):
    width_plot = group_metrics[["group", "avg_width_90", "n"]].copy().sort_values("group")
    fig2 = px.bar(
        width_plot,
        x="group",
        y="avg_width_90",
        color="n",
        title="Figure 2. Average Interval Width by Grade (90%)",
        labels={"group": "Grade", "avg_width_90": "Average width", "n": "N by grade"},
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Figure 2. Eficiencia de intervalos por grade (ancho medio al 90%).")

if not monthly.empty and {"month", "coverage_90", "coverage_95"}.issubset(monthly.columns):
    month_plot = monthly.copy().sort_values("month")
    fig3 = px.line(
        month_plot,
        x="month",
        y=["coverage_90", "coverage_95"],
        markers=True,
        title="Figure 3. Monthly OOT Coverage",
        labels={"month": "Mes", "value": "Coverage"},
        template=PLOTLY_TEMPLATE,
    )
    fig3.add_hline(y=0.90, line_dash="dash", line_color="orange")
    fig3.add_hline(y=0.95, line_dash="dot", line_color="green")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Figure 3. Trayectoria temporal de cobertura en OOT.")

if not benchmark.empty and {"variant", "avg_width", "min_group_coverage", "coverage"}.issubset(benchmark.columns):
    fig4 = px.scatter(
        benchmark,
        x="avg_width",
        y="min_group_coverage",
        color="variant",
        size="coverage",
        hover_data=["coverage", "coverage_gap", "std_group_coverage"],
        title="Figure 4. Variant Trade-off (Efficiency vs Min Group Coverage)",
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Figure 4. Trade-off entre eficiencia y cobertura minima por grupo.")

if not monthly_grade.empty and {"month", "grade", "gap_90"}.issubset(monthly_grade.columns):
    mg_plot = monthly_grade.copy()
    heat = mg_plot.pivot_table(index="month", columns="grade", values="gap_90", aggfunc="mean")
    fig5 = px.imshow(
        heat,
        title="Figure 5. Monthly Coverage Gap by Grade (90%)",
        labels={"x": "Grade", "y": "Month", "color": "Gap 90%"},
        text_auto=".3f",
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Figure 5. Mapa de gap mensual de cobertura al 90% por grade.")

st.markdown("### Tablas principales")
col_t1, col_t2 = st.columns(2)
with col_t1:
    st.markdown("**Table 1. Group Metrics (Mondrian)**")
    if group_metrics.empty:
        st.info("No se encontro `conformal_group_metrics_mondrian.parquet`.")
    else:
        st.dataframe(group_metrics, use_container_width=True, hide_index=True)
        download_table(group_metrics, "paper3_table1_group_metrics.csv")

with col_t2:
    st.markdown("**Table 2. Monthly Backtest**")
    if monthly.empty:
        st.info("No se encontro `conformal_backtest_monthly.parquet`.")
    else:
        st.dataframe(monthly, use_container_width=True, hide_index=True)
        download_table(monthly, "paper3_table2_monthly_backtest.csv")

with st.expander("Appendix Tables"):
    st.markdown("**Table A1. Monthly Backtest by Grade**")
    if monthly_grade.empty:
        st.info("No se encontro `conformal_backtest_monthly_grade.parquet`.")
    else:
        st.dataframe(monthly_grade, use_container_width=True, hide_index=True)
        download_table(monthly_grade, "paper3_tableA1_monthly_grade.csv")

    st.markdown("**Table A2. Operational Alerts**")
    if alerts.empty:
        st.info("No se encontro `conformal_backtest_alerts.parquet`.")
    else:
        st.dataframe(alerts, use_container_width=True, hide_index=True)
        download_table(alerts, "paper3_tableA2_alerts.csv")

    st.markdown("**Table A3. Variant Benchmark (Global)**")
    if benchmark.empty:
        st.info("No se encontro `conformal_variant_benchmark.parquet`.")
    else:
        st.dataframe(benchmark, use_container_width=True, hide_index=True)
        download_table(benchmark, "paper3_tableA3_variant_benchmark.csv")

    st.markdown("**Table A4. Variant Benchmark by Group**")
    if benchmark_by_group.empty:
        st.info("No se encontro `conformal_variant_benchmark_by_group.parquet`.")
    else:
        st.dataframe(benchmark_by_group, use_container_width=True, hide_index=True)
        download_table(benchmark_by_group, "paper3_tableA4_benchmark_by_group.csv")

st.markdown("## 7) Discussion")
st.markdown(
    """
Mondrian mejora la lectura operativa de equidad de cobertura al forzar trazabilidad por
subgrupo. El costo es mayor varianza de cobertura en grupos pequenos y sensibilidad a drift.
El valor aplicado del paper es convertir estas tensiones en reglas de monitoreo claras.
"""
)

st.markdown("## 8) Threats to Validity")
st.markdown(
    """
- Efectos de muestra finita en grades con bajo `n` pueden inflar varianza de cobertura.
- Exchangeability puede romperse en shocks macro, afectando garantia empirica mensual.
- Umbrales de alerta (`epsilon`, `n_min`) requieren ajuste por institucion y apetito de riesgo.
"""
)

st.markdown("## 9) Reproducibility Package")
st.code(
    "\n".join(
        [
            "uv run dvc repro generate_conformal benchmark_conformal_variants backtest_conformal_coverage validate_conformal_policy",
            "uv run python scripts/run_paper_notebook_suite.py",
            "uv run pytest -q tests/test_models/test_conformal.py",
        ]
    ),
    language="bash",
)

st.markdown("## 10) Fases del Paper y Estado Actual")
render_phase_tracker(
    [
        {
            "Fase": "1. Problem Framing",
            "Estado": "Completada",
            "Evidencia": "Secciones 0-2",
            "Criterio de cierre": "Confirmar novelty claim frente a literatura mas reciente.",
        },
        {
            "Fase": "2. Related Work + Positioning",
            "Estado": "Completada",
            "Evidencia": "Tabla de related work (Seccion 3)",
            "Criterio de cierre": "Agregar referencias sugeridas por profesor.",
        },
        {
            "Fase": "3. Metrics + Protocol",
            "Estado": "Completada",
            "Evidencia": "Secciones 4-5 y Ecuaciones 1-3",
            "Criterio de cierre": "Validar definiciones finales de alertas y cobertura.",
        },
        {
            "Fase": "4. Results + Ablations",
            "Estado": "Completada",
            "Evidencia": "Figuras 1-5 y Tablas 1-2-A1-A2-A3-A4",
            "Criterio de cierre": "Refinar narrativa de trade-offs para version final.",
        },
        {
            "Fase": "5. Discussion + Limitations",
            "Estado": "En progreso",
            "Evidencia": "Secciones 7-8",
            "Criterio de cierre": "Ajustar alcance y limites tras revision experta.",
        },
    ]
)

st.markdown("### Puntos a Revisar / Complementar")
st.markdown(
    """
- **Section 1 (Abstract)**: ajustar claim de "group-conditional" segun lectura del profesor.
- **Section 2 (Introduction)**: reforzar caso de uso operativo en riesgo crediticio real.
- **Section 3 (Related Work / Table)**: decidir baseline adicional (CQR o ACP) como comparador principal.
- **Section 5 (Methods / Eq. 1)**: validar tratamiento de truncamiento [0,1] en intervalos de PD.
- **Section 5 (Methods / Eq. 3)**: documentar seleccion final de `epsilon` y `n_min`.
- **Figure 1**: añadir bandas de error por grade si el revisor lo considera necesario.
- **Figure 2**: evaluar normalizacion de ancho por cobertura efectiva para comparabilidad.
- **Figure 3**: considerar rolling windows adicionales (6m) para estabilidad temporal.
- **Figure 4**: marcar explicitamente la variante seleccionada como recomendada.
- **Figure 5**: revisar escala de color para enfatizar episodios criticos de under-coverage.
- **Table 1 / Table 2 / Table A1-A4**: separar claramente tablas core vs apendice tecnico.
"""
)
