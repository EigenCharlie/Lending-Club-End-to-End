"""Paper 1 draft: Conformal Prediction + Robust Portfolio Optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.components.paper_scaffold import render_phase_tracker
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import (
    download_table,
    format_number,
    format_pct,
    try_load_json,
    try_load_parquet,
)

st.title("🧪 Paper 1 — Working Draft")
st.caption(
    "Conformal Prediction Intervals as Uncertainty Sets for Robust Credit Portfolio Optimization"
)
st.warning(
    "Borrador de trabajo para revisión académica. Los claims y tablas están ligados a artefactos "
    "canónicos del proyecto y pueden refinarse tras feedback del profesor."
)

pipeline_summary = try_load_json("pipeline_summary", directory="data", default={})
conformal_status = try_load_json("conformal_policy_status", directory="models", default={})
model_comparison = try_load_json("model_comparison", directory="data", default={})

robust_summary = try_load_parquet("portfolio_robustness_summary")
robust_frontier = try_load_parquet("portfolio_robustness_frontier")
variant_benchmark = try_load_parquet("conformal_variant_benchmark")
variant_benchmark_by_group = try_load_parquet("conformal_variant_benchmark_by_group")

pipeline = pipeline_summary.get("pipeline", {})
conformal = pipeline_summary.get("conformal", {})
pd_metrics = model_comparison.get("final_test_metrics", {})
best_calibration = str(model_comparison.get("best_calibration", "calibración seleccionada"))

coverage_90 = float(conformal_status.get("coverage_90", conformal.get("coverage_90", np.nan)))
coverage_95 = float(conformal_status.get("coverage_95", conformal.get("coverage_95", np.nan)))
width_90 = float(conformal_status.get("avg_width_90", pipeline.get("interval_width_mean", np.nan)))
robust_return = float(pipeline.get("robust_return", np.nan))
nonrobust_return = float(pipeline.get("nonrobust_return", np.nan))
price_of_robustness = float(pipeline.get("price_of_robustness", np.nan))
robust_funded = int(pipeline.get("robust_funded", 0) or 0)
nonrobust_funded = int(pipeline.get("nonrobust_funded", 0) or 0)

st.markdown("## 0) Metadata de Propuesta")
meta_df = pd.DataFrame(
    [
        {"Campo": "Estado", "Valor": "Working Draft"},
        {"Campo": "Venue sugerido", "Valor": "European Journal of Operational Research (EJOR)"},
        {"Campo": "Dataset", "Valor": "Lending Club (split OOT)"},
        {"Campo": "Pregunta", "Valor": "Cómo optimizar portafolio crediticio bajo incertidumbre de PD"},
        {"Campo": "AUC PD", "Valor": f"{pd_metrics.get('auc_roc', np.nan):.4f}"},
        {"Campo": "Cobertura CP 90%", "Valor": format_pct(coverage_90, 2) if np.isfinite(coverage_90) else "N/D"},
        {"Campo": "Price of Robustness", "Valor": format_number(price_of_robustness, prefix="$")},
    ]
)
st.dataframe(meta_df, use_container_width=True, hide_index=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Cobertura 90%", format_pct(coverage_90, 2) if np.isfinite(coverage_90) else "N/D")
k2.metric("Cobertura 95%", format_pct(coverage_95, 2) if np.isfinite(coverage_95) else "N/D")
k3.metric("Retorno Robusto", format_number(robust_return, prefix="$"))
k4.metric("Retorno No Robusto", format_number(nonrobust_return, prefix="$"))

st.markdown("## 1) Abstract (Draft)")
st.markdown(
    f"""
Proponemos un framework **predict-then-optimize** para asignación de capital crediticio que
integra intervalos de **Mondrian Conformal Prediction** como conjuntos de incertidumbre para
optimización robusta vía Pyomo/HiGHS. En evaluación OOT, el modelo PD calibrado alcanza
AUC={pd_metrics.get('auc_roc', np.nan):.4f}; los intervalos conformales obtienen cobertura
{coverage_90:.2%} al nivel nominal 90% (ancho medio {width_90:.3f}).

En la frontera robusta, el portafolio no robusto logra {format_number(nonrobust_return, prefix='$')}
frente a {format_number(robust_return, prefix='$')} en versión robusta, con
{robust_funded} vs {nonrobust_funded} préstamos financiados (robusto vs no robusto), cuantificando
el costo de robustez en {format_number(price_of_robustness, prefix='$')}.
"""
)

st.markdown("## 2) Introduction")
st.markdown(
    """
Los pipelines crediticios tradicionales optimizan decisiones usando **predicciones puntuales** de PD,
ignorando la incertidumbre de estimación. Eso induce decisiones frágiles: pequeñas desviaciones de
calibración o shift temporal degradan la política de portafolio. Este paper aborda ese gap con tres
aportes: (i) cuantificación distribution-free de incertidumbre por loan grade, (ii) integración directa
al optimizador robusto y (iii) medición explícita del costo económico de robustez.
"""
)

st.markdown("## 3) Related Work (Resumen para borrador)")
related = pd.DataFrame(
    [
        ["Elmachtoub & Grigas (2022)", "SPO+", "Objetivo orientado a decisión"],
        ["Johnstone et al. (2021)", "Conformal uncertainty sets", "Conjuntos conformales para robust optimization"],
        ["Patel et al. (2024)", "Contextual CRO", "Conjuntos conformales contextuales"],
        ["Bertsimas & Sim (2004)", "Price of Robustness", "Marco clásico de robustez"],
    ],
    columns=["Referencia", "Eje", "Relevancia para este draft"],
)
st.dataframe(related, use_container_width=True, hide_index=True)

st.markdown("## 4) Data and Experimental Protocol")
st.markdown(
    f"""
- Split temporal OOT ya fijado en pipeline productivo del repositorio.
- Entrenamiento PD + calibración {best_calibration} + conformal Mondrian por grade.
- Evaluación robusta/no robusta sobre mismo universo candidato.
- Reporte de resultados en frontera por tolerancia de riesgo.
"""
)

st.markdown("## 5) Methods")
st.markdown("### 5.1 Conformal Prediction")
st.latex(
    r"\widehat{C}_{1-\alpha}(x,g) = [\max(0,\widehat{p}(x)-q_{1-\alpha,g}),\; \min(1,\widehat{p}(x)+q_{1-\alpha,g})]"
)
st.caption("Ecuación 1. Intervalo conformal por subgrupo Mondrian (grade g).")

st.markdown("### 5.2 Robust Portfolio Optimization")
st.latex(
    r"\max_{x \in \{0,1\}^n} \sum_i x_i\,(r_i - \lambda\,u_i) \quad \text{s.a.}\quad \sum_i x_i a_i \le B,\; \sum_i x_i p_i^{H} a_i \le \tau B + s"
)
st.caption("Ecuación 2. Formulación robusta simplificada con peor caso de PD.")

st.markdown("## 6) Results")

if not robust_summary.empty:
    fig1 = px.bar(
        robust_summary.melt(
            id_vars=["risk_tolerance"],
            value_vars=["baseline_nonrobust_return", "best_robust_return"],
            var_name="policy",
            value_name="return_net",
        ),
        x="risk_tolerance",
        y="return_net",
        color="policy",
        barmode="group",
        title="Figure 1. Net Return by Risk Tolerance (Robust vs Non-Robust)",
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Figure 1. Comparación de retorno neto por tolerancia de riesgo.")

    fig2 = px.line(
        robust_summary,
        x="risk_tolerance",
        y="price_of_robustness_pct",
        markers=True,
        title="Figure 2. Price of Robustness (%)",
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Figure 2. Costo relativo de robustez en la frontera.")

    fig3 = px.line(
        robust_summary,
        x="risk_tolerance",
        y=["best_robust_funded", "baseline_nonrobust_funded"],
        markers=True,
        title="Figure 3. Funded Loans by Policy",
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Figure 3. Tamaño de portafolio financiado bajo política robusta y no robusta.")

if not variant_benchmark.empty:
    fig4 = px.scatter(
        variant_benchmark,
        x="avg_width",
        y="min_group_coverage",
        color="variant",
        size="coverage",
        hover_data=["coverage", "coverage_gap", "std_group_coverage"],
        title="Figure 4. Conformal Variant Trade-off",
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Figure 4. Trade-off entre eficiencia (ancho) y cobertura mínima por grupo.")

st.markdown("### Tablas principales")
col_t1, col_t2 = st.columns(2)
with col_t1:
    st.markdown("**Table 1. Robustness Summary**")
    if robust_summary.empty:
        st.info("No se encontró `portfolio_robustness_summary.parquet`.")
    else:
        st.dataframe(robust_summary, use_container_width=True, hide_index=True)
        download_table(robust_summary, "paper1_table1_robustness_summary.csv")

with col_t2:
    st.markdown("**Table 2. Conformal Variant Benchmark**")
    if variant_benchmark.empty:
        st.info("No se encontró `conformal_variant_benchmark.parquet`.")
    else:
        st.dataframe(variant_benchmark, use_container_width=True, hide_index=True)
        download_table(variant_benchmark, "paper1_table2_conformal_variant_benchmark.csv")

with st.expander("Appendix Tables"):
    if not variant_benchmark_by_group.empty:
        st.markdown("**Table A1. Variant Benchmark by Group**")
        st.dataframe(variant_benchmark_by_group, use_container_width=True, hide_index=True)
        download_table(variant_benchmark_by_group, "paper1_tableA1_benchmark_by_group.csv")
    if not robust_frontier.empty:
        st.markdown("**Table A2. Robustness Frontier (full)**")
        st.dataframe(robust_frontier, use_container_width=True, hide_index=True)
        download_table(robust_frontier, "paper1_tableA2_robustness_frontier.csv")

st.markdown("## 7) Discussion")
st.markdown(
    """
Los resultados sugieren que la robustez reduce exposición al peor caso plausible de PD, pero a costo
significativo de retorno y volumen financiado. El aporte práctico está en transparentar ese trade-off
para decisiones de política crediticia y apetito de riesgo.
"""
)

st.markdown("## 8) Threats to Validity")
st.markdown(
    """
- Dependencia de hiperparámetros robustos y restricciones LP.
- Validez externa limitada a estructura de Lending Club hasta re-calibración en otros portafolios.
- Cobertura conformal no garantiza por sí sola optimalidad económica: se requieren ambos lentes.
"""
)

st.markdown("## 9) Reproducibility Package")
st.code(
    "\n".join(
        [
            "uv run dvc repro generate_conformal benchmark_conformal_variants optimize_portfolio optimize_portfolio_tradeoff",
            "uv run python scripts/run_paper_notebook_suite.py",
            "uv run pytest -q",
        ]
    ),
    language="bash",
)

st.markdown("## 10) Fases del Paper y Estado Actual")
render_phase_tracker(
    [
        {
            "Fase": "1. Scope + Research Question",
            "Estado": "Completada",
            "Evidencia": "Pregunta y claim definidos en draft",
            "Criterio de cierre": "Claim principal validado por revisión experta.",
        },
        {
            "Fase": "2. Related Work",
            "Estado": "Completada",
            "Evidencia": "Tabla de related work en esta página",
            "Criterio de cierre": "Agregar comparación final con papers que indique el profesor.",
        },
        {
            "Fase": "3. Methods + Results",
            "Estado": "Completada",
            "Evidencia": "Ecuaciones + figuras + tablas ligadas a artefactos",
            "Criterio de cierre": "Refinar notación y texto de interpretación.",
        },
        {
            "Fase": "4. Discussion + Validity",
            "Estado": "En progreso",
            "Evidencia": "Secciones 7 y 8",
            "Criterio de cierre": "Expandir riesgos, límites y recomendaciones por feedback.",
        },
    ]
)

st.markdown("### Puntos a Revisar / Complementar")
st.markdown(
    """
- **Section 1 (Abstract)**: ajustar wording para claim más conservador o más fuerte según criterio del profesor.
- **Section 2 (Introduction)**: decidir foco principal (metodológico vs impacto operativo).
- **Section 3 (Related Work / Table)**: incluir o eliminar baselines sugeridos por revisor.
- **Section 5 (Methods / Eq. 1)**: validar notación de cuantiles conformales y supuestos.
- **Section 5 (Methods / Eq. 2)**: confirmar formulación final (LP/MILP y definición de slack).
- **Figure 1**: revisar escala y normalización por capital para comparabilidad entre tolerancias.
- **Figure 2**: evaluar si reportar % y valor absoluto juntos en figura dual.
- **Figure 3**: añadir IC o bandas para funded count bajo escenarios.
- **Figure 4**: anotar claramente variante seleccionada y criterio de selección.
- **Table 1/2/A1/A2**: recortar columnas no críticas para versión de main paper y mover resto a apéndice.
"""
)
