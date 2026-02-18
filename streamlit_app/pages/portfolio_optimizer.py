"""Optimización de portafolio de crédito con robustez."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import next_page_teaser
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import (
    format_number,
    get_notebook_image_path,
    load_json,
    load_parquet,
    try_load_parquet,
)

st.title("💼 Optimizador de Portafolio")
st.caption(
    "Decisión de asignación de capital usando predicción de riesgo y "
    "bandas de incertidumbre conformal."
)

with st.expander("¿Qué es optimización de portafolio de crédito?", expanded=False):
    st.markdown(
        """
### La idea en simple

Imagina que tienes **$1 millón** para prestar y **miles de solicitudes** de crédito.
Cada préstamo tiene un retorno esperado (intereses) y un riesgo (probabilidad de default).
¿A quién le prestas para **maximizar tus ganancias sin asumir demasiado riesgo**?

Eso es exactamente lo que resuelve la optimización de portafolio:

$$\\max \\sum_i (retorno_i - riesgo_i) \\times x_i$$

Sujeto a: presupuesto total ≤ $1M, concentración máxima por préstamo ≤ 25%, PD portafolio ≤ 10%

### ¿Cómo entra el conformal prediction?

- **Sin conformal**: el optimizador usa PD = 12% como si fuera exacto
- **Con conformal**: el optimizador sabe que la PD real podría ser hasta 16% (PD_high)
  y se protege contra ese peor caso plausible

### Price of Robustness = el costo del seguro

La diferencia de retorno entre la solución "optimista" (usa PD puntual) y la "robusta"
(usa PD_high) es el **Price of Robustness**. Es el precio que pagas por protegerte
de la incertidumbre del modelo — exactamente como un seguro.
"""
    )

st.markdown(
    """
Este bloque traduce machine learning en investigación de operaciones aplicada. La PD calibrada aporta una expectativa
de riesgo por préstamo, mientras que Conformal aporta un rango plausible (`pd_low`, `pd_high`) que captura incertidumbre
de estimación. Con esos insumos, el modelo de optimización decide asignación de capital bajo restricciones de presupuesto,
riesgo y factibilidad. La comparación robusto vs no robusto no es decorativa: cuantifica el costo económico de proteger
el downside cuando las predicciones no son perfectamente ciertas.
"""
)

summary = load_json("pipeline_summary")
pipeline = summary.get("pipeline", {})

alloc = load_parquet("portfolio_allocations")
rob_summary = load_parquet("portfolio_robustness_summary")
rob_frontier = load_parquet("portfolio_robustness_frontier")
efficient_frontier = try_load_parquet("efficient_frontier")
if efficient_frontier.empty:
    nonrobust = rob_frontier[rob_frontier["policy"] == "nonrobust"].copy()
    if not nonrobust.empty:
        efficient_frontier = pd.DataFrame(
            {
                "pd_cap": nonrobust["risk_tolerance"],
                "risk_wpd": nonrobust["point_pd"],
                "return": nonrobust["expected_return_net_point"],
            }
        ).sort_values("pd_cap")

kpi_row(
    [
        {"label": "Retorno robusto", "value": format_number(pipeline.get("robust_return", 0), prefix="$")},
        {"label": "Retorno no robusto", "value": format_number(pipeline.get("nonrobust_return", 0), prefix="$")},
        {"label": "Price of Robustness", "value": format_number(pipeline.get("price_of_robustness", 0), prefix="$")},
        {"label": "Aprobados robusto", "value": str(int(pipeline.get("robust_funded", 0)))},
        {"label": "Aprobados no robusto", "value": str(int(pipeline.get("nonrobust_funded", 0)))},
        {"label": "Tamaño batch", "value": str(int(pipeline.get("batch_size", 0)))},
    ],
    n_cols=3,
)

st.dataframe(
    pd.DataFrame(
        [
            {
                "Métrica": "Retorno neto",
                "Significado técnico": "Ingreso esperado menos pérdida esperada bajo restricciones.",
                "Significado negocio": "Resultado económico de la política de aprobación.",
            },
            {
                "Métrica": "Price of Robustness",
                "Significado técnico": "Diferencia entre objetivo no robusto y robusto.",
                "Significado negocio": "Costo explícito por proteger desempeño en peor caso.",
            },
            {
                "Métrica": "N aprobados",
                "Significado técnico": "Tamaño de portafolio factible bajo constraints de riesgo.",
                "Significado negocio": "Capacidad comercial compatible con apetito de riesgo.",
            },
        ]
    ),
    use_container_width=True,
    hide_index=True,
)

st.markdown(
    """
Este módulo conecta investigación de operaciones con ML:
- PD puntual ordena oportunidades.
- Intervalos conformal definen conjunto de incertidumbre.
- Pyomo/HiGHS optimiza retorno sujeto a presupuesto y riesgo.
"""
)

col_img1, col_img2 = st.columns(2)
with col_img1:
    img = get_notebook_image_path("08_portfolio_optimization", "cell_021_out_54.png")
    if img.exists():
        st.image(
            str(img),
            caption="Notebook 08: frontera eficiente con punto robusto vs no robusto.",
            use_container_width=True,
        )
    else:
        if efficient_frontier.empty:
            st.info("No hay frontera eficiente disponible en artefactos actuales.")
        else:
            fig = px.line(
                efficient_frontier.sort_values("pd_cap"),
                x="risk_wpd",
                y="return",
                markers=True,
                title="Fallback: frontera eficiente (artefacto actual)",
                labels={"risk_wpd": "Riesgo (PD ponderada)", "return": "Retorno esperado"},
            )
            fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Imagen de notebook no encontrada; se muestra la frontera reconstruida desde parquet."
            )
with col_img2:
    img = get_notebook_image_path("08_portfolio_optimization", "cell_015_out_14.png")
    if img.exists():
        st.image(
            str(img),
            caption="Notebook 08: sensibilidad retorno/aprobaciones al límite de PD.",
            use_container_width=True,
        )
    else:
        fig = px.bar(
            rob_summary,
            x="risk_tolerance",
            y="price_of_robustness_pct",
            title="Fallback: sensibilidad del price of robustness",
            labels={"risk_tolerance": "Tolerancia de riesgo", "price_of_robustness_pct": "Price of Robustness (%)"},
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Imagen de notebook no encontrada; se muestra sensibilidad usando resumen robusto actual.")

st.subheader("1) Perfil de asignación sobre préstamos")
alloc_plot = alloc.copy()
alloc_plot["financiado"] = alloc_plot["alloc"] > 0
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        alloc_plot,
        x="alloc",
        color="financiado",
        nbins=40,
        title="Distribución de pesos de asignación",
        labels={"alloc": "Peso asignado", "financiado": "Financiado"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Propósito: entender sparsidad de asignación. Insight: el optimizador concentra capital en un subconjunto reducido "
        "de préstamos con mejor trade-off riesgo-retorno."
    )
    st.markdown(
        "**Resultado:** La distribución muestra que la mayoría de préstamos recibe asignación cero "
        "— el optimizador es selectivo. Solo aquellos con combinación favorable de tasa y PD baja "
        "son elegidos, actuando como un filtro cuantitativo de aprobación."
    )

with col2:
    sample = alloc_plot.sample(min(3500, len(alloc_plot)), random_state=22)
    fig = px.scatter(
        sample,
        x="pd_point",
        y="alloc",
        color="int_rate",
        opacity=0.45,
        title="Asignación vs PD puntual",
        labels={"pd_point": "PD puntual", "alloc": "Asignación", "int_rate": "Tasa interés"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Propósito: observar regla de selección en el plano PD-retorno implícito. Insight: asignaciones altas tienden a "
        "zonas de PD contenida y tasa atractiva."
    )
    st.markdown(
        "**Resultado:** El scatter revela una frontera clara: asignaciones altas se concentran en "
        "PD < 15% con tasas > 10%. Los puntos con alta tasa pero PD alta (esquina superior derecha) "
        "reciben baja asignación — el riesgo no compensa el retorno."
    )

st.subheader("2) Frontera eficiente clásica")
if efficient_frontier.empty:
    st.info("No hay `efficient_frontier.parquet`; usando frontera no robusta derivada de tradeoff cuando aplica.")
else:
    fig = px.line(
        efficient_frontier.sort_values("pd_cap"),
        x="risk_wpd",
        y="return",
        markers=True,
        title="Trade-off riesgo-retorno (frontera eficiente)",
        labels={"risk_wpd": "Riesgo (PD ponderada)", "return": "Retorno esperado"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Propósito: mostrar frontera eficiente sin robustez explícita. Insight: mayor retorno exige mayor riesgo promedio "
        "de cartera."
    )
    st.markdown(
        """
**Interpretación de la frontera eficiente:**
- Cada punto representa una configuración de portafolio óptima para un nivel de riesgo dado.
- La pendiente decreciente indica **rendimientos marginales**: los primeros incrementos de riesgo aportan
  mucho retorno; más allá de cierto punto, el retorno adicional es mínimo.
- Para un comité de riesgo, esta curva permite elegir el punto de operación que mejor equilibre
  apetito de riesgo institucional con objetivos de rentabilidad.
"""
    )

st.subheader("3) Frontera robusta vs no robusta")
risk_levels = sorted(rob_frontier["risk_tolerance"].unique().tolist())
selected_risk = st.selectbox("Nivel de tolerancia de riesgo", risk_levels, index=0)
frontier_slice = rob_frontier[rob_frontier["risk_tolerance"] == selected_risk].copy()
frontier_slice["incertidumbre"] = frontier_slice["uncertainty_aversion"].astype(str)

fig = px.line(
    frontier_slice,
    x="uncertainty_aversion",
    y="expected_return_net_point",
    color="policy",
    markers=True,
    title=f"Retorno neto esperado por aversión a incertidumbre (tolerancia={selected_risk:.2f})",
    labels={
        "uncertainty_aversion": "Aversión a incertidumbre",
        "expected_return_net_point": "Retorno neto esperado",
        "policy": "Política",
    },
)
fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Propósito: cuantificar efecto de aversión a incertidumbre sobre retorno. Insight: mayor robustez reduce retorno esperado, "
    "pero estabiliza desempeño frente a error de PD."
)
st.markdown(
    """
**Interpretación de la frontera robusta:**
- La línea **no robusta** muestra el retorno si el modelo fuera perfecto (usa PD puntual).
- La línea **robusta** usa PD_high del conformal — protege contra el peor caso plausible.
- La brecha entre ambas líneas es el **Price of Robustness**: cuánto retorno se sacrifica por protección.
- A mayor aversión a incertidumbre, el portafolio robusto aprueba menos préstamos pero con mayor confianza
  de que el retorno se materialice incluso si las PDs reales son peores que las estimadas.
"""
)

st.subheader("4) Síntesis de robustez por tolerancia")
summary_view = rob_summary[
    [
        "risk_tolerance",
        "baseline_nonrobust_return",
        "best_robust_return",
        "best_robust_funded",
        "price_of_robustness",
        "price_of_robustness_pct",
    ]
].copy()
summary_view["price_of_robustness_pct"] = summary_view["price_of_robustness_pct"] / 100
st.dataframe(summary_view, use_container_width=True, hide_index=True)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=rob_summary["risk_tolerance"],
        y=rob_summary["baseline_nonrobust_return"],
        mode="lines+markers",
        name="Retorno no robusto",
    )
)
fig.add_trace(
    go.Scatter(
        x=rob_summary["risk_tolerance"],
        y=rob_summary["best_robust_return"],
        mode="lines+markers",
        name="Mejor retorno robusto",
    )
)
fig.update_layout(
    **PLOTLY_TEMPLATE["layout"],
    title="Costo de robustez según tolerancia de riesgo",
    xaxis_title="Tolerancia de riesgo de portafolio",
    yaxis_title="Retorno",
    height=390,
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Propósito: comparar retorno robusto vs no robusto por tolerancia de riesgo. Insight: el price of robustness no es constante; "
    "depende del apetito de riesgo definido por negocio."
)
st.markdown(
    """
**Resultado clave:** El costo de robustez varía con la tolerancia. A tolerancias bajas (portafolios
conservadores), la penalización por robustez es proporcional. A tolerancias altas (portafolios agresivos),
la brecha se amplifica — proteger un portafolio agresivo contra incertidumbre es más costoso porque hay
más préstamos riesgosos cuya PD podría desviarse al alza.
"""
)

st.markdown(
    """
**Lectura de decisión:**
- La robustez reduce upside esperado, pero protege contra escenarios adversos de PD.
- El proyecto no oculta este costo: lo reporta explícitamente como una métrica de política.
- Este output alimenta discusión ejecutiva de apetito de riesgo y restricciones regulatorias.
"""
)

st.subheader("5) Contexto histórico: ROI realizado por grade")
st.markdown(
    """
Para anclar las decisiones del optimizador en evidencia histórica, calculamos el **retorno realizado**
(ROI) sobre préstamos ya terminados (Fully Paid + Charged Off) del dataset completo 2007-2020.
"""
)

roi_grade = load_parquet("roi_by_grade")
roi_term = load_parquet("roi_by_grade_term")

if not roi_grade.empty:
    col_roi1, col_roi2 = st.columns(2)
    with col_roi1:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=roi_grade["grade"],
                y=roi_grade["roi_mean"],
                name="ROI medio",
                marker_color="#00D4AA",
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": (roi_grade["roi_p90"] - roi_grade["roi_mean"]).tolist(),
                    "arrayminus": (roi_grade["roi_mean"] - roi_grade["roi_p10"]).tolist(),
                },
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#FF6B6B")
        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            title="ROI medio por grade (con rango P10-P90)",
            height=400,
        )
        fig.update_layout(yaxis_title="ROI", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "ROI = (total recibido - monto fondeado) / monto fondeado. "
            "Barras de error muestran percentiles 10 y 90 — la dispersión crece con el riesgo."
        )

    with col_roi2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=roi_grade["default_rate"],
                y=roi_grade["roi_mean"],
                mode="markers+text",
                text=roi_grade["grade"],
                textposition="top center",
                marker={"size": roi_grade["n_loans"] / roi_grade["n_loans"].max() * 40 + 8, "color": "#0B5ED7"},
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#FF6B6B")
        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            title="Frontera riesgo-retorno histórica",
            height=400,
        )
        fig.update_layout(
            xaxis_title="Default rate",
            yaxis_title="ROI medio",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".1%",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Tamaño del punto proporcional al volumen. Grade G es el único con ROI medio negativo. "
            "La frontera histórica valida las decisiones del optimizador."
        )

    st.markdown(
        f"""
**Lectura del ROI histórico:**
- **Grades A-B**: ROI medio positivo ({roi_grade.iloc[0]['roi_mean']:.1%} y {roi_grade.iloc[1]['roi_mean']:.1%})
  con dispersión moderada. Son los segmentos donde el optimizador concentra capital.
- **Grades D-F**: ROI medio aún positivo pero con **enorme dispersión** (P10 negativo, P90 alto).
  La incertidumbre justifica el enfoque robusto: sin protección, estos segmentos generan volatilidad.
- **Grade G**: ROI medio **negativo** ({roi_grade.iloc[6]['roi_mean']:.1%}) — default rate ~49% destruye valor
  incluso con tasas altas. El optimizador robusto los excluye correctamente.
- La frontera riesgo-retorno histórica confirma que la relación no es lineal: más riesgo no siempre
  compensa con más retorno, validando la necesidad de optimización formal.
"""
    )

if not roi_term.empty:
    with st.expander("ROI por grade y plazo (36 vs 60 meses)"):
        roi_term_display = roi_term.copy()
        roi_term_display["term"] = roi_term_display["term"].str.strip()
        roi_term_display = roi_term_display.rename(columns={
            "grade": "Grade", "term": "Plazo", "n_loans": "Préstamos",
            "default_rate": "Default rate", "roi_mean": "ROI medio",
        })
        roi_term_display["Default rate"] = roi_term_display["Default rate"].map("{:.1%}".format)
        roi_term_display["ROI medio"] = roi_term_display["ROI medio"].map("{:.2%}".format)
        roi_term_display["Préstamos"] = roi_term_display["Préstamos"].map("{:,}".format)
        st.dataframe(roi_term_display, use_container_width=True, hide_index=True)
        st.markdown(
            "Los préstamos a **60 meses** tienen consistentemente mayor default rate y menor ROI. "
            "El plazo largo amplifica la exposición temporal al riesgo, lo que se alinea con los "
            "hallazgos del análisis de supervivencia (mayor PD lifetime)."
        )

st.markdown(
    """
Como mensaje de cierre, esta página muestra por qué la optimización no es una capa "extra", sino el punto donde los outputs
de ML y Conformal se convierten en una política accionable. Aquí se hace visible el costo real de la prudencia y se ofrece un
marco cuantitativo para discutir apetito de riesgo con transparencia económica.
"""
)

next_page_teaser(
    "Provisiones IFRS9",
    "Traducción de riesgo y escenarios a provisiones regulatorias.",
    "pages/ifrs9_provisions.py",
)
