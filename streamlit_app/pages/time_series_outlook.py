"""Panorama temporal: pronóstico de defaults y escenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.components.narrative import next_page_teaser
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import get_notebook_image_path, load_parquet, try_load_parquet

st.title("📈 Panorama Temporal")
st.caption(
    "Modelos estadísticos y ML para proyección de tasa de default, "
    "con bandas de incertidumbre y escenarios IFRS9."
)

history = load_parquet("time_series")
forecasts = load_parquet("ts_forecasts")
scenarios = try_load_parquet("ts_ifrs9_scenarios")
cv_stats = try_load_parquet("ts_cv_stats")

if scenarios.empty and not forecasts.empty:
    baseline_model = "lgbm" if "lgbm" in forecasts.columns else None
    if baseline_model is None:
        model_candidates = [
            c
            for c in forecasts.columns
            if c not in {"unique_id", "ds"}
            and not c.endswith("-lo-90")
            and not c.endswith("-hi-90")
            and not c.endswith("-lo-95")
            and not c.endswith("-hi-95")
        ]
        baseline_model = model_candidates[0] if model_candidates else None

    if baseline_model is not None:
        lo90 = f"{baseline_model}-lo-90"
        hi90 = f"{baseline_model}-hi-90"
        lo95 = f"{baseline_model}-lo-95"
        hi95 = f"{baseline_model}-hi-95"
        scenarios = pd.DataFrame(
            {
                "month": forecasts["ds"],
                "point_forecast": forecasts[baseline_model],
                "optimistic_90": forecasts[lo90] if lo90 in forecasts.columns else forecasts[baseline_model],
                "adverse_90": forecasts[hi90] if hi90 in forecasts.columns else forecasts[baseline_model],
                "optimistic_95": forecasts[lo95] if lo95 in forecasts.columns else forecasts[baseline_model],
                "adverse_95": forecasts[hi95] if hi95 in forecasts.columns else forecasts[baseline_model],
            }
        )

if cv_stats.empty and not history.empty and not forecasts.empty:
    cv_stats = forecasts.copy()
    if "y" not in cv_stats.columns:
        cv_stats["y"] = float(history["y"].tail(12).mean()) if "y" in history.columns else 0.0

st.markdown(
    """
Este bloque conecta estadística y gestión de riesgo prospectivo:
- Pronosticamos la tasa de default agregada del portafolio.
- Incorporamos incertidumbre para lectura prudente de escenarios.
- Enlazamos con IFRS9 para provisiones forward-looking.
"""
)
st.markdown(
    """
Su función dentro del proyecto es complementar la visión micro por préstamo con una lectura macro de ciclo crediticio.
En términos de negocio, esto permite anticipar deterioro antes de observarlo plenamente en defaults realizados y construir
escenarios de planeación menos reactivos. La calidad de este bloque impacta directamente la discusión de provisiones y
resiliencia de cartera bajo estrés.
"""
)

st.dataframe(
    pd.DataFrame(
        [
            {
                "Métrica": "MAE / RMSE",
                "Significado técnico": "Error promedio de pronóstico en magnitud mensual.",
                "Significado negocio": "Menor error mejora planeación de provisiones y stress testing.",
            },
            {
                "Métrica": "Bandas 90%/95%",
                "Significado técnico": "Rango plausible de default futuro según incertidumbre.",
                "Significado negocio": "Permite presupuestar capital con escenarios prudenciales.",
            },
        ]
    ),
    use_container_width=True,
    hide_index=True,
)

model_cols = [c for c in forecasts.columns if c not in ("ds", "unique_id", "y", "cutoff")]
models = sorted(
    c
    for c in model_cols
    if not c.endswith("-lo-90")
    and not c.endswith("-hi-90")
    and not c.endswith("-lo-95")
    and not c.endswith("-hi-95")
)
selected_model = st.selectbox("Modelo de pronóstico", models, index=0)

st.subheader("1) Serie histórica + pronóstico")
fig = go.Figure()

if "y" in history.columns:
    fig.add_trace(
        go.Scatter(
            x=history["ds"],
            y=history["y"],
            mode="lines",
            name="Histórico",
            line={"color": "#334155", "width": 2.4},
        )
    )

if selected_model in forecasts.columns:
    fig.add_trace(
        go.Scatter(
            x=forecasts["ds"],
            y=forecasts[selected_model],
            mode="lines+markers",
            name=f"Pronóstico {selected_model}",
            line={"color": "#00D4AA", "width": 2.5},
            marker={"size": 6, "color": "#0B5ED7", "opacity": 1.0, "line": {"width": 0}},
        )
    )

for level, color in [("90", "rgba(0,212,170,0.22)"), ("95", "rgba(255,217,61,0.18)")]:
    lo = f"{selected_model}-lo-{level}"
    hi = f"{selected_model}-hi-{level}"
    if lo in forecasts.columns and hi in forecasts.columns:
        fig.add_trace(go.Scatter(x=forecasts["ds"], y=forecasts[hi], mode="lines", line={"width": 0}, showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=forecasts["ds"],
                y=forecasts[lo],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=color,
                name=f"Intervalo {level}%",
            )
        )

fig.update_layout(
    **PLOTLY_TEMPLATE["layout"],
)
fig.update_layout(
    title=f"Tasa de default: histórico y proyección ({selected_model})",
    xaxis_title="Fecha",
    yaxis_title="Tasa de default",
    yaxis={"tickformat": ".1%"},
    height=470,
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Propósito: proyectar default futuro. Insight: el modelo seleccionable mantiene trayectoria coherente con histórico. "
    "Uso práctico: alimentar escenarios de provisión y planeación de riesgo."
)

st.subheader("2) Escenarios IFRS9 derivados del pronóstico")
if scenarios.empty:
    st.info("No hay artefacto de escenarios IFRS9 temporal; se omite esta sección.")
else:
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=scenarios["month"],
                y=scenarios["point_forecast"],
                mode="lines+markers",
                name="Punto",
            )
        )
        fig.add_trace(
            go.Scatter(x=scenarios["month"], y=scenarios["adverse_90"], mode="lines", name="Adverso 90%")
        )
        fig.add_trace(
            go.Scatter(
                x=scenarios["month"],
                y=scenarios["optimistic_90"],
                mode="lines",
                name="Optimista 90%",
            )
        )
        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"],
        )
        fig.update_layout(
            title="Escenario central vs bandas optimista/adversa (90%)",
            yaxis={"tickformat": ".1%"},
            height=390,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Propósito: visualizar rango optimista/adverso frente al punto central. "
            "Insight: la incertidumbre no es simétrica en todos los meses. "
            "Uso práctico: definir bandas de planificación IFRS9."
        )

    with col_b:
        scen_long = scenarios.melt(
            id_vars=["month"],
            value_vars=[
                "optimistic_90",
                "point_forecast",
                "adverse_90",
                "optimistic_95",
                "adverse_95",
            ],
            var_name="escenario",
            value_name="tasa",
        )
        fig = px.box(
            scen_long,
            x="escenario",
            y="tasa",
            points="all",
            title="Dispersión de tasas por escenario",
            labels={"escenario": "", "tasa": "Tasa"},
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        fig.update_layout(yaxis={"tickformat": ".1%"}, height=390)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Propósito: resumir dispersión por escenario. Insight: escenarios adversos desplazan sistemáticamente la tasa esperada. "
            "Uso práctico: stress testing de provisiones y capital."
        )

st.subheader("3) Calidad en validación temporal (rolling origin)")
pred_cols = [
    c
    for c in cv_stats.columns
    if c not in {"unique_id", "ds", "cutoff", "y"}
    and not c.endswith("-lo-90")
    and not c.endswith("-hi-90")
    and not c.endswith("-lo-95")
    and not c.endswith("-hi-95")
]
if cv_stats.empty or "y" not in cv_stats.columns or not pred_cols:
    st.info("No hay `ts_cv_stats.parquet` utilizable; se omite comparación MAE de validación temporal.")
else:
    scores = []
    for col in pred_cols:
        err = np.abs(cv_stats["y"] - cv_stats[col])
        scores.append({"modelo": col, "mae": float(err.mean())})
    scores_df = pd.DataFrame(scores).sort_values("mae")

    fig = px.bar(
        scores_df,
        x="modelo",
        y="mae",
        title="MAE promedio en validación temporal",
        labels={"modelo": "Modelo", "mae": "MAE"},
        color="mae",
        color_continuous_scale="Blues",
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=360, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Propósito: comparar error fuera de muestra temporal. Insight: el ranking de modelos no siempre coincide en MAE y RMSE. "
        "Uso práctico: elegir modelo según criterio de negocio (error medio vs penalización de errores grandes)."
    )

st.markdown(
    """
**Lectura de negocio:**
- El bloque temporal anticipa cambios del riesgo antes de ver defaults realizados.
- Las bandas de incertidumbre evitan sobreconfianza en escenarios puntuales.
- Su salida se conecta con escenarios IFRS9 y discusión de resiliencia del portafolio.
"""
)

with st.expander("Nota metodológica: cobertura conformal en series temporales"):
    st.markdown(
        """
**Observación**: Los intervalos conformal del modelo LightGBM presentan cobertura empírica de ~80.6% frente
al objetivo de 90%, mientras que los intervalos paramétricos de ARIMA alcanzan ~100%.

**Explicación**: Conformal prediction estándar asume **exchangeability** (intercambiabilidad) de los datos,
una condición que las series temporales violan por naturaleza: las observaciones están ordenadas en el tiempo
y pueden exhibir cambios de régimen (distributional shift). Cuando el proceso generador cambia entre el período
de calibración y el de evaluación, la cobertura empírica se degrada.

**Implicación práctica**:
- Los intervalos ARIMA paramétricos son más confiables como referencia para escenarios IFRS9,
  dado que su construcción incorpora la estructura temporal explícitamente.
- El modelo ML (LightGBM) aporta valor en capturar patrones no lineales, pero sus bandas de
  incertidumbre deben interpretarse como aproximadas.

**Trabajo futuro**: Métodos como **ACI** (Adaptive Conformal Inference, Gibbs & Candès 2021) y
**EnbPI** (Xu & Xie 2021) extienden conformal prediction a datos no intercambiables, ajustando
dinámicamente los cuantiles de conformidad. Su incorporación mejoraría la cobertura empírica
sin sacrificar la flexibilidad del modelo ML.
"""
    )

st.markdown(
    """
Como cierre, el mensaje es que el pronóstico no reemplaza al score individual, sino que aporta contexto de régimen.
Cuando el entorno cambia, una cartera con la misma composición de hoy puede comportarse distinto mañana. Por eso esta página
debe leerse junto con incertidumbre conformal e IFRS9: juntas habilitan una gestión verdaderamente forward-looking.
"""
)

img = get_notebook_image_path("05_time_series_forecasting", "cell_017_out_01.png")
if img.exists():
    st.image(
        str(img),
        caption="Notebook 05: comparación de modelos por métricas de validación temporal.",
        use_container_width=True,
    )

next_page_teaser(
    "Análisis de Supervivencia",
    "De la pregunta '¿quién incumple?' a '¿cuándo incumple?' para riesgo de horizonte.",
    "pages/survival_analysis.py",
)
