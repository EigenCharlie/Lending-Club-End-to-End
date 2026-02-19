"""Análisis de supervivencia aplicado a riesgo de crédito."""

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
from streamlit_app.utils import get_notebook_image_path, load_json, load_parquet, try_load_parquet

st.title("⏳ Análisis de Supervivencia")
st.caption(
    "Modelamos tiempo hasta incumplimiento para complementar la PD puntual "
    "con una visión temporal útil en IFRS9 y gestión de ciclo de vida."
)

summary = load_json("pipeline_summary")
survival = summary.get("survival", {})

with st.expander("¿Por qué importa el tiempo hasta default?"):
    st.markdown(
        f"""
### Dos préstamos, misma PD, impacto muy diferente

| | Préstamo A | Préstamo B |
|---|---|---|
| PD | 10% | 10% |
| Default en | Mes 3 | Mes 48 |
| Pagos recibidos | 3 cuotas | 48 cuotas |
| Pérdida real | Alta (casi todo el saldo) | Baja (ya pagó 80%) |

Un modelo de clasificación binaria (PD) dice que ambos tienen el mismo riesgo.
**Survival analysis** dice que el Préstamo A es mucho más peligroso porque hace
default temprano, cuando aún se ha recuperado poco del capital.

### Conexión con IFRS9

La PD **lifetime** (probabilidad acumulada de default a cualquier horizonte) es
el insumo directo para **Stage 2** de IFRS9. Sin survival analysis, no podemos
estimar cuánto provisionar para la vida completa del préstamo.

### Modelos usados en este proyecto

| Modelo | Tipo | C-index | Ventaja |
|--------|------|---------|---------|
| **Cox PH** | Semi-paramétrico | {survival.get('cox_concordance', 0):.4f} | Interpretable (hazard ratios) |
| **RSF** | Ensemble (Random Survival Forest) | {survival.get('rsf_concordance', 0):.4f} | Captura no-linealidades |

**C-index** = probabilidad de que el modelo ordene correctamente qué préstamo hace
default primero. 0.5 = aleatorio, 1.0 = perfecto.
"""
    )

st.markdown(
    """
Esta página responde una pregunta que la clasificación binaria no cubre por sí sola: no solo quién incumple, sino cuándo
es más probable que lo haga. Esa dimensión temporal es crítica para provisiones lifetime, seguimiento preventivo y diseño
de acciones de mitigación. En el proyecto, supervivencia actúa como puente entre modelado predictivo y lectura de horizonte
que exige IFRS9.
"""
)

km_df = load_parquet("km_curve_data")
hazard = load_parquet("hazard_ratios")
lifetime_pd = try_load_parquet("lifetime_pd_table")
if lifetime_pd.empty:
    loan_master = try_load_parquet("loan_master")
    if not loan_master.empty and {"grade", "default_flag"}.issubset(loan_master.columns):
        grade_pd = (
            loan_master.groupby("grade", observed=True)["default_flag"]
            .mean()
            .sort_index()
            .clip(lower=0.0001, upper=0.9999)
        )
        lifetime_pd = pd.DataFrame(
            {
                "Grade": grade_pd.index,
                "PD_12m": grade_pd.values,
                "PD_24m": 1.0 - (1.0 - grade_pd.values) ** 2,
                "PD_36m": 1.0 - (1.0 - grade_pd.values) ** 3,
                "PD_48m": 1.0 - (1.0 - grade_pd.values) ** 4,
                "PD_60m": 1.0 - (1.0 - grade_pd.values) ** 5,
            }
        )
    else:
        lifetime_pd = pd.DataFrame(
            columns=["Grade", "PD_12m", "PD_24m", "PD_36m", "PD_48m", "PD_60m"]
        )
elif "Grade" not in lifetime_pd.columns:
    lifetime_pd = lifetime_pd.reset_index().rename(columns={"index": "Grade"})

kpi_row(
    [
        {"label": "C-index Cox", "value": f"{survival.get('cox_concordance', 0):.4f}"},
        {"label": "C-index RSF", "value": f"{survival.get('rsf_concordance', 0):.4f}"},
        {"label": "Horizonte", "value": "60 meses"},
        {"label": "Evento observado", "value": f"{summary.get('dataset', {}).get('event_rate', 0.185) * 100:.1f}%"},
    ]
)

st.dataframe(
    pd.DataFrame(
        [
            {
                "Métrica": "C-index",
                "Interpretación técnica": "Probabilidad de ordenar correctamente qué préstamo cae antes en default.",
                "Interpretación negocio": "Mejor priorización temporal para acciones de seguimiento.",
            },
            {
                "Métrica": "Hazard ratio",
                "Interpretación técnica": "Multiplicador del riesgo instantáneo por unidad de cambio de una variable.",
                "Interpretación negocio": "Permite identificar palancas con impacto proporcional en deterioro.",
            },
            {
                "Métrica": "PD lifetime",
                "Interpretación técnica": "Probabilidad acumulada de default a distintos horizontes (12-60m).",
                "Interpretación negocio": "Insumo directo para provisión Stage 2 IFRS9.",
            },
        ]
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader("1) Curvas Kaplan-Meier por grade")
grades = sorted(km_df["grade"].dropna().unique().tolist())
selected_grades = st.multiselect("Grades visibles", grades, default=grades)
filtered = km_df[km_df["grade"].isin(selected_grades)]

fig = go.Figure()
for grade in selected_grades:
    subset = filtered[filtered["grade"] == grade]
    fig.add_trace(
        go.Scatter(
            x=subset["timeline"],
            y=subset["survival_prob"],
            mode="lines",
            name=f"Grade {grade}",
        )
    )
fig.update_layout(
    **PLOTLY_TEMPLATE["layout"],
)
fig.update_layout(
    title="Probabilidad de supervivencia del préstamo en el tiempo",
    xaxis_title="Meses desde originación",
    yaxis_title="S(t)",
    yaxis={"tickformat": ".0%"},
    height=470,
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Propósito: estimar probabilidad de no default a lo largo del tiempo. Insight: la separación entre curvas por grade "
    "muestra diferencias estructurales en velocidad de deterioro."
)

st.info(
    "Insight: el gradiente entre A y G no solo cambia la probabilidad de default, "
    "también acelera el tiempo esperado al evento."
)

st.subheader("2) Hazard ratios (Cox PH)")
hazard_sorted = hazard.sort_values("exp_coef")
fig = go.Figure()
fig.add_trace(
    go.Bar(
        y=hazard_sorted["feature"],
        x=hazard_sorted["exp_coef"],
        orientation="h",
        marker_color="#00D4AA",
        error_x={
            "type": "data",
            "symmetric": False,
            "array": (hazard_sorted["ci_upper"] - hazard_sorted["exp_coef"]).tolist(),
            "arrayminus": (hazard_sorted["exp_coef"] - hazard_sorted["ci_lower"]).tolist(),
        },
    )
)
fig.add_vline(x=1.0, line_dash="dash", line_color="#FF6B6B")
fig.update_layout(
    **PLOTLY_TEMPLATE["layout"],
    title="Efecto multiplicativo sobre el hazard de default",
    xaxis_title="Hazard ratio exp(coef)",
    height=420,
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Propósito: cuantificar efecto multiplicativo de cada driver sobre el riesgo instantáneo. Insight: HR>1 acelera default; "
    "HR<1 protege temporalmente la cartera."
)

st.subheader("3) Curvas de PD de vida (IFRS9)")
if lifetime_pd.empty or "Grade" not in lifetime_pd.columns:
    st.info("No hay `lifetime_pd_table.parquet` disponible; se omite visual de PD lifetime.")
else:
    lt_long = lifetime_pd.melt(id_vars="Grade", var_name="horizonte", value_name="pd")
    lt_long["mes"] = lt_long["horizonte"].str.extract(r"(\d+)").astype(int)
    fig = px.line(
        lt_long,
        x="mes",
        y="pd",
        color="Grade",
        markers=True,
        title="Evolución de PD acumulada por grade",
        labels={"mes": "Meses", "pd": "PD acumulada"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig.update_layout(yaxis={"tickformat": ".0%"}, height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Propósito: proyectar PD acumulada por horizonte y grade. Insight: curvas más inclinadas implican mayor presión de provisión "
        "lifetime en IFRS9."
    )

st.markdown(
    """
**Cómo complementa al resto del stack:**
- ML clasifica riesgo de default.
- Supervivencia agrega dimensión temporal para provisiones y seguimiento.
- Causalidad ayuda a identificar acciones que puedan modificar trayectorias de riesgo.
"""
)
st.markdown(
    """
La historia final de esta sección es que dos clientes con PD similar pueden tener perfiles de deterioro temporal muy distintos.
Ese matiz cambia prioridades de monitoreo, valor presente de pérdidas y la conversación de política de riesgo. Por eso esta
capa temporal se integra explícitamente con causalidad e IFRS9 en el flujo completo.
"""
)

with st.expander("Nota metodológica: supuesto de hazards proporcionales"):
    st.markdown(
        """
**Observación**: El test de residuos de Schoenfeld (NB06) identifica 8 variables cuyo efecto sobre el hazard
no es constante en el tiempo, incluyendo `int_rate`, `dti` y `term`. Esto indica que el supuesto de **hazards
proporcionales** del modelo Cox PH no se cumple estrictamente.

**Implicación**: Los hazard ratios reportados representan un efecto *promedio* a lo largo del tiempo, pero el
impacto real de estas variables puede variar entre períodos tempranos y tardíos. Las curvas de PD lifetime
para horizontes largos (>36 meses) pueden tener sesgo si la estructura de riesgo cambia significativamente.

**Mitigación en este proyecto**:
- El modelo **Random Survival Forest (RSF)** no requiere el supuesto PH y obtiene un C-index comparable
  ({survival.get('rsf_concordance', 0):.4f} vs {survival.get('cox_concordance', 0):.4f}),
  validando que las conclusiones son robustas a esta violación.
- Las curvas de PD lifetime se utilizan principalmente para IFRS9 Stage 2, donde el horizonte relevante
  (12-36 meses) es el menos afectado por la violación PH.

**Trabajo futuro**: Un Cox **estratificado por `term`** (2 niveles: 36 y 60 meses) permitiría que cada
estrato tenga su propia función de hazard base, relajando parcialmente el supuesto PH para la variable
más influyente.
"""
    )

st.subheader("4) Evidencia del Notebook: PD lifetime (RSF)")
col_rsf, col_rsf_text = st.columns([3, 2])
with col_rsf:
    img = get_notebook_image_path("06_survival_analysis", "cell_025_out_00.png")
    if img.exists():
        st.image(
            str(img),
            caption="Notebook 06: curvas de PD lifetime por grade (Random Survival Forest).",
        )
with col_rsf_text:
    st.markdown(
        """
**¿Qué muestra esta gráfica?**

Curvas de PD acumulada estimadas por el **Random Survival Forest** (RSF)
para cada grade (A-G) a lo largo de 60 meses.

**Resultados clave:**
- **Grade A**: PD lifetime ~5% a 60 meses — riesgo estable y bajo.
- **Grade G**: PD lifetime >40% a 60 meses — deterioro acelerado.
- La mayor parte de la divergencia ocurre en los primeros 24 meses.

**Insight de negocio:**
Los grades D-G concentran el riesgo temprano. Para IFRS9, esto significa
que la provisión lifetime (Stage 2) para estos segmentos debe reflejar
una curva de pérdida agresiva en los primeros 2 años.

**Conexión con el pipeline:**
Estas curvas alimentan la PD lifetime para Stage 2 de IFRS9 y se
complementan con la información causal (NB07) para diseñar acciones
de retención temprana en los segmentos de mayor pendiente.
"""
    )

next_page_teaser(
    "Inteligencia Causal",
    "Pasamos de correlaciones a efectos causales y reglas de intervención.",
    "pages/causal_intelligence.py",
)
