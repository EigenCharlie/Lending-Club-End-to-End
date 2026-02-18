"""Historia de datos del portafolio Lending Club."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.components.audience_toggle import audience_selector
from streamlit_app.components.narrative import narrative_block, next_page_teaser
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import (
    format_number,
    format_pct,
    get_notebook_image_path,
    load_json,
    load_parquet,
    query_duckdb,
)

st.title("📊 Historia de Datos")
st.caption(
    "Radiografía del dataset: composición, riesgo por segmento y dinámica temporal. "
    "Analiza el split de entrenamiento (1.35M préstamos, 2007-2017) del total de 1.86M resueltos."
)
audience = audience_selector()

narrative_block(
    audience,
    general="Esta página presenta el dataset Lending Club: qué contiene, cómo se distribuyen los "
    "préstamos y qué patrones de riesgo existen. Es la base para entender todo lo que sigue.",
    business="El dataset cubre 2007-2020 con múltiples regímenes de mercado. Los patrones aquí "
    "documentados justifican cada decisión metodológica posterior: qué variables usar, "
    "cómo segmentar, y por qué validar temporalmente.",
    technical="EDA fundacional: no es decorativo. Documenta heterogeneidad de riesgo, no estacionariedad, "
    "composición de producto y señales de data leakage. Justifica OOT split, feature selection y "
    "controles de fuga de información.",
)

eda = load_json("eda_summary")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Préstamos", format_number(eda.get("n_loans", 0)))
with col2:
    st.metric("Default global", format_pct(eda.get("default_rate", 0)))
with col3:
    st.metric("Monto medio", format_number(eda.get("loan_amnt", {}).get("mean", 0), prefix="$"))
with col4:
    st.metric("Ingreso mediano", format_number(eda.get("annual_inc", {}).get("median", 0), prefix="$"))

st.info(
    "**Nota sobre las cifras:** El CSV original tiene 2.93M registros. Tras eliminar ~1.06M préstamos "
    "sin resolución final (Current, Late, Grace Period, Issued), quedan **1.86M préstamos resueltos**. "
    "Esta página analiza el **split de entrenamiento** (1.35M, 2007-2017). Los splits de calibración "
    "(238K) y test OOT (277K, 2018-2020) se reservan para validación."
)

st.markdown(
    """
**Lending Club** fue la plataforma de préstamos peer-to-peer más grande de Estados Unidos.
El dataset cubre todo el ciclo crediticio 2007-2020, capturando múltiples regímenes de mercado:
crisis subprime (2007-09), recuperación post-crisis, expansión fintech, y el inicio de COVID-19.
"""
)

# ── Conoce el Dataset ──
st.subheader("Conoce el Dataset: variables más importantes")
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
dict_path = DATA_DIR / "dataset_dictionary.json"
if dict_path.exists():
    with open(dict_path) as f:
        var_dict = json.load(f)
    with st.expander("Diccionario de variables (clic para expandir)", expanded=False):
        df_dict = pd.DataFrame(var_dict)[["variable", "nombre", "descripcion", "tipo", "rango", "relevancia"]]
        df_dict.columns = ["Variable", "Nombre", "Descripción", "Tipo", "Rango típico", "Relevancia para riesgo"]
        st.dataframe(df_dict, use_container_width=True, hide_index=True, height=400)

# ── Distributions ──
st.subheader("Distribuciones de variables clave")
loan_master_full = load_parquet("loan_master")

col_d1, col_d2 = st.columns(2)
with col_d1:
    fig = px.histogram(
        loan_master_full, x="loan_amnt", nbins=50,
        title="Distribución del monto del préstamo",
        labels={"loan_amnt": "Monto ($)", "count": "Frecuencia"},
        color_discrete_sequence=["#0B5ED7"],
    )
    fig.add_vline(x=loan_master_full["loan_amnt"].median(), line_dash="dash", line_color="#D93025",
                  annotation_text=f"Mediana: ${loan_master_full['loan_amnt'].median():,.0f}")
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Distribución sesgada a la derecha con picos en múltiplos redondos ($5K, $10K, $15K, $20K, $25K).")

with col_d2:
    inc_clip = loan_master_full["annual_inc"].clip(upper=200000)
    fig = px.histogram(
        inc_clip, x="annual_inc", nbins=50,
        title="Distribución del ingreso anual (hasta $200K)",
        labels={"annual_inc": "Ingreso anual ($)", "count": "Frecuencia"},
        color_discrete_sequence=["#0F9D58"],
    )
    fig.add_vline(x=loan_master_full["annual_inc"].median(), line_dash="dash", line_color="#D93025",
                  annotation_text=f"Mediana: ${loan_master_full['annual_inc'].median():,.0f}")
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Concentrado entre $30K-$80K con cola larga. Los ingresos extremos son autorreportados.")

col_d3, col_d4 = st.columns(2)
with col_d3:
    dti_clip = loan_master_full["dti"].clip(upper=60)
    fig = px.histogram(
        dti_clip, x="dti", nbins=50,
        title="Distribución del DTI (Debt-to-Income)",
        labels={"dti": "DTI", "count": "Frecuencia"},
        color_discrete_sequence=["#F5A623"],
    )
    fig.add_vline(x=loan_master_full["dti"].median(), line_dash="dash", line_color="#D93025",
                  annotation_text=f"Mediana: {loan_master_full['dti'].median():.1f}")
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Distribución aproximadamente normal con mediana ~15. DTI >30 señala sobreendeudamiento.")

with col_d4:
    fig = px.histogram(
        loan_master_full, x="int_rate", nbins=50,
        title="Distribución de la tasa de interés",
        labels={"int_rate": "Tasa de interés (%)", "count": "Frecuencia"},
        color_discrete_sequence=["#D93025"],
    )
    fig.add_vline(x=loan_master_full["int_rate"].median(), line_dash="dash", line_color="#0B5ED7",
                  annotation_text=f"Mediana: {loan_master_full['int_rate'].median():.1f}%")
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Mayor densidad entre 8-16%. La tasa refleja el riesgo percibido por el prestamista.")

# ── PD vs Interest Rate ──
st.subheader("Relación tasa de interés vs probabilidad de default")
rate_bins = loan_master_full.assign(
    rate_bin=pd.cut(loan_master_full["int_rate"], bins=20)
).groupby("rate_bin", observed=True).agg(
    tasa_media=("int_rate", "mean"),
    pd_observada=("default_flag", "mean"),
    n=("default_flag", "count"),
).reset_index()

fig = px.scatter(
    rate_bins, x="tasa_media", y="pd_observada", size="n",
    title="PD observada por bucket de tasa de interés",
    labels={"tasa_media": "Tasa de interés media (%)", "pd_observada": "PD observada", "n": "Préstamos"},
    trendline="ols",
    color_discrete_sequence=["#0B5ED7"],
)
fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=380)
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Relación casi-lineal: tasas de 5-7% → PD ~2%; tasas de 20-30% → PD ~24%. "
    "Lending Club cobra más a quienes tienen mayor probabilidad de incumplir."
)

# ── Data Quality Story ──
with st.expander("Calidad de datos: de 142 a 60 variables"):
    st.markdown(
        """
| Paso | Variables removidas | Razón |
|------|:-------------------:|-------|
| Leakage (post-loan) | 15 | total_pymnt, recoveries, out_prncp, etc. — solo existen post-default |
| Alta nulidad (>80%) | 43 | Datos insuficientes para aprender patrones |
| IDs y constantes | 8 | id, member_id, policy_code (valor único) |
| Texto libre | 3 | emp_title, title, desc — requeriría NLP |
| Redundantes/reemplazadas | 16+ | Timestamps crudos, sub_grade (→ grade_woe), etc. |
| **Feature engineering** | **+15** | Ratios, logs, WOE, buckets, flags |
| **Features finales** | **= 60** | CatBoost features seleccionadas por IV y SHAP |
"""
    )

st.subheader("1) Gradiente de riesgo por calificación")
grade_default = pd.DataFrame(
    {
        "grade": list(eda.get("default_rate_by_grade", {}).keys()),
        "default_rate": list(eda.get("default_rate_by_grade", {}).values()),
        "n_loans": [eda.get("loan_count_by_grade", {}).get(g, 0) for g in eda.get("default_rate_by_grade", {})],
    }
)

if not grade_default.empty:
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(
            grade_default,
            x="grade",
            y="default_rate",
            title="Tasa de default por grade",
            labels={"grade": "Grade", "default_rate": "Tasa de default"},
            color="default_rate",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=380, coloraxis_showscale=False)
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Propósito: cuantificar gradiente de riesgo por calificación. Insight: el salto de default entre grades altos y bajos "
            "justifica segmentación de política y pricing."
        )

    with col_b:
        fig = px.pie(
            grade_default,
            names="grade",
            values="n_loans",
            title="Composición del portafolio por grade",
            hole=0.45,
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Propósito: ubicar dónde se concentra volumen del portafolio. Insight: concentración en ciertos grades define impacto "
            "económico potencial de cualquier cambio en política."
        )

st.info(
    "Insight clave: el aumento de riesgo desde A hacia G es pronunciado. Esto justifica políticas "
    "diferenciadas de pricing, provisiones y asignación de capital."
)

st.subheader("2) Evolución temporal: volumen y default")
monthly_df: pd.DataFrame
try:
    monthly_df = query_duckdb(
        """
        SELECT date_trunc('month', issue_d) AS mes,
               count(*) AS n_prestamos,
               avg(default_flag) AS tasa_default
        FROM main_staging.stg_loan_master
        GROUP BY 1
        ORDER BY 1
        """
    )
except Exception:
    loan_master = load_parquet("loan_master")
    loan_master["issue_d"] = pd.to_datetime(loan_master["issue_d"], errors="coerce")
    monthly_df = (
        loan_master.dropna(subset=["issue_d"])
        .assign(mes=lambda d: d["issue_d"].dt.to_period("M").dt.to_timestamp())
        .groupby("mes", as_index=False)
        .agg(
            n_prestamos=("id", "count"),
            tasa_default=("default_flag", "mean"),
        )
    )

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=monthly_df["mes"],
        y=monthly_df["n_prestamos"],
        name="Núm. préstamos",
        marker_color="#4ECDC4",
        opacity=0.55,
    )
)
fig.add_trace(
    go.Scatter(
        x=monthly_df["mes"],
        y=monthly_df["tasa_default"],
        name="Tasa default",
        mode="lines",
        line={"width": 2.5, "color": "#FF6B6B"},
        yaxis="y2",
    )
)
fig.update_layout(
    **PLOTLY_TEMPLATE["layout"],
)
fig.update_layout(
    title="Dinámica mensual del portafolio con contexto macroeconómico",
    height=480,
    yaxis={"title": "Núm. préstamos"},
    yaxis2={"title": "Tasa default", "overlaying": "y", "side": "right", "tickformat": ".1%"},
)
# Add macro context annotations
macro_path = DATA_DIR / "macro_context.json"
if macro_path.exists():
    with open(macro_path) as f:
        macro_events = json.load(f)
    colors_map = {"crisis": "#D93025", "recovery": "#0F9D58", "regulation": "#F5A623", "market": "#0B5ED7"}
    for evt in macro_events:
        fig.add_vline(
            x=evt["date"], line_dash="dot",
            line_color=colors_map.get(evt["type"], "#94A3B8"),
            line_width=1,
        )
        fig.add_annotation(
            x=evt["date"], y=1.05, yref="paper",
            text=evt["event"].split(":")[0][:25], showarrow=False,
            font={"size": 8, "color": colors_map.get(evt["type"], "#94A3B8")},
            textangle=-45,
        )
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Líneas punteadas: eventos macroeconómicos clave. Rojo=crisis, verde=recuperación, "
    "naranja=regulación, azul=mercado. Los períodos de mayor volumen no siempre coinciden con menor default."
)

st.subheader("3) Geografía del riesgo: mapa por estado")

state_agg = load_parquet("state_aggregates")
if not state_agg.empty:
    col_map1, col_map2 = st.columns(2)
    with col_map1:
        fig = px.choropleth(
            state_agg,
            locations="addr_state",
            locationmode="USA-states",
            color="n_loans",
            scope="usa",
            color_continuous_scale="Blues",
            title="Volumen de préstamos por estado",
            labels={"addr_state": "Estado", "n_loans": "Préstamos"},
            hover_data={"addr_state": True, "n_loans": ":,", "avg_loan": ":$,.0f"},
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=400, coloraxis_colorbar={"title": "Préstamos"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "California, Texas, Nueva York y Florida concentran el mayor volumen. "
            "La distribución geográfica refleja densidad poblacional y penetración fintech."
        )

    with col_map2:
        fig = px.choropleth(
            state_agg,
            locations="addr_state",
            locationmode="USA-states",
            color="default_rate",
            scope="usa",
            color_continuous_scale="RdYlGn_r",
            title="Tasa de default por estado",
            labels={"addr_state": "Estado", "default_rate": "Default rate"},
            hover_data={"addr_state": True, "default_rate": ":.1%", "n_loans": ":,"},
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=400, coloraxis_colorbar={"title": "Default", "tickformat": ".0%"})
        st.plotly_chart(fig, use_container_width=True)

        top3_risk = state_agg.nlargest(3, "default_rate")
        top3_safe = state_agg.nsmallest(3, "default_rate")
        st.caption(
            f"Mayor riesgo: {', '.join(top3_risk['addr_state'].tolist())}. "
            f"Menor riesgo: {', '.join(top3_safe['addr_state'].tolist())}. "
            "Las diferencias estatales reflejan condiciones económicas locales y perfil de solicitante."
        )

    st.markdown(
        """
**Insight geográfico:** La tasa de default varía entre ~17% y ~22% según el estado. Estados con mayor
volumen (CA, TX) no son necesariamente los de mayor riesgo. Nueva York y Florida combinan alto volumen
con tasas de default superiores al promedio, lo que los hace segmentos de especial atención para
política de riesgo y provisiones IFRS9 diferenciadas.
"""
    )

st.subheader("4) Mezcla de producto y perfil de solicitante")
loan_master = load_parquet("loan_master")

purpose_top = (
    loan_master.groupby("purpose", as_index=False)
    .agg(n=("id", "count"), tasa_default=("default_flag", "mean"), monto_prom=("loan_amnt", "mean"))
    .sort_values("n", ascending=False)
    .head(12)
)

col_x, col_y = st.columns(2)
with col_x:
    fig = px.bar(
        purpose_top.sort_values("n"),
        x="n",
        y="purpose",
        orientation="h",
        title="Top propósitos por volumen",
        labels={"n": "Préstamos", "purpose": "Propósito"},
        color="tasa_default",
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=430, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Propósito: comparar mezcla de propósito y riesgo. Insight: no todos los usos de crédito aportan el mismo perfil de default."
    )

with col_y:
    sample = loan_master.sample(min(25000, len(loan_master)), random_state=42)
    fig = px.scatter(
        sample,
        x="int_rate",
        y="dti",
        color="default_flag",
        opacity=0.35,
        title="Relación tasa de interés vs DTI",
        labels={"int_rate": "Tasa interés (%)", "dti": "DTI", "default_flag": "Default"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=430)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Propósito: observar interacción tasa vs carga financiera (DTI). Insight: regiones con tasa alta y DTI alta muestran "
        "mayor densidad de incumplimiento."
    )

st.markdown(
    """
**Conexión con el resto del pipeline:**
- Estos patrones alimentan el diseño de variables para ML.
- La estacionalidad de defaults conecta con el bloque de series de tiempo.
- La heterogeneidad entre segmentos motiva análisis causal y decisiones robustas.
"""
)

with st.expander("Vista rápida de datos (muestra)"):
    cols = ["id", "issue_d", "grade", "loan_amnt", "int_rate", "dti", "annual_inc", "purpose", "default_flag"]
    st.dataframe(loan_master[cols].sample(min(150, len(loan_master)), random_state=7), use_container_width=True)

col_img1, col_img2 = st.columns(2)
with col_img1:
    img = get_notebook_image_path("01_eda_lending_club", "cell_024_out_00.png")
    if img.exists():
        st.image(
            str(img),
            caption="Notebook 01: default por bucket de tasa y volumen de originación.",
            use_container_width=True,
        )
with col_img2:
    img = get_notebook_image_path("01_eda_lending_club", "cell_025_out_00.png")
    if img.exists():
        st.image(
            str(img),
            caption="Notebook 01: gradiente de default por grade y plazo.",
            use_container_width=True,
        )

st.markdown(
    """
La historia que deja esta sección es clara: el riesgo en Lending Club está altamente estratificado por calidad crediticia,
precio y composición de producto, y además cambia en el tiempo. Por eso el proyecto adopta validación out-of-time, ingeniería
de variables orientada a interpretabilidad y una cadena de decisiones que no depende de un único indicador agregado.
"""
)

next_page_teaser(
    "Laboratorio de Modelos",
    "Entrenamiento, comparación y calibración del modelo PD que sustenta todo el flujo.",
    "pages/model_laboratory.py",
)
