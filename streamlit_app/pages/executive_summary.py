"""Resumen ejecutivo del proyecto integral de riesgo de credito."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
from streamlit_flow.layouts import ManualLayout
from streamlit_flow.state import StreamlitFlowState

from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import next_page_teaser
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import format_number, format_pct, load_json


def _first_valid(*values: float | int | None) -> float:
    for value in values:
        if value is not None and value != 0:
            return float(value)
    return 0.0


_ARCH_NODES = [
    {"id": "raw", "label": "Lending Club Raw", "layer": "data", "detail": "2.93M filas · 142 cols", "icon": "\U0001F4E6"},
    {"id": "clean", "label": "Limpieza + QA", "layer": "data", "detail": "1.86M filas · 110 cols", "icon": "\U0001F9F9"},
    {"id": "fe", "label": "Feature Engineering", "layer": "data", "detail": "WOE/IV · ratios · buckets", "icon": "\u2699\uFE0F"},
    {"id": "pd", "label": "PD Model", "layer": "model", "detail": "CatBoost + Platt · AUC OOT", "icon": "\U0001F3AF"},
    {"id": "conformal", "label": "Conformal Mondrian", "layer": "model", "detail": "Intervalos PD · Coverage 90/95", "icon": "\U0001F4D0"},
    {"id": "causal", "label": "Causalidad (DML/CATE)", "layer": "model", "detail": "+1pp tasa \u2192 +0.787pp default", "icon": "\U0001F9EC"},
    {"id": "survival", "label": "Forecast + Survival", "layer": "model", "detail": "Cox/RSF concordance", "icon": "\u23F3"},
    {"id": "optim", "label": "Optimizaci\u00f3n Robusta", "layer": "decision", "detail": "Pyomo + HiGHS", "icon": "\U0001F4BC"},
    {"id": "ifrs9", "label": "IFRS9 (Stage + ECL)", "layer": "decision", "detail": "4 escenarios · PD\u00d7LGD", "icon": "\U0001F3E6"},
    {"id": "governance", "label": "Gobernanza", "layer": "decision", "detail": "Drift · Fairness · Policy", "icon": "\U0001F6E1\uFE0F"},
    {"id": "streamlit", "label": "Libro Din\u00e1mico Streamlit", "layer": "output", "detail": "17 p\u00e1ginas interactivas", "icon": "\U0001F4CA"},
]
_ARCH_EDGES = [
    {"source": "raw", "target": "clean", "desc": "Filtro resueltos"},
    {"source": "clean", "target": "fe", "desc": "WOE/IV"},
    {"source": "fe", "target": "pd", "desc": "60 features"},
    {"source": "pd", "target": "conformal", "desc": "PD calibrada"},
    {"source": "conformal", "target": "optim", "desc": "Uncertainty sets"},
    {"source": "causal", "target": "optim", "desc": "CATE \u2192 pol\u00edticas"},
    {"source": "survival", "target": "ifrs9", "desc": "Lifetime PD"},
    {"source": "pd", "target": "governance", "desc": "M\u00e9tricas"},
    {"source": "conformal", "target": "governance", "desc": "Cobertura"},
    {"source": "optim", "target": "streamlit", "desc": "Portafolio"},
    {"source": "ifrs9", "target": "streamlit", "desc": "Provisiones"},
    {"source": "governance", "target": "streamlit", "desc": "Policy checks"},
]
_LAYER_COLORS = {
    "data": {"bg": "#F1F5F9", "border": "#64748B", "accent": "#334155"},
    "model": {"bg": "#EFF6FF", "border": "#3B82F6", "accent": "#1D4ED8"},
    "decision": {"bg": "#ECFDF5", "border": "#10B981", "accent": "#047857"},
    "output": {"bg": "#EEF2FF", "border": "#6366F1", "accent": "#4338CA"},
}
_LAYER_BADGES = {
    "data": "\U0001F4BE DATOS",
    "model": "\U0001F9E0 MODELO",
    "decision": "\U0001F4A1 DECISI\u00d3N",
    "output": "\U0001F680 SALIDA",
}
_NODE_MAP = {n["id"]: n for n in _ARCH_NODES}
_EXEC_ARCH_STATE_KEY = "exec_arch_flow_state_manual"
# Topologia fija de la arquitectura para render estable en portada.
_EXEC_ARCH_MANUAL_POSITIONS = {
    "raw": (20, 160),
    "clean": (280, 160),
    "fe": (540, 160),
    "pd": (800, 160),
    "conformal": (1060, 190),
    "causal": (1060, 325),
    "survival": (1060, 460),
    "governance": (1320, 60),
    "optim": (1320, 320),
    "ifrs9": (1320, 460),
    "streamlit": (1580, 320),
}


st.title("🏠 Resumen Ejecutivo")
st.caption(
    "Riesgo de crédito end-to-end: datos, ML, incertidumbre conformal, causalidad, "
    "optimización robusta e IFRS9 sobre un mismo dataset."
)

st.success(
    "**Lending Club**: la plataforma de préstamos peer-to-peer más grande de EE.UU. "
    "2.26 millones de préstamos entre 2007 y 2020. Este proyecto construye un sistema "
    "completo de gestión de riesgo de crédito end-to-end."
)

st.markdown(
    """
### ¿Por qué este proyecto importa?

1. **Decisiones bajo incertidumbre**: No basta con predecir quién hará default — hay que cuantificar
   cuánto confiar en esa predicción y tomar decisiones que funcionen incluso si el modelo se equivoca.

2. **Trazabilidad regulatoria**: IFRS9 y Basilea III exigen que los modelos de riesgo sean auditables,
   calibrados y robustos. Este proyecto implementa gobernanza completa con fairness, drift y policy checks.

3. **Puente ML + Operations Research**: La innovación está en conectar conformal prediction (incertidumbre)
   con optimización robusta (decisión), algo que rara vez se hace en la industria.
"""
)

st.markdown(
    """
Esta portada resume el hilo completo y sirve como punto de entrada. La lectura recomendada es secuencial:
contexto de datos, ingeniería de features, modelo PD, incertidumbre, causalidad, decisión robusta, provisión IFRS9 y gobernanza.
"""
)

summary = load_json("pipeline_summary")
eda = load_json("eda_summary")
comparison = load_json("model_comparison")
policy = load_json("conformal_policy_status", directory="models")
governance = load_json("modeva_governance_status", directory="models")

pipeline = summary.get("pipeline", {})
pd_model = summary.get("pd_model", {})
final_metrics = comparison.get("final_test_metrics", {})

auc = _first_valid(
    final_metrics.get("auc_roc"),
    pipeline.get("pd_auc"),
    pd_model.get("final_auc"),
)
gini = _first_valid(final_metrics.get("gini"), pd_model.get("final_gini"))
brier = _first_valid(final_metrics.get("brier_score"), pd_model.get("final_brier"))
ece = _first_valid(final_metrics.get("ece"), pd_model.get("final_ece"))

st.subheader("Dimensión del problema")
kpi_row(
    [
        {
            "label": "Préstamos (raw)",
            "value": "2.93M",
            "help": "Total de registros en el CSV de Kaggle (incluye préstamos sin resolución: Current, Late, Grace Period)",
        },
        {
            "label": "Préstamos (limpio)",
            "value": "1.86M",
            "help": "Solo préstamos con resolución final (Fully Paid + Charged Off + Default). Se excluyen ~1.06M sin resolución (Current, Late, Grace Period, Issued)",
        },
        {"label": "Tasa de default", "value": format_pct(eda.get("default_rate", 0))},
        {
            "label": "Horizonte temporal",
            "value": "2007-2020",
            "help": "Junio 2007 a Septiembre 2020 (incluye crisis subprime y COVID)",
        },
    ]
)

kpi_row(
    [
        {
            "label": "Variables originales",
            "value": "142",
            "help": "Columnas en el CSV crudo de Kaggle",
        },
        {
            "label": "Tras limpieza",
            "value": "110",
            "help": "Columnas en lending_club_cleaned.parquet",
        },
        {
            "label": "Features modelo PD",
            "value": "60",
            "help": "Features finales para CatBoost (train_fe.parquet)",
        },
        {
            "label": "Train / Cal / Test",
            "value": "1.35M / 238K / 277K",
            "help": "Split out-of-time: train (2007-2017), calibración (2017), test (2018-2020)",
        },
    ]
)

st.subheader("Calidad del modelo predictivo")
kpi_row(
    [
        {"label": "AUC OOT", "value": f"{auc:.4f}"},
        {"label": "Gini", "value": f"{gini:.4f}"},
        {"label": "Brier", "value": f"{brier:.4f}"},
        {"label": "ECE", "value": f"{ece:.4f}"},
    ]
)

st.subheader("Incertidumbre y decisión")
kpi_row(
    [
        {"label": "Cobertura 90%", "value": format_pct(policy.get("coverage_90", 0))},
        {"label": "Cobertura 95%", "value": format_pct(policy.get("coverage_95", 0))},
        {
            "label": "Price of Robustness",
            "value": format_number(pipeline.get("price_of_robustness", 0), prefix="$"),
        },
        {
            "label": "Estado de gobernanza",
            "value": "OK" if governance.get("overall_pass", False) else "Revisión",
            "help": f"{governance.get('checks_passed', 0)}/{governance.get('checks_total', 0)} checks",
        },
    ]
)

st.subheader("Impacto económico: robusto vs no robusto")
impact_df = pd.DataFrame(
    [
        {
            "modo": "No robusto",
            "retorno_neto": pipeline.get("nonrobust_return", 0.0),
            "ecl": pipeline.get("ecl_expected", 0.0),
            "n_aprobados": pipeline.get("nonrobust_funded", 0),
        },
        {
            "modo": "Robusto",
            "retorno_neto": pipeline.get("robust_return", 0.0),
            "ecl": pipeline.get("ecl_conservative", 0.0),
            "n_aprobados": pipeline.get("robust_funded", 0),
        },
    ]
)

col_1, col_2 = st.columns(2)
with col_1:
    fig = px.bar(
        impact_df,
        x="modo",
        y="retorno_neto",
        color="modo",
        title="Retorno neto esperado por política",
        labels={"modo": "", "retorno_neto": "USD"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False, height=360)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Propósito: comparar impacto económico de la política robusta frente a la puntual. "
        "Insight: la política robusta sacrifica parte del upside para estabilizar resultados en escenarios adversos."
    )

with col_2:
    fig = px.bar(
        impact_df,
        x="modo",
        y="ecl",
        color="modo",
        title="Pérdida esperada (ECL) asociada",
        labels={"modo": "", "ecl": "USD"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False, height=360)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Propósito: visualizar costo esperado de crédito por política. "
        "Insight: la decisión de aprobación no cambia solo retorno, también el nivel de provisión implícita."
    )

st.markdown(
    """
**Lectura ejecutiva:**
- El stack no termina en `AUC`: convierte predicción e incertidumbre en decisiones de portafolio auditables.
- Conformal prediction agrega bandas de riesgo con cobertura empírica superior a las metas (90% y 95%).
- Optimización robusta cuantifica explícitamente el costo de proteger el downside.
- IFRS9, causalidad y gobernanza completan una visión integral para riesgo de crédito.
"""
)

st.subheader("Arquitectura funcional")

_flow_nodes = []
for _n in _ARCH_NODES:
    _lc = _LAYER_COLORS[_n["layer"]]
    _ntype = "input" if _n["id"] == "raw" else ("output" if _n["id"] == "streamlit" else "default")
    _flow_nodes.append(
        StreamlitFlowNode(
            id=_n["id"],
            pos=_EXEC_ARCH_MANUAL_POSITIONS[_n["id"]],
            data={"content": f"**{_n['icon']} {_n['label']}**\n\n{_n['detail']}"},
            node_type=_ntype,
            source_position="right",
            target_position="left",
            style={
                "background": _lc["bg"],
                "border": f"2.5px solid {_lc['border']}",
                "borderRadius": "10px",
                "padding": "10px 14px",
                "fontSize": "12px",
                "fontFamily": "Inter, Arial, sans-serif",
                "color": _lc["accent"],
                "width": 195,
                "textAlign": "center",
                "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
            },
        )
    )

_flow_edges = []
for _i, _e in enumerate(_ARCH_EDGES):
    _src_layer = _NODE_MAP[_e["source"]]["layer"]
    _color = _LAYER_COLORS[_src_layer]["border"]
    _flow_edges.append(
        StreamlitFlowEdge(
            id=f"e{_i}",
            source=_e["source"],
            target=_e["target"],
            edge_type="smoothstep",
            label=_e["desc"],
            label_show_bg=True,
            label_bg_style={
                "fill": "#FFFFFF",
                "fillOpacity": 0.92,
                "stroke": _color,
                "strokeWidth": 0.5,
                "rx": 4,
                "ry": 4,
            },
            label_style={
                "fontSize": "10px",
                "fontFamily": "Inter, Arial, sans-serif",
                "fontWeight": "600",
                "fill": _color,
            },
            marker_end={"type": "arrowclosed"},
            animated=False,
            style={"stroke": _color, "strokeWidth": 2, "opacity": 0.7},
        )
    )

if _EXEC_ARCH_STATE_KEY not in st.session_state:
    st.session_state[_EXEC_ARCH_STATE_KEY] = StreamlitFlowState(nodes=_flow_nodes, edges=_flow_edges)

_arch_state = streamlit_flow(
    "exec_architecture_flow",
    st.session_state[_EXEC_ARCH_STATE_KEY],
    layout=ManualLayout(),
    fit_view=True,
    show_minimap=True,
    show_controls=True,
    allow_zoom=True,
    pan_on_drag=True,
    hide_watermark=True,
    height=580,
    get_node_on_click=True,
    style={"backgroundColor": "#FAFBFC"},
)

st.session_state[_EXEC_ARCH_STATE_KEY] = _arch_state
_selected = _arch_state.selected_id
if _selected and _selected in _NODE_MAP:
    _sel_node = _NODE_MAP[_selected]
    st.info(f"**{_sel_node['icon']} {_sel_node['label']}** — {_sel_node['detail']}")

_legend_cols = st.columns(4)
for _col, (_lk, _lv) in zip(_legend_cols, _LAYER_COLORS.items()):
    with _col:
        st.markdown(
            f'<div style="background:{_lv["bg"]}; border:2px solid {_lv["border"]}; '
            f'border-radius:8px; padding:8px 12px; text-align:center; font-size:13px; '
            f'color:{_lv["accent"]}; font-weight:600;">{_LAYER_BADGES[_lk]}</div>',
            unsafe_allow_html=True,
        )

st.caption(
    "Lectura del diagrama: el proyecto no termina en un score. La PD calibrada alimenta una capa de incertidumbre "
    "conformal, esa incertidumbre se convierte en restricciones de decisión robusta y en rangos de provisión IFRS9, "
    "mientras causalidad y supervivencia aportan acción e interpretación temporal."
)
st.markdown(
    """
La contribución metodológica del proyecto está en **encadenar técnicas que normalmente se presentan por separado**.
Primero, el modelo PD (CatBoost calibrado) transforma señales tabulares complejas en una probabilidad utilizable; luego,
Conformal Mondrian agrega una garantía empírica de cobertura por segmentos, evitando la sobreconfianza de una PD puntual.
Ese intervalo no se queda en el plano académico: entra como insumo directo en la optimización robusta de cartera, donde el
`PD_high` representa un peor caso plausible y hace explícito el costo de robustez en términos de retorno.

En paralelo, la capa causal (DML/CATE) responde una pregunta distinta: no solo quién es riesgoso, sino **qué intervención
podría cambiar ese riesgo** y con qué valor económico esperado. Forecasting y supervivencia aportan la dimensión temporal,
que es esencial para IFRS9 y para comités de riesgo orientados a horizonte. El resultado final es un sistema coherente en el
que cada técnica entrega un output que se reutiliza como input en la siguiente capa, cerrando el puente entre analítica,
decisión y gobernanza.
"""
)
st.markdown(
    """
Como cierre ejecutivo, la pregunta central no es solo “¿cuál modelo clasifica mejor?”, sino “¿qué tan defendible es la
decisión final cuando existe incertidumbre de modelo?”. Esta portada anticipa la respuesta del resto del recorrido:
el valor surge de integrar predicción, incertidumbre, causalidad y optimización en una misma narrativa operativa.
"""
)

next_page_teaser(
    "Historia de Datos",
    "Comenzamos por el comportamiento del portafolio: composición, señales de riesgo y patrones temporales.",
    "pages/data_story.py",
)
