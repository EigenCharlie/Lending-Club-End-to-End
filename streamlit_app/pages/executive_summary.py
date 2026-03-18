"""Resumen ejecutivo del proyecto integral de riesgo de credito."""
# ruff: noqa: E402, I001

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

from streamlit_app.components.context_help import term_popover
from streamlit_app.components.dvc_kpi_spine import render_global_kpi_spine
from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import next_page_teaser, storytelling_intro
from streamlit_app.components.story_shell import (
    render_decision_box,
    render_key_takeaway,
    render_page_feedback,
    render_page_header,
)
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import (
    format_number,
    format_pct,
    get_operational_threshold,
    get_pd_internal_threshold,
    load_gpu_replay_summary,
    load_rapids_ifrs9_mc_tail_metrics,
    load_rapids_pd_benchmark_stage_table,
    load_rapids_insight_stage_table,
    load_rapids_stage_comparison,
    load_rapids_tradeoff_full_ab_status,
    load_rapids_tradeoff_full_policy,
    try_load_json,
)


def _first_valid(*values: float | int | None) -> float:
    for value in values:
        if value is not None and value != 0:
            return float(value)
    return 0.0


_ARCH_NODES = [
    {
        "id": "raw",
        "label": "Lending Club Raw",
        "layer": "data",
        "detail": "2.93M filas · 142 cols",
        "icon": "\U0001f4e6",
    },
    {
        "id": "clean",
        "label": "Limpieza + QA",
        "layer": "data",
        "detail": "1.86M filas · 110 cols",
        "icon": "\U0001f9f9",
    },
    {
        "id": "fe",
        "label": "Feature Engineering",
        "layer": "data",
        "detail": "WOE/IV · ratios · buckets",
        "icon": "\u2699\ufe0f",
    },
    {
        "id": "pd",
        "label": "PD Model",
        "layer": "model",
        "detail": "CatBoost + calibración · AUC OOT",
        "icon": "\U0001f3af",
    },
    {
        "id": "conformal",
        "label": "Conformal Mondrian",
        "layer": "model",
        "detail": "Intervalos PD · Coverage 90/95",
        "icon": "\U0001f4d0",
    },
    {
        "id": "causal",
        "label": "Causalidad (DoWhy + CausalForestDML)",
        "layer": "model",
        "detail": "Policy simulation bajo CATE local",
        "icon": "\U0001f9ec",
    },
    {
        "id": "survival",
        "label": "Forecast + Survival",
        "layer": "model",
        "detail": "Cox/RSF concordance",
        "icon": "\u23f3",
    },
    {
        "id": "optim",
        "label": "Optimizaci\u00f3n Robusta",
        "layer": "decision",
        "detail": "Pyomo + HiGHS",
        "icon": "\U0001f4bc",
    },
    {
        "id": "ifrs9",
        "label": "IFRS9 (Stage + ECL)",
        "layer": "decision",
        "detail": "4 escenarios · PD\u00d7LGD",
        "icon": "\U0001f3e6",
    },
    {
        "id": "governance",
        "label": "Gobernanza",
        "layer": "decision",
        "detail": "Drift · Fairness · Policy",
        "icon": "\U0001f6e1\ufe0f",
    },
    {
        "id": "streamlit",
        "label": "Libro Din\u00e1mico Streamlit",
        "layer": "output",
        "detail": "17 p\u00e1ginas interactivas",
        "icon": "\U0001f4ca",
    },
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
    "data": "\U0001f4be DATOS",
    "model": "\U0001f9e0 MODELO",
    "decision": "\U0001f4a1 DECISI\u00d3N",
    "output": "\U0001f680 SALIDA",
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

page_contract = get_page_contract("executive_summary")
render_page_header(page_contract)
render_key_takeaway(
    "El valor del proyecto no está en una sola métrica: está en encadenar PD, incertidumbre, optimización e IFRS9 con una fuente canónica común."
)
term_popover("canónico", label="Qué significa 'canónico' en este dashboard")
storytelling_intro(
    page_goal="Entender cómo se encadenan score, incertidumbre, causalidad, optimización e IFRS9 en una sola historia.",
    business_value="Permite leer el dashboard de arriba hacia abajo sin perder el hilo entre métricas técnicas y decisiones.",
    key_decision="Usar esta portada para validar el estado canónico del run antes de defender cualquier claim puntual.",
    how_to_read=[
        "Empieza por la espina KPI y luego baja por arquitectura.",
        "Usa esta página como índice antes de entrar a módulos especializados.",
        "Confirma que el run canónico y sus artefactos estén alineados.",
    ],
)

st.success(
    "**Lending Club**: la plataforma de préstamos peer-to-peer más grande de EE.UU. "
    "Este proyecto transforma su histórico de préstamos en un sistema completo "
    "de decisión de riesgo de crédito, desde score hasta provisiones IFRS9."
)

render_decision_box(
    "Evaluar la política de riesgo con KPIs canónicos (DVC) y después bajar al detalle por módulo para defender trade-offs.",
    owner="Riesgo / Data Science",
    cadence="snapshot por commit o release",
)
render_global_kpi_spine("executive")
st.caption(
    "Thresholds canónicos: "
    f"interno PD `{get_pd_internal_threshold():.2f}` vs operativo fairness/aprobación "
    f"`{get_operational_threshold():.2f}`."
)

rapids_summary = load_gpu_replay_summary()
rapids_compare = load_rapids_stage_comparison()
rapids_ifrs9 = load_rapids_ifrs9_mc_tail_metrics()
rapids_insights = load_rapids_insight_stage_table()
rapids_full_ab = load_rapids_tradeoff_full_ab_status()
rapids_full_policy = load_rapids_tradeoff_full_policy()
pd_benchmark = load_rapids_pd_benchmark_stage_table()
if not rapids_compare.empty:
    rapids_wins = rapids_compare[rapids_compare["speedup_gpu_vs_cpu"] > 1].copy()
    top_rapids = (
        rapids_wins.sort_values("speedup_gpu_vs_cpu", ascending=False).iloc[0]
        if not rapids_wins.empty
        else None
    )
    st.markdown("### Nuevo carril RAPIDS ya validado")
    kpi_row(
        [
            {"label": "GPU replay", "value": rapids_summary.get("run_tag", "N/D")},
            {"label": "Stages GPU PASS", "value": str(len(rapids_compare))},
            {
                "label": "Mayor speedup",
                "value": (
                    f"{top_rapids['stage']} · {top_rapids['speedup_gpu_vs_cpu']:.1f}x"
                    if top_rapids is not None
                    else "N/D"
                ),
            },
            {
                "label": "IFRS9 MC GPU",
                "value": f"{rapids_ifrs9.get('speedup_gpu_vs_cpu', 0):.1f}x",
            },
        ],
        n_cols=4,
    )
    st.markdown(
        """
    El proyecto ya tiene dos carriles claros:

    - **canónico CPU** para promoción, gobernanza y storytelling central;
    - **RAPIDS/GPU** para etapas donde la aceleración es material.

    La lectura final es útil para negocio y academia: `cuopt` acelera fuerte OR, `cupy` acelera Monte Carlo IFRS9, y el replay muestra honestamente que `PD` completo todavía no gana throughput end-to-end.
    """
    )
    show_cols = ["stage", "cpu_seconds", "gpu_seconds", "speedup_gpu_vs_cpu"]
    st.dataframe(
        rapids_compare[show_cols].rename(
            columns={
                "stage": "Stage",
                "cpu_seconds": "CPU seconds",
                "gpu_seconds": "GPU seconds",
                "speedup_gpu_vs_cpu": "GPU speedup vs CPU",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    if not rapids_insights.empty:
        cugraph_row = rapids_insights[rapids_insights["stage"] == "cugraph_similarity"]
        cugraph_speedup = (
            float(cugraph_row.iloc[0]["speedup_gpu_vs_cpu"]) if not cugraph_row.empty else 0.0
        )
        st.info(
            f"RAPIDS ya entró también a la fábrica de insights: `cuGraph` ganó ~{cugraph_speedup:.1f}x en similarity graph, mientras `cuDF ETL` y `cuML KMeans` mostraron por qué esta sección debe ser selectiva, no propagandística."
        )
    if rapids_full_ab:
        st.caption(
            "El experimento full de `tradeoff + A/B` confirmó otra lección: la GPU resuelve la escala, pero el criterio económico sigue favoreciendo una política casi no robusta. El cuello residual ya no es computacional."
        )
    if not pd_benchmark.empty:
        fit_speedup = pd_benchmark.loc[
            pd_benchmark["stage"] == "fit_only", "speedup_gpu_vs_cpu"
        ]
        full_speedup = pd_benchmark.loc[
            pd_benchmark["stage"] == "full_stage", "speedup_gpu_vs_cpu"
        ]
        hpo_row = pd_benchmark.loc[pd_benchmark["stage"] == "hpo"]
        if "gpu_status" in pd_benchmark.columns:
            hpo_gpu_status = hpo_row["gpu_status"]
            hpo_status_label = str(hpo_gpu_status.iloc[0]) if not hpo_gpu_status.empty else "N/D"
        else:
            hpo_gpu_seconds = hpo_row["gpu_seconds"] if "gpu_seconds" in hpo_row.columns else pd.Series(dtype=float)
            hpo_status_label = (
                "inestable"
                if not hpo_gpu_seconds.empty and pd.isna(hpo_gpu_seconds.iloc[0])
                else "N/D"
            )
        st.caption(
            "PD ya está separado honestamente en tres problemas: "
            f"fit-only ({float(fit_speedup.iloc[0]) if not fit_speedup.empty else 0:.2f}x), "
            f"full-stage ({float(full_speedup.iloc[0]) if not full_speedup.empty else 0:.2f}x) "
            f"y HPO GPU ({hpo_status_label})."
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

st.markdown("### Métrica ML vs KPI de negocio")
st.dataframe(
    pd.DataFrame(
        [
            {
                "Capa": "Ranking",
                "Métrica ML": "AUC / KS",
                "KPI de negocio asociado": "Priorización de originación y revisión",
                "Riesgo de mala lectura": "Asumir que buen AUC implica probabilidad confiable",
            },
            {
                "Capa": "Probabilidad",
                "Métrica ML": "Brier / ECE (proper scoring)",
                "KPI de negocio asociado": "Pricing, límites y provisión IFRS9",
                "Riesgo de mala lectura": "Confundir ranking alto con calibración adecuada",
            },
            {
                "Capa": "Incertidumbre",
                "Métrica ML": "Coverage / Interval width",
                "KPI de negocio asociado": "Robustez de decisión y buffers prudenciales",
                "Riesgo de mala lectura": "Leer bandas como CI de parámetros y no como PI operativo",
            },
            {
                "Capa": "Decisión",
                "Métrica ML": "Price of Robustness / funded loans",
                "KPI de negocio asociado": "Retorno ajustado por riesgo y resiliencia",
                "Riesgo de mala lectura": "Optimizar retorno puntual sin costo de incertidumbre",
            },
        ]
    ),
    width="stretch",
    hide_index=True,
)

st.markdown(
    """
Esta portada resume el hilo completo y sirve como punto de entrada. La lectura recomendada es secuencial:
contexto de datos, ingeniería de features, modelo PD, incertidumbre, causalidad, decisión robusta, provisión IFRS9 y gobernanza.
"""
)

summary = try_load_json("pipeline_summary")
eda = try_load_json("eda_summary")
comparison = try_load_json("model_comparison")
policy = try_load_json("conformal_policy_status", directory="models")
governance = try_load_json("governance_status", directory="models")
causal_effect = try_load_json("causal_effect_status", directory="models")

pipeline = summary.get("pipeline", {})
pd_model = summary.get("pd_model", {})
causal_summary = summary.get("causal", {})
final_metrics = comparison.get("final_test_metrics", {})

causal_ate = causal_effect.get("ate", causal_summary.get("ate"))
if causal_ate not in (None, ""):
    try:
        _NODE_MAP["causal"]["detail"] = f"+1pp tasa -> {float(causal_ate):+.3f}pp default"
    except Exception:
        _NODE_MAP["causal"]["detail"] = "ATE artefact-driven"

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
    st.plotly_chart(fig, width="stretch")
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
    st.plotly_chart(fig, width="stretch")
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
    st.session_state[_EXEC_ARCH_STATE_KEY] = StreamlitFlowState(
        nodes=_flow_nodes, edges=_flow_edges
    )

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
for _col, (_lk, _lv) in zip(_legend_cols, _LAYER_COLORS.items(), strict=False):
    with _col:
        st.markdown(
            f'<div style="background:{_lv["bg"]}; border:2px solid {_lv["border"]}; '
            f"border-radius:8px; padding:8px 12px; text-align:center; font-size:13px; "
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

En paralelo, la capa causal (DoWhy + CausalForestDML) responde una pregunta distinta: no solo quién es riesgoso, sino **qué intervención
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
render_page_feedback("executive_summary")

next_page_teaser(
    "Historia de Datos",
    "Comenzamos por el comportamiento del portafolio: composición, señales de riesgo y patrones temporales.",
    "pages/data_story.py",
)
