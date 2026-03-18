"""Inteligencia causal para políticas de riesgo de crédito."""

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
from streamlit_app.components.narrative import next_page_teaser, storytelling_intro
from streamlit_app.components.story_shell import (
    render_caveats,
    render_key_takeaway,
    render_page_feedback,
    render_page_header,
)
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import get_notebook_image_path, try_load_json, try_load_parquet


def _first_numeric(*values: object) -> float | None:
    for value in values:
        try:
            if value is None:
                continue
            out = float(value)
        except Exception:
            continue
        if pd.notna(out):
            return out
    return None


def _format_pp(value: float | None) -> str:
    if value is None:
        return "N/D"
    return f"{value:+.3f}pp"


def _format_money(value: float | None) -> str:
    if value is None:
        return "N/D"
    return f"${value:,.0f}"


def _build_causal_snapshot() -> dict:
    pipeline_summary = try_load_json("pipeline_summary", default={})
    pipeline_causal = pipeline_summary.get("causal", {})
    effect_status = try_load_json("causal_effect_status", directory="models", default={})
    rule_status = try_load_json("causal_policy_rule", directory="models", default={})
    oot_status = try_load_json("causal_policy_oot_status", directory="models", default={})
    portfolio_status = try_load_json("cate_portfolio_status", directory="models", default={})
    refutation_summary = try_load_json("causal_refutation_summary", directory="models", default={})
    causal_oot_tail_risk = try_load_parquet("causal_oot_tail_risk")
    selected_metrics = rule_status.get("selected_metrics", {})
    return {
        "effect_status": effect_status,
        "rule_status": rule_status,
        "oot_status": oot_status,
        "portfolio_status": portfolio_status,
        "ate": _first_numeric(effect_status.get("ate"), pipeline_causal.get("ate")),
        "cate_mean": _first_numeric(
            effect_status.get("cate_mean"), pipeline_causal.get("cate_mean")
        ),
        "selected_rule": str(
            rule_status.get("selected_rule") or pipeline_causal.get("selected_rule") or "N/D"
        ),
        "total_net_value": _first_numeric(
            selected_metrics.get("total_net_value"),
            pipeline_causal.get("total_net_value"),
        ),
        "bootstrap_p05_net": _first_numeric(
            selected_metrics.get("bootstrap_p05_net"),
            pipeline_causal.get("bootstrap_p05_net"),
        ),
        "avg_action_rate": _first_numeric(
            oot_status.get("avg_action_rate"),
            selected_metrics.get("action_rate"),
            pipeline_causal.get("avg_action_rate"),
        ),
        "official_method": effect_status.get("official_method", {})
        or pipeline_causal.get("official_method", {}),
        "refutation_summary": refutation_summary,
        "causal_oot_tail_risk": causal_oot_tail_risk,
    }

st.title("🧬 Inteligencia Causal")
st.caption(
    "Estimación de efectos causales heterogéneos para orientar políticas de precio "
    "y acciones de mitigación de riesgo."
)
page_contract = get_page_contract("causal_intelligence")
render_page_header(page_contract)
render_key_takeaway(
    "La meta aquí no es describir correlaciones sino estimar efectos causales heterogéneos útiles para políticas de precio/intervención."
)
causal_snapshot = _build_causal_snapshot()
ate_text = _format_pp(causal_snapshot["ate"])
net_text = _format_money(causal_snapshot["total_net_value"])
rule_name = causal_snapshot["selected_rule"]
storytelling_intro(
    page_goal=(
        "Distinguir correlación de causalidad para saber qué acción realmente reduce default."
    ),
    business_value=(
        "Evita políticas costosas que parecen razonables por correlación, pero no cambian el riesgo real."
    ),
    key_decision=(
        "Definir segmentos donde conviene intervenir en precio o tratamiento y cuantificar su impacto neto."
    ),
    how_to_read=[
        "Empieza por el bloque de ATE/CATE y su interpretación de negocio.",
        "Revisa la regla causal seleccionada y su valor neto económico.",
        "Contrasta supuestos causales con límites antes de operacionalizar la política.",
    ],
)

with st.expander("¿Por qué no basta con correlación? — La trampa del scoring tradicional"):
    st.markdown(
        f"""
### El problema

En el dataset, los préstamos con **tasas altas tienen más defaults**. Pero, ¿subir la tasa
*causa* más defaults? ¿O simplemente le cobramos más a quienes ya eran riesgosos?

**Correlación**: "Los préstamos con tasa >20% tienen 24% de default"
**Causalidad**: "Si movemos la tasa 1pp, el efecto esperado sobre default se estima con diseño causal y debe salir del artefacto oficial"

Son preguntas fundamentalmente distintas. La primera describe el pasado; la segunda
permite diseñar intervenciones futuras.

### ¿Dónde se usa inferencia causal en la industria?

| Industria | Aplicación |
|-----------|------------|
| **Banca** | Pricing dinámico: ¿cuánto puedo subir la tasa sin aumentar defaults? |
| **Telecoms** | Campañas de retención: ¿a quién le funciona el descuento? |
| **Regulación** | Stress testing: ¿qué pasa si sube la tasa de referencia? |
| **Seguros** | Efecto de franquicias en reclamaciones |
| **Tech** | A/B testing causal (Uber, Lyft, Netflix) |

### Resultado clave del proyecto

> En el snapshot canónico actual: **+1 punto porcentual en tasa de interés → {ate_text} en probabilidad de default**

La política seleccionada en el artefacto oficial es **`{rule_name}`** y su
valor neto esperado reportado es **{net_text}**. La semántica operativa no es
un SCM exacto: es una **simulación de política bajo CATE local** usada para
priorizar dónde una intervención de precio parece defendible.
"""
    )

st.markdown(
    """
La capa causal aborda una limitación frecuente en analítica de crédito: confundir correlación con intervención útil.
Que una variable esté asociada a más default no implica que mover esa variable cambie el resultado en la misma magnitud.
Por eso se combina diseño causal (DoWhy) con estimación heterogénea (EconML), buscando reglas que sean simultáneamente
plausibles desde identificación y rentables desde impacto económico. En el flujo del proyecto, causalidad no reemplaza
al score PD: lo complementa para decidir **dónde actuar** y no solo **a quién clasificar**.
"""
)
st.markdown(
    """
### Qué técnica causal se está usando y cómo leerla
- **ATE** (Average Treatment Effect): efecto promedio de mover una palanca (ej. tasa) sobre default.
- **CATE** (Conditional ATE): ese efecto, pero condicionado por perfil de cliente/segmento.
- **DoWhy** aporta identificación backdoor, DAG y refutaciones.
- **EconML CausalForestDML** estima efectos heterogéneos con flexibilidad no lineal en tabular.

Interpretación en este proyecto:
- `CATE > 0`: aumentar tasa empeora default (conviene bajar o no subir en ese segmento).
- `CATE < 0`: subir tasa no incrementa default o incluso puede asociarse a mejor selección.
- Una política causal útil combina magnitud de efecto con restricción económica (`net_value`).
"""
)
st.markdown(
    """
### Supuestos y límites que sí declaramos
Para interpretar estos efectos como causales, se asumen condiciones estándar de inferencia observacional:
- **Ignorabilidad condicional**: tras controlar covariables relevantes, la asignación de tratamiento es as-if aleatoria.
- **Overlap**: existen observaciones comparables entre niveles de tratamiento en cada subgrupo.
- **Consistencia temporal**: las covariables usadas estaban disponibles al momento de originación.

En términos prácticos: estos resultados son evidencia causal aplicada y útil para política, pero siempre deben leerse
como estimaciones sujetas a supuestos, no como “verdad absoluta” independiente del diseño de datos.
"""
)

portfolio_status = causal_snapshot["portfolio_status"]
if portfolio_status:
    st.markdown("### Estado operativo actual del CATE portfolio")
    st.info(
        f"El proyecto ya separa con claridad política causal y portfolio causal. En el champion oficial, `cate_portfolio` quedó en `{portfolio_status.get('cate_policy_mode', 'N/D')}` con `promotion_eligible={portfolio_status.get('promotion_eligible', False)}`. La GPU ya acelera ese bloque con cuOpt, pero esa aceleración no cambia todavía su estatus económico: sigue siendo un carril de investigación."
    )

cate_df = try_load_parquet("cate_estimates")
segment_summary = try_load_parquet("causal_policy_segment_summary")
grade_summary = try_load_parquet("causal_policy_grade_summary")
rule_selected = try_load_parquet("causal_policy_rule_selected")
rule_candidates = try_load_parquet("causal_policy_rule_candidates")
simulation = try_load_parquet("causal_policy_simulation")
selected = (
    rule_selected.iloc[0]
    if not rule_selected.empty
    else pd.Series(
        {
            "rule_name": rule_name,
            "action_rate": causal_snapshot["avg_action_rate"] or 0.0,
            "total_net_value": causal_snapshot["total_net_value"] or 0.0,
            "total_loss_reduction": 0.0,
            "total_revenue_impact": 0.0,
            "bootstrap_p05_net": causal_snapshot["bootstrap_p05_net"] or 0.0,
        }
    )
)

kpi_row(
    [
        {"label": "Regla elegida", "value": str(selected.get("rule_name", "N/D"))},
        {"label": "Action rate", "value": f"{selected.get('action_rate', 0) * 100:.1f}%"},
        {"label": "Valor neto total", "value": f"${selected.get('total_net_value', 0):,.0f}"},
        {"label": "Reducción pérdida", "value": f"${selected.get('total_loss_reduction', 0):,.0f}"},
    ]
)

portfolio_status = causal_snapshot["portfolio_status"]
if portfolio_status.get("warning") or (
    portfolio_status and not portfolio_status.get("feasible_adjusted", True)
):
    st.warning(
        "Integración causal no utilizable en este run. "
        + str(portfolio_status.get("warning") or "").strip()
    )

st.dataframe(
    pd.DataFrame(
        [
            {
                "Métrica": "ATE/CATE",
                "Significado técnico": "Efecto causal promedio/heterogéneo de la tasa sobre default.",
                "Significado negocio": "Permite diseñar pricing diferenciado por sensibilidad real.",
            },
            {
                "Métrica": "Action rate",
                "Significado técnico": "Proporción de clientes donde se recomienda intervención.",
                "Significado negocio": "Tamaño operacional de la política causal.",
            },
            {
                "Métrica": "Valor neto",
                "Significado técnico": "Pérdida evitada - costo de intervención en ingresos.",
                "Significado negocio": "Justifica económicamente la política seleccionada.",
            },
        ]
    ),
    width="stretch",
    hide_index=True,
)
st.markdown(
    """
En lenguaje de implementación, el pipeline causal produce una estimación por préstamo de sensibilidad (`cate`), luego
simula reglas operativas (quién recibe intervención, de cuánto, y con qué impacto esperado en pérdidas/ingresos) y por
último selecciona la regla que maximiza valor neto bajo restricciones de cobertura y downside. Es decir, no se queda en
estimación de efecto: llega hasta política accionable.
"""
)

st.subheader("1) Distribución de efectos heterogéneos (CATE)")
if cate_df.empty:
    st.info("No hay `cate_estimates.parquet` disponible para visualizar la distribución de efectos.")
else:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            cate_df.sample(min(120000, len(cate_df)), random_state=21),
            x="cate",
            nbins=70,
            title="Distribución CATE",
            labels={"cate": "Efecto causal estimado de tasa sobre default"},
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
        fig.update_traces(marker_color="#00D4AA")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Propósito: observar heterogeneidad de sensibilidad causal. Insight: una distribución ancha de CATE confirma que "
            "una política única de tasa no es óptima para todos los clientes."
        )

    with col2:
        if "grade" in cate_df.columns:
            fig = px.box(
                cate_df.sample(min(120000, len(cate_df)), random_state=27),
                x="grade",
                y="cate",
                title="CATE por grade",
                labels={"grade": "Grade", "cate": "CATE"},
            )
            fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("El artefacto CATE actual no incluye `grade`; se omite el boxplot por segmento.")
        st.caption(
            "Propósito: comparar sensibilidad causal por grade. Insight: algunos segmentos concentran mayor potencial de reducción "
            "de default ante ajuste de tasa."
        )

st.subheader("2) Impacto de política por segmento")
if segment_summary.empty or grade_summary.empty:
    st.info("Faltan resúmenes de simulación causal para renderizar los gráficos por segmento/grade.")
else:
    col3, col4 = st.columns(2)
    with col3:
        fig = px.bar(
            segment_summary,
            x="segment",
            y="total_net_value",
            color="action_rate",
            title="Valor neto total por segmento",
            labels={
                "segment": "Segmento",
                "total_net_value": "Valor neto (USD)",
                "action_rate": "Action rate",
            },
            color_continuous_scale="Tealgrn",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Propósito: priorizar segmentos por valor económico esperado. Insight: no siempre coincide el mayor valor con el mayor "
            "action rate, por lo que la regla debe optimizar ambos."
        )

    with col4:
        fig = px.bar(
            grade_summary.sort_values("grade"),
            x="grade",
            y="action_rate",
            color="avg_pd_reduction",
            title="Action rate y reducción de PD por grade",
            labels={"grade": "Grade", "action_rate": "Action rate", "avg_pd_reduction": "Δ PD"},
            color_continuous_scale="Sunsetdark",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390, coloraxis_showscale=False)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Propósito: medir intensidad de intervención por grade. Insight: action rate alto con baja mejora de PD puede no ser "
            "económicamente eficiente."
        )

st.markdown(
    """
Lectura conjunta de las gráficas de esta sección:
1. Histograma/boxplot de CATE: muestran heterogeneidad real de sensibilidad (base para personalización de política).
2. Barras por segmento/grade: traducen esa heterogeneidad a valor económico y factibilidad operativa.
3. Frontera de reglas: explicita el trade-off entre amplitud de intervención y retorno neto esperable.
4. Waterfall: deja auditable la composición de valor (pérdida evitada vs costo comercial).
"""
)
st.markdown(
    """
La lógica económica detrás de esta lectura es directa: primero identificamos dónde existe sensibilidad causal real,
luego traducimos esa sensibilidad a reglas accionables y, finalmente, medimos si el beneficio en pérdidas evitadas supera
el costo comercial de la intervención. Si ese último paso no cierra, la regla se descarta aunque el efecto causal sea alto.
"""
)

st.subheader("3) Frontera de reglas candidatas")
if rule_candidates.empty:
    st.info("No hay reglas candidatas disponibles. Ejecuta la validación causal para poblar esta sección.")
else:
    fig = px.scatter(
        rule_candidates,
        x="action_rate",
        y="total_net_value",
        color="pass_all",
        size="n_selected",
        text="rule_name",
        title="Trade-off entre cobertura de acción y valor económico",
        labels={
            "action_rate": "Action rate",
            "total_net_value": "Valor neto (USD)",
            "pass_all": "Cumple constraints",
        },
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Propósito: evaluar frontera de reglas candidatas. Insight: la mejor regla no es la de mayor cobertura, sino la que "
        "maximiza valor cumpliendo restricciones."
    )

    st.dataframe(
        rule_candidates.sort_values(["pass_all", "total_net_value"], ascending=[False, False]),
        width="stretch",
        hide_index=True,
    )

st.subheader("4) Descomposición económica de la regla seleccionada")
fig = go.Figure(
    go.Waterfall(
        measure=["relative", "relative", "total"],
        x=["Reducción de pérdida", "Impacto en ingresos", "Valor neto"],
        y=[
            selected.get("total_loss_reduction", 0),
            selected.get("total_revenue_impact", 0),
            selected.get("total_net_value", 0),
        ],
        connector={"line": {"color": "#A0AEC0"}},
    )
)
fig.update_layout(
    **PLOTLY_TEMPLATE["layout"], height=360, title="Cómo se forma el valor causal neto"
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Propósito: descomponer creación de valor en ahorro de pérdida vs impacto comercial. Insight: hace auditable el trade-off "
    "de una política causal antes de implementarla."
)

st.markdown(
    """
**Mensaje metodológico:**
- El modelo predictivo indica quién tiene más riesgo.
- El bloque causal estima qué palancas pueden cambiar ese riesgo.
- La optimización convierte ese aprendizaje en reglas de portafolio económicamente coherentes.
"""
)
st.markdown(
    """
Como cierre, esta capa responde una pregunta estratégica del proyecto: “¿qué decisiones cambian realmente el riesgo y
con qué costo-beneficio esperado?”. La respuesta causal completa el stack porque evita confundir patrones observacionales
con decisiones efectivas. En términos narrativos, aquí pasamos de diagnosticar riesgo a diseñar intervención defendible.
"""
)
st.markdown(
    """
La implicación para negocio es fuerte: dos clientes con PD similar pueden requerir decisiones distintas si su sensibilidad
causal difiere. Por eso la capa causal no compite con el score, sino que agrega una dimensión de política que permite
asignar acciones donde realmente producen valor neto, en lugar de aplicar reglas uniformes por conveniencia operativa.
"""
)

col_i, col_j = st.columns(2)
with col_i:
    img = get_notebook_image_path("07_causal_inference", "cell_020_out_01.png")
    if img.exists():
        st.image(
            str(img),
            caption="Notebook 07: correlación vs causalidad para efecto de tasa sobre default.",
            width="stretch",
        )
with col_j:
    img = get_notebook_image_path("07_causal_inference", "cell_026_out_01.png")
    if img.exists():
        st.image(
            str(img),
            caption="Notebook 07: sensibilidad de tasa y recomendación de política por segmento.",
            width="stretch",
        )

_cate_seg_img = get_notebook_image_path("07_causal_inference", "cate_segment_heterogeneity.png")
_cate_oot_img = get_notebook_image_path("07_causal_inference", "cate_distribution_train_vs_oot.png")
_cate_kpi_img = get_notebook_image_path("07_causal_inference", "causal_policy_kpis.png")
_ate_ci_img = get_notebook_image_path("07_causal_inference", "ate_confidence_interval.png")
_causal_named = [
    (_cate_seg_img, "NB07: heterogeneidad CATE por segmento — quién es más sensible al ajuste de tasa."),
    (_cate_oot_img, "NB07: distribución CATE train vs OOT — estabilidad de efectos en test temporal."),
    (_cate_kpi_img, "NB07: KPIs de política causal — acción rate, valor neto esperado y reducción de pérdida."),
    (_ate_ci_img, "NB07: ATE con intervalo de confianza bootstrap — magnitud e incertidumbre del efecto promedio."),
]
_causal_named_valid = [(p, c) for p, c in _causal_named if p.exists()]
if _causal_named_valid:
    with st.expander("Figuras del notebook: heterogeneidad CATE y KPIs de política", expanded=True):
        _cn_cols = st.columns(min(len(_causal_named_valid), 2))
        for _ci, (_p, _cap) in enumerate(_causal_named_valid):
            with _cn_cols[_ci % 2]:
                st.image(str(_p), caption=_cap, width="stretch")

with st.expander("Muestra de simulación de política por préstamo"):
    cols = [
        "id",
        "segment",
        "grade",
        "base_rate_pp",
        "recommended_delta_rate_pp",
        "expected_pd_reduction",
        "net_value",
        "recommended_action",
    ]
    if simulation.empty:
        st.info("No hay simulación causal cargada para mostrar ejemplos por préstamo.")
    else:
        available_cols = [col for col in cols if col in simulation.columns]
        st.dataframe(
            simulation[available_cols].sample(min(120, len(simulation)), random_state=3),
            width="stretch",
            hide_index=True,
        )

st.subheader("5) Impacto en optimización de portafolio")
cate_comparison = try_load_parquet("cate_portfolio_comparison")
if not cate_comparison.empty and len(cate_comparison) == 2:
    base = cate_comparison[cate_comparison["scenario"] == "baseline"].iloc[0]
    adj = cate_comparison[cate_comparison["scenario"] == "cate_adjusted"].iloc[0]
    delta_obj = float(adj["objective_value"] - base["objective_value"])
    delta_funded = int(adj["n_funded"] - base["n_funded"])
    kpi_row(
        [
            {"label": "Objetivo baseline", "value": f"${base['objective_value']:,.0f}"},
            {"label": "Objetivo CATE-adj", "value": f"${adj['objective_value']:,.0f}"},
            {"label": "Δ objetivo", "value": f"${delta_obj:+,.0f}"},
            {"label": "Δ loans funded", "value": f"{delta_funded:+d}"},
        ],
        n_cols=4,
    )
    fig = px.bar(
        cate_comparison.melt(id_vars="scenario", var_name="metric", value_name="value"),
        x="metric",
        y="value",
        color="scenario",
        barmode="group",
        title="Baseline vs CATE-adjusted portfolio",
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=350)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Este análisis cierra el ciclo causal→portafolio: los efectos heterogéneos de NB07 "
        "se traducen en ajustes de tasa que mejoran la asignación de capital en NB08."
    )
else:
    st.info("Ejecuta `scripts/optimize_cate_portfolio.py` para comparar portafolios baseline vs CATE-adjusted.")

_refutation_summary = causal_snapshot.get("refutation_summary", {})
if _refutation_summary:
    with st.expander("Refutaciones DoWhy + tail risk OOT en CATE", expanded=False):
        refs = _refutation_summary.get("refutation_tests", [])
        verdict = _refutation_summary.get("refutation_verdict", "")
        if verdict:
            st.info(verdict)
        if refs:
            ref_df = pd.DataFrame([{
                "Test": r.get("test", ""),
                "Status": r.get("status", ""),
                "Interpretación": r.get("interpretation", ""),
            } for r in refs])
            st.dataframe(ref_df, hide_index=True, width="stretch")
        oot_summ = _refutation_summary.get("oot_cate_summary", {})
        if oot_summ:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("N OOT loans", f"{oot_summ.get('n_obs', 0):,}")
            c2.metric("ATE OOT", f"{float(oot_summ.get('ate_oot', 0)):.5f}")
            c3.metric("CATE P5 (tail riesgo)", f"{oot_summ.get('percentiles', {}).get('p5', 0):.5f}")
            c4.metric("CATE P95 (mayor beneficio)", f"{oot_summ.get('percentiles', {}).get('p95', 0):.5f}")
            st.caption(oot_summ.get("tail_risk_interpretation", ""))
        _tail_df = causal_snapshot.get("causal_oot_tail_risk", pd.DataFrame())
        if isinstance(_tail_df, pd.DataFrame) and not _tail_df.empty:
            st.markdown("**Distribución CATE por grade (OOT test set)**")
            st.dataframe(_tail_df.round(5), hide_index=True, width="stretch")

render_caveats(
    [
        "Los efectos causales dependen de supuestos de identificación y cobertura de covariables observadas.",
        "Una política basada en CATE requiere validación operativa y guardrails antes de despliegue.",
    ]
)
render_page_feedback("causal_intelligence")

next_page_teaser(
    "Optimizador de Portafolio",
    "Integramos PD, incertidumbre y restricciones para decidir asignación de capital.",
    "pages/portfolio_optimizer.py",
)
