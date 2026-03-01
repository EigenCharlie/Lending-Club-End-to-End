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
from streamlit_app.utils import get_notebook_image_path, load_parquet, try_load_parquet

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
        """
### El problema

En el dataset, los préstamos con **tasas altas tienen más defaults**. Pero, ¿subir la tasa
*causa* más defaults? ¿O simplemente le cobramos más a quienes ya eran riesgosos?

**Correlación**: "Los préstamos con tasa >20% tienen 24% de default"
**Causalidad**: "Si subimos la tasa 1pp a un cliente, su probabilidad de default sube +0.787pp"

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

> **+1 punto porcentual en tasa de interés → +0.787pp en probabilidad de default**

Esto tiene implicación económica directa: por cada punto de tasa que subes,
pierdes ~0.8% más de préstamos al default. La política óptima seleccionada
(regla `high_plus_medium_positive`) genera un valor neto de **$5.857M** en
pérdidas evitadas, enfocando intervenciones donde el efecto causal es mayor.
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
- **DoWhy** aporta el marco de identificación causal (supuestos, DAG y refutación conceptual).
- **EconML (DML/Causal Forest)** estima efectos heterogéneos con flexibilidad no lineal en tabular.

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

cate_df = load_parquet("cate_estimates")
segment_summary = load_parquet("causal_policy_segment_summary")
grade_summary = load_parquet("causal_policy_grade_summary")
rule_selected = load_parquet("causal_policy_rule_selected")
rule_candidates = load_parquet("causal_policy_rule_candidates")
simulation = load_parquet("causal_policy_simulation")

selected = rule_selected.iloc[0]

kpi_row(
    [
        {"label": "Regla elegida", "value": str(selected.get("rule_name", "N/D"))},
        {"label": "Action rate", "value": f"{selected.get('action_rate', 0) * 100:.1f}%"},
        {"label": "Valor neto total", "value": f"${selected.get('total_net_value', 0):,.0f}"},
        {"label": "Reducción pérdida", "value": f"${selected.get('total_loss_reduction', 0):,.0f}"},
    ]
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
    fig = px.box(
        cate_df.sample(min(120000, len(cate_df)), random_state=27),
        x="grade",
        y="cate",
        title="CATE por grade",
        labels={"grade": "Grade", "cate": "CATE"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Propósito: comparar sensibilidad causal por grade. Insight: algunos segmentos concentran mayor potencial de reducción "
        "de default ante ajuste de tasa."
    )

st.subheader("2) Impacto de política por segmento")
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

with st.expander("Muestra de simulación contrafactual por préstamo"):
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
    st.dataframe(
        simulation[cols].sample(min(120, len(simulation)), random_state=3),
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
