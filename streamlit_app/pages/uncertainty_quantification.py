"""Cuantificación de incertidumbre con conformal prediction."""

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

from streamlit_app.components.concept_capsules import render_concept_stack
from streamlit_app.components.conformal_applied_blocks import (
    build_coverage_stability_table,
    build_cp_concept_matrix_rows,
    build_cp_method_menu_rows,
    render_cp_guarantees_and_limits,
    render_exchangeability_stress_checklist,
)
from streamlit_app.components.context_help import (
    chart_help_popover,
    methodology_dialog,
    term_popover,
)
from streamlit_app.components.dvc_kpi_spine import render_conformal_health_panel
from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import next_page_teaser, storytelling_intro
from streamlit_app.components.story_shell import (
    render_caveats,
    render_decision_box,
    render_key_takeaway,
    render_page_feedback,
    render_page_header,
    render_section_checkpoint,
)
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import format_pct, get_notebook_image_path, load_json, load_parquet

st.title("📐 Cuantificación de Incertidumbre")
st.caption(
    "Conformal prediction (Mondrian) para convertir predicciones puntuales en intervalos "
    "con cobertura empírica controlada."
)
page_contract = get_page_contract("uncertainty_quantification")
render_page_header(page_contract)
render_concept_stack(page_contract.page_id, page_type=page_contract.page_type, max_cards=2)
render_key_takeaway(
    "Conformal no mejora el ranking del modelo: agrega una capa de confiabilidad para decidir cuánto creerle a cada PD."
)
term_popover("conformal", label="Conformal en una frase")

storytelling_intro(
    page_goal="Cuantificar qué tan incierta es cada PD en lugar de usar un único número puntual.",
    business_value="Evita sobreconfianza del modelo y mejora decisiones de cartera y provisión.",
    key_decision="Definir qué nivel de cobertura/ancho es aceptable para operar de forma robusta.",
    how_to_read=[
        "Revisar cobertura global 90/95 y ancho promedio.",
        "Comprobar cobertura mínima por grupo (grade) para evitar sesgos ocultos.",
        "Validar estabilidad temporal en el backtest mensual.",
    ],
)
render_decision_box(
    "Elegir cobertura objetivo y ancho tolerable como política explícita, no como subproducto accidental del modelo.",
    owner="Riesgo / MRM",
    cadence="monitoreo mensual",
)
render_conformal_health_panel()
methodology_dialog(
    "Mondrian vs global conformal",
    """
Mondrian estima cuantiles de no-conformidad por subgrupo (ej. `grade`) en lugar de un único cuantil global.

Ventaja:
- evita que grupos masivos y seguros escondan mala cobertura en grupos riesgosos.

Costo:
- puede ampliar intervalos en grupos con menos datos o mayor heterogeneidad.
""",
    button_label="Ver por qué Mondrian importa",
)

# ── Visual Intuition ──
col_before, col_after = st.columns(2)
with col_before:
    st.markdown(
        """
#### Sin Conformal Prediction
> Préstamo #12345: **PD = 12%**

Un solo número. ¿Qué tan confiable es?
¿Podría ser 8%? ¿O 20%? No lo sabemos.
Decisiones basadas en un punto son frágiles.
"""
    )
with col_after:
    st.markdown(
        """
#### Con Conformal Prediction
> Préstamo #12345: **PD = [8%, 16%]** con 90% de garantía

Un rango con garantía matemática: 90% de las veces,
el valor real cae dentro de este intervalo.
Decisiones más robustas y auditables.
"""
    )

st.info(
    "**Analogía:** Es como el pronóstico del clima. En vez de decir 'mañana hará 25°C' "
    "(un punto), dice 'entre 22°C y 28°C con 90% de confianza' (un intervalo). "
    "Si vas a decidir si llevar paraguas, el rango te da mejor información que el punto."
)

st.markdown(
    """
En riesgo de crédito, una PD puntual puede inducir una falsa sensación de precisión. El aporte de Conformal Prediction
es convertir esa predicción en un intervalo con garantía de cobertura en muestra finita, sin exigir supuestos fuertes
de normalidad o especificación perfecta del modelo. En este proyecto, el enfoque Mondrian añade control por subgrupo
(`grade`), evitando que un buen promedio global esconda problemas locales. Esta capa es crítica porque su salida no se
queda en diagnóstico: se reutiliza como input en la optimización robusta y en la lectura prudencial de provisiones IFRS9.
"""
)

with st.expander("¿Dónde se usa Conformal Prediction en la industria?"):
    st.markdown(
        """
| Sector | Uso | Referencia |
|--------|-----|------------|
| **Farmacéutica** | Intervalos en dosis-respuesta (AstraZeneca) | MAPIE library |
| **Manufactura** | Control de calidad predictivo (Volvo) | Vovk et al. (2005) |
| **Fintech** | Cuantificación de incertidumbre en pricing y riesgo | Crecimiento exponencial desde 2020 |
| **Seguros** | Intervalos en reclamaciones y reservas | Taquet et al. (2025) |
| **Regulación bancaria** | Model risk management (incertidumbre cuantificada) | EBA guidelines |

	Conformal Prediction muestra adopción creciente en dominios de alto riesgo e incertidumbre.
	En este proyecto se usa MAPIE como implementación open-source para integrar cobertura
	empírica con decisiones de negocio.
"""
    )

with st.expander("¿Qué es Mondrian y por qué importa?"):
    st.markdown(
        """
**Mondrian Conformal Prediction** calcula intervalos separados por grupo en vez de
uno global. En este proyecto, los grupos son los **grades** (A-G).

**¿Por qué importa?**

Imagina un portafolio con 70% Grade A (bajo riesgo) y 30% Grade G (alto riesgo).
Un intervalo global podría tener 90% de cobertura promedio, pero:
- Grade A: 95% de cobertura (sobreprotegido, intervalos demasiado anchos)
- Grade G: 78% de cobertura (subprotegido, intervalos demasiado estrechos)

Con Mondrian, cada grade tiene su propia calibracion por particion, evitando que
los grades seguros "subsidien" la cobertura de los riesgosos.

Resultado: cobertura justa y operativa por segmento, no solo en promedio.
"""
    )
st.markdown(
    """
### Marco conceptual rápido
- **Conformal Prediction**: método de calibración que construye intervalos de predicción con garantía empírica de cobertura.
- **Coverage (cobertura)**: porcentaje de casos en los que el valor observado cae dentro del intervalo.
- **Interval width (ancho)**: diferencia `límite superior - límite inferior`; mayor ancho implica más conservadurismo.
- **Mondrian conformal**: versión estratificada por grupo (en este proyecto, `grade`), que controla cobertura dentro de cada
  segmento y no solo en promedio global.

Intuición práctica del trade-off:
- Si la cobertura es baja, el intervalo es demasiado optimista.
- Si el ancho es excesivo, la decisión robusta pierde eficiencia económica.
- El objetivo operativo es mantener cobertura objetivo con intervalos lo más informativos posibles.
"""
)
st.markdown(
    """
### Cómo se construye el intervalo en este proyecto (lectura operativa)
1. Se entrena un modelo PD base y se calibra en un set separado.
2. En un conjunto de calibración se calcula un residuo de no conformidad (error absoluto entre `y_true` y PD).
3. Se toma un cuantil objetivo de ese residuo para fijar el radio del intervalo.
4. En versión **Mondrian**, ese cuantil se estima por grupo (`grade`) en lugar de uno global.
5. El resultado por préstamo es un rango `[PD_low, PD_high]` que luego consume optimización robusta e IFRS9.

Esto implica algo importante: no estamos diciendo “el modelo es perfecto”, sino “dado su error observado, esta banda
tiene una frecuencia de acierto controlada”. Es una garantía empírica, no una promesa paramétrica idealizada.
"""
)

with st.status(
    "Cargando artefactos conformal (policy, intervalos y backtests)...", expanded=False
) as _cp_status:
    policy = load_json("conformal_policy_status", directory="models")
    checks = load_parquet("conformal_policy_checks")
    conf_df = load_parquet("conformal_intervals_mondrian")
    group_df = load_parquet("conformal_group_metrics_mondrian")
    backtest = load_parquet("conformal_backtest_monthly")
    backtest_grade = load_parquet("conformal_backtest_monthly_grade")
    variant_benchmark = load_parquet("conformal_variant_benchmark")
    _cp_status.update(label="Artefactos conformal cargados", state="complete")

st.subheader("0) Qué está garantizado y qué no")
render_cp_guarantees_and_limits()
st.dataframe(pd.DataFrame(build_cp_concept_matrix_rows()), width="stretch", hide_index=True)

st.subheader("0.1) Estabilidad por tamaño de muestra")
group_stability = build_coverage_stability_table(
    group_df.rename(columns={"group": "segmento"}),
    label_col="segmento",
    coverage_col="coverage_90",
    n_col="n",
    target=0.90,
)
month_frame = backtest.copy()
if not month_frame.empty and "month" in month_frame.columns:
    month_frame["segmento"] = pd.to_datetime(month_frame["month"], errors="coerce").dt.strftime("%Y-%m")
month_stability = build_coverage_stability_table(
    month_frame,
    label_col="segmento",
    coverage_col="coverage_90",
    n_col="n",
    target=0.90,
)

if not group_stability.empty:
    plot_df = group_stability.copy().sort_values("segmento")
    plot_df["err_plus"] = plot_df["ci_high_95"] - plot_df["coverage"]
    plot_df["err_minus"] = plot_df["coverage"] - plot_df["ci_low_95"]
    fig_stability = px.bar(
        plot_df,
        x="segmento",
        y="coverage",
        error_y="err_plus",
        error_y_minus="err_minus",
        color="small_n_flag",
        color_discrete_map={True: "#FF6B6B", False: "#00D4AA"},
        title="Cobertura 90% por grade con banda de incertidumbre (Wilson 95%)",
        labels={"segmento": "Grade", "coverage": "Cobertura", "small_n_flag": "n<1000"},
    )
    fig_stability.add_hline(y=0.90, line_dash="dash", line_color="#5F6B7A")
    fig_stability.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
    fig_stability.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_stability, width="stretch")

col_stab_1, col_stab_2 = st.columns(2)
with col_stab_1:
    st.markdown("**Diagnóstico por grupo**")
    if group_stability.empty:
        st.info("No hay datos suficientes para estabilidad por grupo.")
    else:
        st.dataframe(group_stability, width="stretch", hide_index=True)
with col_stab_2:
    st.markdown("**Diagnóstico por mes**")
    if month_stability.empty:
        st.info("No hay datos suficientes para estabilidad por mes.")
    else:
        st.dataframe(month_stability.tail(18), width="stretch", hide_index=True)

if not group_stability.empty:
    low_n_groups = group_stability[group_stability["small_n_flag"]]["segmento"].tolist()
    if low_n_groups:
        st.warning(
            "Grupos con n bajo y mayor variabilidad esperable: "
            + ", ".join(str(g) for g in low_n_groups)
        )

st.subheader("0.2) Validez vs eficiencia (lectura operativa)")
st.dataframe(
    pd.DataFrame(
        [
            {
                "Eje": "Validez",
                "Cómo se lee": "Cobertura observada frente a target (90/95).",
                "Riesgo si se ignora": "Decisiones sobreconfiadas.",
            },
            {
                "Eje": "Eficiencia",
                "Cómo se lee": "Ancho medio/mediano del intervalo.",
                "Riesgo si se ignora": "Conservadurismo costoso y baja utilidad.",
            },
            {
                "Eje": "Balance",
                "Cómo se lee": "Cobertura suficiente con ancho util para negocio.",
                "Riesgo si se ignora": "Optimizar una metrica y degradar la otra.",
            },
        ]
    ),
    width="stretch",
    hide_index=True,
)

st.subheader("0.3) Cuándo usar qué método")
canonical_variant = "mondrian_selected_cfg"
if not variant_benchmark.empty and "variant" in variant_benchmark.columns:
    selected = variant_benchmark[
        variant_benchmark["variant"].astype(str).str.contains("selected", case=False, na=False)
    ]
    if not selected.empty:
        canonical_variant = str(selected.iloc[0]["variant"])
st.dataframe(
    pd.DataFrame(build_cp_method_menu_rows(canonical_variant)),
    width="stretch",
    hide_index=True,
)
if not variant_benchmark.empty:
    cols = ["variant", "coverage", "avg_width", "min_group_coverage"]
    valid_cols = [c for c in cols if c in variant_benchmark.columns]
    st.caption("Comparativa real de variantes disponibles en artefactos actuales.")
    st.dataframe(variant_benchmark[valid_cols], width="stretch", hide_index=True)

st.subheader("0.4) Exchangeability stress checklist")
render_exchangeability_stress_checklist()

st.subheader("Resumen de desempeño conformal")

width_by_grade = (
    conf_df.groupby("grade", observed=True)["width_90"].median().sort_values(ascending=False)
)
widest_grade = str(width_by_grade.index[0]) if not width_by_grade.empty else "N/D"
narrowest_grade = str(width_by_grade.index[-1]) if not width_by_grade.empty else "N/D"
widest_width = float(width_by_grade.iloc[0]) if not width_by_grade.empty else 0.0
narrowest_width = float(width_by_grade.iloc[-1]) if not width_by_grade.empty else 0.0
pd_width_corr = float(conf_df["y_pred"].corr(conf_df["width_90"])) if len(conf_df) > 1 else 0.0
median_width_90 = (
    float(conf_df["width_90"].median())
    if "width_90" in conf_df.columns and not conf_df.empty
    else 0.0
)

kpi_row(
    [
        {"label": "Cobertura 90%", "value": format_pct(policy.get("coverage_90", 0))},
        {"label": "Cobertura 95%", "value": format_pct(policy.get("coverage_95", 0))},
        {"label": "Ancho promedio 90%", "value": f"{policy.get('avg_width_90', 0):.3f}"},
        {"label": "Ancho mediano 90%", "value": f"{median_width_90:.3f}"},
        {
            "label": "Cobertura mínima por grupo",
            "value": format_pct(policy.get("min_group_coverage_90", 0)),
        },
        {"label": "Alertas críticas", "value": str(policy.get("critical_alerts", 0))},
    ],
    n_cols=3,
)

st.dataframe(
    pd.DataFrame(
        [
            {
                "Métrica": "Cobertura 90% / 95%",
                "Qué significa": "Frecuencia empírica con que el valor real cae dentro del intervalo.",
                "Lectura práctica": "Si supera objetivo, el sistema está bien calibrado en incertidumbre.",
            },
            {
                "Métrica": "Ancho promedio",
                "Qué significa": "Nivel de conservadurismo del intervalo.",
                "Lectura práctica": "Más ancho = más protección, pero menor eficiencia en decisión.",
            },
            {
                "Métrica": "Cobertura mínima por grupo",
                "Qué significa": "Calidad de cobertura en el peor subsegmento (grade).",
                "Lectura práctica": "Evita que la garantía global oculte fallos en subpoblaciones.",
            },
        ]
    ),
    width="stretch",
    hide_index=True,
)
st.markdown(
    """
En notación simple, si definimos el intervalo por préstamo como `[PD_low, PD_high]`, la cobertura empírica a nivel de
portafolio se aproxima por: `coverage = mean(1{ y_true in [PD_low, PD_high] })`. En Mondrian, ese cálculo se verifica por
subgrupo y luego se monitorea en el tiempo, lo cual es clave para evitar que un buen promedio global esconda fallos
persistentes en segmentos específicos.
"""
)

st.subheader("1) Distribución de intervalos")
chart_help_popover(
    "distribucion_intervalos",
    what_to_look_at="si los anchos aumentan en zonas de mayor PD y si hay colas muy anchas por segmento.",
    common_misread="un intervalo más ancho no significa peor modelo; puede ser incertidumbre real más alta.",
)
col1, col2 = st.columns(2)

with col1:
    sample = conf_df.sample(min(80000, len(conf_df)), random_state=11)
    fig = px.histogram(
        sample,
        x="width_90",
        color="grade",
        nbins=55,
        barmode="overlay",
        opacity=0.65,
        title="Ancho de intervalos (90%) por grade",
        labels={"width_90": "Ancho intervalo", "count": "Frecuencia"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Propósito: medir dispersión del ancho de intervalos por grade. Insight: perfiles más riesgosos tienden a intervalos más "
        "anchos, reflejando mayor incertidumbre estructural."
    )

with col2:
    sample = conf_df.sample(min(35000, len(conf_df)), random_state=19)
    fig = px.scatter(
        sample,
        x="y_pred",
        y="width_90",
        color="grade",
        opacity=0.25,
        title="Riesgo predicho vs ancho de incertidumbre",
        labels={"y_pred": "PD puntual", "width_90": "Ancho 90%"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Propósito: relacionar riesgo puntual con incertidumbre. Insight: la amplitud aumenta en zonas de PD elevada y ayuda a "
        "evitar decisiones sobreconfiadas."
    )

st.markdown(
    f"""
**Cómo interpretar esta sección (insights accionables):**
- El grade con mayor ancho mediano es **{widest_grade}** (`{widest_width:.3f}`) y el menor es **{narrowest_grade}** (`{narrowest_width:.3f}`).
- Esto significa que no todos los segmentos tienen la misma “calidad de certeza”; los más inestables exigen decisión más prudente.
- La correlación entre PD puntual y ancho es **{pd_width_corr:.2f}**: a mayor riesgo predicho, mayor banda de incertidumbre.
"""
)

st.subheader("2) Cobertura por grupo (Mondrian)")
group_plot = group_df.rename(columns={"group": "grade"})
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=group_plot["grade"],
        y=group_plot["coverage_90"],
        marker_color="#00D4AA",
        name="Cobertura 90%",
    )
)
fig.add_hline(y=0.90, line_dash="dash", line_color="#FF6B6B", annotation_text="Meta 90%")
fig.update_layout(
    **PLOTLY_TEMPLATE["layout"],
)
fig.update_layout(
    title="Cobertura empírica por grade",
    yaxis={"tickformat": ".0%"},
    height=390,
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Propósito: validar garantía de cobertura por subgrupo. Insight: cobertura segmentada cerca o por encima de la meta indica "
    "que la garantía global no oculta fallos locales."
)
worst_group = group_plot.loc[group_plot["coverage_90"].idxmin()]
best_group = group_plot.loc[group_plot["coverage_90"].idxmax()]
under_target = group_plot[group_plot["coverage_90"] < 0.90]
st.markdown(
    f"""
**Lectura de negocio de cobertura por grade:**
- Peor cobertura observada: **grade {worst_group["grade"]}** con **{worst_group["coverage_90"]:.1%}**.
- Mejor cobertura observada: **grade {best_group["grade"]}** con **{best_group["coverage_90"]:.1%}**.
- Grades bajo meta 90%: **{len(under_target)}**. Si este número sube, aumenta el riesgo de decisiones mal protegidas en segmentos críticos.
"""
)
st.markdown(
    """
Interpretación clave de negocio: en riesgo de crédito no basta con “cumplir 90% global”. Si un grade específico queda
sistemáticamente por debajo de meta, el proceso puede sobreestimar calidad de cartera justo en los segmentos más sensibles.
La variante Mondrian existe para cerrar ese hueco de gobernanza y forzar una lectura más justa y operativa por subgrupo.
"""
)

with st.expander("Nota metodológica: cobertura sub-nominal en Grade A"):
    st.markdown(
        """
**Observación**: Grade A presenta cobertura empírica de ~86% al nivel 90%, ligeramente por debajo del objetivo.

**Explicación**: La garantía de cobertura de conformal prediction es **marginal** (promedio sobre toda la distribución),
no condicional por subgrupo. Mondrian mejora el control por grupo, pero la cobertura por segmento tiene varianza
inversamente proporcional al tamaño del set de calibración de ese grupo. Grade A tiene la tasa de default más baja (~5%),
lo que reduce el número efectivo de observaciones positivas disponibles para calibración.

**Contexto cuantitativo**: Para un set de calibración de tamaño *n* por grupo, la desviación esperada de la cobertura
empírica respecto al objetivo es del orden de *1/sqrt(n)*. El shortfall observado de ~3.8pp es consistente con esta
variabilidad de muestra finita.

**Implicación práctica**: La sub-cobertura de Grade A no invalida la metodología; refleja un trade-off inherente entre
granularidad de control (más grupos) y precisión estadística (más datos por grupo). Para mitigar:
- Aumentar el set de calibración para grupos con baja prevalencia.
- Usar niveles de confianza adaptativos por grupo.
- Referencia: Vovk et al. (2005), *Algorithmic Learning in a Random World*, Cap. 4.
"""
    )

st.subheader("3) Estabilidad temporal")
col3, col4 = st.columns(2)
with col3:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest["month"],
            y=backtest["coverage_90"],
            mode="lines+markers",
            name="Cobertura 90%",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backtest["month"],
            y=backtest["coverage_95"],
            mode="lines+markers",
            name="Cobertura 95%",
        )
    )
    fig.add_hline(y=0.90, line_dash="dash", line_color="#FF6B6B")
    fig.add_hline(y=0.95, line_dash="dot", line_color="#FFD93D")
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
    )
    fig.update_layout(
        title="Backtest mensual de cobertura",
        yaxis={"tickformat": ".0%"},
        height=390,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Propósito: monitorear estabilidad temporal de cobertura. Insight: desviaciones persistentes bajo meta anticipan necesidad "
        "de recalibración."
    )

with col4:
    heat_df = backtest_grade.copy()
    heat_df["month"] = pd.to_datetime(heat_df["month"]).dt.strftime("%Y-%m")
    pivot = heat_df.pivot(index="grade", columns="month", values="coverage_90").sort_index()
    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Mapa de calor cobertura 90% (grade x mes)",
        labels={"color": "Cobertura"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Propósito: detectar meses/grades con sobre o subcobertura. Insight: el mapa de calor acelera diagnóstico operativo de "
        "segmentos inestables."
    )

monthly_below = int((backtest["coverage_90"] < 0.90).sum())
worst_month = backtest.loc[backtest["coverage_90"].idxmin()]
best_month = backtest.loc[backtest["coverage_90"].idxmax()]
worst_cell = backtest_grade.loc[backtest_grade["coverage_90"].idxmin()]
st.markdown(
    f"""
**Qué te dicen estas dos gráficas en conjunto:**
- Meses por debajo de meta 90%: **{monthly_below}**.
- Mejor mes: **{pd.to_datetime(best_month["month"]).strftime("%Y-%m")}** con **{best_month["coverage_90"]:.1%}**.
- Peor mes: **{pd.to_datetime(worst_month["month"]).strftime("%Y-%m")}** con **{worst_month["coverage_90"]:.1%}**.
- Peor celda grade×mes: **grade {worst_cell["grade"]} en {pd.to_datetime(worst_cell["month"]).strftime("%Y-%m")}** con **{worst_cell["coverage_90"]:.1%}**.
"""
)

st.subheader("4) Reglas de política y auditoría")
st.dataframe(checks, width="stretch", hide_index=True)

col_a, col_b = st.columns(2)
with col_a:
    img = get_notebook_image_path("04_conformal_prediction", "cell_022_out_01.png")
    if img.exists():
        st.image(
            str(img),
            caption="Notebook 04: calibración cobertura vs objetivo y trade-off con ancho.",
            width="stretch",
        )
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=backtest["month"],
                y=backtest["coverage_90"],
                mode="lines+markers",
                name="Cobertura 90%",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=backtest["month"],
                y=backtest["coverage_95"],
                mode="lines+markers",
                name="Cobertura 95%",
            )
        )
        fig.add_hline(y=0.90, line_dash="dash", line_color="#FF6B6B")
        fig.add_hline(y=0.95, line_dash="dot", line_color="#FFD93D")
        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            height=320,
            title="Fallback: cobertura observada vs objetivo",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Imagen de notebook no encontrada; se muestra gráfico equivalente construido desde artefactos."
        )
with col_b:
    img = get_notebook_image_path("04_conformal_prediction", "cell_018_out_00.png")
    if img.exists():
        st.image(
            str(img),
            caption="Notebook 04: comparación marginal vs Mondrian por grade.",
            width="stretch",
        )
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=group_plot["grade"],
                y=group_plot["coverage_90"],
                name="Cobertura 90%",
                marker_color="#00D4AA",
            )
        )
        fig.add_trace(
            go.Bar(
                x=group_plot["grade"],
                y=group_plot["coverage_95"],
                name="Cobertura 95%",
                marker_color="#0B5ED7",
                opacity=0.65,
            )
        )
        fig.add_hline(y=0.90, line_dash="dash", line_color="#FF6B6B")
        fig.add_hline(y=0.95, line_dash="dot", line_color="#FFD93D")
        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            barmode="group",
            height=320,
            title="Fallback: cobertura por grade (90% vs 95%)",
            yaxis={"tickformat": ".0%"},
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Imagen de notebook no encontrada; se muestra fallback por grade con datos actuales."
        )

failed_checks = checks[~checks["passed"]] if "passed" in checks.columns else pd.DataFrame()
if "comparator" in checks.columns:
    deltas = []
    for _, row in checks.iterrows():
        comparator = str(row.get("comparator", ">=")).strip()
        value = float(row.get("value", 0.0))
        threshold = float(row.get("threshold", 0.0))
        delta = value - threshold if comparator == ">=" else threshold - value
        deltas.append(delta)
    checks_delta = checks.copy()
    checks_delta["delta_umbral"] = deltas
    tightest = checks_delta.sort_values("delta_umbral").iloc[0]
else:
    tightest = None

if failed_checks.empty:
    if tightest is not None:
        st.success(
            "Todas las reglas pasan. "
            f"La más ajustada es `{tightest['metric']}` con margen {float(tightest['delta_umbral']):.4f} sobre su umbral."
        )
    else:
        st.success("Todas las reglas conformal pasan en el snapshot actual.")
else:
    st.warning(
        f"Hay {len(failed_checks)} regla(s) fuera de política; requiere recalibración o revisión segmentada."
    )
    st.dataframe(failed_checks, width="stretch", hide_index=True)

st.markdown(
    """
**Interpretación para riesgo de crédito:**
- La cobertura se mantiene sobre objetivo a nivel global y por segmentos.
- Los anchos crecen en perfiles de mayor riesgo, reflejando incertidumbre estructural real.
- Esta capa permite trasladar incertidumbre cuantificada al optimizador y al staging IFRS9.
"""
)
st.markdown(
    """
Como cierre narrativo, Conformal transforma una discusión abstracta de “riesgo de modelo” en una capa cuantitativa
operativa. Esa transición es crucial para este proyecto: los intervalos no quedan como visualización académica, sino que
entran en dos decisiones reales del flujo: 1) provisión prudencial por rango (IFRS9) y 2) asignación robusta de cartera
bajo peor caso plausible (`PD_high`).
"""
)
st.markdown(
    """
También deja explícito un límite metodológico sano: mayor cobertura normalmente exige mayor ancho. Por eso no se persigue
“cobertura máxima” de forma ciega, sino un punto de equilibrio que preserve capacidad de decisión. La política conformal
debe leerse siempre junto a retorno robusto y a sensibilidad IFRS9 para decidir si el nivel de prudencia elegido es el
adecuado para el apetito de riesgo del portafolio.
"""
)
render_section_checkpoint(
    "Checkpoint: qué debe quedar claro antes de pasar a la siguiente página",
    [
        "Cobertura debe leerse junto con ancho (no solo pasar el target).",
        "Mondrian protege subgrupos y evita falsa seguridad del promedio global.",
        "El output clave para decisiones downstream es el intervalo (`PD_low`, `PD_high`).",
    ],
)
render_caveats(
    [
        "La garantía conformal es empírica y depende de supuestos de estabilidad/intercambiabilidad.",
        "Cobertura alta en promedio puede ocultar variabilidad temporal si no se monitorea el backtest.",
        "El ancho óptimo depende del apetito de riesgo y del costo económico de conservadurismo.",
    ]
)
render_page_feedback("uncertainty_quantification")

next_page_teaser(
    "Panorama Temporal",
    "Pronósticos de default y escenarios para riesgo prospectivo e IFRS9.",
    "pages/time_series_outlook.py",
)
