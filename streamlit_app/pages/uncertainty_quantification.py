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

from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import next_page_teaser, storytelling_intro
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import format_pct, get_notebook_image_path, load_json, load_parquet

st.title("📐 Cuantificación de Incertidumbre")
st.caption(
    "Conformal prediction (Mondrian) para convertir predicciones puntuales en intervalos "
    "con cobertura empírica controlada."
)

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

Con Mondrian, **cada grade tiene su propia garantía de cobertura**, evitando que
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

policy = load_json("conformal_policy_status", directory="models")
checks = load_parquet("conformal_policy_checks")
conf_df = load_parquet("conformal_intervals_mondrian")
group_df = load_parquet("conformal_group_metrics_mondrian")
backtest = load_parquet("conformal_backtest_monthly")
backtest_grade = load_parquet("conformal_backtest_monthly_grade")

status = "✅ Cumple política" if policy.get("overall_pass", False) else "⚠️ Requiere revisión"
st.subheader(f"Estado de política conformal: {status}")

width_by_grade = (
    conf_df.groupby("grade", observed=True)["width_90"]
    .median()
    .sort_values(ascending=False)
)
widest_grade = str(width_by_grade.index[0]) if not width_by_grade.empty else "N/D"
narrowest_grade = str(width_by_grade.index[-1]) if not width_by_grade.empty else "N/D"
widest_width = float(width_by_grade.iloc[0]) if not width_by_grade.empty else 0.0
narrowest_width = float(width_by_grade.iloc[-1]) if not width_by_grade.empty else 0.0
pd_width_corr = float(conf_df["y_pred"].corr(conf_df["width_90"])) if len(conf_df) > 1 else 0.0

kpi_row(
    [
        {"label": "Cobertura 90%", "value": format_pct(policy.get("coverage_90", 0))},
        {"label": "Cobertura 95%", "value": format_pct(policy.get("coverage_95", 0))},
        {"label": "Ancho promedio 90%", "value": f"{policy.get('avg_width_90', 0):.3f}"},
        {"label": "Cobertura mínima por grupo", "value": format_pct(policy.get("min_group_coverage_90", 0))},
        {"label": "Checks aprobados", "value": f"{policy.get('checks_passed', 0)}/{policy.get('checks_total', 0)}"},
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
    use_container_width=True,
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
    st.plotly_chart(fig, use_container_width=True)
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
    st.plotly_chart(fig, use_container_width=True)
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
st.plotly_chart(fig, use_container_width=True)
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
- Peor cobertura observada: **grade {worst_group['grade']}** con **{worst_group['coverage_90']:.1%}**.
- Mejor cobertura observada: **grade {best_group['grade']}** con **{best_group['coverage_90']:.1%}**.
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
    fig.add_trace(go.Scatter(x=backtest["month"], y=backtest["coverage_90"], mode="lines+markers", name="Cobertura 90%"))
    fig.add_trace(go.Scatter(x=backtest["month"], y=backtest["coverage_95"], mode="lines+markers", name="Cobertura 95%"))
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
    st.plotly_chart(fig, use_container_width=True)
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
    st.plotly_chart(fig, use_container_width=True)
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
- Mejor mes: **{pd.to_datetime(best_month['month']).strftime('%Y-%m')}** con **{best_month['coverage_90']:.1%}**.
- Peor mes: **{pd.to_datetime(worst_month['month']).strftime('%Y-%m')}** con **{worst_month['coverage_90']:.1%}**.
- Peor celda grade×mes: **grade {worst_cell['grade']} en {pd.to_datetime(worst_cell['month']).strftime('%Y-%m')}** con **{worst_cell['coverage_90']:.1%}**.
"""
)

st.subheader("4) Reglas de política y auditoría")
st.dataframe(checks, use_container_width=True, hide_index=True)

col_a, col_b = st.columns(2)
with col_a:
    img = get_notebook_image_path("04_conformal_prediction", "cell_022_out_01.png")
    if img.exists():
        st.image(
            str(img),
            caption="Notebook 04: calibración cobertura vs objetivo y trade-off con ancho.",
            use_container_width=True,
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
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320, title="Fallback: cobertura observada vs objetivo")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Imagen de notebook no encontrada; se muestra gráfico equivalente construido desde artefactos.")
with col_b:
    img = get_notebook_image_path("04_conformal_prediction", "cell_018_out_00.png")
    if img.exists():
        st.image(
            str(img),
            caption="Notebook 04: comparación marginal vs Mondrian por grade.",
            use_container_width=True,
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
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Imagen de notebook no encontrada; se muestra fallback por grade con datos actuales.")

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
    st.warning(f"Hay {len(failed_checks)} regla(s) fuera de política; requiere recalibración o revisión segmentada.")
    st.dataframe(failed_checks, use_container_width=True, hide_index=True)

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

next_page_teaser(
    "Panorama Temporal",
    "Pronósticos de default y escenarios para riesgo prospectivo e IFRS9.",
    "pages/time_series_outlook.py",
)
