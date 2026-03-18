"""Provisiones IFRS9 con escenarios y sensibilidad."""

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

from streamlit_app.components.audience_toggle import audience_selector
from streamlit_app.components.context_help import methodology_dialog, term_popover
from streamlit_app.components.decision_panels import decision_checklist, tradeoff_panel
from streamlit_app.components.dvc_kpi_spine import render_global_kpi_spine
from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import narrative_block, next_page_teaser, storytelling_intro
from streamlit_app.components.story_shell import (
    render_caveats,
    render_decision_box,
    render_key_takeaway,
    render_page_feedback,
    render_page_header,
)
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import (
    format_number,
    get_notebook_image_path,
    load_rapids_ifrs9_correlated_metrics,
    load_rapids_ifrs9_mc_tail_metrics,
    page_error_boundary,
    try_load_parquet,
)

st.title("🏦 Provisiones IFRS9")
st.caption(
    "IFRS9 es la norma internacional que obliga a los bancos a reservar dinero hoy "
    "por las pérdidas que esperan tener mañana. "
    "Esta página cuantifica cuánto reservar (ECL) por tipo de préstamo y escenario económico, "
    "y cómo la incertidumbre del modelo afecta esa estimación."
)

page_contract = get_page_contract("ifrs9_provisions")
render_page_header(page_contract)
render_key_takeaway(
    "IFRS9 no es un cálculo aislado: es la traducción contable de PD, incertidumbre y horizonte temporal en provisiones defendibles."
)
term_popover("canónico", label="Snapshot canónico y consistencia IFRS9")

audience = audience_selector()

narrative_block(
    audience,
    general="IFRS9 es la norma contable internacional que obliga a los bancos a provisionar "
    "las pérdidas que esperan tener en sus préstamos, no solo las que ya ocurrieron. "
    "Esta página muestra cuánto se debe provisionar bajo diferentes escenarios.",
    business="Esta página traduce resultados analíticos a lenguaje contable y regulatorio. "
    "Muestra cómo la calidad de PD, la incertidumbre y el horizonte temporal impactan "
    "provisiones esperadas (ECL) por stage y escenario.",
    technical="ECL = PD x LGD x EAD x DF. Staging por PD thresholds + conformal width como SICR signal. "
    "4 escenarios con multiplicadores de PD y LGD. Sensibilidad bivariada PD x LGD.",
)

storytelling_intro(
    page_goal="Traducir modelos de riesgo a provisiones contables IFRS9 por stage y escenario.",
    business_value="Conecta analítica con impacto directo en P&L, reservas y solvencia.",
    key_decision="Definir nivel de provisión y sensibilidad a escenarios macro (baseline vs severe).",
    how_to_read=[
        "Entender Stage 1/2/3 y fórmula ECL.",
        "Comparar ECL baseline vs severe y shares por stage.",
        "Usar sensibilidad PD/LGD para evaluar resiliencia de capital.",
    ],
)
render_decision_box(
    "Definir baseline de provisión y buffer bajo severe usando métricas canónicas compartidas con el resto del pipeline.",
    owner="Finanzas / Riesgo",
    cadence="cierre mensual",
)
render_global_kpi_spine("ifrs9")
tradeoff_panel(
    "Trade-off IFRS9",
    upside="Mayor prudencia y resiliencia contable ante deterioro macro.",
    downside="Más provisión impacta P&L y métricas de rentabilidad en el corto plazo.",
    monitoring="ECL baseline, ECL severe, uplift severe, shares por stage y sensibilidad PD×LGD.",
    color="#FFF7ED",
)
methodology_dialog(
    "Cómo leer el uplift severe",
    """
`ifrs9.severe_uplift_pct` resume cuánto crece la provisión total al pasar de baseline a escenario severe.

Lectura:
- alto uplift -> alta sensibilidad del portafolio al shock macro y/o a la calidad de los segmentos.
- bajo uplift -> mayor resiliencia, o menor severidad relativa del escenario.

No reemplaza el análisis por stage ni por grade; es un KPI de síntesis.
""",
    button_label="Ver interpretación del uplift severe",
)

# ── IFRS9 for Non-Accountants ──
with st.expander("IFRS9 para no contadores — ¿qué es y por qué importa?", expanded=False):
    st.markdown(
        """
### ¿Qué es IFRS9?

**IFRS 9** (International Financial Reporting Standard 9) es la norma contable global que
desde enero 2018 obliga a las instituciones financieras a provisionar **pérdidas esperadas**,
no solo pérdidas ya incurridas. Aplica en más de 140 países.

### La fórmula central: Expected Credit Loss

$$ECL = PD \\times LGD \\times EAD \\times DF$$

| Componente | Significado | Ejemplo |
|:----------:|-------------|---------|
| **PD** | Probabilidad de que el cliente no pague | 5% |
| **LGD** | % del monto que se pierde si hay default | 40% |
| **EAD** | Monto expuesto al momento del default | $10,000 |
| **DF** | Factor de descuento a valor presente | ~0.95 |
| **ECL** | Pérdida esperada = provisión requerida | **$190** |

### Los 3 Stages (etapas de deterioro)

| Stage | Estado del préstamo | PD usada | Provisión |
|:-----:|---------------------|----------|-----------|
| **1** | Sin deterioro | PD a **12 meses** | Pérdida a 1 año |
| **2** | Deterioro significativo (SICR) | PD **lifetime** | Pérdida de vida completa |
| **3** | Default confirmado (90+ DPD) | PD ≈ **100%** | Pérdida total esperada |

### ¿Qué es SICR?

**Significant Increase in Credit Risk** — el trigger que manda un préstamo de Stage 1 a Stage 2.
Puede dispararse por: aumento significativo de PD, morosidad temprana, o (nuestra innovación)
**aumento del ancho del intervalo conformal**.

### Conexión con Basilea III

- **IFRS9** determina **provisiones** (pérdida esperada → reservas contables)
- **Basilea III** determina **capital regulatorio** (pérdida inesperada → colchón de capital)
- Ambos usan PD, LGD y EAD pero con horizontes y definiciones distintas
- Mejor PD calibrada → provisiones más precisas → menor volatilidad de capital
"""
    )

st.info(
    "**Innovación del proyecto:** Usamos el ancho del intervalo conformal (PD_high - PD_point) "
    "como señal adicional de SICR. Si la incertidumbre del modelo crece significativamente para "
    "un préstamo, eso puede indicar deterioro antes de que la PD puntual lo capture."
)

with page_error_boundary("Provisiones IFRS9"):
    scenarios = try_load_parquet("ifrs9_scenario_summary")
    scenario_grade = try_load_parquet("ifrs9_scenario_grade_summary")
    sensitivity = try_load_parquet("ifrs9_sensitivity_grid")
    input_quality = try_load_parquet("ifrs9_input_quality")
    ifrs9_mc = load_rapids_ifrs9_mc_tail_metrics()
    ifrs9_mc_correlated = load_rapids_ifrs9_correlated_metrics()
    ecl_comp = try_load_parquet("ifrs9_ecl_comparison")
    if ecl_comp.empty:
        baseline_by_grade = scenario_grade[scenario_grade["scenario"] == "baseline"].copy()
        if baseline_by_grade.empty:
            ecl_comp = pd.DataFrame(columns=["Grade", "ECL_Stage1", "ECL_Stage2", "Stage2/Stage1"])
        else:
            stage1_proxy = baseline_by_grade["total_ecl"] * (
                1.0 - baseline_by_grade["stage2_share"] - baseline_by_grade["stage3_share"]
            )
            stage2_proxy = baseline_by_grade["total_ecl"] * (
                baseline_by_grade["stage2_share"] + baseline_by_grade["stage3_share"]
            )
            ecl_comp = pd.DataFrame(
                {
                    "Grade": baseline_by_grade["grade"],
                    "ECL_Stage1": stage1_proxy.clip(lower=0.0),
                    "ECL_Stage2": stage2_proxy.clip(lower=0.0),
                }
            )
            ecl_comp["Stage2/Stage1"] = ecl_comp["ECL_Stage2"] / (ecl_comp["ECL_Stage1"] + 1e-9)

    if scenarios.empty:
        base = {"total_ecl": 0.0, "stage2_share": 0.0, "stage3_share": 0.0}
        severe = {"total_ecl": 0.0, "stage2_share": 0.0, "stage3_share": 0.0}
    else:
        base_rows = scenarios[scenarios["scenario"] == "baseline"]
        severe_rows = scenarios[scenarios["scenario"] == "severe"]
        base = base_rows.iloc[0] if not base_rows.empty else scenarios.iloc[0]
        severe = severe_rows.iloc[0] if not severe_rows.empty else scenarios.iloc[-1]

    if input_quality.empty:
        input_quality = pd.DataFrame([{"n_rows": 0, "pd_current_mean": 0.0, "pd_orig_mean": 0.0}])

kpi_row(
    [
        {"label": "ECL baseline", "value": format_number(base["total_ecl"], prefix="$")},
        {"label": "ECL severe", "value": format_number(severe["total_ecl"], prefix="$")},
        {"label": "Stage 2 baseline", "value": f"{base['stage2_share'] * 100:.1f}%"},
        {"label": "Stage 3 baseline", "value": f"{base['stage3_share'] * 100:.1f}%"},
        {"label": "PD promedio", "value": f"{(input_quality.iloc[0]['pd_current_mean'] * 100 if not input_quality.empty else 0.0):.1f}%"},
        {"label": "N préstamos IFRS9", "value": f"{int(input_quality.iloc[0]['n_rows']) if not input_quality.empty else 0:,}"},
    ],
    n_cols=3,
)

st.dataframe(
    pd.DataFrame(
        [
            {
                "Métrica": "Total ECL",
                "Significado técnico": "Pérdida esperada agregada considerando PD, LGD, EAD y descuento.",
                "Significado negocio": "Nivel de provisión contable requerido.",
            },
            {
                "Métrica": "Stage2/Stage1",
                "Significado técnico": "Cuánto se amplifica pérdida al pasar a horizonte lifetime.",
                "Significado negocio": "Sensibilidad de capital ante deterioro significativo de riesgo.",
            },
            {
                "Métrica": "Uplift en escenario severe",
                "Significado técnico": "Elasticidad de ECL ante shocks de PD/LGD.",
                "Significado negocio": "Impacto potencial en resultados y solvencia.",
            },
        ]
    ),
    width="stretch",
    hide_index=True,
)

st.markdown(
    """
Este módulo muestra cómo la capa de modelado se traduce en provisiones contables:
- Stage 1: pérdida a 12 meses.
- Stage 2: pérdida de vida remanente (SICR).
- Stage 3: exposición deteriorada.
"""
)

if ifrs9_mc:
    st.markdown("### Extensión nueva: IFRS9 Monte Carlo masivo en GPU")
    kpi_row(
        [
            {"label": "Loans", "value": format_number(float(ifrs9_mc.get("n_loans", 0)))},
            {"label": "Escenarios", "value": format_number(float(ifrs9_mc.get("n_scenarios", 0)))},
            {"label": "CPU", "value": f"{ifrs9_mc.get('cpu_seconds', 0):.2f}s"},
            {"label": "GPU", "value": f"{ifrs9_mc.get('gpu_seconds', 0):.2f}s"},
            {"label": "Speedup", "value": f"{ifrs9_mc.get('speedup_gpu_vs_cpu', 0):.1f}x"},
            {
                "label": "Diff medio relativo",
                "value": f"{ifrs9_mc.get('mean_rel_diff_total_ecl_pct', 0):.4f}%",
            },
        ],
        n_cols=3,
    )
    st.markdown(
        """
El pipeline canónico sigue usando sensibilidad determinista por escenarios pequeños porque es más simple de auditar y explicar.
La capa nueva RAPIDS añade un carril **Monte Carlo** para estudiar distribución completa de ECL:

- media, desviación y percentiles de cola,
- `expected shortfall`,
- miles de escenarios con shocks compartidos CPU/GPU para comparación justa.
"""
    )
    tail_df = pd.DataFrame(
        [
            {"Métrica": k, "CPU": v, "GPU": ifrs9_mc.get("gpu_tail", {}).get(k)}
            for k, v in ifrs9_mc.get("cpu_tail", {}).items()
        ]
    )
    if not tail_df.empty:
        st.dataframe(tail_df, width="stretch", hide_index=True)
    if ifrs9_mc_correlated:
        st.markdown("#### Variante RAPIDS más rica: shocks correlacionados")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Correlation profile": ifrs9_mc_correlated.get("correlation_profile"),
                        "Antithetic": ifrs9_mc_correlated.get("antithetic"),
                        "Scenarios": ifrs9_mc_correlated.get("n_scenarios"),
                        "CPU seconds": ifrs9_mc_correlated.get("cpu_seconds"),
                        "GPU seconds": ifrs9_mc_correlated.get("gpu_seconds"),
                        "Speedup": ifrs9_mc_correlated.get("speedup_gpu_vs_cpu"),
                        "Mean relative diff (%)": ifrs9_mc_correlated.get(
                            "mean_rel_diff_total_ecl_pct"
                        ),
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.markdown(
            """
La siguiente mejora metodológica ya quedó demostrada: introducir correlación entre shocks de `PD`, `LGD`, `EAD`
y descuento, junto con variates antitéticas, sigue dejando diferencias CPU/GPU prácticamente nulas.

Eso cambia la lectura de la sección RAPIDS:
- ya no es solo “más escenarios”;
- ahora es una extensión de provisiones con distribución de cola más defendible para investigación.
"""
        )
        st.caption(
            "Siguiente paso natural: calibrar esos perfiles de correlación con drivers macro y reportar colas por `grade`, `stage` y `term`."
        )

col_nb_img, col_nb_text = st.columns([3, 2])
with col_nb_img:
    img = get_notebook_image_path("09_end_to_end_pipeline", "cell_009_out_02.png")
    if img.exists():
        st.image(
            str(img),
            caption="Notebook 09: distribución de stages y rango ECL con señal conformal.",
        )
    else:
        stage_fallback = scenarios[
            ["scenario", "stage1_share", "stage2_share", "stage3_share"]
        ].copy()
        stage_long = stage_fallback.melt(
            id_vars=["scenario"],
            value_vars=["stage1_share", "stage2_share", "stage3_share"],
            var_name="stage",
            value_name="share",
        )
        fig = px.bar(
            stage_long,
            x="scenario",
            y="share",
            color="stage",
            title="Fallback: distribución de stages IFRS9 por escenario",
            labels={"scenario": "Escenario", "share": "Participación"},
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320, yaxis={"tickformat": ".0%"})
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Imagen de notebook no encontrada; se muestra fallback construido desde escenarios IFRS9."
        )
with col_nb_text:
    st.markdown(
        """
**¿Qué muestra esta imagen?**

La distribución de préstamos por **stage IFRS9** y el rango de ECL
asociado, incorporando la señal del intervalo conformal.

**Resultados clave:**
- La mayoría del portafolio permanece en **Stage 1** (sin deterioro),
  lo cual es esperado en un portafolio diversificado.
- La migración a Stage 2 captura préstamos con **SICR** — incluyendo
  aquellos detectados por aumento del ancho conformal.
- El rango de ECL (mínimo-máximo) muestra la **incertidumbre** en la
  provisión: no es un número fijo sino una banda.

**Insight de negocio:**
El conformal interval width como señal de SICR puede detectar deterioro
**antes** de que se refleje en morosidad observable, actuando como
sistema de alerta temprana para el comité de riesgo.
"""
    )

st.subheader("1) ECL por grade y stage")
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=ecl_comp["Grade"],
        y=ecl_comp["ECL_Stage1"],
        name="Stage 1 (12m)",
        marker_color="#00D4AA",
    )
)
fig.add_trace(
    go.Bar(
        x=ecl_comp["Grade"],
        y=ecl_comp["ECL_Stage2"],
        name="Stage 2 (vida)",
        marker_color="#FF6B6B",
    )
)
fig.update_layout(
    **PLOTLY_TEMPLATE["layout"],
    barmode="group",
    title="Expected Credit Loss por grade",
    yaxis_title="ECL (USD)",
    height=430,
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Propósito: separar provisión 12m vs lifetime por grade. Insight: Stage 2 concentra la mayor presión de capital en "
    "segmentos de mayor riesgo."
)

st.markdown(
    """
**Interpretación del gráfico ECL por grade:**
- **Grades A-C**: Stage 1 y Stage 2 tienen ECL relativamente bajo — son los segmentos de menor presión contable.
- **Grades D-G**: Stage 2 (lifetime) es significativamente mayor que Stage 1 (12m), reflejando la acumulación
  de riesgo cuando se provisiona a horizonte completo.
- La **diferencia entre barras** (Stage 2 vs Stage 1) es lo que impacta capital cuando un préstamo migra
  por SICR: cada migración genera un salto discreto en provisiones.
"""
)

if "Stage2/Stage1" in ecl_comp.columns:
    st.info(
        f"Multiplicador promedio Stage2/Stage1: **{ecl_comp['Stage2/Stage1'].mean():.1f}x**. "
        "La migración a Stage 2 impacta materialmente el capital contable."
    )

st.subheader("2) Escenarios macro: baseline a severe")
if scenarios.empty:
    st.info(
        "No hay `ifrs9_scenario_summary.parquet` disponible. Se omite comparación de escenarios."
    )
else:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            scenarios,
            x="scenario",
            y="total_ecl",
            color="scenario",
            title="ECL total por escenario",
            labels={"scenario": "Escenario", "total_ecl": "ECL total (USD)"},
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False, height=390)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Propósito: cuantificar sensibilidad macro de ECL total. Insight: el salto baseline->severe muestra vulnerabilidad "
            "de reservas ante estrés."
        )

    with col2:
        stage_long = scenarios.melt(
            id_vars=["scenario"],
            value_vars=["stage1_share", "stage2_share", "stage3_share"],
            var_name="stage",
            value_name="share",
        )
        fig = px.bar(
            stage_long,
            x="scenario",
            y="share",
            color="stage",
            title="Composición de stages por escenario",
            labels={"scenario": "Escenario", "share": "Participación"},
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        fig.update_layout(yaxis={"tickformat": ".0%"}, height=390)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Propósito: visualizar migración de stages por escenario. Insight: el aumento de Stage 2/3 explica gran parte del uplift "
            "de provisiones."
        )

st.markdown(
    """
**Lectura integrada de escenarios:**
- El escenario **baseline** representa condiciones normales del portafolio.
- El escenario **severo** simula un deterioro macroeconómico (aumentos de PD y LGD simultáneos).
- El **uplift** baseline→severe cuantifica la vulnerabilidad del portafolio: cuánto capital adicional
  se necesitaría bajo estrés.
- La **migración de stages** bajo estrés es el principal driver del aumento de ECL — más préstamos
  pasan de Stage 1 a Stage 2/3, activando provisiones lifetime.
"""
)

st.subheader("3) Heatmaps de sensibilidad")
col3, col4 = st.columns(2)
with col3:
    if sensitivity.empty:
        st.info("No hay `ifrs9_sensitivity_grid.parquet`; se omite heatmap PD x LGD.")
    else:
        sens_matrix = sensitivity.pivot_table(
            index="pd_mult",
            columns="lgd_mult",
            values="total_ecl",
            aggfunc="mean",
        )
        fig = px.imshow(
            sens_matrix,
            color_continuous_scale="Reds",
            title="ECL promedio por multiplicadores PD x LGD",
            labels={"x": "LGD mult", "y": "PD mult", "color": "ECL"},
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Propósito: medir elasticidad de ECL ante shocks de PD y LGD. Insight: permite construir mapas de materialidad para "
            "stress testing interno."
        )

with col4:
    if scenario_grade.empty:
        st.info("No hay `ifrs9_scenario_grade_summary.parquet`; se omite heatmap por grade.")
    else:
        scen_grade = scenario_grade.copy()
        heat = scen_grade.pivot(index="grade", columns="scenario", values="avg_ecl")
        fig = px.imshow(
            heat,
            color_continuous_scale="YlOrRd",
            title="ECL promedio por grade y escenario",
            labels={"x": "Escenario", "y": "Grade", "color": "ECL promedio"},
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Propósito: identificar segmentos más sensibles al escenario macro. Insight: grades bajos presentan mayor incremento "
            "de ECL relativo."
        )

st.markdown(
    """
**Lectura de los heatmaps:**
- El heatmap PD×LGD muestra la **elasticidad de ECL**: ¿cuánto cambia la provisión si PD o LGD se
  desvían de la estimación base? Las zonas más rojas indican combinaciones que disparan provisiones muy altas.
- El heatmap por grade×escenario revela qué **segmentos son más sensibles** al macro: grades bajos (E, F, G)
  muestran los mayores saltos de ECL, lo que los convierte en candidatos prioritarios para planes de
  contingencia y acciones preventivas.
- Estos mapas son herramientas directas de **stress testing interno** para el comité de riesgo.
"""
)

st.subheader("4) Definiciones IFRS9 usadas en el proyecto")
defs = pd.DataFrame(
    [
        {"Stage": "1", "Trigger": "Sin SICR", "PD usada": "12 meses", "Horizonte ECL": "12 meses"},
        {
            "Stage": "2",
            "Trigger": "SICR detectado",
            "PD usada": "Lifetime",
            "Horizonte ECL": "Vida remanente",
        },
        {
            "Stage": "3",
            "Trigger": "Deterioro / default",
            "PD usada": "≈1.0",
            "Horizonte ECL": "Pérdida total esperada",
        },
    ]
)
st.dataframe(defs, width="stretch", hide_index=True)

st.markdown(
    """
**Conexión con el pipeline:**
- Pronósticos y conformal enriquecen lectura forward-looking.
- Survival aporta estructura temporal de PD.
- El resultado IFRS9 cierra el puente entre ciencia de datos y requerimiento regulatorio.
"""
)
st.markdown(
    """
Como conclusión, IFRS9 deja de verse como un cálculo aislado y pasa a entenderse como salida natural de un sistema integrado.
Si la PD está mejor calibrada y la incertidumbre está explícitamente cuantificada, la provisión resultante es más defendible
técnicamente y más útil para planificación prudencial bajo escenarios macro.
"""
)
decision_checklist(
    "Checklist para comité IFRS9",
    [
        "Comparar ECL baseline vs severe y acordar buffer prudencial explícito.",
        "Revisar concentración de ECL por stage/grade para planes de mitigación.",
        "Confirmar que supuestos PD/LGD y señales SICR estén documentados y trazables.",
    ],
)
render_caveats(
    [
        "Las provisiones dependen de supuestos de escenario y multiplicadores PD/LGD, no solo del modelo base.",
        "El uso de señal conformal como apoyo SICR es útil pero no sustituye políticas regulatorias formales.",
        "Los KPIs agregados deben complementarse con lectura por stage y por segmento.",
    ]
)
render_page_feedback("ifrs9_provisions")

next_page_teaser(
    "Gobernanza del Modelo",
    "Monitoreo de drift, fairness, robustez y contrato de modelo.",
    "pages/model_governance.py",
)
