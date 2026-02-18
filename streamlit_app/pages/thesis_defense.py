"""Mapa integrado de métodos para riesgo de crédito."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import next_page_teaser
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import format_number, format_pct, load_json, load_parquet

st.title("🧩 Mapa Integrado de Métodos")
st.caption(
    "Síntesis de cómo se complementan machine learning, estadística, análisis causal "
    "e investigación de operaciones en el pipeline."
)
st.markdown(
    """
Esta página funciona como puente entre módulos: no evalúa una sola técnica, sino cómo se encadenan para transformar
datos históricos en decisiones de riesgo con impacto económico y regulatorio. El objetivo es mostrar que el valor del
proyecto no está en una métrica aislada, sino en la **complementariedad metodológica** entre predicción, incertidumbre,
causalidad, optimización y cumplimiento IFRS9.
"""
)

summary = load_json("pipeline_summary")
model_cmp = load_json("model_comparison")
conformal = load_json("conformal_policy_status", directory="models")
governance = load_json("modeva_governance_status", directory="models")

pipeline = summary.get("pipeline", {})
final = model_cmp.get("final_test_metrics", {})
rule = load_parquet("causal_policy_rule_selected").iloc[0]
ifrs9 = load_parquet("ifrs9_scenario_summary")

kpi_row(
    [
        {"label": "AUC ML", "value": f"{final.get('auc_roc', 0):.4f}"},
        {"label": "Cobertura 90%", "value": format_pct(conformal.get("coverage_90", 0))},
        {"label": "C-index RSF", "value": f"{summary.get('survival', {}).get('rsf_concordance', 0):.4f}"},
        {"label": "Valor causal neto", "value": format_number(rule.get("total_net_value", 0), prefix="$")},
        {"label": "Retorno robusto", "value": format_number(pipeline.get("robust_return", 0), prefix="$")},
        {"label": "Gobernanza", "value": "OK" if governance.get("overall_pass", False) else "Revisión"},
    ],
    n_cols=3,
)

st.subheader("Qué aporta cada disciplina")
methods = pd.DataFrame(
    [
        {
            "Disciplina": "Machine Learning",
            "Técnica principal": "CatBoost calibrado + SHAP",
            "Pregunta que responde": "¿Qué préstamos tienen mayor probabilidad de default?",
            "Artefacto": "model_comparison.json / shap_summary.parquet",
            "Valor para riesgo": "Priorización y explicación de riesgo individual",
        },
        {
            "Disciplina": "Estadística de incertidumbre",
            "Técnica principal": "Conformal Mondrian",
            "Pregunta que responde": "¿Con qué banda de confianza estimamos la PD?",
            "Artefacto": "conformal_intervals_mondrian.parquet",
            "Valor para riesgo": "Cobertura empírica y control de sobreconfianza",
        },
        {
            "Disciplina": "Series + Supervivencia",
            "Técnica principal": "Forecasting + KM/Cox/RSF",
            "Pregunta que responde": "¿Cómo evoluciona el riesgo en el tiempo?",
            "Artefacto": "time_series.parquet / lifetime_pd_table.parquet",
            "Valor para riesgo": "Forward-looking y horizonte de provisiones",
        },
        {
            "Disciplina": "Inferencia causal",
            "Técnica principal": "CausalForestDML + policy learning",
            "Pregunta que responde": "¿Qué acciones cambian realmente el riesgo?",
            "Artefacto": "causal_policy_rule_selected.parquet",
            "Valor para riesgo": "Intervenciones con impacto económico estimado",
        },
        {
            "Disciplina": "Investigación de operaciones",
            "Técnica principal": "Optimización robusta (Pyomo/HiGHS)",
            "Pregunta que responde": "¿Cómo asignar capital bajo incertidumbre?",
            "Artefacto": "portfolio_robustness_frontier.parquet",
            "Valor para riesgo": "Trade-off explícito retorno vs robustez",
        },
    ]
)
st.dataframe(methods, use_container_width=True, hide_index=True)

st.subheader("Cadena de valor analítica")
st.markdown(
    """
Para evitar interpretaciones engañosas, aquí no mezclamos en una sola escala números de naturaleza distinta
(`AUC` entre 0 y 1 frente a impactos en millones de USD). Se muestran por separado:
1. **Calidad técnica** del sistema (predicción, cobertura, tiempo-a-evento).
2. **Impacto económico/regulatorio** (retorno robusto, valor causal, ECL IFRS9).
"""
)
st.markdown(
    """
La lectura correcta de esta cadena es secuencial y no decorativa. Primero verificamos que el bloque técnico sea confiable:
si el ranking de riesgo no separa bien (`AUC`), si la incertidumbre no cubre lo prometido (`coverage`) o si el horizonte
temporal no discrimina bien (`C-index`), cualquier cálculo económico posterior queda expuesto a error estructural.
Solo después tiene sentido leer los impactos en valor neto, asignación de capital y provisiones IFRS9.
"""
)

tech_chain = pd.DataFrame(
    [
        {"métrica": "AUC OOT", "valor": final.get("auc_roc", 0.0), "bloque": "Predicción"},
        {"métrica": "Cobertura 90%", "valor": conformal.get("coverage_90", 0.0), "bloque": "Incertidumbre"},
        {"métrica": "C-index RSF", "valor": summary.get("survival", {}).get("rsf_concordance", 0.0), "bloque": "Horizonte"},
    ]
)
value_chain = pd.DataFrame(
    [
        {"etapa": "Retorno robusto", "valor_usd": pipeline.get("robust_return", 0.0), "tipo": "Impacto económico"},
        {"etapa": "Valor causal neto", "valor_usd": float(rule.get("total_net_value", 0.0)), "tipo": "Impacto económico"},
        {"etapa": "IFRS9 baseline", "valor_usd": float(ifrs9[ifrs9["scenario"] == "baseline"]["total_ecl"].iloc[0]), "tipo": "Impacto regulatorio"},
        {"etapa": "IFRS9 severe", "valor_usd": float(ifrs9[ifrs9["scenario"] == "severe"]["total_ecl"].iloc[0]), "tipo": "Impacto regulatorio"},
    ]
)

col_a, col_b = st.columns(2)
with col_a:
    fig = px.bar(
        tech_chain,
        x="métrica",
        y="valor",
        color="bloque",
        title="Calidad técnica por bloque",
        labels={"métrica": "", "valor": "Valor"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390, showlegend=True)
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Propósito: validar que cada bloque técnico cumpla su función antes de monetizar resultados. "
        "Insight: cuando AUC, cobertura y C-index se mantienen en niveles consistentes, la cadena de decisión "
        "aguas abajo es más defendible."
    )
with col_b:
    fig = px.bar(
        value_chain,
        x="etapa",
        y="valor_usd",
        color="tipo",
        barmode="group",
        title="Impacto económico/regulatorio (USD)",
        labels={"etapa": "", "valor_usd": "USD"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Propósito: traducir desempeño analítico a impacto económico y regulatorio. "
        "Insight: retorno robusto y valor causal muestran creación de valor, mientras IFRS9 refleja carga prudencial "
        "bajo escenarios macro."
    )

st.dataframe(
    pd.DataFrame(
        [
            {
                "Elemento": "AUC / Cobertura / C-index",
                "Lectura": "Miden calidad técnica del sistema en separación, incertidumbre y horizonte temporal.",
                "Por qué importa": "Sin calidad técnica, el impacto económico downstream no es confiable.",
            },
            {
                "Elemento": "Retorno robusto y valor causal",
                "Lectura": "Cuantifican creación de valor de política bajo incertidumbre y con intervenciones específicas.",
                "Por qué importa": "Conectan modelado con decisiones rentables y defendibles.",
            },
            {
                "Elemento": "ECL IFRS9 baseline/severe",
                "Lectura": "Miden sensibilidad contable/regulatoria ante escenarios macro.",
                "Por qué importa": "Traducen resultados analíticos en requerimientos de provisión y capital.",
            },
        ]
    ),
    use_container_width=True,
    hide_index=True,
)
st.markdown(
    """
En conjunto, ambas vistas responden tres preguntas de comité de riesgo:
1. ¿El sistema técnico es creíble?
2. ¿Qué valor económico genera cuando se toma decisión bajo incertidumbre?
3. ¿Qué implicación regulatoria deja en provisiones y capital?
Ese puente técnico-negocio es la esencia de esta cadena de valor analítica.
"""
)

st.subheader("Matriz de complementariedad")
matrix = pd.DataFrame(
    [
        {"Módulo": "Historia de datos", "Alimenta a": "ML / Causal / OR", "Producto": "Segmentación y drivers base"},
        {"Módulo": "Modelos PD", "Alimenta a": "Conformal / OR / IFRS9", "Producto": "Probabilidades calibradas"},
        {"Módulo": "Conformal", "Alimenta a": "OR / IFRS9", "Producto": "Intervalos de incertidumbre"},
        {"Módulo": "Series de tiempo", "Alimenta a": "IFRS9", "Producto": "Escenarios forward-looking"},
        {"Módulo": "Supervivencia", "Alimenta a": "IFRS9", "Producto": "Estructura temporal de PD"},
        {"Módulo": "Causalidad", "Alimenta a": "OR / negocio", "Producto": "Reglas de intervención"},
        {"Módulo": "Optimización", "Alimenta a": "Comité de riesgo", "Producto": "Política de asignación"},
        {"Módulo": "Gobernanza", "Alimenta a": "Control interno", "Producto": "Validación y trazabilidad"},
    ]
)
st.dataframe(matrix, use_container_width=True, hide_index=True)

st.subheader("Diferenciación vs. ecosistema público")
st.markdown(
    """
Se analizaron **más de 60 notebooks públicos** en Kaggle sobre el mismo dataset de Lending Club
(5 versiones del dataset, múltiples autores). El panorama es claro:
"""
)
diff_data = pd.DataFrame(
    [
        {"Técnica": "EDA y visualización", "Kaggle (60+ notebooks)": "Ampliamente cubierto", "Este proyecto": "Cubierto + contexto macro + geografía"},
        {"Técnica": "Clasificación binaria (RF, XGBoost, LogReg)", "Kaggle (60+ notebooks)": "Estándar en ~80% de notebooks", "Este proyecto": "CatBoost + calibración Platt (ECE=0.0128)"},
        {"Técnica": "SHAP / explicabilidad", "Kaggle (60+ notebooks)": "1-2 notebooks en detalle", "Este proyecto": "Cubierto en NB03 + Streamlit"},
        {"Técnica": "Validación out-of-time", "Kaggle (60+ notebooks)": "Ninguno (todos usan random split)", "Este proyecto": "Split temporal 2007-2017 / 2017 / 2018-2020"},
        {"Técnica": "WOE/IV feature engineering", "Kaggle (60+ notebooks)": "Ninguno", "Este proyecto": "OptBinning con supervisión monotónica"},
        {"Técnica": "Calibración de probabilidades", "Kaggle (60+ notebooks)": "Ninguno", "Este proyecto": "Platt vs Isotonic vs Venn-Abers"},
        {"Técnica": "Conformal prediction", "Kaggle (60+ notebooks)": "Ninguno", "Este proyecto": "MAPIE Mondrian con cobertura garantizada"},
        {"Técnica": "Survival analysis", "Kaggle (60+ notebooks)": "Ninguno", "Este proyecto": "Cox PH + RSF para PD lifetime"},
        {"Técnica": "Inferencia causal", "Kaggle (60+ notebooks)": "Ninguno", "Este proyecto": "DML + Causal Forest (ATE + CATE)"},
        {"Técnica": "Portfolio optimization", "Kaggle (60+ notebooks)": "Ninguno (1 notebook con threshold simple)", "Este proyecto": "Pyomo/HiGHS robusta con uncertainty sets"},
        {"Técnica": "IFRS9 / ECL / staging", "Kaggle (60+ notebooks)": "Ninguno", "Este proyecto": "4 escenarios + sensibilidad + conformal SICR"},
        {"Técnica": "Predict-then-optimize", "Kaggle (60+ notebooks)": "Ninguno", "Este proyecto": "Pipeline completo PD → Conformal → Pyomo"},
    ]
)
st.dataframe(diff_data, use_container_width=True, hide_index=True, height=460)
st.info(
    "**Conclusión:** Las técnicas que definen este proyecto — conformal prediction, optimización robusta, "
    "causalidad, survival analysis, IFRS9 y el pipeline predict-then-optimize — no aparecen en ningún "
    "notebook público de Kaggle sobre este dataset. La contribución metodológica es genuinamente diferenciada."
)

st.markdown(
    """
**Mensaje final del proyecto:**
- No es un conjunto de notebooks aislados: es un sistema analítico coherente.
- Cada técnica aporta una perspectiva distinta del mismo problema de riesgo.
- La combinación mejora explicabilidad, decisión y gobernabilidad del proceso completo.
"""
)
st.markdown(
    """
Como historia completa, la lectura es esta: partimos de datos heterogéneos, construimos señal predictiva calibrada,
cuantificamos incertidumbre con garantías empíricas, identificamos palancas causales de intervención y finalmente tomamos
decisiones robustas de cartera bajo restricciones reales. La aportación del proyecto no es solo "predecir mejor", sino
demostrar cómo integrar técnicas poco combinadas en una misma cadena de valor para riesgo de crédito aplicado.
"""
)

next_page_teaser(
    "Historia de Datos",
    "Volver al inicio del recorrido analítico y navegar el pipeline completo.",
    "pages/data_story.py",
)
