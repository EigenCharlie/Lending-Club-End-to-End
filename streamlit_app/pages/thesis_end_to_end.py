"""Narrativa completa del pipeline end-to-end."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.components.audience_toggle import audience_selector
from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import narrative_block, next_page_teaser
from streamlit_app.components.story_shell import (
    render_key_takeaway,
    render_page_feedback,
    render_page_header,
    render_section_checkpoint,
)
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import (
    format_number,
    format_pct,
    get_notebook_image_path,
    load_gpu_replay_summary,
    load_rapids_ifrs9_mc_tail_metrics,
    load_rapids_pd_benchmark_stage_table,
    load_rapids_insight_stage_table,
    load_rapids_stage_comparison,
    load_rapids_tradeoff_full_ab_status,
    load_rapids_tradeoff_full_policy,
    page_error_boundary,
    try_load_json,
    try_load_parquet,
)

st.title("🧭 Visión End-to-End")
st.caption(
    "Motivación, base metodológica, implementación en Lending Club y lectura de resultados "
    "con interpretación técnica y de negocio."
)
page_contract = get_page_contract("thesis_end_to_end")
render_page_header(page_contract)
render_key_takeaway(
    "Esta página integra el hilo completo: el valor del proyecto aparece cuando lees cómo cada módulo alimenta decisiones downstream, no cuando miras cada técnica aislada."
)
st.markdown(
    """
Esta página es el relato maestro del proyecto. Su propósito es conectar en una sola narrativa los componentes que normalmente
se muestran por separado: modelado predictivo, incertidumbre, causalidad, optimización e IFRS9. La lectura está diseñada como
un capítulo integrador: qué problema resolvemos, por qué esta combinación metodológica es relevante y cómo los resultados se
traducen en decisiones concretas.
"""
)
audience = audience_selector()

summary = try_load_json("pipeline_summary")
comparison = try_load_json("model_comparison")
policy = try_load_json("conformal_policy_status", directory="models")
eda = try_load_json("eda_summary")

pipeline = summary.get("pipeline", {})
final_metrics = comparison.get("final_test_metrics", {})
causal_summary = summary.get("causal", {})
causal_rule_df = try_load_parquet("causal_policy_rule_selected")
causal_rule = (
    causal_rule_df.iloc[0]
    if not causal_rule_df.empty
    else pd.Series(
        {
            "rule_name": causal_summary.get("selected_rule", "N/D"),
            "total_net_value": causal_summary.get("total_net_value", 0.0),
        }
    )
)
ifrs9_scenarios = try_load_parquet("ifrs9_scenario_summary")
robust_summary = try_load_parquet("portfolio_robustness_summary")
rapids_summary = load_gpu_replay_summary()
rapids_compare = load_rapids_stage_comparison()
rapids_ifrs9 = load_rapids_ifrs9_mc_tail_metrics()
rapids_insights = load_rapids_insight_stage_table()
rapids_full_ab = load_rapids_tradeoff_full_ab_status()
rapids_full_policy = load_rapids_tradeoff_full_policy()
pd_benchmark = load_rapids_pd_benchmark_stage_table()

narrative_block(
    audience,
    general=(
        "El problema no es solo predecir defaults: es tomar mejores decisiones de portafolio "
        "cuando existe incertidumbre del modelo."
    ),
    business=(
        "Este pipeline reduce brecha entre analítica y decisión: conecta score, provisiones IFRS9 "
        "y asignación de capital en una narrativa única."
    ),
    technical=(
        "Diseño central: `PD calibrada -> intervalos conformal -> optimización robusta`, "
        "complementado por series temporales, supervivencia y causalidad."
    ),
)

kpi_row(
    [
        {"label": "Préstamos analizados", "value": format_number(eda.get("n_loans", 0))},
        {"label": "AUC OOT", "value": f"{final_metrics.get('auc_roc', 0):.4f}"},
        {"label": "ECE", "value": f"{final_metrics.get('ece', 0):.4f}"},
        {"label": "Cobertura 90%", "value": format_pct(policy.get("coverage_90", 0))},
        {
            "label": "Retorno robusto",
            "value": format_number(pipeline.get("robust_return", 0), prefix="$"),
        },
        {
            "label": "Valor causal neto",
            "value": format_number(causal_rule.get("total_net_value", 0), prefix="$"),
        },
    ],
    n_cols=3,
)

if audience == "General":
    st.markdown(
        """
### Lectura para público general
Piensa este proyecto como una cadena de decisiones:
1. Entender bien los datos.
2. Estimar riesgo de cada préstamo.
3. Medir qué tan incierta es esa estimación.
4. Decidir cartera de forma prudente.
5. Traducirlo a provisiones y políticas accionables.
"""
    )
elif audience == "Negocio":
    st.markdown(
        """
### Lectura para negocio/comité de riesgo
La propuesta está orientada a gestión:
- **Calidad de score** para segmentar cartera.
- **Bandas de incertidumbre** para evitar sobreconfianza.
- **Optimización robusta** para cuantificar costo de prudencia.
- **IFRS9 + causalidad** para conectar regulación y acción comercial.
"""
    )
else:
    st.markdown(
        """
### Lectura técnica (modelo, teoría y supuestos)
- CatBoost se usa por robustez en tabular heterogéneo y manejo nativo de missing/categorías.
- Conformal Mondrian aporta cobertura empírica segmentada sin asumir normalidad.
- DoWhy + CausalForestDML permiten estimar efecto de intervención y heterogeneidad.
- OR robusta usa `PD_high` para proteger el objetivo en peor caso plausible.

Formulación simplificada del bloque robusto:
"""
    )
    st.latex(r"\max_x \sum_i x_i \cdot L_i \cdot (r_i - LGD_i \cdot PD_i)")
    st.latex(
        r"\text{s.a. } \sum_i x_i L_i \le B,\quad PD_i \in [PD_{low,i},PD_{high,i}],\quad \text{restricciones de riesgo}"
    )
    st.markdown(
        """
En modo robusto se penaliza retorno esperado para ganar estabilidad ante error de modelo.
"""
    )

st.markdown(
    """
En la práctica, estos módulos suelen vivir separados: el equipo de modelado reporta `AUC`, el equipo financiero calcula
provisiones y el equipo de originación toma decisiones comerciales con reglas heurísticas. Este proyecto propone lo contrario:
una **columna vertebral cuantitativa única** donde cada output se vuelve input del siguiente bloque.
La PD calibrada entrega una probabilidad interpretable; Conformal agrega un intervalo con cobertura empírica finita; la
optimización robusta usa ese intervalo para decidir asignación bajo peor caso plausible; causalidad estima dónde una acción
de precio cambia realmente el riesgo; y IFRS9 traduce todo a impacto contable/regulatorio. Esta integración, más que cualquier
métrica individual, es la contribución central del trabajo.
"""
)

if not rapids_compare.empty:
    st.markdown("### Carril complementario: replay RAPIDS sobre el proyecto real")
    st.markdown(
        """
Este capítulo ahora tiene un complemento importante. Ya no solo existe el pipeline end-to-end CPU promovido; también existe un **replay GPU controlado** sobre las etapas donde la aceleración sí vale la pena.

- `PD` y `LGD/EAD` se reejecutan sobre full-data como comparación honesta.
- `portfolio`, `tradeoff`, `A/B` y `CATE` usan `cuopt` nativo.
- `IFRS9` ganó una extensión nueva: Monte Carlo masivo con `cupy`, sin reemplazar el carril regulatorio canónico.
"""
    )
    gpu_story = rapids_compare[
        ["stage", "cpu_seconds", "gpu_seconds", "speedup_gpu_vs_cpu"]
    ].copy()
    st.dataframe(
        gpu_story.rename(
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
    st.info(
        f"Hallazgo estructural: la GPU gana sobre todo en OR e IFRS9 Monte Carlo. El speedup de Monte Carlo ya quedó en {rapids_ifrs9.get('speedup_gpu_vs_cpu', 0):.1f}x, mientras que el replay completo de PD sigue siendo más útil como benchmark comparativo que como aceleración end-to-end."
    )
    if not rapids_insights.empty:
        st.caption(
            "La siguiente capa RAPIDS ya no vive solo en el pipeline: la insight factory mostró que `cuGraph` sí abre análisis nuevos, mientras `cuDF ETL` y `cuML KMeans` no deben venderse como victorias automáticas."
        )
    if rapids_full_ab:
        st.caption(
            "También ya quedó medido el frontier OR a escala full. La GPU resolvió la escala, pero la policy seleccionada sigue colapsando a una variante casi no robusta; ese siguiente problema es económico, no computacional."
        )
    if rapids_full_policy:
        balanced = rapids_full_policy.get("selected_policy_balanced_robustness", {})
        st.caption(
            f"Cuando forzamos robustez explícita, el selector balanceado sube a `gamma={balanced.get('gamma', 'N/D')}`; aun así, el costo económico en A/B full sigue siendo alto. Eso convierte la política robusta en un problema de diseño, no de solver."
        )
    if not pd_benchmark.empty:
        st.caption(
            "PD también cambió de lectura: la comparación RAPIDS ya se divide en fit-only, HPO y full-stage para separar lo que realmente gana en GPU del overhead alrededor."
        )

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Motivación y contexto",
        "Diseño metodológico",
        "Resultados e interpretación",
        "Evidencia visual de notebooks",
        "Teoría y referencias",
    ]
)

with tab1:
    st.markdown(
        """
### ¿Qué problema resuelve este pipeline?
En crédito, un enfoque tradicional suele terminar en una probabilidad puntual (`PD`) y una métrica agregada (AUC).
Eso deja tres vacíos:
1. **No cuantifica incertidumbre** de la predicción por préstamo.
2. **No traduce incertidumbre a decisión** de asignación de capital.
3. **No integra temporalidad ni causalidad** para políticas de riesgo más accionables.

### Motivación aplicada a Lending Club
- Dataset histórico grande, heterogéneo y con cambios de régimen.
- Necesidad de separar lo que predice bien de lo que decide bien.
- Necesidad de explicar resultados para perfil técnico y de negocio.
"""
    )

    literature = pd.DataFrame(
        [
            {
                "Bloque": "Modelado tabular",
                "Base conceptual": "Gradient boosting para datos tabulares (CatBoost)",
                "Brecha que cubre": "Mejor discriminación con tratamiento robusto de variables heterogéneas.",
            },
            {
                "Bloque": "Calibración",
                "Base conceptual": "Platt/Isotonic para calidad probabilística",
                "Brecha que cubre": "Convertir buen ranking en probabilidades útiles para IFRS9 y pricing.",
            },
            {
                "Bloque": "Conformal prediction",
                "Base conceptual": "Garantías de cobertura finita (split/Mondrian conformal)",
                "Brecha que cubre": "Cuantificar incertidumbre sin asumir distribución paramétrica rígida.",
            },
            {
                "Bloque": "Inferencia causal",
                "Base conceptual": "DoWhy backdoor + CausalForestDML",
                "Brecha que cubre": "Pasar de correlaciones a políticas de intervención.",
            },
            {
                "Bloque": "Investigación de operaciones",
                "Base conceptual": "Optimización robusta bajo incertidumbre",
                "Brecha que cubre": "Decidir asignación de cartera con control explícito de downside.",
            },
        ]
    )
    st.dataframe(literature, width="stretch", hide_index=True)

with tab2:
    st.markdown("### Contrato del pipeline y artefactos")
    flow = pd.DataFrame(
        [
            {
                "Paso": "Datos y limpieza",
                "Artefacto principal": "data/interim/lending_club_cleaned.parquet",
                "Problema que resuelve": "Calidad y consistencia de base histórica.",
            },
            {
                "Paso": "Ingeniería de variables",
                "Artefacto principal": "data/processed/train_fe.parquet",
                "Problema que resuelve": "Señales predictivas interpretables (WOE/IV, ratios, buckets).",
            },
            {
                "Paso": "Modelado PD + calibración",
                "Artefacto principal": "models/pd_canonical.cbm + pd_canonical_calibrator.pkl",
                "Problema que resuelve": "Probabilidades útiles para decisión y provisiones.",
            },
            {
                "Paso": "Conformal Mondrian",
                "Artefacto principal": "data/processed/conformal_intervals_mondrian.parquet",
                "Problema que resuelve": "Intervalos de incertidumbre con cobertura empírica.",
            },
            {
                "Paso": "IFRS9 / sensibilidad",
                "Artefacto principal": "data/processed/ifrs9_scenario_summary.parquet",
                "Problema que resuelve": "Rango de provisiones bajo escenarios macro.",
            },
            {
                "Paso": "Causal policy",
                "Artefacto principal": "data/processed/causal_policy_rule_selected.parquet",
                "Problema que resuelve": "Definir acciones con efecto económico esperado.",
            },
            {
                "Paso": "Optimización robusta",
                "Artefacto principal": "data/processed/portfolio_robustness_summary.parquet",
                "Problema que resuelve": "Trade-off retorno vs robustez explícito.",
            },
        ]
    )
    st.dataframe(flow, width="stretch", hide_index=True)

    st.markdown(
        """
**Cadena funcional crítica del proyecto:** `PD calibrada -> [PD_low, PD_high] conformal -> set de incertidumbre ->
optimización robusta`.

Esta cadena es el puente entre analítica predictiva y decisión cuantitativa. Sin calibración, la PD no es confiable para
negocio; sin intervalos conformal, la optimización ignora incertidumbre; sin optimización robusta, la política final no
internaliza el costo de escenarios adversos.
"""
    )
    st.markdown("### Arquitectura técnica (Graphviz)")
    st.graphviz_chart(
        """
digraph Pipeline {
    rankdir=LR;
    graph [pad="0.2", nodesep="0.45", ranksep="0.75", fontsize=15, fontname="Arial"];
    node [shape=box, style="rounded,filled", fillcolor="#F6F8FB", color="#CBD5E1", fontname="Arial", fontsize=12];
    edge [color="#64748B", penwidth=1.4, arrowsize=0.75];

    data [label="Datos limpios\\n(train/cal/test)"];
    fe [label="Feature engineering\\n(WOE/IV + ratios)"];
    pd [label="PD model\\n(CatBoost + calibración)"];
    conf [label="Conformal\\n(Mondrian)"];
    ts [label="Series de tiempo"];
    surv [label="Supervivencia"];
    causal [label="Causalidad\\n(DoWhy + CausalForestDML)"];
    opt [label="Optimización robusta\\n(Pyomo/HiGHS)"];
    ifrs [label="IFRS9\\n(ECL por escenario)"];
    gov [label="Gobernanza\\n(drift/fairness/robustez)"];
    app [label="Streamlit + DuckDB + dbt + Feast"];

    data -> fe -> pd -> conf;
    conf -> opt;
    conf -> ifrs;
    ts -> ifrs;
    surv -> ifrs;
    causal -> opt;
    pd -> gov;
    conf -> gov;
    opt -> app;
    ifrs -> app;
    gov -> app;
}
""",
        width="stretch",
    )
    st.caption(
        "Propósito: mostrar acoplamiento de módulos. Insight: la calidad de decisión depende de la calidad del score, "
        "de la incertidumbre y de la capa de optimización/gobernanza de forma conjunta."
    )

with tab3:
    _severe_slice = ifrs9_scenarios[ifrs9_scenarios["scenario"] == "severe"]["total_ecl"] if not ifrs9_scenarios.empty and "severe" in ifrs9_scenarios["scenario"].values else pd.Series(dtype=float)
    _baseline_slice = ifrs9_scenarios[ifrs9_scenarios["scenario"] == "baseline"]["total_ecl"] if not ifrs9_scenarios.empty and "baseline" in ifrs9_scenarios["scenario"].values else pd.Series(dtype=float)
    severe = float(_severe_slice.iloc[0]) if not _severe_slice.empty else 0.0
    baseline = float(_baseline_slice.iloc[0]) if not _baseline_slice.empty else 0.0
    uplift = (severe / baseline - 1) if baseline else 0

    interpretation = pd.DataFrame(
        [
            {
                "Métrica": "AUC OOT",
                "Valor": f"{final_metrics.get('auc_roc', 0):.4f}",
                "Lectura técnica": "Capacidad de ranking del modelo PD fuera de muestra temporal.",
                "Lectura de negocio": "Mejor priorización de créditos más riesgosos.",
            },
            {
                "Métrica": "ECE",
                "Valor": f"{final_metrics.get('ece', 0):.4f}",
                "Lectura técnica": "Error de calibración promedio de probabilidades.",
                "Lectura de negocio": "Menor distorsión en pricing y provisiones.",
            },
            {
                "Métrica": "Cobertura 90%",
                "Valor": format_pct(policy.get("coverage_90", 0)),
                "Lectura técnica": "Cumplimiento empírico de banda de incertidumbre.",
                "Lectura de negocio": "Mayor confianza para decisiones prudenciales.",
            },
            {
                "Métrica": "Price of Robustness",
                "Valor": format_number(pipeline.get("price_of_robustness", 0), prefix="$"),
                "Lectura técnica": "Costo de usar peor caso (PD_high) frente a política puntual.",
                "Lectura de negocio": "Prima pagada por resiliencia en escenarios adversos.",
            },
            {
                "Métrica": "ECL uplift severe vs baseline",
                "Valor": format_pct(uplift),
                "Lectura técnica": "Sensibilidad de provisión ante estrés macro.",
                "Lectura de negocio": "Impacto de capital y planeación de reservas.",
            },
            {
                "Métrica": "Valor causal neto",
                "Valor": format_number(causal_rule.get("total_net_value", 0), prefix="$"),
                "Lectura técnica": "Ganancia esperada de regla causal seleccionada.",
                "Lectura de negocio": "Potencial de intervención rentable en pricing.",
            },
        ]
    )
    st.dataframe(interpretation, width="stretch", hide_index=True)

    compare = pd.DataFrame(
        [
            {
                "modo": "No robusto",
                "retorno": pipeline.get("nonrobust_return", 0.0),
                "tipo": "Retorno",
            },
            {"modo": "Robusto", "retorno": pipeline.get("robust_return", 0.0), "tipo": "Retorno"},
            {"modo": "Baseline IFRS9", "retorno": baseline, "tipo": "ECL"},
            {"modo": "Severe IFRS9", "retorno": severe, "tipo": "ECL"},
        ]
    )
    fig = px.bar(
        compare,
        x="modo",
        y="retorno",
        color="tipo",
        barmode="group",
        title="Magnitudes clave de decisión y provisión",
        labels={"modo": "", "retorno": "USD"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Propósito: contrastar magnitudes de retorno y provisión en una misma vista. Insight: optimizar cartera sin "
        "leer simultáneamente el costo IFRS9 puede generar decisiones parciales."
    )

    st.markdown("### Robustez por tolerancia de riesgo")
    st.dataframe(
        robust_summary[
            [
                "risk_tolerance",
                "baseline_nonrobust_return",
                "best_robust_return",
                "best_robust_funded",
                "price_of_robustness_pct",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    if audience == "General":
        st.markdown(
            """
**Lectura simple:** hay una prima por prudencia. A mayor protección frente a incertidumbre, menor retorno esperado,
pero también menor exposición a sorpresas negativas.
"""
        )
    elif audience == "Negocio":
        st.markdown(
            """
**Lectura de comité:** la tabla define una curva de política. Cada nivel de tolerancia implica:
- un retorno objetivo distinto,
- una escala distinta de originación (`funded`),
- y un costo explícito de robustez.
Esto facilita fijar apetito de riesgo con números auditables.
"""
        )
    else:
        st.markdown(
            """
**Lectura técnica:** `price_of_robustness_pct` resume la brecha entre solución nominal y robusta para cada restricción
de riesgo. En términos de optimización robusta, equivale al costo dual de proteger la factibilidad/objetivo bajo
incertidumbre en PD.
"""
        )

with tab4:
    st.markdown(
        """
Esta sección organiza la evidencia visual con lógica narrativa, no como una galería suelta.
Cada figura responde una pregunta: qué problema técnico resolvió, qué hallazgo cuantitativo produjo y por qué ese hallazgo
es relevante para negocio o gobernanza de riesgo.
"""
    )
    narrative_gallery = [
        {
            "stem": "03_pd_modeling",
            "file": "cell_013_out_00.png",
            "title": "Optimización de hiperparámetros (PD)",
            "question": "¿Cómo sabemos que el desempeño no es resultado de una configuración arbitraria?",
            "insight": (
                "La trayectoria de Optuna y la importancia de hiperparámetros muestran convergencia estable, con "
                "`learning_rate` como principal palanca de variación del AUC validado."
            ),
            "impact": (
                "Esta estabilidad reduce riesgo de sobreajuste y fortalece la credibilidad del score que alimenta "
                "Conformal, IFRS9 y optimización."
            ),
        },
        {
            "stem": "03_pd_modeling",
            "file": "cell_017_out_00.png",
            "title": "Explicabilidad SHAP",
            "question": "¿Qué variables están moviendo realmente la predicción de default?",
            "insight": (
                "Tasa de interés, plazo y calidad crediticia concentran el mayor impacto absoluto, en línea con teoría "
                "de riesgo de crédito y con los resultados de WOE/IV."
            ),
            "impact": (
                "La consistencia entre explicación estadística y lógica de negocio mejora adopción en comité y auditoría."
            ),
        },
        {
            "stem": "09_end_to_end_pipeline",
            "file": "cell_009_out_02.png",
            "title": "IFRS9 con rango conformal",
            "question": "¿Qué cambia cuando llevamos incertidumbre de PD a provisiones?",
            "insight": (
                "El rango entre escenario esperado y conservador es material, evidenciando que una sola cifra puntual de "
                "ECL puede subestimar riesgo de cola."
            ),
            "impact": (
                "Esto habilita planeación prudencial de reservas y conversación más realista sobre sensibilidad macro."
            ),
        },
        {
            "stem": "09_end_to_end_pipeline",
            "file": "cell_018_out_00.png",
            "title": "Dashboard integrado E2E",
            "question": "¿Qué valor añade integrar score, incertidumbre y decisión en una sola lectura?",
            "insight": (
                "La vista unificada hace explícito el trade-off entre retorno, cobertura conformal y costo de robustez, "
                "evitando decisiones tomadas por métricas aisladas."
            ),
            "impact": (
                "Convierte resultados técnicos en una narrativa accionable para estrategia de cartera y gobernanza."
            ),
        },
    ]

    for card in narrative_gallery:
        img = get_notebook_image_path(card["stem"], card["file"])
        if not img.exists():
            continue
        with st.container(border=True):
            st.markdown(f"#### {card['title']}")
            col_img, col_txt = st.columns([1.2, 1.0], gap="large")
            with col_img:
                st.image(str(img), caption=f"{card['stem']} | {card['file']}", width=760)
            with col_txt:
                st.markdown(f"**Pregunta que responde:** {card['question']}")
                st.markdown(f"**Insight principal:** {card['insight']}")
                st.markdown(f"**Relevancia práctica:** {card['impact']}")

    st.markdown(
        """
En conjunto, esta evidencia muestra por qué la propuesta no se limita a construir un clasificador:
la cadena completa **PD calibrada -> Conformal -> Optimización robusta -> IFRS9** crea una arquitectura de decisión
bajo incertidumbre que rara vez se implementa de forma integrada en proyectos de riesgo de crédito aplicados.
"""
    )

with tab5:
    st.markdown(
        """
### Referencias primarias (documentación y papers)
- CatBoost paper: https://arxiv.org/abs/1706.09516
- CatBoost docs oficiales: https://catboost.ai/en/docs/
- MAPIE docs: https://mapie.readthedocs.io/
- DoWhy docs: https://www.pywhy.org/dowhy/
- EconML docs: https://www.pywhy.org/EconML/
- Double Machine Learning (Chernozhukov et al.): https://www.nber.org/papers/w23564
- Generalized Random Forests / Causal Forest: https://arxiv.org/abs/1610.01271
- Pyomo docs: https://pyomo.readthedocs.io/
- IFRS 9 (estándar): https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/
"""
    )
    if audience == "General":
        st.markdown(
            """
### Marco conceptual (versión general)
1. **Predicción**: estimar probabilidad de default por préstamo.
2. **Incertidumbre**: reconocer que toda predicción tiene error.
3. **Decisión**: elegir cartera equilibrando riesgo y retorno.
4. **Regulación**: traducir esa decisión a provisiones IFRS9.
"""
        )
    elif audience == "Negocio":
        st.markdown(
            """
### Marco conceptual (versión negocio)
1. **Score calibrado** mejora pricing, límites y priorización comercial.
2. **Conformal** reduce sobre-aprobación por exceso de confianza en PD puntual.
3. **Optimización robusta** hace explícito el costo/beneficio del apetito de riesgo.
4. **Causalidad** habilita políticas activas de tasa con impacto económico esperado.
5. **IFRS9** conecta estrategia comercial con impacto contable y de capital.
"""
        )
    else:
        st.markdown(
            """
### Marco conceptual (versión técnica)
1. **CatBoost + calibración**: separación (AUC/KS) y probabilidad bien calibrada (Brier/ECE).
2. **Conformal Mondrian**: cobertura finita por subgrupo sin supuestos paramétricos fuertes.
3. **DoWhy + CausalForestDML**: identificación backdoor y estimación de CATE bajo supuestos de ignorabilidad/overlap.
4. **OR robusta**: maximin/peor-caso en intervalo de PD para controlar downside.
5. **IFRS9**: acople de PD 12m/lifetime + LGD/EAD en escenarios macro.
"""
        )

st.subheader("Conclusión de valor práctico")
st.markdown(
    """
Este stack end-to-end permite responder preguntas que un score aislado no puede responder:
1. ¿Qué tan riesgoso es este préstamo? (PD calibrada)
2. ¿Qué tan incierta es esa estimación? (Conformal)
3. ¿Qué decisión de cartera es más resiliente? (Optimización robusta)
4. ¿Qué impacto tiene en provisiones regulatorias? (IFRS9)
5. ¿Qué acción puede modificar causalmente el riesgo? (CATE / policy learning)
"""
)

render_section_checkpoint(
    "Checkpoint final del recorrido E2E",
    [
        "PD calibrada aporta señal probabilística utilizable.",
        "Conformal aporta incertidumbre cuantificada con cobertura controlada.",
        "Optimización e IFRS9 traducen la analítica a decisiones económicas y regulatorias.",
        "Gobernanza valida si el sistema completo sigue siendo operable.",
    ],
)
st.info(
    "Fairness conformal en profundidad (cobertura condicional por subgrupo, trade-offs y agenda online) "
    "se desarrolla en `research_landscape.py` y `thesis_contribution.py`. "
    "En especialización se reporta como señal de gobernanza, no como claim central."
)
render_page_feedback("thesis_end_to_end")

next_page_teaser(
    "Arquitectura y Linaje de Datos",
    "De dónde vienen los datos, cómo se transforman y por qué se crean datasets especializados.",
    "pages/data_architecture.py",
)
