"""Panorama de investigacion: estado del arte y propuestas de publicacion."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from streamlit_app.components.context_help import methodology_dialog
from streamlit_app.components.narrative import next_page_teaser
from streamlit_app.components.story_shell import (
    render_caveats,
    render_key_takeaway,
    render_page_header,
)
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.utils import try_load_json


def _safe_page_link(page_path: str, label: str, icon: str) -> None:
    """AppTest-safe wrapper for page links outside multipage runtime."""
    try:
        st.page_link(page_path, label=label, icon=icon)
    except Exception:
        st.caption(f"{icon} {label} → {page_path}")


st.title("🔬 Panorama de Investigación")
st.caption(
    "Estado del arte, papers clave y propuestas de publicación "
    "derivadas de este proyecto de riesgo de crédito."
)
page_contract = get_page_contract("research_landscape")
render_page_header(page_contract)
render_key_takeaway(
    "Esta página debe leerse como registro maestro de posicionamiento académico: concentra el mapa temático y reduce duplicación entre los drafts de paper."
)
_causal_status = try_load_json("causal_effect_status", directory="models", default={})
_causal_ate = _causal_status.get("ate")
try:
    _causal_ate_text = f"{float(_causal_ate):+.3f}pp"
except Exception:
    _causal_ate_text = "ATE artefact-driven"

st.markdown(
    """
Esta página presenta el panorama académico que fundamenta nuestro pipeline
**predict-then-optimize con conformal prediction**. Para cada disciplina, identificamos
los papers seminales, el estado actual de la investigación, y cómo este proyecto
se posiciona respecto a la literatura existente.
"""
)
st.info(
    "Regla narrativa del proyecto: en `tesis_especializacion.py` fairness conformal se reporta "
    "como diagnóstico de gobernanza; la profundidad metodológica (cobertura condicional, trade-offs, "
    "online/adaptive CP) se concentra aquí."
)
methodology_dialog(
    "Cómo usar esta página (modo experto)",
    """
Uso recomendado para lector técnico:
- Usa la matriz temática como índice de gaps y claims antes de entrar a cada sección.
- Revisa las propuestas de publicación solo después de confirmar el posicionamiento frente a literatura.
- Toma esta página como registro maestro para evitar duplicar referencias y claims en cada draft.
""",
    button_label="Ver guía de lectura (experto)",
)

st.subheader("0) Matriz temática y de posicionamiento (registro maestro)")
theme_matrix = pd.DataFrame(
    [
        {
            "Tema": "ML + calibración",
            "Pregunta central": "Cómo pasar de ranking a probabilidad utilizable",
            "Gap dominante": "Calibración y UQ subreportadas en credit scoring",
            "Artefacto del proyecto": "PD calibrada + selección de calibrador",
            "Draft principal": "Paper 1 / Paper 2",
        },
        {
            "Tema": "Conformal Prediction (Mondrian)",
            "Pregunta central": "Cómo garantizar cobertura útil por subgrupo",
            "Gap dominante": "Poca evidencia en crédito real a gran escala",
            "Artefacto del proyecto": "Coverage global+grupo, backtest mensual, alerts",
            "Draft principal": "Paper 3",
        },
        {
            "Tema": "Conformal + Robust Optimization",
            "Pregunta central": "Cómo traducir incertidumbre en decisiones robustas",
            "Gap dominante": "Muy poca intersección aplicada en crédito",
            "Artefacto del proyecto": "Frontera robusta + price of robustness",
            "Draft principal": "Paper 1",
        },
        {
            "Tema": "IFRS9 E2E con UQ",
            "Pregunta central": "Cómo llevar incertidumbre a staging/ECL",
            "Gap dominante": "Pipelines IFRS9 con UQ distribution-free son raros",
            "Artefacto del proyecto": "ECL por escenario + SICR con conformal width",
            "Draft principal": "Paper 2",
        },
        {
            "Tema": "Causalidad / Survival / Forecasting",
            "Pregunta central": "Qué acción, cuándo y bajo qué régimen cambia el riesgo",
            "Gap dominante": "Integración con decisión robusta y provisión",
            "Artefacto del proyecto": "CATE + PD lifetime + escenarios",
            "Draft principal": "Tesis / extensiones futuras",
        },
    ]
)
st.dataframe(theme_matrix, width="stretch", hide_index=True)

st.subheader("0.1) Matriz de deduplicación entre drafts")
dedup_matrix = pd.DataFrame(
    [
        {
            "Elemento narrativo": "Fundamentos de conformal (Vovk 2005, garantía finita)",
            "Página canónica": "research_landscape",
            "Paper 1": "resumen corto + link",
            "Paper 2": "solo lo necesario para IFRS9",
            "Paper 3": "detalle completo",
        },
        {
            "Elemento narrativo": "Calibración (Platt/Isotonic/Venn-Abers)",
            "Página canónica": "model_laboratory + research_landscape",
            "Paper 1": "impacto en optimización",
            "Paper 2": "impacto en ECL/SICR",
            "Paper 3": "solo prerequisito",
        },
        {
            "Elemento narrativo": "Price of Robustness (Bertsimas & Sim)",
            "Página canónica": "portfolio_optimizer + research_landscape",
            "Paper 1": "detalle completo",
            "Paper 2": "mención si conecta a provisiones",
            "Paper 3": "no central",
        },
        {
            "Elemento narrativo": "Mondrian coverage por grupo y drift temporal",
            "Página canónica": "uncertainty_quantification + research_landscape",
            "Paper 1": "solo como input de robustez",
            "Paper 2": "solo como calidad de incertidumbre",
            "Paper 3": "detalle completo",
        },
    ]
)
st.dataframe(dedup_matrix, width="stretch", hide_index=True)
st.caption(
    "Regla práctica: evita reescribir teoría completa en cada draft; cada paper debe reutilizar posicionamiento y concentrarse en su novelty claim."
)

st.subheader("0.2) Pitfalls operativos de Conformal Prediction")
cp_pitfalls = pd.DataFrame(
    [
        {
            "Pitfall": "Confundir cobertura marginal con cobertura condicional exacta",
            "Riesgo operativo": "Sobreprometer garantia en cada observacion individual.",
            "Mitigacion": "Explicitar alcance de validez y monitorear por segmento/tiempo.",
        },
        {
            "Pitfall": "Fragmentacion de grupos en Mondrian",
            "Riesgo operativo": "Cuantiles inestables en grupos con n bajo.",
            "Mitigacion": "Usar min_group_size, fusion de grupos o particion jerarquica.",
        },
        {
            "Pitfall": "Romper exchangeability bajo drift",
            "Riesgo operativo": "Cobertura fuera de target pese a buen historico.",
            "Mitigacion": "Backtest mensual + alertas + politica de recalibracion/escalamiento.",
        },
        {
            "Pitfall": "Perseguir solo cobertura y olvidar ancho",
            "Riesgo operativo": "Intervalos poco utiles por conservadurismo excesivo.",
            "Mitigacion": "Optimizar balance validez-eficiencia con KPIs conjuntos.",
        },
    ]
)
st.dataframe(cp_pitfalls, width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. ML en Credit Scoring
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("1) Machine Learning en credit scoring")

st.markdown(
    """
El credit scoring moderno ha convergido en **gradient boosting** como familia dominante.
Una revisión sistemática de 63 papers (2018-2024) confirma que CatBoost, XGBoost y
LightGBM superan consistentemente a la regresión logística tradicional en AUC y KS,
especialmente con datos heterogéneos y features categóricas.

**Gap identificado**: la mayoría de papers se enfocan en métricas de discriminación
(AUC, F1) pero **ignoran la calibración** de las probabilidades y la
**cuantificación de incertidumbre**. Un modelo con AUC=0.72 que produce probabilidades
sesgadas es peligroso para decisiones de portafolio.
"""
)

with st.expander("Papers clave"):
    st.markdown(
        """
- **Ayari, Guetari & Kraiem (2025/2026)** — *Machine learning powered financial credit
  scoring: a systematic literature review*, Artificial Intelligence Review.
  Revisión 2018-2024 con 63 estudios y énfasis en brechas de interpretabilidad/fairness.
- **Lessmann et al. (2015)** — *Benchmarking state-of-the-art classification algorithms
  for credit scoring*, European Journal of Operational Research. Benchmark de 41 métodos.
- **Soni et al. (2024)** — *Latest Advancements in Credit Risk Assessment with ML and
  Deep Learning Techniques*, Cybernetics and Information Technologies.
- **Liu et al. (2026)** — *A two-stage ML method for personal credit risk scoring with
  fragmentary data*, Expert Systems with Applications.
"""
    )

# ══════════════════════════════════════════════════════════════════════════════
# 2. Calibración
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("2) Calibración de probabilidades")

st.markdown(
    """
La calibración transforma scores de ML en probabilidades que reflejan frecuencias
reales. Es un paso **imprescindible** antes de usar probabilidades en decisiones
financieras o como input para incertidumbre.
"""
)

cal_data = pd.DataFrame(
    [
        {
            "Método": "Platt Scaling (1999)",
            "Mecanismo": "Sigmoid: P(y=1) = 1/(1+exp(-az-b))",
            "Ventaja": "Suave, generalizable, 2 parámetros",
            "Limitación": "Asume relación sigmoide",
        },
        {
            "Método": "Isotonic Regression",
            "Mecanismo": "Step function monotónica no-paramétrica",
            "Ventaja": "Flexible, sin supuestos",
            "Limitación": "Overfitting con calibration sets pequeños",
        },
        {
            "Método": "Venn-Abers (Vovk & Petej, 2014)",
            "Mecanismo": "Dos isotonic (y=0, y=1) → intervalo [p_low, p_high]",
            "Ventaja": "Intervalos con validez probabilística",
            "Limitación": "Computacionalmente costoso, menos conocido",
        },
    ]
)
st.dataframe(cal_data, width="stretch", hide_index=True)

with st.expander("Papers clave"):
    st.markdown(
        """
- **Platt (1999)** — *Probabilistic Outputs for SVMs*, Advances in Large Margin Classifiers.
- **Zadrozny & Elkan (2002)** — *Transforming classifier scores into accurate multiclass
  probability estimates*, KDD.
- **Vovk & Petej (2014)** — *Venn-Abers Predictors*, UAI. Calibración con garantías de validez.
- **Bellini et al. (2024)** — *Practical Credit Risk and Capital Modeling*, Springer.
  Calibración en contexto regulatorio IFRS9/Basel.
- **Kull et al. (2017)** — *Beyond Temperature Scaling: Obtaining Well-Calibrated Multi-Class
  Probabilities with Dirichlet Calibration*, NeurIPS Workshop. Referencia moderna para calibración robusta.
"""
    )

# ══════════════════════════════════════════════════════════════════════════════
# 3. Conformal Prediction
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("3) Conformal Prediction")

st.markdown(
    """
Conformal Prediction (CP) es un framework de **cuantificación de incertidumbre
distribution-free** con garantías de cobertura en muestra finita. A diferencia de
los intervalos bootstrap (asintóticos) o bayesianos (dependientes del prior), CP solo
requiere **exchangeability** de los datos.

La variante **Split Conformal** usa un set de calibración separado para calcular
nonconformity scores, logrando eficiencia computacional sin sacrificar la garantía
teórica. **Mondrian Conformal** extiende la garantía a nivel de subgrupo (e.g.,
loan grade A, B, ..., G), asegurando cobertura condicional por categoría.
"""
)

st.info(
    "**Garantía formal**: P(Y ∈ C(X)) ≥ 1 - α para todo n finito, "
    "sin supuestos distribucionales. Solo requiere exchangeability."
)

with st.expander("Papers clave"):
    st.markdown(
        """
- **Vovk, Gammerman & Shafer (2005)** — *Algorithmic Learning in a Random World*,
  Springer. Libro fundacional de conformal prediction.
- **Romano, Patterson & Candès (2019)** — *Conformalized Quantile Regression*,
  NeurIPS. CQR para intervalos adaptativos.
- **Ding et al. (2023)** — *Class-Conditional Conformal Prediction with Many Classes*,
  NeurIPS. Mondrian para garantías group-conditional.
- **Angelopoulos & Bates (2023)** — *Conformal Prediction: A Gentle Introduction*.
  Tutorial accesible para practitioners.
- **Gibbs & Candès (2021)** — *Adaptive Conformal Inference Under Distribution Shift*.
  ACI para datos no-exchangeable (series temporales).
- **Plassier et al. (2024)** — *Probabilistic Conformal Prediction with Approximate
  Conditional Validity*, arXiv.
"""
    )

# ══════════════════════════════════════════════════════════════════════════════
# 4. Predict-then-Optimize
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("4) Predict-then-Optimize")

st.markdown(
    """
El paradigma tradicional en ML aplicado es **predict → decide**: entrenar un modelo
que minimiza error de predicción (MSE, log-loss) y luego usar esas predicciones como
input fijo para un optimizador. El problema: **minimizar error de predicción no
minimiza error de decisión**.

**Smart Predict-then-Optimize (SPO+)** de Elmachtoub & Grigas (2022) propone una
loss function que mide directamente el **costo de la decisión subóptima** causada
por el error de predicción. Cuando el problema downstream es un LP, SPO+ se puede
computar eficientemente.
"""
)

with st.expander("Papers clave"):
    st.markdown(
        """
- **Elmachtoub & Grigas (2022)** — *Smart "Predict, then Optimize"*,
  Management Science 68(1):9-26. Paper fundacional. SPO+ loss para LPs.
- **Mandi et al. (2023/2024)** — *Decision-Focused Learning: Foundations, State of the Art,
  Benchmark and Future Opportunities* (survey de referencia del área).
- **Donti, Amos & Kolter (2017)** — *Task-based End-to-end Model Learning in Stochastic
  Optimization*, NeurIPS. Diferenciación a través del optimizador.
- **Capitaine et al. (2025)** — *Online Decision-Focused Learning with Large Language Models*,
  arXiv. Evidencia reciente sobre aprendizaje secuencial orientado a decisión.
"""
    )

# ══════════════════════════════════════════════════════════════════════════════
# 5. Conformal + Optimization (our contribution)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("5) Conformal Prediction + Optimización Robusta")

st.markdown(
    """
La intersección de **conformal prediction** y **optimización robusta** es un área
emergente con muy pocos trabajos publicados. La idea central: usar los intervalos
conformal como **conjuntos de incertidumbre** para formulaciones robustas, en lugar
de los tradicionales conjuntos elipsoidales o box sets heurísticos.
"""
)

st.success(
    "**Nuestra contribución**: Este proyecto conecta Mondrian Conformal Prediction "
    "con optimización robusta de portafolio crediticio via Pyomo en un setting aplicado "
    "de crédito al consumo a gran escala. Los intervalos [PD_low, PD_high] por loan grade "
    "alimentan box uncertainty sets con garantía de cobertura finita para decisiones robustas."
)

with st.expander("Papers clave"):
    st.markdown(
        """
- **Johnstone et al. (2021)** — *Conformal Uncertainty Sets for Robust Optimization*,
  PMLR Vol. 152. Usa Mahalanobis distance como conformity score para generar conjuntos
  elipsoidales.
- **Patel, Rayan & Tewari (2024)** — *Conformal Contextual Robust Optimization*,
  AISTATS (oral). Extiende al setting contextual (condicional); conjuntos data-dependent.
- **Bao et al. (2025)** — *Optimal Model Selection for Conformalized Robust Optimization*,
  arXiv. Introduce CROMS para elegir modelos por riesgo de decisión robusta.
- **Kato (2024)** — *Conformal Predictive Portfolio Selection*, arXiv.
  Intervalos de predicción para retornos como input directo a selección de portafolio.
- **Bertsimas & Sim (2004)** — *The Price of Robustness*, Operations Research.
  Framework clásico de robust optimization con uncertainty budgets.
  Nuestra implementación cuantifica el "price of robustness" empíricamente
  comparando portafolios con PD_point vs PD_high.
"""
    )

# ══════════════════════════════════════════════════════════════════════════════
# 6. Causal ML
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("6) Inferencia Causal en crédito")

st.markdown(
    f"""
La correlación entre tasa de interés y default no implica causalidad — borrowers de
mayor riesgo reciben tasas más altas (selection bias). En el pipeline oficial usamos
**DoWhy** para identificación/refutación backdoor y **CausalForestDML** para efectos
heterogéneos, separando el claim causal canónico del notebook exploratorio.

En el snapshot oficial del proyecto: ATE estimado = +1pp en tasa → **{_causal_ate_text}
en probabilidad de default**, con heterogeneidad observable por grade y DTI.
"""
)

with st.expander("Papers clave"):
    st.markdown(
        """
- **Chernozhukov et al. (2018)** — *Double/Debiased ML for Treatment and Structural
  Parameters*, The Econometrics Journal. Framework DML con orthogonalization.
- **Athey & Wager (2019)** — *Estimating Treatment Effects with Causal Forests*,
  Annals of Statistics. Causal Forest para HTEs.
- **Causal Inference for Banking, Finance, and Insurance Survey (2023)** — arXiv.
  Backdoor adjustment, IVs, causal forests en finanzas.
- **Prescriptive Analytics for Sustainable Financial Systems (2024)** — MDPI.
  Framework causal-ML para evaluación de políticas crediticias.
"""
    )

# ══════════════════════════════════════════════════════════════════════════════
# 7. Survival Analysis
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("7) Análisis de Supervivencia en crédito")

st.markdown(
    """
El análisis de supervivencia estima **cuándo** un préstamo incumple, no solo si lo hará.
Esto es crítico para IFRS9 Stage 2, donde se requiere la **PD lifetime** (probabilidad
de default durante toda la vida del préstamo) para provisionar deterioro significativo.

**Cox PH** es el estándar semiparamétrico, pero las violaciones del supuesto de hazards
proporcionales (detectadas via Schoenfeld test) motivan el uso complementario de
**Random Survival Forests**, que no requiere este supuesto.
"""
)

with st.expander("Papers clave"):
    st.markdown(
        """
- **Time to Default Benchmark (2016)** — Journal of the Operational Research Society.
  Compara Cox PH, spline-based y mixture cure models; splines recomendados.
- **Bárcena Saavedra et al. (2024)** — *Probability of default for lifetime credit loss
  for IFRS 9 using machine learning competing risks survival analysis models*,
  Expert Systems with Applications.
- **Botha & Verster (2025)** — *Approaches for modelling the term-structure of default
  risk under IFRS 9: a tutorial using discrete-time survival analysis*, arXiv.
- **Ptak-Chmielewska & Kopciuszewski (2024)** — *Random survival forests and Cox
  regression in loss given default estimation*, Journal of Credit Risk.
"""
    )

# ══════════════════════════════════════════════════════════════════════════════
# 8. Time Series
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("8) Series de Tiempo en crédito")

st.markdown(
    """
El forecasting de tasas de default agregadas conecta el riesgo individual con
el riesgo sistémico. Modelos clásicos como ARIMA/ETS/Theta siguen siendo una
línea base fuerte para series cortas; híbridos STL+CatBoost y modelos globales
de panel aportan valor cuando existe suficiente estructura cross-series. En este
proyecto, las covariables macro o exógenas solo son defendibles si existe un
contrato explícito de futuro; sin ese artefacto, la narrativa oficial del
forecast debe ser univariada.

Los **intervalos conformal para time series** siguen siendo un área activa: la
violación de exchangeability en datos temporales degrada la cobertura, por lo que
intervalos estadísticos bien calibrados suelen ser la referencia oficial mientras
ACI, EnbPI y variantes online se evalúan como research backlog.
"""
)

with st.expander("Papers clave"):
    st.markdown(
        """
- **Schlembach et al. (2025)** — *Conformal multistep-ahead multivariate time-series
  forecasting*, Machine Learning (Springer). Método nmtCP para no-exchangeability
  y horizonte multistep.
- **Wang & Hyndman (2024)** — *Online conformal inference for multi-step time series
  forecasting* (working paper + paquete `conformalForecast`).
- **JANET (2024)** — *Joint Adaptive Conformal Inference and Transductive Calibration
  for multi-step time series forecasting*, arXiv.
- **Incorporating Macroeconomic Scenarios in Credit Loss Forecasting** —
  práctica IFRS9: desempleo, GDP y spreads como drivers de PD/LGD/EAD.
- **Nixtla (2023-2025)** — Ecosystem open-source: statsforecast, mlforecast,
  hierarchicalforecast. Modular, rápido, compatible con conformal.
"""
    )

# ══════════════════════════════════════════════════════════════════════════════
# 9. Publication Proposals
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("9) Propuestas de publicación")

st.markdown(
    """
Basándose en los resultados de este proyecto, identificamos tres propuestas concretas
de publicación que abordan gaps en la literatura existente.
"""
)

tab1, tab2, tab3 = st.tabs(["Paper 1: CP + Robust Opt", "Paper 2: IFRS9 E2E", "Paper 3: Mondrian"])

with tab1:
    st.markdown(
        """
### Conformal Prediction Intervals as Uncertainty Sets for Robust Credit Portfolio Optimization

**Target**: European Journal of Operational Research (EJOR)

**Abstract sketch**: Proponemos un framework que integra intervalos de Mondrian Conformal
Prediction como conjuntos de incertidumbre box para optimización robusta de portafolio
crediticio. A diferencia de los conjuntos elipsoidales de Johnstone et al. (2021),
nuestros box sets son group-conditional (por grade de riesgo) y tienen garantía de
cobertura finita sin supuestos distribucionales. Evaluamos en 1.35M préstamos de
Lending Club, cuantificando el trade-off retorno-robustez (price of robustness)
y demostrando que la protección conformal produce portafolios estables bajo estrés.

**Contribución clave**: Primer framework que conecta Mondrian CP → box uncertainty sets
→ Pyomo robust LP para asignación de crédito con cobertura garantizada por segmento.

**Metodología**: CatBoost PD → calibración probabilística (método seleccionado en pipeline) → MAPIE SplitConformalRegressor
(Mondrian by grade) → Box sets [PD_low, PD_high] → Pyomo LP + HiGHS → Frontera de
robustez empírica.
"""
    )
    _safe_page_link(
        "pages/paper_1_cp_robust_opt.py",
        "Abrir workspace completo del Paper 1",
        "🧪",
    )

with tab2:
    st.markdown(
        """
### An End-to-End ML Pipeline for IFRS9 Compliance with Distribution-Free Uncertainty

**Target**: Journal of Banking & Finance

**Abstract sketch**: Presentamos un pipeline end-to-end que integra ML, conformal
prediction y optimización para compliance regulatorio IFRS9. El pipeline produce
ECL (Expected Credit Loss) con intervalos de incertidumbre: ECL_low y ECL_high
derivados de PD conformal × LGD × EAD. Introducimos el ancho del intervalo conformal
(PD_high - PD_point) como señal adicional de SICR (Significant Increase in Credit Risk),
complementando el criterio estándar de incremento en PD. Validamos en 277K préstamos
out-of-time (2018-2020) con controles formales de calidad y monitoreo temporal.

**Contribución clave**: Conformal interval width como señal SICR + ECL por rango para
lectura prudencial — ambos conceptos nuevos en la literatura IFRS9.

**Metodología**: PD calibrada → Conformal intervals → IFRS9 staging (con CP width como
señal) → ECL_point, ECL_low, ECL_high → Stress testing bajo escenarios macro.
"""
    )
    _safe_page_link(
        "pages/paper_2_ifrs9_e2e.py",
        "Abrir workspace completo del Paper 2",
        "🏦",
    )

with tab3:
    st.markdown(
        """
### Mondrian Conformal Prediction for Group-Conditional Credit Risk Coverage

**Target**: COPA Conference (Conformal Prediction) o NeurIPS Workshop on Distribution-Free UQ

**Abstract sketch**: Aplicamos Mondrian Conformal Prediction para obtener garantías de
cobertura condicional por segmento de riesgo (loan grades A-G) en un dataset de 1.35M
préstamos. Documentamos el trade-off entre granularidad de grupos y varianza de cobertura:
Grade A (baja tasa de default) exhibe under-coverage de ~3.8pp al 90%, dentro de la
tolerancia de muestra finita. Comparamos con Split Conformal marginal y Conformalized
Quantile Regression (CQR), mostrando que Mondrian provee intervalos más justos
operativamente (anchos diferentes por grade) a costa de mayor varianza en grupos pequeños.

**Contribución clave**: Primer estudio empírico a gran escala de Mondrian CP en credit
risk, con análisis detallado de coverage por subgrupo y recomendaciones prácticas.

**Metodología**: ProbabilityRegressor wrapper → MAPIE SplitConformalRegressor →
Mondrian by grade → Coverage validation (global + group-conditional) → Width analysis.
"""
    )
    _safe_page_link(
        "pages/paper_3_mondrian.py",
        "Abrir workspace completo del Paper 3",
        "📐",
    )

# ══════════════════════════════════════════════════════════════════════════════
# Reference Table
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Referencias completas")

refs = pd.DataFrame(
    [
        [
            "Elmachtoub & Grigas",
            2022,
            "Management Science",
            "SPO+ loss — predict-then-optimize fundacional",
        ],
        [
            "Vovk, Gammerman & Shafer",
            2005,
            "Springer (libro)",
            "Conformal prediction — teoría fundacional",
        ],
        ["Romano, Patterson & Candès", 2019, "NeurIPS", "Conformalized Quantile Regression"],
        [
            "Johnstone et al.",
            2021,
            "PMLR Vol. 152",
            "Conformal uncertainty sets para robust optimization",
        ],
        ["Patel et al.", 2024, "AISTATS", "Conformal contextual robust optimization"],
        ["Bao et al.", 2025, "arXiv", "CROMS para selección de modelo en CRO"],
        ["Kato", 2024, "arXiv", "Conformal predictive portfolio selection"],
        ["Ding et al.", 2023, "NeurIPS", "Mondrian (class-conditional) conformal prediction"],
        ["Chernozhukov et al.", 2018, "Econometrics Journal", "Double/Debiased ML"],
        ["Athey & Wager", 2019, "Annals of Statistics", "Causal Forests para treatment effects"],
        ["Bellini et al.", 2024, "Springer", "Practical Credit Risk and Capital Modeling"],
        [
            "Bertsimas & Sim",
            2004,
            "Operations Research",
            "Price of Robustness — robust optimization",
        ],
        ["Gibbs & Candès", 2021, "NeurIPS", "Adaptive Conformal Inference bajo distribution shift"],
        ["Ayari et al.", 2026, "AI Review (Springer)", "Revisión sistemática: 63 papers, GB > LR"],
        ["Soni et al.", 2024, "Cybernetics and IT", "Review reciente de ML/DL para credit risk"],
        [
            "Liu et al.",
            2026,
            "Expert Systems with Applications",
            "Credit scoring con datos fragmentarios",
        ],
        [
            "Bárcena Saavedra et al.",
            2024,
            "Expert Systems with Applications",
            "IFRS9 lifetime PD con competing risks",
        ],
        ["Botha & Verster", 2025, "arXiv", "Tutorial survival discreto para term-structure IFRS9"],
        [
            "Schlembach et al.",
            2025,
            "Machine Learning (Springer)",
            "Conformal multistep para series temporales",
        ],
        ["Wang & Hyndman", 2024, "arXiv / package", "Online conformal para multi-step forecasting"],
    ],
    columns=["Referencia", "Año", "Venue", "Relevancia para este proyecto"],
)
st.dataframe(refs, width="stretch", hide_index=True)

_safe_page_link(
    "pages/research_best_practices.py",
    "Abrir guía de buenas prácticas y herramientas para drafts",
    "🧰",
)

render_caveats(
    [
        "La literatura evoluciona rápido; este mapa debe actualizarse antes de enviar versiones finales.",
        "El objetivo es posicionamiento y foco narrativo, no exhaustividad bibliográfica absoluta.",
    ],
    title="Límites del panorama y mantenimiento del registro maestro",
)

next_page_teaser(
    "Paper 1: CP + Robust Opt",
    "Inicio del workspace de redacción con secciones completas y resultados exportables.",
    "pages/paper_1_cp_robust_opt.py",
)
