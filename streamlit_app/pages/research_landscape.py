"""Panorama de investigacion: estado del arte y propuestas de publicacion."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from streamlit_app.components.narrative import next_page_teaser

st.title("🔬 Panorama de Investigación")
st.caption(
    "Estado del arte, papers clave y propuestas de publicación "
    "derivadas de este proyecto de riesgo de crédito."
)

st.markdown(
    """
Esta página presenta el panorama académico que fundamenta nuestro pipeline
**predict-then-optimize con conformal prediction**. Para cada disciplina, identificamos
los papers seminales, el estado actual de la investigación, y cómo este proyecto
se posiciona respecto a la literatura existente.
"""
)

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
- **ML Credit Scoring Survey (2025)** — Revisión sistemática, Springer.
  63 papers analizados. Consenso: gradient boosting > logistic regression.
  Gap: poca atención a incertidumbre.
- **Lessmann et al. (2015)** — *Benchmarking state-of-the-art classification algorithms
  for credit scoring*, European Journal of Operational Research. Benchmark de 41 métodos.
- **Xia et al. (2017)** — *Boosted tree models for credit scoring*, Expert Systems with
  Applications. CatBoost y XGBoost en credit risk.
- **Credit Scoring Using ML and Deep Learning (2024)** — AIMS Press.
  Neural networks complementan pero no superan GB en datasets tabulares.
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
st.dataframe(cal_data, use_container_width=True, hide_index=True)

with st.expander("Papers clave"):
    st.markdown(
        """
- **Platt (1999)** — *Probabilistic Outputs for SVMs*, Advances in Large Margin Classifiers.
- **Zadrozny & Elkan (2002)** — *Transforming classifier scores into accurate multiclass
  probability estimates*, KDD.
- **Vovk & Petej (2014)** — *Venn-Abers Predictors*, UAI. Calibración con garantías de validez.
- **Bellini et al. (2024)** — *Practical Credit Risk and Capital Modeling*, Springer.
  Calibración en contexto regulatorio IFRS9/Basel.
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
- **Taquet et al. (2025)** — *MAPIE: an open-source library for distribution-free
  uncertainty quantification*. La librería que usamos.
- **Gibbs & Candès (2021)** — *Adaptive Conformal Inference Under Distribution Shift*.
  ACI para datos no-exchangeable (series temporales).
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
- **Mandi et al. (2024)** — *Decision-Focused Learning: Foundations, State of the Art,
  Benchmark and Future Opportunities*, JAIR. Survey completo del área.
- **Donti, Amos & Kolter (2017)** — *Task-based End-to-end Model Learning in Stochastic
  Optimization*, NeurIPS. Diferenciación a través del optimizador.
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
    "con optimización robusta de portafolio crediticio via Pyomo — una combinación "
    "no explorada en la literatura existente. Los intervalos [PD_low, PD_high] por "
    "loan grade alimentan directamente box uncertainty sets con garantía de cobertura "
    "finita, produciendo portafolios matemáticamente robustos."
)

with st.expander("Papers clave"):
    st.markdown(
        """
- **Johnstone et al. (2021)** — *Conformal Uncertainty Sets for Robust Optimization*,
  PMLR Vol. 152. Usa Mahalanobis distance como conformity score para generar conjuntos
  elipsoidales. **Diferencia con nuestro trabajo**: ellos usan parámetros continuos;
  nosotros usamos Mondrian conformal para grupos discretos (loan grades).
- **Patel et al. (2024)** — *Conformal Contextual Robust Optimization*. Extiende al
  setting contextual (condicional); conjuntos data-dependent.
- **Conformal Predictive Portfolio Selection (2024)** — arXiv. Intervalos de predicción
  para retornos de activos como input directo a selección de portafolio.
- **Bertsimas & Sim (2004)** — *The Price of Robustness*, Operations Research.
  Framework clásico de robust optimization con uncertainty budgets.
  Nuestra implementación: el "price of robustness" se cuantifica empíricamente
  comparando portafolios con PD_point vs PD_high.
"""
    )

# ══════════════════════════════════════════════════════════════════════════════
# 6. Causal ML
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("6) Inferencia Causal en crédito")

st.markdown(
    """
La correlación entre tasa de interés y default no implica causalidad — borrowers de
mayor riesgo reciben tasas más altas (selection bias). **Double/Debiased ML** (DML)
y **Causal Forests** permiten estimar efectos causales heterogéneos eliminando el
sesgo de confusión con garantías semiparamétricas.

En nuestro proyecto: ATE estimado = +1pp en tasa → **+0.787pp en probabilidad de
default**, con efectos heterogéneos por grade y DTI.
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

**Cox PH** es el estándar semiparamétrico (C-index=0.677 en nuestros datos), pero
las violaciones del supuesto de hazards proporcionales (detectadas via Schoenfeld test)
motivan el uso complementario de **Random Survival Forests** (C-index=0.684),
que no requiere este supuesto.
"""
)

with st.expander("Papers clave"):
    st.markdown(
        """
- **Time to Default Benchmark (2016)** — Journal of the Operational Research Society.
  Compara Cox PH, spline-based y mixture cure models; splines recomendados.
- **Probability of Default using ML Competing Risks (2024)** — Expert Systems with
  Applications. RSF + competing risks para IFRS9 lifetime PD.
- **Discrete-time Hazard Models for IFRS9 (2025)** — arXiv tutorial.
  Modelos de hazard discreto para 12-month vs lifetime PD.
"""
    )

# ══════════════════════════════════════════════════════════════════════════════
# 8. Time Series
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("8) Series de Tiempo en crédito")

st.markdown(
    """
El forecasting de tasas de default agregadas conecta el riesgo individual con
el riesgo sistémico. Modelos como ARIMA capturan tendencias y estacionalidad,
mientras que LightGBM (via Nixtla mlforecast) incorpora features macroeconómicas.

Los **intervalos conformal para time series** son un área activa: la violación de
exchangeability en datos temporales degrada la cobertura, motivando enfoques
como **Adaptive Conformal Inference (ACI)** de Gibbs & Candès (2021).
"""
)

with st.expander("Papers clave"):
    st.markdown(
        """
- **Macroeconomic-Sensitive Credit Risk Forecasting (2026)** — Preprints.
  Regime-switching (Markov) con 22.7% menos error que modelos sin régimen.
- **Incorporating Macroeconomic Scenarios in Credit Loss Forecasting** — Banking Exchange.
  Macro links (desempleo, GDP, spreads) a PD, LGD, EAD.
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

**Metodología**: CatBoost PD → Platt calibration → MAPIE SplitConformalRegressor
(Mondrian by grade) → Box sets [PD_low, PD_high] → Pyomo LP + HiGHS → Frontera de
robustez empírica.
"""
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
out-of-time (2018-2020) con policy gates formales y monitoreo temporal.

**Contribución clave**: Conformal interval width como señal SICR + ECL por rango para
lectura prudencial — ambos conceptos nuevos en la literatura IFRS9.

**Metodología**: PD calibrada → Conformal intervals → IFRS9 staging (con CP width como
señal) → ECL_point, ECL_low, ECL_high → Stress testing bajo escenarios macro.
"""
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
        ["Patel et al.", 2024, "arXiv", "Conformal contextual robust optimization"],
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
        [
            "Taquet et al.",
            2025,
            "JMLR (pendiente)",
            "MAPIE: librería open-source de conformal prediction",
        ],
        ["Gibbs & Candès", 2021, "NeurIPS", "Adaptive Conformal Inference bajo distribution shift"],
        ["ML Credit Scoring Survey", 2025, "Springer", "Revisión sistemática: 63 papers, GB > LR"],
    ],
    columns=["Referencia", "Año", "Venue", "Relevancia para este proyecto"],
)
st.dataframe(refs, use_container_width=True, hide_index=True)

next_page_teaser(
    "Visión End-to-End",
    "Narrativa completa del pipeline con métricas detalladas por componente.",
    "pages/thesis_end_to_end.py",
)
