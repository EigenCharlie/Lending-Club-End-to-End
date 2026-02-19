"""Glosario y fundamentos: términos, técnicas y fórmulas clave del proyecto."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from streamlit_app.components.narrative import next_page_teaser
from streamlit_app.utils import load_json

st.title("📖 Glosario y Fundamentos")
st.caption(
    "Referencia rápida de todos los conceptos, métricas, técnicas y fórmulas "
    "utilizados en este proyecto de riesgo de crédito end-to-end."
)
st.markdown(
    """
Esta página funciona como diccionario de consulta. Antes de explorar los resultados analíticos,
aquí puedes familiarizarte con los términos financieros, regulatorios, de machine learning y
optimización que aparecen en todo el recorrido. Cada término incluye una definición accesible
y su conexión con el proyecto.
"""
)

# ── Glossary Data ──
comparison = load_json("model_comparison")
final_metrics = comparison.get("final_test_metrics", {})
policy = load_json("conformal_policy_status", directory="models")
pipeline_summary = load_json("pipeline_summary")
pipeline_metrics = pipeline_summary.get("pipeline", {})
survival_metrics = pipeline_summary.get("survival", {})
ifrs9_baseline = float(pipeline_metrics.get("ecl_expected", 0.0))
ifrs9_severe = float(pipeline_metrics.get("ecl_conservative", 0.0))
ifrs9_uplift = (ifrs9_severe / ifrs9_baseline - 1.0) if ifrs9_baseline else 0.0
nonrobust_return = float(pipeline_metrics.get("nonrobust_return", 0.0))
price_of_robustness = float(pipeline_metrics.get("price_of_robustness", 0.0))
por_pct = (price_of_robustness / (abs(nonrobust_return) + 1e-6) * 100.0) if nonrobust_return else 0.0

GLOSSARY = [
    # Financial terms
    {"termino": "PD", "categoria": "Financiero", "definicion": "Probability of Default. Probabilidad de que un préstamo entre en incumplimiento. Es la salida principal del modelo CatBoost calibrado.", "en_proyecto": f"Modelo PD con AUC={final_metrics.get('auc_roc', 0):.4f}, calibrado con Platt Sigmoid (ECE={final_metrics.get('ece', 0):.4f})."},
    {"termino": "LGD", "categoria": "Financiero", "definicion": "Loss Given Default. Porcentaje del monto expuesto que se pierde cuando ocurre un default. Complemento de la tasa de recuperación.", "en_proyecto": "Modelado sobre préstamos en default (~88% nulls esperados en no-defaults)."},
    {"termino": "EAD", "categoria": "Financiero", "definicion": "Exposure at Default. Monto expuesto al momento del incumplimiento. Para préstamos amortizables, es el saldo pendiente.", "en_proyecto": "Dataset especializado ead_dataset.parquet solo con defaults."},
    {"termino": "ECL", "categoria": "Financiero", "definicion": "Expected Credit Loss. Pérdida esperada = PD × LGD × EAD × Factor de descuento. Métrica central de IFRS9.", "en_proyecto": f"ECL baseline ${ifrs9_baseline / 1e6:,.1f}M, escenario severo ${ifrs9_severe / 1e9:,.3f}B ({ifrs9_uplift:+.2%})."},
    {"termino": "DTI", "categoria": "Financiero", "definicion": "Debt-to-Income ratio. Pagos mensuales de deuda divididos entre ingreso mensual. Mediana en el dataset: ~15.", "en_proyecto": "Feature clave del modelo PD. DTI alto (>30) señala sobreendeudamiento."},
    {"termino": "NPL", "categoria": "Financiero", "definicion": "Non-Performing Loan. Préstamo con pagos vencidos >90 días o en reestructuración. Equivale a Stage 3 en IFRS9.", "en_proyecto": "Préstamos Charged Off + Default + Late 31-120 en el dataset."},
    {"termino": "SICR", "categoria": "Financiero", "definicion": "Significant Increase in Credit Risk. Evento que dispara la migración de Stage 1 a Stage 2 en IFRS9.", "en_proyecto": "Innovación: ancho del intervalo conformal (PD_high - PD_point) como señal adicional de SICR."},
    {"termino": "Write-off", "categoria": "Financiero", "definicion": "Castigo contable: reconocimiento formal de que un préstamo es irrecuperable. Genera pérdida directa en resultados.", "en_proyecto": "Estado 'Charged Off' en el dataset de Lending Club."},
    {"termino": "Spread", "categoria": "Financiero", "definicion": "Diferencia entre la tasa cobrada al prestatario y el costo de fondeo. Representa el margen bruto del préstamo.", "en_proyecto": "int_rate del dataset menos costo de fondeo estimado."},
    {"termino": "Grade", "categoria": "Financiero", "definicion": "Calificación de riesgo de Lending Club (A-G). Grade A default ~2%, Grade G ~37%. Variable de mayor poder predictivo.", "en_proyecto": "Usada en Mondrian conformal para intervalos por grupo y en segmentación IFRS9."},
    # Regulatory terms
    {"termino": "IFRS 9", "categoria": "Regulatorio", "definicion": "International Financial Reporting Standard 9. Norma contable que requiere provisionar pérdidas esperadas (no solo incurridas). Vigente desde enero 2018.", "en_proyecto": "Página completa de provisiones IFRS9 con 4 escenarios y análisis de sensibilidad."},
    {"termino": "Basilea III", "categoria": "Regulatorio", "definicion": "Marco regulatorio bancario que define requerimientos mínimos de capital. Los bancos deben mantener capital suficiente para absorber pérdidas inesperadas.", "en_proyecto": "IFRS9 determina provisiones (pérdida esperada); Basilea III determina capital (pérdida inesperada)."},
    {"termino": "Stage 1", "categoria": "Regulatorio", "definicion": "Préstamos sin deterioro significativo. Se provisiona ECL a 12 meses (PD 12m × LGD × EAD).", "en_proyecto": "Mayoría del portafolio. PD a 12 meses del modelo CatBoost."},
    {"termino": "Stage 2", "categoria": "Regulatorio", "definicion": "Préstamos con incremento significativo de riesgo (SICR). Se provisiona ECL lifetime (PD lifetime × LGD × EAD).", "en_proyecto": "Migración Stage 1→2 analizada con conformal width como señal SICR."},
    {"termino": "Stage 3", "categoria": "Regulatorio", "definicion": "Préstamos deteriorados (default, 90+ DPD). PD ~ 1.0, se provisiona pérdida total esperada.", "en_proyecto": "Préstamos en Charged Off/Default del dataset."},
    {"termino": "Stress Test", "categoria": "Regulatorio", "definicion": "Ejercicio regulatorio que evalúa resiliencia de un portafolio bajo escenarios macroeconómicos adversos.", "en_proyecto": f"4 escenarios IFRS9 (baseline, mild, adverse, severe) con uplift severo actual {ifrs9_uplift:+.2%}."},
    # ML terms
    {"termino": "AUC", "categoria": "Machine Learning", "definicion": "Area Under the ROC Curve. Mide la capacidad del modelo para separar defaults de no-defaults. 0.5=aleatorio, 1.0=perfecto. En banca, AUC >0.70 se considera aceptable.", "en_proyecto": f"CatBoost calibrado: AUC={final_metrics.get('auc_roc', 0):.4f} en test out-of-time."},
    {"termino": "Gini", "categoria": "Machine Learning", "definicion": "Coeficiente Gini = 2×AUC - 1. Escala de 0 (sin poder) a 1 (perfecto). Métrica estándar en credit scoring bancario.", "en_proyecto": f"Gini={final_metrics.get('gini', 0):.4f}, consistente con modelos de crédito al consumo."},
    {"termino": "KS", "categoria": "Machine Learning", "definicion": "Kolmogorov-Smirnov statistic. Máxima separación entre distribuciones acumuladas de buenos y malos. KS >0.30 es buen poder discriminante.", "en_proyecto": f"KS={final_metrics.get('ks_statistic', 0):.4f} en test OOT."},
    {"termino": "Brier Score", "categoria": "Machine Learning", "definicion": "Error cuadrático medio de las probabilidades predichas vs outcomes reales. Menor es mejor. Combina discriminación y calibración.", "en_proyecto": f"Brier={final_metrics.get('brier_score', 0):.4f} post-calibración."},
    {"termino": "ECE", "categoria": "Machine Learning", "definicion": "Expected Calibration Error. Mide qué tan bien las probabilidades predichas reflejan las frecuencias reales de default. ECE=0 es calibración perfecta.", "en_proyecto": f"ECE={final_metrics.get('ece', 0):.4f} con Platt Sigmoid (seleccionado sobre Isotonic)."},
    {"termino": "SHAP", "categoria": "Machine Learning", "definicion": "SHapley Additive exPlanations. Método de teoría de juegos que atribuye la contribución de cada variable a cada predicción individual.", "en_proyecto": "Top drivers: int_rate, grade, term, loan_to_income, revol_util."},
    {"termino": "CatBoost", "categoria": "Machine Learning", "definicion": "Algoritmo de gradient boosting que maneja variables categóricas nativamente y es robusto a overfitting. Desarrollado por Yandex. Dominante en competencias de datos tabulares.", "en_proyecto": "Modelo final: CatBoost tuneado con Optuna (1000+ trials) + calibración Platt."},
    {"termino": "Gradient Boosting", "categoria": "Machine Learning", "definicion": "Técnica de ensamble que construye secuencialmente árboles de decisión, donde cada nuevo árbol corrige los errores del anterior.", "en_proyecto": "CatBoost, XGBoost y LightGBM son variantes de gradient boosting."},
    {"termino": "Calibración", "categoria": "Machine Learning", "definicion": "Ajuste post-entrenamiento para que las probabilidades predichas sean consistentes con las frecuencias observadas. Si predice PD=10%, ~10% deben hacer default.", "en_proyecto": "Platt Sigmoid seleccionada (ECE=0.0128) sobre Isotonic."},
    {"termino": "Cross-validation", "categoria": "Machine Learning", "definicion": "Técnica de evaluación que divide datos en K subconjuntos para entrenar y validar el modelo K veces, reduciendo sesgo de evaluación.", "en_proyecto": "No usada para split final (se usa OOT temporal), sí para Optuna."},
    {"termino": "WOE", "categoria": "Machine Learning", "definicion": "Weight of Evidence. Transformación de variables categóricas/binneadas que captura la relación monotónica con el default. Estándar en credit scoring.", "en_proyecto": "Aplicado a grade, purpose, home_ownership via OptBinning."},
    {"termino": "IV", "categoria": "Machine Learning", "definicion": "Information Value. Mide el poder predictivo global de una variable. IV <0.02 débil, 0.02-0.1 útil, 0.1-0.3 fuerte, >0.3 muy fuerte.", "en_proyecto": "Usado para ranking y selección de features en NB02."},
    # Uncertainty terms
    {"termino": "Conformal Prediction", "categoria": "Incertidumbre", "definicion": "Marco estadístico que produce intervalos de predicción con garantía de cobertura finita sin asumir distribución paramétrica. Solo requiere intercambiabilidad de datos.", "en_proyecto": f"MAPIE 1.3.0 SplitConformalRegressor. Cobertura 90%={policy.get('coverage_90', 0):.4f}, 95%={policy.get('coverage_95', 0):.4f}."},
    {"termino": "Coverage (Cobertura)", "categoria": "Incertidumbre", "definicion": "Proporción de valores reales que caen dentro del intervalo predicho. Cobertura 90% = 90% de los valores reales están en [PD_low, PD_high].", "en_proyecto": f"90%: {policy.get('coverage_90', 0):.4f}, 95%: {policy.get('coverage_95', 0):.4f}, checks {int(policy.get('checks_passed', 0))}/{int(policy.get('checks_total', 0))}."},
    {"termino": "Mondrian CP", "categoria": "Incertidumbre", "definicion": "Variante de conformal prediction que calcula intervalos por grupo (e.g., por grade), garantizando cobertura condicional por segmento.", "en_proyecto": f"Intervalos por grade. Min cobertura grupo: {policy.get('min_group_coverage_90', 0):.4f} (meta ≥0.88)."},
    {"termino": "Interval Width", "categoria": "Incertidumbre", "definicion": "Ancho del intervalo conformal (PD_high - PD_low). Intervalos más estrechos son más informativos pero con mismo nivel de cobertura.", "en_proyecto": f"Ancho promedio 90%: {policy.get('avg_width_90', 0):.4f} (meta <0.80). Usado como señal SICR."},
    {"termino": "Split Conformal", "categoria": "Incertidumbre", "definicion": "Método que usa un conjunto de calibración separado para calcular residuos conformales. Más eficiente computacionalmente que full conformal.", "en_proyecto": "SplitConformalRegressor con calibration set temporal separado."},
    # Causal terms
    {"termino": "ATE", "categoria": "Causal", "definicion": "Average Treatment Effect. Efecto promedio de una intervención sobre toda la población. Responde: ¿cuánto cambia Y si aplicamos tratamiento T?", "en_proyecto": "+1pp en tasa de interés → +0.787pp en probabilidad de default."},
    {"termino": "CATE", "categoria": "Causal", "definicion": "Conditional Average Treatment Effect. Efecto causal que varía por subgrupo/individuo. Permite intervenciones personalizadas.", "en_proyecto": "Distribución amplia de CATE justifica política diferenciada por segmento."},
    {"termino": "DML", "categoria": "Causal", "definicion": "Double/Debiased Machine Learning. Método de Chernozhukov et al. (2018) que usa ML para controlar confounders y estimar efectos causales sin sesgo.", "en_proyecto": "EconML LinearDML para estimación robusta del efecto tasa → default."},
    {"termino": "Causal Forest", "categoria": "Causal", "definicion": "Extensión de Random Forest para estimar efectos de tratamiento heterogéneos (CATE). Basado en Athey & Wager (2019).", "en_proyecto": "Modelo de 337MB entrenado para CATE heterogéneo por segmento."},
    {"termino": "Counterfactual", "categoria": "Causal", "definicion": "Escenario hipotético: ¿qué hubiera pasado si hubiéramos aplicado una intervención diferente? Base del análisis causal.", "en_proyecto": "Simulación contrafactual de políticas de intervención por regla."},
    # OR terms
    {"termino": "Optimización Robusta", "categoria": "Operations Research", "definicion": "Enfoque de optimización que incorpora incertidumbre en los parámetros del modelo. En vez de optimizar para el caso esperado, protege contra el peor caso plausible.", "en_proyecto": "PD_high (conformal) como peor caso -> Pyomo/HiGHS resuelve asignación robusta."},
    {"termino": "Uncertainty Set", "categoria": "Operations Research", "definicion": "Conjunto de valores posibles para parámetros inciertos. En optimización robusta, el modelo se protege contra todos los escenarios dentro del conjunto.", "en_proyecto": "[PD_low, PD_high] del conformal define el uncertainty set por préstamo."},
    {"termino": "Price of Robustness", "categoria": "Operations Research", "definicion": "Diferencia en retorno esperado entre la solución óptima sin incertidumbre y la robusta. Cuantifica el costo de proteger el downside.", "en_proyecto": f"{por_pct:.2f}% de reducción de retorno @ tolerancia 0.10 en snapshot actual. Es el 'costo del seguro'."},
    {"termino": "Efficient Frontier", "categoria": "Operations Research", "definicion": "Curva que muestra las mejores combinaciones posibles de riesgo y retorno. No se puede mejorar retorno sin asumir más riesgo.", "en_proyecto": "Frontera eficiente robusta vs no-robusta comparada en la página de portafolio."},
]

# ── Search & Filter ──
st.subheader("Buscar términos")
col_search, col_cat = st.columns([2, 1])
with col_search:
    search = st.text_input("Buscar por nombre o descripción", placeholder="Ej: conformal, PD, IFRS...")
with col_cat:
    categories = sorted({g["categoria"] for g in GLOSSARY})
    selected_cat = st.selectbox("Filtrar por categoría", ["Todas"] + categories)

filtered = GLOSSARY
if search:
    search_lower = search.lower()
    filtered = [
        g for g in filtered
        if search_lower in g["termino"].lower()
        or search_lower in g["definicion"].lower()
        or search_lower in g["en_proyecto"].lower()
    ]
if selected_cat != "Todas":
    filtered = [g for g in filtered if g["categoria"] == selected_cat]

st.markdown(f"**{len(filtered)}** términos encontrados")

df_glossary = pd.DataFrame(filtered)
if not df_glossary.empty:
    df_display = df_glossary.rename(columns={
        "termino": "Término",
        "categoria": "Categoría",
        "definicion": "Definición",
        "en_proyecto": "En este proyecto",
    })
    st.dataframe(df_display, use_container_width=True, hide_index=True, height=500)

# ── Industry Usage ──
st.subheader("Técnicas y su uso en la industria")
st.markdown(
    """
Las técnicas empleadas en este proyecto no son experimentales: son herramientas utilizadas
activamente en bancos, fintechs y aseguradoras de primer nivel a nivel mundial.
"""
)

industry_data = [
    {"Técnica": "CatBoost / XGBoost / LightGBM", "Uso en la industria": "Credit scoring en >70% de instituciones financieras digitales. Dominantes en competencias Kaggle de datos tabulares. Adoptados por JPMorgan, Capital One, Nubank, Mercado Libre.", "En este proyecto": "Modelo PD principal (CatBoost tuneado + Platt)"},
    {"Técnica": "WOE / IV (Weight of Evidence)", "Uso en la industria": "Estándar de facto en credit scoring bancario desde los años 90. Requerido por algunos reguladores para scorecard interpretable.", "En este proyecto": "Feature engineering: grade_woe, purpose_woe, home_ownership_woe"},
    {"Técnica": "SHAP (Explicabilidad)", "Uso en la industria": "Estándar de explicabilidad ML en banca (requerido por EBA, OCC). Usado para explicar decisiones individuales de crédito.", "En este proyecto": "Top 20 features con SHAP, dependence plots"},
    {"Técnica": "Conformal Prediction", "Uso en la industria": "Adoptado en farmacéutica (AstraZeneca), manufactura (Volvo), fintech (cuantificación de incertidumbre en modelos de pricing y riesgo). Crecimiento exponencial desde 2020.", "En este proyecto": "MAPIE Mondrian: intervalos PD con cobertura garantizada por grade"},
    {"Técnica": "Inferencia Causal (DML/CATE)", "Uso en la industria": "Pricing dinámico en Uber/Lyft, campañas de retención en telecoms, análisis de impacto de políticas en banca central (BIS, Fed).", "En este proyecto": "Efecto tasa→default, políticas de intervención por segmento"},
    {"Técnica": "Survival Analysis", "Uso en la industria": "Estimación de lifetime PD para IFRS9 Stage 2 en todos los bancos bajo IFRS. Modelos de churn en telecoms y seguros.", "En este proyecto": f"Cox PH (C={survival_metrics.get('cox_concordance', 0):.4f}) y RSF (C={survival_metrics.get('rsf_concordance', 0):.4f}) para PD lifetime por grade"},
    {"Técnica": "Optimización Robusta (Pyomo)", "Uso en la industria": "Asignación de capital en fondos de inversión, planificación de supply chain (Amazon, Walmart), gestión de portafolio en asset management.", "En este proyecto": "Asignación de préstamos con uncertainty sets conformales + HiGHS solver"},
    {"Técnica": "IFRS9 / ECL Modeling", "Uso en la industria": "Obligatorio para todas las instituciones financieras bajo IFRS (>140 países). Cada banco tiene modelos internos de ECL por stage.", "En este proyecto": "4 escenarios, sensibilidad PD×LGD, staging con conformal width"},
    {"Técnica": "dbt + DuckDB", "Uso en la industria": "dbt: estándar de transformación de datos en startups y empresas data-driven (Spotify, GitLab). DuckDB: análisis local sin servidor (reemplaza SQLite para analítica).", "En este proyecto": "19 modelos dbt sobre DuckDB local con linaje verificable"},
]
st.dataframe(pd.DataFrame(industry_data), use_container_width=True, hide_index=True)

# ── Key Formulas ──
st.subheader("Fórmulas clave")

col_f1, col_f2 = st.columns(2)

with col_f1:
    st.markdown("**Expected Credit Loss (ECL)**")
    st.latex(r"ECL = PD \times LGD \times EAD \times DF")
    st.caption("Donde DF = factor de descuento. PD a 12 meses (Stage 1) o lifetime (Stage 2).")

    st.markdown("**Coeficiente Gini**")
    st.latex(r"Gini = 2 \times AUC - 1")
    st.caption("Escala: 0 (sin poder discriminante) a 1 (discriminación perfecta).")

    st.markdown("**Brier Score**")
    st.latex(r"Brier = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)^2")
    st.caption("Error cuadrático medio de probabilidades. Menor es mejor.")

with col_f2:
    st.markdown("**Cobertura Conformal**")
    st.latex(r"Coverage = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[y_i \in C(x_i)]")
    st.caption("Proporción de valores reales dentro del intervalo predicho. Meta: ≥90% y ≥95%.")

    st.markdown("**Price of Robustness**")
    st.latex(r"PoR = \frac{R_{nominal} - R_{robust}}{R_{nominal}} \times 100\%")
    st.caption("Porcentaje de retorno sacrificado por protección contra incertidumbre.")

    st.markdown("**Information Value (IV)**")
    st.latex(r"IV = \sum_{i=1}^{B}(D_i\% - ND_i\%) \times \ln\left(\frac{D_i\%}{ND_i\%}\right)")
    st.caption("Poder predictivo global de una variable. >0.3 = muy fuerte.")

# ── Reading Guide ──
st.subheader("Guía de lectura del dashboard")
st.markdown(
    """
| Orden | Sección | Página | Pregunta que responde |
|:-----:|---------|--------|-----------------------|
| 1 | Inicio | 🏠 Resumen Ejecutivo | ¿Qué problema resolvemos y con qué resultados? |
| 2 | Inicio | 📖 Glosario y Fundamentos (esta página) | ¿Qué significa cada término y técnica? |
| 3 | Recorrido E2E | 🧭 Visión End-to-End | ¿Cuál es la narrativa completa del proyecto? |
| 4 | Recorrido E2E | 🗂️ Arquitectura y Linaje de Datos | ¿Cómo fluyen los datos a través del sistema? |
| 5 | Recorrido E2E | 🧩 Mapa Integrado de Métodos | ¿Cómo se conectan las técnicas entre sí? |
| 6 | Recorrido E2E | 📚 Atlas de Evidencia | ¿Dónde está la evidencia de cada notebook? |
| 7 | Analítica | 🔧 Ingeniería de Features | ¿Cómo se transformaron las variables para el modelo? |
| 8 | Analítica | 📊 Historia de Datos | ¿Qué contiene el dataset y qué patrones existen? |
| 9 | Analítica | 🔬 Laboratorio de Modelos | ¿Qué modelo se eligió y por qué? |
| 10 | Analítica | 📐 Cuantificación de Incertidumbre | ¿Cómo cuantificamos la incertidumbre de las predicciones? |
| 11 | Analítica | 📈 Panorama Temporal | ¿Cómo evolucionan los defaults en el tiempo? |
| 12 | Analítica | ⏳ Análisis de Supervivencia | ¿Cuándo ocurren los defaults? |
| 13 | Analítica | 🧬 Inteligencia Causal | ¿Qué intervenciones pueden reducir el riesgo? |
| 14 | Decisiones | 💼 Optimizador de Portafolio | ¿Cómo asignar capital bajo incertidumbre? |
| 15 | Decisiones | 🏦 Provisiones IFRS9 | ¿Cuánto provisionar bajo diferentes escenarios? |
| 16 | Gobernanza | 🛡️ Gobernanza del Modelo | ¿Es el modelo confiable y justo? |
| 17 | Exploración | 💬 Chat con Datos | Exploración libre por SQL |
"""
)

next_page_teaser(
    "Historia de Datos",
    "Explora el dataset: distribuciones, patrones de riesgo y dinámica temporal de 1.35M préstamos.",
    "pages/data_story.py",
)
