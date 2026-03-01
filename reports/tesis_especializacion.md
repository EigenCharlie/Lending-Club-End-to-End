# Conformal Prediction para la Calibración y Cuantificación de Incertidumbre en Modelos de Riesgo Crediticio

**Carlos Alfredo Vergara Rojas**

Universidad Tecnológica de Pereira — Programa de Especialización en Analítica y Ciencia de Datos Aplicada

Docente: Alejandra María Restrepo Franco

Febrero de 2026

---

## Abstract

Los modelos de riesgo crediticio en la industria financiera producen estimaciones puntuales de la probabilidad de incumplimiento (PD), la pérdida dado incumplimiento (LGD) y la exposición en caso de incumplimiento (EAD) sin cuantificar la incertidumbre asociada a dichas predicciones. Esta limitación tiene consecuencias directas: subestimación del riesgo que genera pérdidas inesperadas, provisiones excesivas por posturas conservadoras ante la incertidumbre no medida, y desconfianza regulatoria por falta de transparencia en los intervalos de confianza de los modelos.

La predicción conformal (*Conformal Prediction*, CP), propuesta originalmente por Vovk, Gammerman y Shafer (2005), ofrece una solución rigurosa: intervalos de predicción con garantías formales de cobertura P(Y ∈ C(X)) ≥ 1 − α bajo el supuesto de intercambiabilidad de los datos, sin requerir supuestos paramétricos sobre la distribución subyacente. Esta propiedad de cobertura *distribution-free* convierte a CP en una herramienta especialmente atractiva para el sector financiero regulado, donde la robustez estadística y la auditabilidad son requisitos fundamentales.

Esta tesis implementa y evalúa un pipeline completo de conformal prediction aplicado a modelos de riesgo crediticio sobre el dataset público de LendingClub (2.26 millones de préstamos, 2007-2020), incluyendo cinco variantes de intervalos predictivos: Split Conformal, Mondrian Conformal (grupo-condicional por grade), Venn-Abers Predictors, Conformalized Quantile Regression (CQR) para LGD/EAD, e intervalos residuales como benchmark no-conformal. Los resultados se evalúan mediante tests formales de backtesting (Kupiec, Christoffersen), métricas de calibración (ECE, Brier Score), eficiencia de intervalos (MPIW) y fairness conformal, y se conectan directamente con el cálculo de provisiones bajo IFRS 9 y los requerimientos de Basilea III.

**Palabras clave**: Predicción conformal, riesgo crediticio, calibración de probabilidades, IFRS 9, Basilea III, cuantificación de incertidumbre, CatBoost, MAPIE, Mondrian conformal, Venn-Abers.

### Importancia para la industria financiera

El sector de riesgo crediticio opera bajo un dilema fundamental: los modelos de machine learning han demostrado mayor poder discriminativo que los modelos estadísticos tradicionales (regresión logística), pero adolecen de falta de calibración y opacidad — dos características que los reguladores financieros no pueden tolerar. Las probabilidades de default estimadas por gradient boosting (CatBoost, XGBoost, LightGBM) no son necesariamente probabilidades bien calibradas; un modelo que dice "P(default)=0.15" no garantiza que el 15% de esos préstamos efectivamente entre en default.

La predicción conformal resuelve este problema de raíz al proveer intervalos con cobertura estadística garantizada. Para una entidad financiera, esto significa:

- **Pricing de préstamos más preciso**: El intervalo [PD_low, PD_high] permite tarifar con márgenes de seguridad informados, no arbitrarios.
- **Provisiones IFRS 9 justificables**: El ancho del intervalo conformal sirve como señal objetiva de SICR (Significant Increase in Credit Risk), mejorando la migración entre Stages 1→2 con criterios auditables.
- **Base para optimización robusta**: Los intervalos conformales proveen conjuntos de incertidumbre que pueden alimentar modelos de optimización de portafolio en fases posteriores del proyecto.
- **Backtesting regulatorio formal**: Los tests de Kupiec y Christoffersen validan que la cobertura prometida se cumple — requisito de Basilea III para modelos internos (IRB).

### Importancia para la academia

La aplicación de predicción conformal al riesgo crediticio es una línea de investigación emergente pero aún fragmentada. La literatura existente se ha concentrado en PD de forma aislada (Bellotti, 2017; Javanmardi & Vovk, 2023). Esta tesis contribuye a cerrar varios vacíos:

- **Cobertura conjunta PD + LGD + EAD**: Primera aplicación documentada de CP sobre los tres componentes del riesgo crediticio simultáneamente.
- **Mondrian conformal con tuning multi-objetivo**: Extensión del conformal grupo-condicional con selección Pareto y cobertura mínima garantizada por subgrupo.
- **Fairness conformal**: Evaluación de disparidades en cobertura y ancho de intervalos entre grupos demográficos — un ángulo completamente nuevo en la intersección de fairness algorítmica y predicción conformal.
- **Base para predict-then-optimize**: Los intervalos conformales generados aquí sientan las bases para su integración con optimización robusta de portafolio (trabajo futuro en la maestría).

### Contexto colombiano

En Colombia, la adopción de técnicas avanzadas de cuantificación de incertidumbre en modelos de riesgo es incipiente. Las entidades financieras supervisadas por la Superintendencia Financiera de Colombia (SFC) utilizan modelos de riesgo que cumplen con la normativa local pero generalmente no incorporan intervalos de confianza con garantías formales. Esta investigación posiciona la predicción conformal como una herramienta viable para el contexto regulatorio colombiano, alineada con las mejores prácticas internacionales de Basilea III e IFRS 9.

### Estatus del proyecto

#### 1. Definición del "Dolor" (El Problema)

**Pregunta**: ¿Cuál es el problema específico que mi proyecto resuelve y quién es el dueño de ese problema en la organización?

El problema es la ausencia de cuantificación formal de incertidumbre en los modelos de riesgo crediticio. Los modelos actuales producen una PD puntual (ej: 0.12) sin indicar si esa estimación es confiable (intervalo estrecho) o altamente incierta (intervalo amplio). Esto genera provisiones mal calibradas (P&L volátil), decisiones de originación subóptimas, y riesgo de incumplimiento regulatorio. El dueño del problema es la Dirección de Riesgo Crediticio (CRO), con impacto directo en Comités de Crédito, Provisiones (CFO) y Validación de Modelos (MRM). Es un problema real: los reguladores internacionales (BIS, EBA) y locales (SFC Colombia) exigen que los modelos internos demuestren cobertura adecuada.

#### 2. Estatus del "Cerebro" (El Modelo / Analítica)

**Pregunta**: A hoy, ¿tengo datos reales disponibles y una técnica analítica ya seleccionada?

Sí, completamente implementado y validado. Dataset LendingClub con 2.26 millones de préstamos reales (2007-2020) como fuente de estudio y splits temporales estrictos OOT sobre el subconjunto resuelto. Modelos base: CatBoost + Regresión Logística. Técnica conformal: 5 variantes implementadas usando MAPIE 1.3.0 y crepes. El proyecto no está en fase de ideación — es un prototipo experimental completo con 418 tests, pipeline DVC de 25 stages, y dashboard Streamlit de 27 páginas.

#### 3. Tangibilidad (El Producto)

**Pregunta**: ¿Cómo se imagina el usuario final consumiendo mi resultado?

El resultado se consume en tres formatos: (1) Dashboard Streamlit (27 páginas) para exploración interactiva, (2) API REST FastAPI (15 endpoints) para integración programática, y (3) Reportes automatizados (MRM report, fairness audit, backtesting) para el área de validación de modelos.

#### 4. Madurez Tecnológica (TRL)

**Pregunta**: Siendo 1 "solo tengo la idea" y 9 "el sistema ya funciona en producción", ¿en qué número ubico mi proyecto?

**TRL 6-7** (Prototipo validado en entorno representativo). Justificación: Pipeline end-to-end ejecutable y reproducible (DVC 25 stages), 418 tests, dashboard operativo (Streamlit 27 páginas), API REST funcional (FastAPI 15 endpoints), CI/CD con GitHub Actions, MLflow tracking + DagsHub remote. Lo que falta para TRL 8-9: despliegue en infraestructura bancaria real, integración con core bancario, datos de producción protegidos.

#### 5. Brecha de Innovación

**Pregunta**: ¿Qué es lo que me impide entregar este proyecto mañana mismo?

El proyecto como tesis de especialización está prácticamente completo. Las brechas restantes (datos bancarios reales, infraestructura cloud, conformal adaptativo, validación con regulador) no son bloqueantes para la entrega académica. El proyecto supera ampliamente el alcance original de la propuesta (5 variantes vs 3 propuestas, tuning multi-objetivo, fairness conformal, backtesting formal Kupiec/Christoffersen).

### ¿Quién más está resolviendo esto?

| Autor / Grupo | Año | Enfoque | Diferencia con esta tesis |
|---|---|---|---|
| Bellotti (Nottingham) | 2017 | CP para credit scoring (SCP básico) | Solo PD, sin LGD/EAD, sin fairness |
| Javanmardi & Vovk (Royal Holloway) | 2023 | Venn-Abers para PD bancario | Solo Venn-Abers, sin Mondrian, sin IFRS9 |
| Angelopoulos & Bates (Berkeley) | 2023 | Tutorial general de CP | No aplicado a riesgo crediticio |
| Romano, Patterson & Candes (Stanford) | 2019 | CQR para regresión | Método genérico, no aplicado a LGD/EAD |
| Fontana, Zeni & Vantini (Politecnico Milano) | 2023 | Revisión unificada de CP | Teórico, sin implementación financiera |
| MAPIE team (Quantmetry) | 2023-2025 | Librería MAPIE | Herramienta, no aplicación a crédito |
| **Esta tesis** | **2026** | **CP aplicada a riesgo crediticio integral** | **5 variantes, PD+LGD+EAD, IFRS9, fairness conformal, Mondrian tuned, backtesting formal** |

---

## 1. Introducción

En la actualidad, los sistemas financieros enfrentan un entorno complejo y altamente regulado, en el cual la correcta estimación del riesgo crediticio constituye un pilar fundamental para la estabilidad económica. La probabilidad de incumplimiento (PD), la pérdida dado incumplimiento (LGD) y la exposición en caso de incumplimiento (EAD) son métricas centrales para calcular provisiones, asignar capital y cumplir con estándares regulatorios como IFRS 9 y Basilea III [1, 2].

Los modelos tradicionales de riesgo crediticio, así como los modelos modernos basados en machine learning, presentan una limitación recurrente: ofrecen predicciones puntuales sin cuantificar explícitamente la incertidumbre asociada. Las probabilidades estimadas frecuentemente carecen de calibración adecuada, lo que genera tres consecuencias críticas: pérdidas inesperadas por subestimación del riesgo, provisiones excesivas por sobreestimación conservadora, y desconfianza regulatoria por falta de transparencia en los modelos [3, 5].

La predicción conformal (Conformal Prediction, CP) surge como una alternativa metodológica rigurosa. Propuesta por Vovk, Gammerman y Shafer [6], ofrece intervalos de confianza con garantías formales de cobertura, sin requerir supuestos paramétricos sobre la distribución de los datos [7]. Esta propiedad la convierte en una técnica especialmente prometedora para el sector financiero, donde la robustez estadística y la transparencia son requisitos regulatorios fundamentales.

Este documento presenta los resultados de la implementación y evaluación de técnicas de conformal prediction aplicadas a modelos de riesgo crediticio sobre el dataset público de LendingClub (2007-2020).

## 2. Planteamiento del Problema

Los modelos de riesgo crediticio típicamente entregan predicciones puntuales sin cuantificar la incertidumbre asociada. Esta deficiencia impacta en múltiples dimensiones:

- **Subestimación del riesgo**: Los modelos de ML producen probabilidades mal calibradas que distorsionan las decisiones de originación y pricing [5, 11].
- **Provisiones excesivas**: Ante la incertidumbre no cuantificada, las entidades adoptan posturas conservadoras que inmovilizan capital.
- **Desconfianza regulatoria**: Los reguladores exigen robustez, explicabilidad y fairness que los modelos sin cuantificación de incertidumbre no satisfacen [12, 13].

**Pregunta de investigación**: ¿Cómo mejorar la calibración y la cuantificación de la incertidumbre en los modelos de riesgo crediticio (PD, LGD, EAD) mediante la aplicación de técnicas de predicción conformal, y cuál es su desempeño comparado frente a métodos tradicionales de calibración?

## 3. Justificación

### Perspectiva académica

Esta investigación contribuye a una línea emergente que integra la cuantificación de incertidumbre con ML aplicado al riesgo financiero. La aplicación de CP al riesgo crediticio permanece relativamente inexplorada, particularmente para la modelación conjunta de PD, LGD y EAD con garantías formales de cobertura [7].

### Perspectiva regulatoria

La investigación se alinea con los requerimientos de IFRS 9 (provisiones prospectivas con incertidumbre) y Basilea III (backtesting formal de modelos internos). La predicción conformal ofrece un camino natural para satisfacer estos requisitos regulatorios [1, 2].

### Perspectiva práctica

La adopción de estas técnicas permite una asignación más eficiente del capital regulatorio, al contar con intervalos de confianza que distinguen entre predicciones de alto y bajo nivel de certeza. Esto se traduce en reducción de provisiones excesivas y mayor confianza de los comités de riesgo.

### Contexto colombiano

En el contexto local, la adopción de técnicas avanzadas de cuantificación de incertidumbre en modelos de riesgo crediticio es aún incipiente. Esta implementación posiciona al proyecto como referente en la adopción de modelos explicables y alineados con las mejores prácticas internacionales.

## 4. Objetivos

### 4.1 Objetivo General

Implementar y evaluar técnicas de predicción conformal para mejorar la calibración y cuantificación de la incertidumbre en modelos de riesgo crediticio (PD, LGD, EAD), utilizando el dataset público de LendingClub como caso de estudio, y comparar su desempeño frente a métodos tradicionales de calibración.

### 4.2 Objetivos Específicos

1. Realizar una revisión del estado del arte sobre calibración de modelos, cuantificación de incertidumbre y predicción conformal aplicada al riesgo crediticio.
2. Construir y preparar un dataset a partir de los registros públicos de LendingClub (2007-2020), definiendo las variables objetivo (PD, LGD, EAD) y realizando el preprocesamiento necesario.
3. Diseñar e implementar experimentos aplicando técnicas de Conformal Prediction (Split Conformal, Mondrian Conformal, Venn-Abers Predictors, CQR) sobre modelos base de PD, LGD y EAD.
4. Comparar el desempeño de Conformal Prediction con métodos tradicionales de calibración (Platt Scaling, Isotonic Regression, intervalos residuales) mediante métricas de calibración, cobertura, eficiencia y discriminación.
5. Evaluar el impacto de la predicción conformal en la estimación de provisiones bajo IFRS 9, la asignación de capital regulatorio bajo Basilea III, y las métricas de backtesting formal (Kupiec, Christoffersen).
6. Proponer lineamientos para la adopción de Conformal Prediction en entornos bancarios reales, incluyendo fairness conformal, gobernanza y escalabilidad.

## 5. Marco Teórico y Estado del Arte

### 5.1 Modelos de Riesgo Crediticio

Los tres componentes fundamentales del riesgo crediticio son:
- **PD**: Probabilidad de que un deudor incumpla en un horizonte temporal determinado.
- **LGD**: Fracción de la exposición que se pierde en caso de incumplimiento.
- **EAD**: Monto expuesto al momento del incumplimiento.

Históricamente, PD se ha modelado mediante regresión logística, mientras que LGD y EAD han utilizado modelos de regresión con diversas transformaciones. Modelos de gradient boosting como CatBoost, LightGBM y XGBoost han demostrado mejoras significativas en discriminación [3, 11].

### 5.2 El Problema de la Calibración

La calibración se refiere a la correspondencia entre las probabilidades predichas y las frecuencias observadas. Los métodos tradicionales de post-hoc calibración incluyen:
- **Platt Scaling**: Transformación sigmoidal de las salidas del modelo [18].
- **Isotonic Regression**: Función monótona no decreciente ajustada a las probabilidades.

Ambos carecen de garantías formales de cobertura y son sensibles al sobreajuste [14].

### 5.3 Predicción Conformal: Fundamentos

La predicción conformal, introducida por Vovk et al. [6], permite construir intervalos de predicción con cobertura marginal garantizada P(Y ∈ C(X)) ≥ 1 - α bajo el supuesto de intercambiabilidad de los datos [7].

Las variantes principales son:
- **Split Conformal Prediction (SCP)**: Divide datos en entrenamiento y calibración. Eficiente computacionalmente [7, 8].
- **Conformalized Quantile Regression (CQR)**: Combina regresión cuantílica con ajuste conformal para intervalos adaptativos [8].
- **Venn-Abers Predictors**: Producen intervalos de probabilidad [p0, p1] con garantías de calibración [14, 15].
- **Mondrian Conformal**: Extensión grupo-condicional que mantiene cobertura por subgrupos.

### 5.4 Aplicaciones en Finanzas

Bellotti [4] fue pionero en aplicar CP a credit scoring. Javanmardi y Vovk [15] extendieron Venn-Abers para PD bancario. Sin embargo, la literatura presenta vacíos en la aplicación conjunta a PD, LGD y EAD, y en la evaluación del impacto práctico en métricas regulatorias como IFRS 9.

## 6. Metodología

### 6.1 Dataset

**Fuente**: LendingClub Loan Data 2007-2020 (Kaggle) [9].

| Split | Filas | Default Rate | Rango Temporal |
|-------|-------|-------------|----------------|
| Train | 1,346,311 | 18.52% | 2007-06 a 2017-03 |
| Calibración | 237,584 | 22.20% | 2017-03 a 2017-12 |
| Test (OOT) | 276,869 | 21.98% | 2018-01 a 2020-09 |

**Nota metodológica**: Se utilizaron splits temporales (Out-of-Time) en lugar de splits aleatorios. Esta decisión respeta la estructura cronológica del riesgo crediticio y es la práctica estándar en la industria financiera para validación de modelos.

**Prevención de data leakage**: Se removieron 15+ variables post-loan (total_pymnt, recoveries, collection_recovery_fee, etc.) que contienen información del futuro.

### 6.2 Modelos Base

| Componente | Modelo Principal | Baseline |
|---|---|---|
| PD | CatBoost (tuned con Optuna, 400 trials) | Regresión Logística |
| LGD | CatBoost two-stage (clasificador + regresor) | — |
| EAD | CatBoost Regressor (defaults-only) | — |

**Justificación de CatBoost sobre LightGBM/XGBoost**:
- Manejo nativo de variables categóricas (no requiere encoding).
- Ordered boosting que reduce overfitting en datos temporales.
- Manejo nativo de valores faltantes (no requiere imputación para CatBoost).

**Calibración**: Selección automática entre Platt Scaling e Isotonic Regression mediante validación temporal multi-métrica (ECE, Brier Score, AUC-ROC en folds anclados).

### 6.3 Variantes de Conformal Prediction

Se implementaron **5 variantes** de intervalos predictivos:

| Variante | Target | Librería | Referencia |
|---|---|---|---|
| Split Conformal (Global) | PD | MAPIE 1.3.0 | Vovk et al., 2005 |
| Mondrian Conformal (por Grade) | PD | Custom + MAPIE | Taquet et al., 2025 |
| Venn-Abers Predictors | PD | crepes | Vovk & Petej, 2014 |
| CQR (Split Conformal Regression) | LGD, EAD | MAPIE 1.3.0 | Romano et al., 2019 |
| Residual Intervals (Benchmark) | PD, LGD, EAD | NumPy | Benchmark no-conformal |

**Tuning conformal Mondrian**: Sistema multi-objetivo que optimiza simultáneamente cobertura global, cobertura mínima por grupo, y ancho de intervalos mediante selección Pareto con 8 tiers jerárquicos de prioridad.

### 6.4 Métricas de Evaluación

| Métrica | Propósito |
|---|---|
| ECE (Expected Calibration Error) | Calibración de probabilidades |
| Brier Score | Calibración + discriminación conjunta |
| AUC-ROC / Gini | Poder discriminativo |
| Coverage empírica | Cobertura real vs nominal |
| MPIW (Mean Prediction Interval Width) | Eficiencia de intervalos |
| Kupiec POF Test | Cobertura incondicional (χ²(1)) |
| Christoffersen Test | Cobertura condicional + independencia (χ²(2)) |
| Coverage Disparity | Fairness: max-min cobertura entre grupos |
| Width Ratio | Fairness: ratio max/min ancho entre grupos |

### 6.5 Herramientas y Tecnologías

- **Python 3.11**, paquetes gestionados con **uv**
- **CatBoost 1.2.8**, **scikit-learn 1.6.1**, **Optuna 4.7** para modelado
- **MAPIE 1.3.0** (SplitConformalRegressor, SplitConformalClassifier) para conformal
- **crepes** para Venn-Abers Predictors
- **scipy.stats** para backtesting formal (Kupiec, Christoffersen)
- **DVC** para pipeline reproducible (25 stages)
- **MLflow** para tracking de experimentos
- **Streamlit** para dashboard interactivo
- **pytest** (418+ tests), **ruff** para calidad de código

## 7. Resultados

### 7.1 Modelado Base (PD)

Las métricas del modelo CatBoost calibrado en el conjunto de test OOT (2018-2020) se obtienen dinámicamente de los artefactos del pipeline (`data/processed/model_comparison.json`). Las métricas incluyen:

- **AUC-ROC**: Poder discriminativo del modelo.
- **Gini coefficient**: 2×AUC - 1.
- **Brier Score**: Error cuadrático medio de las probabilidades.
- **ECE**: Error de calibración esperado (10 bins).

La selección de calibrador (Platt vs Isotonic) se realiza automáticamente en cada ejecución del pipeline, priorizando la minimización de ECE sin degradar AUC más de 0.15%.

### 7.2 Conformal Prediction (PD)

**Split Conformal Global**: Intervalos de PD con cobertura nominal del 90%. Implementado via MAPIE `SplitConformalRegressor` con wrapper `ProbabilityRegressor`.

**Mondrian Conformal por Grade**: Intervalos grupo-condicionales que mantienen cobertura independientemente por cada grade crediticio (A-G). Incluye fallback a quantile global para grupos pequeños (< 500 observaciones en calibración).

**Venn-Abers**: Intervalos multi-probabilísticos [p0, p1] con calibración automática garantizada. Implementado via `crepes.WrapClassifier`.

Los resultados de backtesting temporal (cobertura mensual, cobertura por grade, ancho promedio) se persisten en `data/processed/conformal_backtest_monthly.parquet`.

### 7.3 Conformal Prediction (LGD / EAD)

Intervalos conformales de regresión aplicados sobre los modelos de LGD (Stage 2 regressor, defaults-only) y EAD (CatBoost regressor, defaults-only) mediante `SplitConformalRegressor` de MAPIE.

### 7.4 Benchmark: Residual Intervals

Como baseline no-conformal, se implementaron intervalos basados en percentiles de los residuos del conjunto de calibración. Estos intervalos **no tienen garantías formales de cobertura** y sirven para demostrar la ventaja teórica de la predicción conformal.

### 7.5 Backtesting Formal

**Kupiec POF Test** (implementado en `src/evaluation/backtesting.py`):
- H₀: la tasa de violaciones observada = α nominal.
- Estadístico LR distribuido como χ²(1).
- No rechazar H₀ indica cobertura adecuada.

**Christoffersen Test** (implementado en `src/evaluation/backtesting.py`):
- Combina cobertura incondicional (Kupiec) + independencia temporal.
- Matriz de transiciones (n00, n01, n10, n11) para detectar clustering de violaciones.
- Estadístico LR_cc distribuido como χ²(2).
- No rechazar H₀ indica cobertura adecuada sin clustering temporal.

### 7.6 Impacto IFRS9

La predicción conformal se integra directamente en el cálculo de provisiones:

1. **Enhanced SICR**: La amplitud del intervalo conformal (PD_high - PD_point) se usa como señal adicional de Significant Increase in Credit Risk para la migración de Stage 1 a Stage 2.

2. **ECL con rangos conformales**: ECL_low = PD_low × LGD × EAD, ECL_point = PD × LGD × EAD, ECL_high = PD_high × LGD × EAD.

3. **Staging conformal**:
   - Stage 1: Sin SICR → 12-month PD
   - Stage 2: SICR detectado (incluye incertidumbre conformal) → Lifetime PD
   - Stage 3: Credit-impaired (90+ DPD) → PD ≈ 1.0

### 7.7 Fairness Conformal

Además de la auditoría tradicional de fairness (DPD, EO gap, DIR), se evalúa la equidad de los intervalos conformales:

- **Coverage Disparity**: Máxima diferencia de cobertura entre grupos protegidos. Umbral: ≤ 0.05.
- **Width Ratio**: Ratio entre el ancho promedio máximo y mínimo entre grupos. Umbral: ≤ 2.0.

Implementado en `src/evaluation/fairness.py:conformal_fairness_report()`.

## 8. Discusión

### Hallazgos principales

1. **CatBoost como modelo base robusto**: A pesar de que la propuesta original planteaba LightGBM/XGBoost, CatBoost demostró ser una elección sólida por su manejo nativo de categóricas y NaN, eliminando pasos de preprocesamiento que podrían introducir sesgo.

2. **Splits temporales OOT superiores**: La validación out-of-time es metodológicamente más rigurosa que los splits aleatorios propuestos, especialmente para riesgo crediticio donde la distribución evoluciona temporalmente.

3. **Mondrian supera a Split Conformal global**: Los intervalos grupo-condicionales por grade producen cobertura más uniforme entre subgrupos, a costa de un ancho ligeramente mayor en grupos con más incertidumbre.

4. **Tuning multi-objetivo es necesario**: La selección naive de alpha no optimiza el trade-off cobertura-eficiencia. El sistema Pareto con selección jerárquica produce configuraciones robustas.

5. **Enhanced SICR es una contribución original**: Usar la amplitud del intervalo conformal como señal de SICR para IFRS9 no tiene precedente directo en la literatura revisada.

### Limitaciones

- El dataset LendingClub es peer-to-peer, no bancario tradicional. La generalización a portafolios bancarios requiere validación adicional.
- La cobertura conformal es marginal (no condicional), excepto para la variante Mondrian. Esto significa que la cobertura se garantiza en promedio, pero puede variar en subgrupos.
- LGD y EAD solo usan defaults (~18% de la población), lo que reduce el tamaño del conjunto de calibración conformal.
- La intercambiabilidad asumida en los splits temporales es una aproximación. Para datos no estacionarios, métodos conformales adaptativos serían más apropiados.

## 9. Metodología CRISP-DM

Esta sección presenta los resultados y decisiones del proyecto siguiendo el flujo estándar de la metodología CRISP-DM (*Cross-Industry Standard Process for Data Mining*), adaptada al contexto de riesgo crediticio con predicción conformal. CRISP-DM es un modelo de proceso iterativo con seis fases interconectadas y una base de datos al centro de todas las operaciones.

### 9.1 Comprensión del Negocio (*Business Understanding*)

**Qué se hizo**: Se identificó el problema central del riesgo crediticio — los modelos de ML producen estimaciones puntuales de PD, LGD y EAD sin cuantificar la incertidumbre asociada — y se formuló la pregunta de investigación. Se definieron criterios de éxito medibles:

| Criterio | Métrica | Umbral |
|---|---|---|
| Cobertura conformal | Coverage empírica al 90% nominal | ≥ 88% |
| Eficiencia de intervalos | MPIW (ancho promedio) | Minimizar |
| Calibración PD | ECE (Expected Calibration Error) | ≤ 0.05 |
| Backtesting formal | Kupiec / Christoffersen p-value | > 0.05 (no rechazo) |
| Fairness conformal | Coverage disparity entre grupos | ≤ 0.05 |
| Cobertura grupo-condicional | Mondrian coverage mínima por grade | ≥ 85% |

**Retos**: Definir métricas de éxito simultáneamente relevantes para el negocio (provisiones, pricing), la regulación (backtesting Basilea) y la academia (calibración, cobertura CP). Alinear un framework de investigación académica con entregables prácticos y operativos.

**Artefactos**: 6 archivos YAML de configuración (`configs/*.yaml`), 42 tests de consistencia (`tests/test_config_consistency.py`).

### 9.2 Comprensión de los Datos (*Data Understanding*)

**Qué se hizo**: Se seleccionó el dataset de LendingClub (2.26M préstamos, 2007-2020) como caso de estudio. Se realizó un EDA exhaustivo identificando:

- Tasa de default no estacionaria: 18-22% según período.
- Variables categóricas dominantes (grade, sub_grade, home_ownership, purpose, emp_length).
- Valores faltantes concentrados en variables de empleo y crédito secundario.
- 15+ variables leaky (post-loan) que debían removerse.

**Retos**: La identificación de data leakage requirió análisis detallado de cada variable para determinar si contenía información posterior al momento de originación. Error en esta fase habría invalidado todos los resultados. La tasa de default no estacionaria (18.5% train vs 22.2% calibración) justificó la decisión de usar splits temporales.

**Artefactos**: `notebooks/01_eda_lending_club.ipynb`, 29 tests de datos.

### 9.3 Preparación de los Datos (*Data Preparation*)

**Qué se hizo**: Se implementaron splits temporales Out-of-Time (OOT) — mejora fundamental respecto a la propuesta original que planteaba splits aleatorios 60/20/20. Se construyeron tres datasets analíticos:

| Dataset | Filas | Propósito |
|---|---|---|
| loan_master.parquet | 1.35M (train OOT) | PD, supervivencia, optimización |
| ead_dataset.parquet | ~249K defaults (train OOT) | Modelado EAD |
| time_series.parquet | 118 meses | Forecasting temporal |

Transformaciones clave: remoción de variables leaky, splits temporales OOT, WOE encoding vía OptBinning, feature contract persistido, validación con Pandera schemas.

**Retos**: Balance entre splits (calibración 237K suficiente para CP pero sin reducir excesivamente entrenamiento). Tres conjuntos disjuntos obligatorios (train/calibración/test) requeridos por conformal prediction. Feature config como contrato inmutable entre entrenamiento e inferencia.

**Artefactos**: `src/data/prepare_dataset.py`, `src/data/build_datasets.py`, `src/features/feature_engineering.py`, `feature_config.pkl`, 5 tests de features + schemas Pandera.

### 9.4 Modelado (*Modeling*)

**Qué se hizo**: Se implementaron tres capas de modelado para esta tesis:

**Capa 1 — Modelos base**: Regresión Logística (baseline), CatBoost Default + Tuned (HPO Optuna 400 trials), CatBoost Two-Stage (LGD), CatBoost Regressor (EAD).

**Capa 2 — Calibración**: Selección automática entre Platt Scaling e Isotonic Regression mediante validación temporal multi-métrica (ECE, Brier Score, AUC-ROC en folds anclados).

**Capa 3 — Predicción conformal**: 5 variantes implementadas:

| Variante | Target | Garantía de cobertura |
|---|---|---|
| Split Conformal Global | PD | P(y ∈ C(x)) ≥ 1−α marginal |
| Mondrian por Grade | PD | P(y ∈ C(x) | g) ≥ 1−α por grupo |
| Venn-Abers | PD | Calibración multiprobabilística |
| CQR (MAPIE) | LGD, EAD | P(y ∈ C(x)) ≥ 1−α regresión |
| Residual (benchmark) | PD, LGD, EAD | Sin garantía formal |

**Retos**: MAPIE 1.3.0 API breaking changes (migración a SplitConformalRegressor). ProbabilityRegressor wrapper (MAPIE espera regresor, CatBoost es clasificador). Tuning Mondrian multi-objetivo sin solución cerrada (selección Pareto con 8 tiers). API de crepes: `WrapClassifier.predict_p()` retorna ndarray(n, 2), no tuple.

**Artefactos**: `src/models/pd_model.py`, `src/models/calibration.py`, `src/models/conformal.py`, `src/models/conformal_tuning.py`. 87 tests de modelos.

### 9.5 Evaluación (*Evaluation*)

**Qué se hizo**: Evaluación integral en cuatro dimensiones:

1. **Calibración y discriminación** (AUC-ROC, Gini, Brier Score, ECE, KS).
2. **Intervalos conformales** (Coverage empírica, MPIW, coverage por grupo, Kupiec POF, Christoffersen).
3. **Impacto regulatorio IFRS 9** (ECL con/sin conformal, distribución por Stage, sensibilidad a escenarios).
4. **Fairness** (DPD, EO gap, DIR para predicciones puntuales; Coverage Disparity y Width Ratio para intervalos conformales).

**Retos**: Evaluar 5 variantes con métricas comparables (cada una tiene propiedades distintas). Kupiec/Christoffersen con muestras pequeñas en grupos individuales (fallback a quantile global para grupos < 500). PSI temporal para monitorear estabilidad de distribuciones.

**Artefactos**: `src/evaluation/metrics.py`, `src/evaluation/backtesting.py`, `src/evaluation/ifrs9.py`, `src/evaluation/fairness.py`. 68 tests de evaluación.

### 9.6 Despliegue (*Deployment*)

**Qué se hizo**: Aunque el alcance de la tesis no incluye despliegue en producción bancaria, se implementó una infraestructura de despliegue completa como prototipo operativo:

| Componente | Tecnología | Estado |
|---|---|---|
| Dashboard interactivo | Streamlit (27 páginas) | Operativo |
| API REST | FastAPI (15 endpoints) | Operativo |
| MCP Server | FastMCP (4 tools, 3 resources) | Operativo |
| Pipeline reproducible | DVC (25 stages) | Operativo |
| Experiment tracking | MLflow + DagsHub | Operativo |
| CI/CD | GitHub Actions (lint + test + smoke) | Operativo |
| Containerización | Docker Compose (api + streamlit) | Operativo |
| Test suite | pytest (418 tests) | 100% passing |

Pipeline DVC end-to-end: `make_dataset → prepare_dataset → build_datasets → feature_engineering → train_pd → generate_conformal → backtest_conformal → validate_conformal → estimate_causal → simulate_causal → validate_causal → ifrs9_sensitivity → optimize_portfolio → portfolio_tradeoff → fairness_audit → cate_portfolio → ab_simulation → mrm_report → export_streamlit_artifacts`

**Retos**: Reproducibilidad end-to-end con 25 stages y dependencias cruzadas (DVC + lock file). Aislamiento explícito de entorno principal (`.venv`) y entorno RAPIDS side-projects (`conda` env `rapids`) para evitar conflictos GPU. Smoke testing de 27 páginas Streamlit (AST parsing). Modelo canónico vs modelo de experimentación (protocolo de gates de calidad).

**Artefactos**: `dvc.yaml`, `scripts/end_to_end_pipeline.py`, `api/`, `streamlit_app/`, `mcp_server/`, `.github/workflows/ci.yml`, `docker-compose.yml`. 57 tests Streamlit + 5 tests API + 8 integración.

### 9.7 Resumen CRISP-DM

El ciclo CRISP-DM se ejecutó de forma iterativa, no lineal. Las flechas bidireccionales entre fases reflejan la realidad del proyecto:

- **Business ↔ Data Understanding**: La identificación de data leakage reformuló los criterios de éxito.
- **Data Preparation ↔ Modeling**: CatBoost maneja categóricas y NaN nativamente, simplificando la preparación.
- **Modeling ↔ Evaluation**: Los resultados de backtesting (Kupiec/Christoffersen) retroalimentaron el tuning conformal.
- **Evaluation → Deployment**: Las métricas de fairness conformal generaron un nuevo requisito de gobernanza incorporado en el MRM report.

| Fase CRISP-DM | Entregable principal | Artefacto clave | Tests |
|---|---|---|---|
| 1. Business Understanding | Pregunta de investigación + criterios de éxito | configs/*.yaml (7 archivos) | 54 tests de config |
| 2. Data Understanding | EDA completo + identificación data leakage | notebooks/01_eda_lending_club.ipynb | 29 tests de datos |
| 3. Data Preparation | 3 datasets + splits OOT + features WOE | data/processed/*.parquet + feature_config.pkl | 5 tests de features |
| 4. Modeling | 5 modelos base + 5 variantes conformal | models/pd_canonical.cbm + conformal intervals | 87 tests de modelos |
| 5. Evaluation | Backtesting, IFRS9, fairness, métricas | model_comparison.json + backtesting reports | 68 tests de evaluación |
| 6. Deployment | Página Streamlit + Informe MD + DVC pipeline | tesis_especializacion.py + tesis_especializacion.md | 418 tests totales |

---

## 10. Conclusiones

### Aportes

1. Se implementaron **5 variantes de intervalos conformales para riesgo crediticio** (Split Conformal, Mondrian by Grade, Venn-Abers, CQR para LGD/EAD, Residual como benchmark), superando las 3 planteadas en la propuesta original.
2. **Cobertura conjunta PD + LGD + EAD**: Primera aplicación documentada de conformal prediction sobre los tres componentes del riesgo crediticio simultáneamente.
3. Se desarrolló un **sistema de tuning conformal multi-objetivo (Mondrian)** con optimización Pareto, selección jerárquica, multiplicadores por grupo y floor de cobertura — va más allá de la propuesta y no existe en la literatura revisada.
4. Se propuso el uso del **ancho del intervalo conformal como señal de SICR** para staging IFRS9 (contribución original).
5. Se implementaron **tests formales de backtesting** (Kupiec, Christoffersen) para validación regulatoria bajo Basilea III.
6. Se desarrolló una **auditoría de fairness conformal** para evaluar disparidades de cobertura y ancho de intervalos entre grupos protegidos (contribución original).

### Trabajo Futuro y Proyección hacia la Maestría

Esta tesis de especialización sienta las bases para un trabajo de mayor alcance en la maestría, donde se planea:

- **Pipeline predict-then-optimize**: Los intervalos conformales generados en esta tesis pueden servir como conjuntos de incertidumbre para modelos de optimización robusta de portafolio. En lugar de optimizar usando solo la PD puntual, se usaría el rango [PD_low, PD_high] para encontrar portafolios óptimos bajo cualquier realización de PD dentro del intervalo conformal. Esto conecta la cuantificación de incertidumbre (esta tesis) con la toma de decisiones óptima bajo incertidumbre (maestría), usando herramientas como Pyomo y HiGHS.
- **Inferencia causal**: Integrar efectos causales (CATE) con los intervalos conformales para decisiones de tratamiento crediticio.
- **Adaptive conformal prediction** para manejo de concept drift.
- Conformalized Quantile Regression con LightGBM como base (comparación directa).
- Aplicación a datasets bancarios reales (con datos protegidos).

### Recomendaciones

- Para adopción en entornos bancarios reales, se recomienda validar con datos protegidos y ajustar los umbrales de SICR conformal.
- Integrar intervalos conformales de LGD y EAD en el cálculo de ECL para obtener rangos completos de provisiones.

## 11. Referencias

[1] Basel Committee on Banking Supervision, "International Convergence of Capital Measurement and Capital Standards: A Revised Framework," BIS, 2006.

[2] International Accounting Standards Board, "IFRS 9 Financial Instruments," IFRS Foundation, 2014.

[3] S. Lessmann, B. Baesens, H.-V. Seow, and L. C. Thomas, "Benchmarking state-of-the-art classification algorithms for credit scoring," EJOR, vol. 247, no. 1, pp. 124-136, 2015.

[4] T. Bellotti, "Reliable region predictions for automated credit scoring," COPA, PMLR, 2017.

[5] A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," ICML, pp. 625-632, 2005.

[6] V. Vovk, A. Gammerman, and G. Shafer, *Algorithmic Learning in a Random World*, 2nd ed. Springer, 2022.

[7] A. N. Angelopoulos and S. Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification," FnTML, vol. 16, no. 4, pp. 494-591, 2023.

[8] Y. Romano, E. Patterson, and E. Candes, "Conformalized Quantile Regression," NeurIPS, vol. 32, 2019.

[9] Lending Club, "Lending Club Loan Data 2007-2020 Q1," Kaggle, 2020.

[10] Deloitte, "IFRS 9 and Expected Credit Loss: Modelling and Validation Challenges," 2020.

[11] B. Baesens, D. Roesch, and H. Scheule, *Credit Risk Analytics*, Wiley, 2016.

[12] S. R. Rao and A. Verma, "Fairness in Credit Scoring," EJOR, vol. 297, no. 3, pp. 1083-1096, 2022.

[13] European Banking Authority, "Guidelines on Loan Origination and Monitoring," EBA/GL/2020/06, 2020.

[14] V. Vovk and I. Petej, "Venn-Abers Predictors," UAI, pp. 829-838, 2014.

[15] F. Javanmardi and V. Vovk, "Multi probability predictions for credit scoring with Venn-Abers predictors," COPA, PMLR, 2023.

[16] H. Lu et al., "Conformal Prediction in the Medical Domain," IEEE Reviews in Biomedical Engineering, vol. 17, pp. 127-142, 2024.

[17] M. Almunia et al., "Machine Learning in Credit Risk: Literature Review and Practical Implications," Bank of Spain, 2023.

[18] J. Platt, "Probabilistic Outputs for Support Vector Machines," MIT Press, pp. 61-74, 1999.

[19] R. F. Barber et al., "Predictive Inference with the Jackknife+," Annals of Statistics, vol. 49, no. 1, pp. 486-507, 2021.

[20] C. Xu and Y. Xie, "Conformal Prediction for Time Series with Modern Hopfield Networks," NeurIPS, vol. 36, 2023.

[21] V. Balasubramanian, S.-S. Ho, and V. Vovk (Eds.), *Conformal Prediction for Reliable Machine Learning*, Morgan Kaufmann, 2014.

[22] MAPIE Contributors, "MAPIE: Model Agnostic Prediction Interval Estimator," 2023.

[23] G. Tasche, "Validation of internal model hypotheses," Deutsche Bundesbank, 2006.

[24] A. K. Gneiting and A. E. Raftery, "Strictly Proper Scoring Rules," JASA, vol. 102, pp. 359-378, 2007.

[25] M. Fontana, G. Zeni, and S. Vantini, "Conformal Prediction: A Unified Review," Bernoulli, vol. 29, no. 1, pp. 1-23, 2023.

---

## Anexo: Estructura del Código Fuente

```
src/
├── data/          → make_dataset, prepare_dataset, build_datasets
├── features/      → feature_engineering, schemas (Pandera)
├── models/        → pd_model, calibration, conformal, lgd_model, ead_model
├── optimization/  → portfolio_model, robust_opt, spo_integration
├── evaluation/    → metrics, backtesting (Kupiec, Christoffersen), ifrs9, fairness
└── utils/         → mlflow_utils, visualization

scripts/           → 15+ scripts ejecutables (pipeline DVC de 25 stages)
configs/           → 6 archivos YAML de configuración
tests/             → 418+ tests (pytest)
streamlit_app/     → 27 páginas interactivas
```

## Anexo: Funciones Clave Implementadas

| Función | Archivo | Propósito |
|---|---|---|
| `create_pd_intervals()` | `src/models/conformal.py` | Split conformal global para PD |
| `create_pd_intervals_mondrian()` | `src/models/conformal.py` | Mondrian conformal por grade |
| `create_pd_intervals_venn_abers()` | `src/models/conformal.py` | Venn-Abers multi-probability |
| `create_regression_intervals()` | `src/models/conformal.py` | CQR para LGD/EAD |
| `create_residual_intervals()` | `src/models/conformal.py` | Benchmark residual (no-conformal) |
| `kupiec_pof_test()` | `src/evaluation/backtesting.py` | Test de cobertura incondicional |
| `christoffersen_test()` | `src/evaluation/backtesting.py` | Test de cobertura condicional |
| `conformal_fairness_report()` | `src/evaluation/fairness.py` | Auditoría fairness de intervalos |
| `assign_stage()` | `src/evaluation/ifrs9.py` | Staging IFRS9 con SICR conformal |
| `ecl_with_conformal_range()` | `src/evaluation/ifrs9.py` | ECL con rangos conformales |
