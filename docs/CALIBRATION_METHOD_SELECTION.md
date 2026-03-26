# Selección de Método de Calibración PD — Formal Writeup

**Estado**: Canónico post paper-grade run 2026-03-13
**Método seleccionado**: Venn-Abers
**Artefacto fuente**: `models/pd_calibration_diagnostics.json`

---

## El problema de calibración en riesgo de crédito

Un modelo PD bien calibrado satisface: para todos los préstamos con PD predicha $\hat{p}$, la tasa de default observada se aproxima a $\hat{p}$. Formalmente, la *calibración marginal* exige:

$$\mathbb{E}[Y \mid \hat{p}(X) = p] \approx p \quad \forall p \in [0,1]$$

En la práctica se mide con la **Expected Calibration Error (ECE)**:

$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$

donde $B_b$ es el bin $b$, $\text{acc}(B_b)$ es la tasa de default observada en el bin, y $\text{conf}(B_b)$ es la media de probabilidades predichas.

**Por qué importa en crédito:**
- Una PD mal calibrada distorsiona el ECL bajo IFRS9 (`ECL = PD × LGD × EAD`)
- Los intervalos conformales downstream heredan el error de calibración de las PD base
- El pricing basado en riesgo usa PD directamente como input de rentabilidad ajustada

---

## Métodos evaluados (4 candidatos)

### 1. Platt Scaling (Platt 1999)

Entrena una regresión logística sobre los scores brutos del modelo:

$$\hat{p}_{\text{Platt}}(s) = \frac{1}{1 + \exp(As + B)}$$

- **Ventaja**: simple, estable, opera sobre 1 parámetro real
- **Limitación**: asume que la relación score → probabilidad es logística; falla cuando la distribución del score es multimodal o tiene colas pesadas
- **Garantía teórica**: ninguna distribución-libre; requiere que el score base sea bien comportado

### 2. Isotonic Regression (Zadrozny & Elkan 2002)

Ajuste no paramétrico con restricción de monotonía:

$$\hat{p}_{\text{iso}} = \arg\min_{f \text{ monótona}} \sum_i (y_i - f(s_i))^2$$

- **Ventaja**: más flexible que Platt; captura cualquier relación monótona
- **Limitación**: puede sobreajustar con pocos datos; no tiene garantía de cobertura finita
- **Garantía teórica**: consistencia asintótica (requiere $n \to \infty$)

### 3. Venn-Abers Calibration (Vovk & Petej 2012; Vovk et al. 2015)

Método conformal de calibración que produce **intervalos de probabilidad** con garantías de cobertura finita. Para cada nuevo punto $x$, produce un par $(p_0, p_1)$ donde:

$$p_0 = \text{Venn predictor bajo } y=0, \quad p_1 = \text{Venn predictor bajo } y=1$$

La probabilidad calibrada se estima como:

$$\hat{p}_{\text{VA}} = \frac{p_0 + p_1}{2}$$

**Garantía teórica clave**: bajo intercambiabilidad (i.i.d.), Venn-Abers produce probabilidades *marginalmente calibradas* con garantía de cobertura finita — sin hipótesis distribucionales adicionales.

- **Ventaja**: distribución-libre, calibración con garantía finita, produce bounds que cuantifican incertidumbre epistémica de la calibración
- **Ventaja adicional**: el ancho del intervalo VA (`avg_width`) es un indicador de confianza en la calibración por punto
- **Limitación**: overhead computacional O(n log n) vs O(1) de Platt; no aplica `unbiasedness_in_the_large` cuando hay shift de prevalencia

### 4. Beta Calibration (Kull et al. 2017)

Ajuste paramétrico flexible con tres parámetros (a, b, m) que modela la transformación score → probabilidad mediante una distribución Beta:

$$\hat{p}_{\text{Beta}}(s) = \frac{1}{1 + \frac{1-s^a}{s^a} \cdot \frac{1}{e^{b + m \cdot \log(s/(1-s))}}}$$

- **Ventaja**: más flexible que Platt (3 parámetros vs 2), captura asimetrías en la distribución de scores
- **Ventaja adicional**: forma funcional paramétrica → no sobreajusta como Isotonic con muestras pequeñas
- **Limitación**: requiere scores en (0,1); puede ser inestable con scores extremos (< 0.001 o > 0.999)
- **Garantía teórica**: consistencia asintótica; sin garantía distribución-libre finita (a diferencia de Venn-Abers)
- **Implementación**: `betacal.BetaCalibration(parameters="abm")` en `src/models/calibration.py`

---

## Resultados de comparación (paper-grade run 2026-03-13)

| Método | ECE (OOT) | Brier (OOT) | AUC (OOT) |
|--------|-----------|-------------|-----------|
| Platt | 0.0084 | 0.1546 | 0.7129 |
| Isotonic | 0.0062 | 0.1545 | 0.7128 |
| **Venn-Abers** | **0.0061** | **0.1545** | **0.7128** |

**Metadatos Venn-Abers (OOT, n=276,869):**
- `mean_p0 = 0.2164`, `mean_p1 = 0.2170` (prevalencia observada: 21.98%)
- `avg_width = 0.0005` (ancho medio del intervalo VA → calibración muy estable)
- `median_width = 0.0003`
- `unbiasedness_in_the_large = False` (señal de drift leve de prevalencia entre cal/test, esperado en split OOT estricto)

**Interpretación del `avg_width = 0.0005`:** los bounds de Venn-Abers son casi colapsos de punto, indicando que la calibración es altamente estable a través del espacio de scores. Si hubiera incertidumbre epistémica significativa en la calibración, el ancho sería mayor (≥ 0.01).

---

## Rationale de selección

Venn-Abers fue seleccionado como método canónico por tres razones:

1. **Mejor ECE en datos OOT**: ECE=0.0061 vs 0.0062 (Isotonic) vs 0.0084 (Platt). La diferencia respecto a Isotonic es pequeña (~1.6%) pero Venn-Abers logra esta mejora con garantías teóricas adicionales.

2. **Garantía de calibración con muestra finita**: a diferencia de Platt/Isotonic, la calibración Venn-Abers es válida bajo intercambiabilidad sin asunciones distribucionales. Esto es coherente con la filosofía conformal del proyecto.

3. **Compatibilidad arquitectónica**: el proyecto usa MAPIE Mondrian Conformal para los intervalos de PD. Usar Venn-Abers para la calibración base crea una arquitectura de incertidumbre coherente: calibración conformal + intervalos conformales.

**Cuándo revisar esta decisión:**
- Si el `avg_width` Venn-Abers crece sustancialmente (> 0.01) indicando inestabilidad
- Si el AUC drop relativo a uncalibrated supera 0.15%
- Si un nuevo método (e.g., Temperature Scaling + conformal) muestra ECE < 0.005 en datos OOT

---

## Referencias clave

1. **Vovk, V. & Petej, I. (2012)** — Venn prediction. [arXiv:1211.6990]
2. **Vovk, V., Shen, J., Manokhin, V., Xie, M. (2017)** — Nonparametric predictive distributions based on conformal prediction. JMLR.
3. **Platt, J. (1999)** — Probabilistic outputs for SVMs and comparisons to regularized likelihood methods. Advances in Large Margin Classifiers.
4. **Zadrozny, B. & Elkan, C. (2002)** — Transforming classifier scores into accurate multiclass probability estimates. KDD.
5. **Brier, G.W. (1950)** — Verification of forecasts expressed in terms of probability. Monthly Weather Review.
6. **Gneiting, T. & Raftery, A.E. (2007)** — Strictly proper scoring rules, prediction, and estimation. JASA.

---

## Artefactos canónicos

| Artefacto | Ruta | Contenido |
|-----------|------|-----------|
| Diagnósticos completos | `models/pd_calibration_diagnostics.json` | Comparación 4-way, VA bounds, bins de confiabilidad |
| Calibrador serializado | `models/pd_canonical_calibrator.pkl` | Objeto `VennAbersScoreCalibrator` fitted |
| Estado de calibración rare-event | `models/pd_rare_event_calibration_status.json` | ECE por grupo protegido, worst-grade Brier |
| Reporte rare-event | `data/processed/pd_rare_event_calibration_report.parquet` | Por slice: ECE, Brier, PR-AUC |
| Implementación | `src/models/venn_abers.py` | `VennAbersScoreCalibrator` |

---

*Última actualización: 2026-03-16 | Run: `paper-grade-2026-03-13-final-heavy-2026-03-13-230650`*
