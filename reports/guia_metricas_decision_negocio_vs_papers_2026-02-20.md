# Guía pedagógica de métricas y decisiones (negocio vs papers)
Fecha: 2026-02-20

## 0) Objetivo de este documento
Este documento está escrito para una audiencia no experta. Busca responder cuatro preguntas:

1. Qué significa cada métrica del proyecto.
2. Qué impacto real tiene en negocio cuando sube o baja.
3. Cuándo conviene elegir una estrategia u otra (retorno, balanceado, prudente).
4. Qué parte de esto es práctica de negocio y qué parte es más común en papers.

También sirve como guion para explicar la historia en Streamlit sin depender de jerga técnica.

---

## 1) Fuentes y snapshot usado
Artefactos consultados:

- `reports/before_after_recompute_comparison_longrun.json`
- `models/conformal_policy_status.json`
- `data/processed/portfolio_robustness_summary.parquet`
- `data/processed/portfolio_robustness_frontier.parquet`
- `data/processed/ifrs9_scenario_summary.parquet`
- `models/pipeline_results.pkl`

Importante: hay dos fotos temporales distintas.

1. **Before -> After (tabla original de comparación)**: usada para explicar la mejora de pipeline.
2. **Estado actual (recovery con max_candidates=12000)**: usada para operación actual de optimización.

---

## 2) Qué significa cada métrica (en palabras simples)

### 2.1 Métricas PD (modelo de default)

#### AUC
- Qué es: qué tan bien ordena el modelo a los préstamos riesgosos vs seguros.
- Cómo leerlo:
  - 0.50: aleatorio.
  - 1.00: perfecto.
  - más alto = mejor ordenamiento.
- Impacto negocio:
  - mejor AUC suele mejorar decisiones de aprobación, límites y pricing.
  - reduce errores de “aprobar muy riesgoso” o “rechazar buen cliente”.

#### Gini
- Qué es: la misma idea de AUC en escala de riesgo (`Gini = 2*AUC - 1`).
- Cómo leerlo:
  - más alto = mejor separación.
- Impacto negocio:
  - muchas áreas de riesgo usan Gini para reportes ejecutivos/comités.

#### KS
- Qué es: separación máxima entre distribución de score de malos vs buenos pagadores.
- Cómo leerlo:
  - más alto = umbrales de corte más claros.
- Impacto negocio:
  - ayuda a definir políticas de aceptación/rechazo más defendibles.

#### Brier
- Qué es: error cuadrático de probabilidades.
- Cómo leerlo:
  - más bajo = probabilidades más útiles.
- Impacto negocio:
  - mejora cálculo de pérdida esperada y provisión.
  - reduce distorsión en pricing basado en PD.

#### ECE
- Qué es: cuánto difieren probabilidades predichas de frecuencias reales.
- Cómo leerlo:
  - más bajo = mejor calibración.
- Impacto negocio:
  - clave para IFRS9 y decisiones donde importa el nivel de PD (no solo el ranking).

### 2.2 Métricas Conformal (incertidumbre)

#### Coverage 90% y 95%
- Qué es: porcentaje real de casos que cae dentro del intervalo de PD.
- Cómo leerlo:
  - idealmente cerca o por encima de la meta nominal (90%, 95%).
- Impacto negocio:
  - más cobertura = menos riesgo de subestimar incertidumbre.
  - útil para decisiones conservadoras y gobernanza.

#### Avg width 90%
- Qué es: ancho promedio del intervalo.
- Cómo leerlo:
  - más chico = más precisión.
  - pero si se hace demasiado chico, puede caer cobertura.
- Impacto negocio:
  - intervalos más finos mejoran granularidad de decisión.
  - el trade-off natural es “precisión vs protección”.

### 2.3 Métricas IFRS9

#### Baseline total ECL y Severe total ECL
- Qué es: provisión esperada bajo escenario base y severo.
- Cómo leerlo:
  - más alto = más provisión contable requerida.
- Impacto negocio:
  - pega directo en P&L y capital.
  - define cuánto “colchón” financiero debe reservarse.

### 2.4 Métricas de optimización

#### Robust return
- Qué es: retorno neto esperado de la política robusta.
- Impacto:
  - mide la calidad económica de una política protegida contra incertidumbre.

#### Non-robust return
- Qué es: retorno neto esperado de la política sin robustez explícita.
- Impacto:
  - da el “techo” de retorno optimista.

#### Price of Robustness (PoR)
- Qué es: costo económico de pasar de política optimista a robusta.
- Fórmula: `PoR = non_robust_return - robust_return`.
- Impacto:
  - lenguaje de negocio para explicar “costo del seguro de robustez”.

---

## 3) Diferencias observadas en tu proyecto

## 3.1 Before -> After (comparación original)

| Bloque | Métrica | Before | After | Delta | Lectura de negocio |
|---|---:|---:|---:|---:|---|
| PD | AUC | 0.6990 | 0.7172 | +0.0182 | Mejor ordenamiento de riesgo. |
| PD | Gini | 0.3980 | 0.4344 | +0.0363 | Mayor separación score. |
| PD | KS | 0.2942 | 0.3200 | +0.0258 | Umbrales más claros para política. |
| PD | Brier | 0.1572 | 0.1538 | -0.0034 | Probabilidades más útiles. |
| PD | ECE | 0.0130 | 0.0094 | -0.0036 | Calibración mejora fuerte. |
| Conformal | Coverage 90 | 0.8887 | 0.8917 | +0.0030 | Más protección. |
| Conformal | Coverage 95 | 0.9480 | 0.9511 | +0.0031 | Más protección. |
| Conformal | Avg width 90 | 0.7459 | 0.7225 | -0.0234 | Mejor precisión con cobertura algo mayor. |
| IFRS9 | Baseline ECL | 1,010,417,398 | 967,995,472 | -42,421,926 | Menor provisión esperada. |
| IFRS9 | Severe ECL | 1,850,841,855 | 1,779,233,609 | -71,608,245 | Menor provisión en estrés severo. |
| Optimización | Robust return | 2,279.61 | 62,029.96 | +59,750.35 | Política robusta pasó de casi inviable a utilizable. |
| Optimización | Non-robust return | 77,346.30 | 98,235.01 | +20,888.71 | Mejora económica general. |
| Optimización | PoR | 75,066.69 | 36,205.04 | -38,861.64 | Bajó el costo de robustez en ese snapshot. |

## 3.2 Estado actual (recovery de optimización)
Con `max_candidates=12000`, para `risk_tolerance=0.10`:

- `robust_return = 70,050.33`
- `nonrobust_return = 123,451.86`
- `price_of_robustness = 53,401.53` (`43.26%`)

Lectura:
- Subió retorno robusto vs snapshot anterior.
- También subió retorno no robusto.
- El costo de robustez absoluto y relativo volvió a ser más alto, porque ahora el benchmark no robusto también crece.

---

## 4) Cuándo tomar una estrategia u otra

## 4.1 Perfil 1: Retorno
Parámetros de referencia:
- `risk_tolerance=0.12`
- `uncertainty_aversion=0.0`

Cuándo usar:
- objetivo comercial agresivo.
- presión por crecimiento de originación y margen.
- entorno macro relativamente estable.

Impacto esperado:
- mayor retorno.
- más préstamos financiados.
- menos margen de seguridad ante deterioro inesperado.

## 4.2 Perfil 2: Balanceado
Parámetros de referencia:
- `risk_tolerance=0.10`
- `uncertainty_aversion=0.0` (base operativa actual)

Cuándo usar:
- operación normal de negocio.
- metas mixtas: rentabilidad + control de riesgo.
- comité de riesgo busca consistencia mensual sin frenar crecimiento.

Impacto esperado:
- buen equilibrio entre upside y robustez.
- mejor defendible para operación continua.

## 4.3 Perfil 3: Prudente
Parámetros de referencia:
- `risk_tolerance=0.06`
- `uncertainty_aversion=2.0`

Cuándo usar:
- señales de estrés macro o aumento de morosidad.
- prioridad en preservación de capital.
- apetito de riesgo reducido por dirección/comité.

Impacto esperado:
- menor retorno y menor volumen financiado.
- mayor protección en peor caso de PD.

---

## 5) Regla simple de decisión (para explicar fácil)

1. Si el negocio pide **crecer retorno**: usar perfil Retorno.
2. Si pide **sostener crecimiento con control**: usar perfil Balanceado.
3. Si pide **defender capital**: usar perfil Prudente.

Siempre revisar en paralelo:
- conformal policy (checks de cobertura y alertas),
- IFRS9 baseline/severe ECL,
- Price of Robustness y funded count.

---

## 6) Esto lo hacen en negocio o solo en papers

| Práctica | Negocio | Papers | Comentario útil para explicar |
|---|---|---|---|
| AUC / KS para score | Muy común | Muy común | Base de evaluación de riesgo en ambos mundos. |
| Brier / ECE calibración | Común en equipos maduros | Muy común | Vital cuando PD alimenta provisiones y pricing. |
| Conformal prediction | Emergente | Crecimiento fuerte | Más común en investigación; adopción práctica en aumento. |
| Optimización robusta con conjuntos de incertidumbre | Selectiva (casos de alto valor) | Establecida | En negocio se usa, pero no siempre con formalismo académico completo. |
| Price of Robustness como KPI explícito | Menos común | Muy común | En negocio muchas veces se discute implícitamente (sin ese nombre). |
| IFRS9 Stage + escenarios ECL | Obligatorio | Muy estudiado | Es puente directo entre modelado y contabilidad regulatoria. |

Conclusión práctica:
- No es “solo paper”. El núcleo (AUC/KS, calibración, IFRS9) es totalmente de negocio.
- Lo más “de frontera” es formalizar incertidumbre + robustez de forma explícita y trazable.

---

## 7) Guion corto para presentar a alguien no técnico

“Primero medimos qué tan bien el modelo ordena riesgo (AUC/KS).  
Luego validamos que sus probabilidades sean confiables (Brier/ECE).  
Después le agregamos incertidumbre con conformal para no decidir con un solo número puntual.  
Con eso optimizamos cartera en modo robusto y medimos su costo (Price of Robustness).  
Finalmente vemos el impacto contable real en IFRS9 (ECL baseline y severo).”

---

## 8) Qué mirar primero en Streamlit

Orden recomendado para aprendizaje:

1. `streamlit_app/pages/executive_summary.py`
2. `streamlit_app/pages/glossary_fundamentals.py`
3. `streamlit_app/pages/model_laboratory.py`
4. `streamlit_app/pages/portfolio_optimizer.py`
5. `streamlit_app/pages/ifrs9_provisions.py`
6. `streamlit_app/pages/research_landscape.py`

---

## 9) Nota metodológica importante

No optimizar una sola métrica aislada.

Ejemplos:
- subir AUC sin cuidar ECE puede romper provisiones.
- subir cobertura conformal demasiado puede ensanchar intervalos y frenar negocio.
- maximizar retorno sin robustez puede aumentar fragilidad en estrés.

La calidad real del sistema aparece en el balance conjunto de métricas.

---

## 10) Referencias rápidas para sustentar la explicación

Referencias internas del proyecto:

- `streamlit_app/pages/research_landscape.py`
- `streamlit_app/pages/paper_1_cp_robust_opt.py`
- `streamlit_app/pages/glossary_fundamentals.py`

Referencias académicas/base citadas en el proyecto (útiles para defensa):

- Bertsimas & Sim (2004), *The Price of Robustness*.
- Elmachtoub & Grigas (2022), *Smart Predict-then-Optimize (SPO+)*.
- Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random World*.
- Angelopoulos & Bates (2023), *Conformal Prediction: A Gentle Introduction*.
- Chernozhukov et al. (2018), *Double/Debiased Machine Learning*.

Marco regulatorio de negocio:

- IFRS 9 para provisión esperada (ECL).
- Basilea III para marco de capital y riesgo.
