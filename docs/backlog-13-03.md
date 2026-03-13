<!-- cspell:disable -->
<!-- markdownlint-disable -->

# Backlog 13-03

Fecha: 2026-03-13

Estado base:
- Baseline operativo actual: `champion-2026-03-12-mega-definitive`
- Registry oficial: `configs/baselines/canonical_operational_baseline.json`
- Objetivo de este backlog: cerrar pendientes metodológicos y operativos antes de la corrida final paper-grade con `study_name` limpio de PD

## Prioridad global

Orden recomendado:
1. PD conformal estricto
2. Time series intervals
3. Causal policy / CATE
4. A/B más fuerte
5. Governance warnings
6. Cierre de protocolo paper
7. Study limpio de PD + mega corrida final paper-grade

## Resumen ejecutivo

Lo ya promovido:
- PD core con calibración isotónica
- policy champion de portfolio
- survival RSF
- fairness
- governance operativo
- LGD/EAD conformal

Lo pendiente de cierre:
- PD conformal estricto
- intervalos de time series
- decisión causal/CATE
- evidencia A/B más fuerte
- limpieza narrativa de governance
- congelación formal del protocolo final

## Update incorporado de la mega run `champion-2026-03-12-mega-definitive`

### Estado general de la corrida

- Run tag: `champion-2026-03-12-mega-definitive`
- Perfil: `champion_search_max`
- Sampling profile: `mega64plus`
- Estado final: `PASS`
- Gates globales: `overall_pass=true`
- Resultado operativo: promovido como nuevo baseline oficial
- Registry oficial actualizado en:
  - `configs/baselines/canonical_operational_baseline.json`
  - `configs/baselines/core_official_baseline.json`

### Qué quedó actualizado oficialmente

- El baseline operativo anterior `2026-03-11-C-official-selector-v3-freeze` fue reemplazado por `champion-2026-03-12-mega-definitive`
- `champion_registry.json` fue regenerado con el nuevo champion operativo
- La comparación oficial del run quedó en:
  - `reports/run_comparisons/champion-2026-03-12-mega-definitive/comparison.json`
  - `reports/run_comparisons/champion-2026-03-12-mega-definitive/comparison.md`

### Mejores concretas obtenidas en la mega run

#### PD core

- Mejor modelo final: `CatBoost (tuned + calibrated)`
- Calibración elegida: `Isotonic Regression`
- Trials HPO ejecutados: `295`
- Mejor AUC de validación temporal en HPO: `0.7226`
- Métricas finales del modelo promovido:
  - AUC: `0.7128`
  - Gini: `0.4256`
  - Brier: `0.1545`
  - D2 Brier: `0.0988`

Lectura:
- hubo mejora real sobre el baseline anterior
- la mejora de AUC fue marginal pero limpia
- la mejora fuerte siguió viniendo más de calibración y capa de decisión que de AUC puro

#### Quality gate PD vs baseline anterior

- `pd_quality`: `PASS`
- Delta AUC: `+0.001147`
- Delta ECE: `+0.000984`
- Delta D2 Brier: `+0.001419`

Lectura:
- el modelo quedó mejor en discriminación y Brier skill
- ECE subió un poco, pero siguió dentro del contrato aceptable

#### Portfolio champion

- Policy seleccionada:
  - `risk_tolerance=0.18`
  - `policy_mode=capped_blended_uncertainty`
  - `gamma=0.05`
  - `delta_cap_quantile=0.9`
- Resultado económico:
  - diferencia de retorno total: `+6332.14`
  - ratio de financiados: `1.1148`
  - `passed_no_regression=true`

Lectura:
- la policy robusta sí valió la pena
- mejoró retorno total y cantidad financiada
- quedó como champion portfolio oficial

#### A/B económico

- Gate: `no_regression`
- Resultado: `PASS`
- Control:
  - total return: `221297.45`
  - funded: `209`
- Champion robusto:
  - total return: `227629.59`
  - funded: `233`
- Significancia bootstrap:
  - `p_value=0.4495`
  - `significant=false`

Lectura:
- operativamente pasa porque no hay regresión
- metodológicamente todavía no hay evidencia fuerte de superioridad estadística

#### Survival

- `survival_quality`: `PASS`
- `cox_cindex`: sin cambio material (`0.66434`)
- `rsf_cindex`: mejora de `0.66341` a `0.67966`

Lectura:
- survival fue una de las mejoras más valiosas de toda la mega run
- RSF sí mejoró de forma clara frente al baseline anterior

#### Fairness

- `overall_pass=true`
- `n_passed=6/6`
- Threshold operativo seleccionado: `0.35`

Lectura:
- fairness quedó cerrada operativamente
- el threshold de negocio/fairness oficial sigue siendo `0.35`
- esto es distinto al threshold interno de búsqueda PD

#### Governance

- `overall_pass=true`
- `challenger_promotable=true`
- warnings activos:
  - `warn_c2st=true`
  - `warn_distribution_tests=true`

Lectura:
- governance pasa y fue promovido
- pero todavía queda trabajo narrativo/metodológico para explicar mejor esos warnings

#### PD conformal

- `conformal_policy`: `PASS` en comparación operativa
- `conformal_promotion_pass=true`
- `conformal_statistical_warning=true`
- Métricas principales:
  - `coverage_90=0.9257`
  - `coverage_95=0.9516`
  - `min_group_coverage_90=0.8992`
  - `critical_alerts=0`
- Tests todavía fallando:
  - `kupiec_pvalue_90`
  - `kupiec_pvalue_95`
  - `christoffersen_pvalue_90`
  - `christoffersen_pvalue_95`

Lectura:
- mejora suficiente para promoción operativa
- todavía no cerrada para narrativa Q1 estricta

#### LGD/EAD conformal

- LGD seleccionado:
  - variante: `direct_adaptive_grade_temporal`
  - `overall_pass=true`
  - `coverage_90=0.9050`
  - `coverage_95=0.9550`
- EAD:
  - `coverage_90=0.9004`
  - `coverage_95=0.9410`

Lectura:
- LGD/EAD conformal sí quedó lo bastante fuerte para promoción operativa
- esta fue una mejora concreta y promotable de la mega run

#### Time series

- Estado: `warn`
- Point champion:
  - `AutoARIMA`
  - `point_promotable=true`
- Interval champion:
  - `AutoARIMA`
  - `interval_promotable=false`
  - `coverage_90=0.8102`

Lectura:
- el punto sirve y sigue siendo usable
- los intervalos siguen siendo una deuda central del proyecto

#### Causal / CATE

- ATE estimado: `0.00975`
- IC incluye cero
- refutaciones: no concluyentes / no disponibles
- CATE portfolio:
  - `promotion_eligible=false`
  - `cate_policy_mode=research_only_fallback`
  - `objective_change_pct=-4.4706`

Lectura:
- causal y CATE siguen aportando insights
- no entran todavía al camino canónico

### Qué quedó promovido tras esta mega run

Promovido operativamente:
- PD core
- calibración isotónica
- portfolio champion
- survival RSF
- fairness
- governance
- LGD/EAD conformal
- baseline operativo completo del run

Promovido operativamente con warning:
- PD conformal
- IFRS9 CPU canónico cuando dependa de narrativa temporal

No promovido al camino canónico:
- time series interval champion
- causal policy
- CATE portfolio

### Qué valió la pena de esta mega run

- validó la nueva arquitectura `champion_search` como carril de búsqueda pesado real
- mejoró el champion operativo y ya reemplazó el baseline anterior
- consolidó el uso de GPU en portfolio/tradeoff/selector/A-B/CATE/LGD-EAD donde sí aporta
- dejó claro que el mayor upside futuro no está en subir mucho más el AUC PD, sino en:
  - conformal
  - time series intervals
  - causal/CATE
  - A/B
  - governance narrativo

### Cómo usar este update frente a planes anteriores

Si un plan anterior todavía asumía como baseline el snapshot de `2026-03-11-C-official-selector-v3-freeze`, actualizarlo así:
- baseline oficial nuevo: `champion-2026-03-12-mega-definitive`
- PD core nuevo: `CatBoost tuned + calibrated`
- calibración oficial: `Isotonic Regression`
- policy champion nueva: `risk_tolerance=0.18`, `capped_blended_uncertainty`
- survival RSF mejorado
- fairness y governance promovidos
- LGD/EAD conformal promovido
- PD conformal sigue con warning estadístico
- time series intervalos siguen abiertos
- causal/CATE sigue research-only

### Tabla rápida de handoff: antes vs ahora

| Área | Antes | Ahora | Estado actual |
|---|---|---|---|
| Baseline oficial | `2026-03-11-C-official-selector-v3-freeze` | `champion-2026-03-12-mega-definitive` | Promovido y activo |
| PD best model | champion anterior | `CatBoost (tuned + calibrated)` | Promovido |
| Calibración oficial | baseline anterior | `Isotonic Regression` | Promovido |
| PD AUC | `0.7116` | `0.7128` | Mejora marginal real |
| PD Gini | `0.4233` | `0.4256` | Mejora |
| PD Brier | `0.1548` | `0.1545` | Mejora |
| PD HPO best validation AUC | histórico anterior menor | `0.7226` | Mejor HPO de la historia actual |
| Trials Optuna acumulados | menos que el run actual | `295` | Estudio extendido |
| Portfolio champion | policy anterior | `risk_tolerance=0.18`, `capped_blended_uncertainty`, `gamma=0.05` | Promovido |
| A/B total return | `221297.45` control | `227629.59` champion robusto | `no_regression` PASS |
| A/B funded | `209` control | `233` champion robusto | Mejora operativa |
| A/B significancia | no cerrada | `p=0.4495`, no significativa | Pendiente research |
| Survival RSF c-index | `0.66341` | `0.67966` | Mejora fuerte, promovido |
| Fairness | ya importante, no congelado en este run | `6/6` atributos pasan con threshold `0.35` | Promovido |
| Governance | baseline anterior | `overall_pass=true`, `challenger_promotable=true` | Promovido con warnings |
| PD conformal | baseline anterior operativo | cobertura mejorada, `conformal_promotion_pass=true` | Promovido con warning estadístico |
| LGD conformal | variantes previas | `direct_adaptive_grade_temporal` | Promovido |
| EAD conformal | baseline previo | cobertura alineada a target | Promovido |
| Time series point forecast | usable | `AutoARIMA` promotable | Se mantiene usable |
| Time series intervals | abiertos | `interval_promotable=false`, `coverage_90=0.8102` | Pendiente crítico |
| Causal ATE | exploratorio | ATE positivo pequeño con CI cruzando cero | Sigue research-only |
| CATE portfolio | exploratorio | `promotion_eligible=false` | Sigue research-only |
| RAPIDS / GPU path | más fragmentado | integrado en `champion_search` para OR/LGD-EAD/A-B/CATE | Mejorado y validado |

### Mini resumen para pegar en otra sesión

- Nuevo baseline oficial: `champion-2026-03-12-mega-definitive`
- PD promovido: `CatBoost tuned + calibrated`, calibración `Isotonic Regression`
- Mejoras PD: AUC `0.7116 -> 0.7128`, Gini `0.4233 -> 0.4256`, Brier `0.1548 -> 0.1545`
- HPO extendido: `295` trials, mejor validation AUC `0.7226`
- Portfolio champion promovido: `risk_tolerance=0.18`, `capped_blended_uncertainty`, `gamma=0.05`
- A/B operativo pasa: retorno `221297 -> 227630`, funded `209 -> 233`, significancia aún no cerrada
- Survival RSF mejora fuerte: `0.66341 -> 0.67966`
- Fairness promovido: `6/6` atributos pasan, threshold operativo `0.35`
- Governance promovido: pasa, pero con warnings `c2st` y distribution drift
- PD conformal: promotable operativo, no cerrado aún para paper/Q1 por tests estadísticos
- LGD/EAD conformal: promovido
- Time series: point forecast usable, intervalos siguen pendientes
- Causal/CATE: siguen en `insights_factory`, no en camino canónico

## Backlog por pipeline

### 1. `champion_search`

#### 1.1 PD conformal estricto
Objetivo:
- convertir el estado actual de “promotable operativo con warning” en una política conformal defendible también para narrativa Q1

Pendientes:
- ampliar benchmark de variantes PD conformal con:
  - `CrossConformal`
  - `JackknifeAfterBootstrap`
  - `Venn-Abers`
  - variantes localizadas
  - variantes group-weighted
- mantener la variante actual como baseline operativo
- comparar por variante:
  - `coverage_90`
  - `coverage_95`
  - `min_group_coverage_90`
  - `avg_width_90`
  - `avg_width_95`
  - `kupiec_pvalue_90`
  - `kupiec_pvalue_95`
  - `christoffersen_pvalue_90`
  - `christoffersen_pvalue_95`
  - estabilidad temporal por mes
- definir selector explícito de variante conformal:
  - prioridad 1: tests estadísticos
  - prioridad 2: cobertura por grupo
  - prioridad 3: anchura
- dejar artefacto final con:
  - variante elegida
  - tabla comparativa
  - razón de selección

Entregable:
- nueva política conformal PD con estado claro:
  - `operationally_promotable`
  - `research_closed`

#### 1.2 Time series intervals
Objetivo:
- sacar intervalos de `warn` y decidir si entran al carril canónico o quedan como research

Pendientes:
- mantener point forecast actual como baseline
- benchmarkear:
  - `ACI`
  - `EnbPI`
  - `OnlineConformal`
  - variantes Nixtla / StatsForecast
- medir:
  - cobertura 80/90/95
  - sharpness
  - estabilidad rolling
  - degradación por horizonte
  - comportamiento en cambio de régimen
- revisar criterio de selección:
  - horizonte 12 fijo
  - selección multi-horizonte
  - selección a 6 y evaluación a 12
- determinar si la falla viene de:
  - forecast base
  - método conformal
  - shift temporal

Entregable:
- `interval_promotable=true` o decisión formal de dejar intervalos fuera del camino canónico

#### 1.3 Causal policy / CATE
Objetivo:
- decidir si causalidad entra al camino canónico o queda consolidada en `insights_factory`

Pendientes:
- reforzar refutaciones:
  - placebo
  - random common cause
  - subset
  - sensitivity / robustness
- ampliar tuning de `CausalForestDML`:
  - `cv`
  - `mc_iters`
  - `criterion`
  - `min_balancedness_tol`
  - hiperparámetros estructurales del bosque
- revisar validez de diseño:
  - tratamiento continuo
  - overlap
  - confounders
  - effect modifiers
- repetir evaluación OOT:
  - valor neto
  - tail risk
  - robustez por segmentos
  - comparación contra champion portfolio
- fijar criterio binario de promoción:
  - señal causal no trivial
  - refutaciones aceptables
  - valor económico robusto

Entregable:
- decisión final:
  - `canonical_candidate`
  - o `insights_only`

#### 1.4 A/B más fuerte
Objetivo:
- pasar de `no_regression` a una historia económica más convincente

Pendientes:
- aumentar bootstrap y seeds
- correr sensibilidad por:
  - cohortes temporales
  - segmentos de riesgo
  - segmentos de monto
  - segmentos de ingreso
- ampliar reporte con:
  - retorno total
  - retorno por funded
  - variabilidad
  - downside
  - robustez del uplift
- revisar si la policy champion debe optimizar una métrica más alineada al A/B final

Entregable:
- evidencia económica más fuerte que la simple no regresión

### 2. `canonical_rebuild`

#### 2.1 Governance warnings
Objetivo:
- dejar mejor cerrada la historia regulatoria sin cambiar el baseline promovido salvo necesidad real

Pendientes:
- analizar por qué `c2st` y distribution tests disparan warning
- separar drift benigno de drift material
- construir política de materialidad:
  - PSI
  - importancia de feature
  - efecto sobre score
  - efecto sobre decisión
- reforzar narrativa de estabilidad:
  - SHAP
  - reason codes
  - threshold operativo
- dejar disclaimer estándar para:
  - drift estadístico esperado por tiempo
  - ausencia de deterioro operativo material

Entregable:
- governance más defendible para tesis, libro y paper

#### 2.2 Freeze operativo más explícito
Objetivo:
- asegurar que el camino canónico use solo decisiones congeladas del champion actual

Pendientes:
- revisar que `canonical_rebuild` no reabra:
  - HPO
  - fairness frontier search
  - conformal variant search
  - selector económico research
  - survival search
- verificar que el baseline promovido sea la única fuente de verdad operativa
- confirmar que `insights_factory` consuma artefactos canónicos sin sobreescribirlos

Entregable:
- rebuild canónico totalmente reproducible y barato

### 3. `insights_factory`

#### 3.1 Causal y CATE como carril research formal
Objetivo:
- ordenar lo causal como fábrica de insights mientras no sea canónico

Pendientes:
- separar outputs claramente:
  - exploratorio
  - candidate-to-canonical
  - descartado
- producir figuras y tablas comparativas para:
  - ATE
  - heterogeneidad
  - policy uplift
  - robustez

Entregable:
- narrativa causal limpia dentro de `insights_factory`

#### 3.2 RAPIDS y Monte Carlo GPU
Objetivo:
- dejar RAPIDS como evidencia comparativa y de infraestructura, no como ruido suelto

Pendientes:
- consolidar benchmarks CPU vs GPU
- consolidar IFRS9 Monte Carlo GPU como anexo research
- dejar tabla de:
  - speedup
  - estabilidad
  - rol canónico vs research

Entregable:
- anexo técnico reusable para libro y paper

#### 3.3 Notebooks y figuras de evidencia
Objetivo:
- ordenar notebooks para que complementen el libro y no compitan con el pipeline

Pendientes:
- clasificar notebooks en:
  - evidencia reusable
  - exploración histórica
  - side projects
- enlazar cada notebook relevante con:
  - capítulo futuro Quarto
  - artefactos de entrada
  - outputs reutilizables

Entregable:
- inventario de notebooks listo para narrativa editorial

## Cierre de protocolo paper

Objetivo:
- congelar la metodología antes de la corrida final paper-grade

Pendientes:
- fijar:
  - split temporal
  - feature universe
  - training regime PD
  - calibración oficial
  - shortlist conformal
  - survival methodology
  - policy portfolio oficial
  - criterio de promoción
- decidir de forma definitiva:
  - qué queda en pipeline canónico
  - qué queda en `insights_factory`
- escribir documento de protocolo final

Entregable:
- protocolo fijo y versionado para la corrida final

## Corrida final paper-grade

Objetivo:
- ejecutar la corrida final solo cuando el protocolo ya esté congelado

Pendientes:
- crear `study_name` nuevo y limpio para PD
- no mezclar trials históricos en el estudio final
- reutilizar historia previa solo para:
  - rangos
  - semillas
  - intuición del search space
- correr la mega corrida final con:
  - protocolo congelado
  - conformal shortlist cerrada
  - time series definido
  - causal decidido
  - promotion rules finales

Entregable:
- evidencia confirmatoria final para paper/Q1

## Orden recomendado entre sesiones

Sesión 1:
- PD conformal estricto

Sesión 2:
- time series intervals

Sesión 3:
- causal policy / CATE

Sesión 4:
- A/B más fuerte

Sesión 5:
- governance warnings

Sesión 6:
- cierre de protocolo paper

Sesión 7:
- diseño de `study_name` limpio de PD
- preparación de mega corrida final

## Definición de terminado

Antes de la corrida final paper-grade, deben quedar cerrados estos checks:
- PD conformal sin warning crítico o con justificación metodológica explícita y aceptable
- time series con decisión final documentada
- causal/CATE con decisión final documentada
- A/B con evidencia ampliada
- governance con warnings contextualizados
- protocolo final congelado y versionado

## Nota de uso

Este archivo es la referencia principal de pendientes entre sesiones. Si una sesión cambia prioridades o descarta una línea de trabajo, actualizar este documento primero y luego ejecutar cambios de código o corrida.
