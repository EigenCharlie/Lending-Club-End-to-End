<!-- cspell:disable -->
<!-- markdownlint-disable -->

# Backlog 13-03

> **DEPRECATED** — Este backlog fue reemplazado por [`docs/backlog-papers-unified.md`](backlog-papers-unified.md) (2026-03-13). Este archivo se mantiene como referencia histórica.

Fecha: 2026-03-13

Estado base:
- Baseline operativo actual: `champion-2026-03-12-mega-definitive`
- Registry oficial: `configs/baselines/canonical_operational_baseline.json`
- Objetivo de este backlog: cerrar pendientes metodológicos y operativos antes de la corrida final paper-grade con `study_name` limpio de PD
- Regla de verdad vigente:
  - artefactos vivos actuales en `models/`, `data/processed/` y `reports/storytelling_snapshot.json` mandan sobre snapshots históricos
  - la mega run `champion-2026-03-12-mega-definitive` sigue siendo el baseline oficial
  - pero varias decisiones fueron revalidadas y actualizadas después, sin relanzar `champion_search`

## Prioridad global

Orden recomendado:
1. Study limpio de PD + corrida final paper-grade
2. Congelar documentación final del protocolo y narrativa
3. Preparar writing / Quarto / figures publication-grade
4. Mantener research opcional fuera del carril canónico

## Resumen ejecutivo

Lo ya promovido:
- PD core con calibración `Venn-Abers`
- policy champion de portfolio
- survival RSF
- fairness
- governance operativo
- LGD/EAD conformal
- panel de time series regularizado con grilla mensual completa
- semántica canónica de thresholds `0.05` interno PD vs `0.35` operativo fairness/aprobación
- registry oficial de librerías/métodos conformales (`models/conformal_method_registry.json`)
- benchmark binario de `classification sets` como sidecar research (`models/pd_set_prediction_status.json`)
- auditoría de `rare-event calibration` para PD (`models/pd_rare_event_calibration_status.json`)

Lo pendiente de cierre:
- `study_name` limpio de PD para la corrida confirmatoria final
- corrida final paper-grade con protocolo ya congelado
- limpieza editorial final de docs/backlogs/Quarto/papers
- cualquier exploración adicional queda como research opcional, no como bloqueo

## Auditoría hardening/policy

Fuente canónica de esta auditoría:
- `docs/AUDITORIA_HARDENING_GATES_PAPER_GRADE_2026-03-13.md`

Diagnóstico global:
- la arquitectura de `gates`, `thresholds`, `promotion` y `paper-grade closure` ya es coherente en código;
- el pendiente real ya no es rediseñar policies, sino normalizar contratos, metadata y narrativa.

Cierres ejecutados de hardening/policy:
- `P0 contract fix`
  - `run_tag` normalizado en:
    - `models/conformal_policy_status.json`
    - `models/governance_status.json`
    - `models/champion_registry.json`
    - `models/champion_search_bundle.json`
  - `threshold_semantics` propagado a:
    - `models/champion_registry.json`
    - `models/champion_search_bundle.json`
  - resultado: `0.05` interno PD y `0.35` operativo ya conviven explícitamente en artifacts secundarios
- `P1 policy normalization`
  - fallback de `configs/fairness_policy.yaml` alineado a `0.35`
  - `reports/storytelling_snapshot.json` refrescado con `SCHEMA_VERSION` vigente y decisión final TS explícita
- `P1 test hardening`
  - tests nuevos/extendidos para:
    - coherencia semántica entre artifacts
    - `strict` vs `promotion` vs `paper-grade`
    - `research_only` vs `promoted`
    - banner histórico en dossier de promoción
- `P2 editorial cleanup`
  - `docs/PROMOTION_DOSSIER_2026-03-01.md` ya quedó marcado como snapshot histórico

Clasificación de hallazgos:
- `contracto canónico correcto`
  - `threshold_semantics`
  - separación `strict` / `promotion` / `paper-grade`
  - `time_series` como `research_only` solo para intervalos
- `duplicado incoherente`
  - fallback fairness `0.50` vs threshold operativo `0.35`
  - artifacts secundarios que muestran `0.05` sin semántica operativa adjunta
- `regla ad hoc a formalizar`
  - versionado narrativo de `storytelling_snapshot`
  - coherencia semántica entre artifacts, no solo metadata

## Estado vigente tras validación P0

Esta sección reemplaza como verdad operativa actual cualquier lectura más vieja del documento.

### Qué cambió de forma oficial después del baseline

- El baseline oficial sigue siendo `champion-2026-03-12-mega-definitive`
- No se relanzó `champion_search`
- Sí se hicieron reruns focalizados y refreshes de artefactos para validar cambios P0
- La calibración PD canónica vigente pasó de `Isotonic Regression` a `Venn-Abers`
- `PD conformal` dejó de tener lectura de promoción operativa simple y hoy queda como:
  - `promotion_pass=false`
  - `overall_pass=false`
  - `checks_passed=9/13`
  - con semántica nueva:
    - `strict_overall_pass=false`
    - `methodological_justification_pass=true`
    - `non_statistical_checks_pass=true`
    - `winkler_90` ya no queda como deuda abierta, sino como check formal con banda compensada documentada
- `time_series` sigue con:
  - point champion usable/promotable
  - interval champion no promotable
- El carril TS quedó corregido estructuralmente:
  - panel `grade_term` mensual sin gaps internos
  - autorebuild de artifacts viejos/irregulares en `forecast_default_rates.py`

### Métricas vigentes relevantes

#### PD core vigente

- Best model: `CatBoost (tuned + calibrated)`
- Best calibration: `Venn-Abers`
- OOT AUC: `0.712813`
- Brier: `0.154537`
- ECE: `0.006129`

Lectura:
- el fix de `Venn-Abers` sí valió la pena
- la mejora numérica es pequeña pero real
- metodológicamente corrige un bug importante y debe quedarse oficial

#### Thresholds vigentes

- Threshold interno PD: `0.05`
- Threshold operativo fairness/aprobación: `0.35`

Lectura:
- esta separación ya debe tratarse como contrato estable del proyecto

#### PD conformal vigente

- `checks_passed=9/13`
- `overall_pass=false`
- `promotion_pass=false`
- `strict_overall_pass=false`
- `methodological_justification_pass=true`
- `coverage_90=0.9283`
- `coverage_95=0.9605`
- `avg_width_90=0.7569`
- `min_group_coverage_90=0.8873`
- `winkler_90=1.2032`
- `winkler_90_policy_mode=compensated_band`

Lectura:
- el cierre paper-grade ya queda aceptable por regla formal de compensación, no por sensibilidad ad hoc
- el único bloqueo estricto restante sigue siendo estadístico (`Kupiec/Christoffersen`)
- `promotion_pass` permanece en `false`; esto cierra narrativa/método, no promoción operativa automática
- `non_statistical_checks_pass=true`, así que no queda un gap metodológico abierto en el backlog operativo

#### Time series vigente

- Point champion: `AutoARIMA`
- `point_promotable=true`
- Interval champion: `AutoARIMA`
- `interval_promotable=false`
- interval benchmark MAPIE:
  - `best_method=enbpi`
  - sigue en carril diagnóstico

Lectura:
- el fix estructural del panel sí sirve y debe promoverse
- el problema restante no es de integridad del panel sino de calidad de intervalos
- semántica operativa formal:
  - `models/time_series_status.json` gobierna el carril canónico
  - `research_only` no apaga forecasts ni escenarios; solo evita promocionar la capa de intervalos como contrato oficial
  - `promoted` implicaría `interval_champion.promotable=true` y cierre completo del carril TS en snapshot, bundle y protocolo

### Estado de promoción actual

Promover ya a código y pipelines:
- `Venn-Abers`
- `refresh_pd_calibration_artifacts.py`
- `threshold_semantics`
- panel TS mensual regularizado
- autorebuild defensivo de `forecast_default_rates.py`
- floor conformal más conservador (`group_coverage_floor_target_90=0.92`)
- selector conformal con `stability_over_time` y sidecar `cross_conformal_score_space`
- method registry y descarte explícito de `Nonconformist` / `Fortuna` / `NeuralProphet`
- set prediction binario y auditoría de rare-event calibration como carriles diagnósticos

No promover todavía como configuración ganadora:
- `PD conformal` actual
- `time series intervals` actuales

## Estado de cierre paper-grade

Según [paper_grade_protocol_status.json](/home/eigenlinux/projects/lending-club-risk-project/models/paper_grade_protocol_status.json), los checks de cierre metodológico ya están cerrados:

- `PD conformal`: cerrado para paper-grade
  - `strict_overall_pass=false`
  - `canonical_methodological_justification_pass=true`
  - `promotion_pass=false`
- `time_series`: decisión final documentada como `research_only`
- `causal/CATE`: decisión final documentada y cerrada
- `A/B`: evidencia ampliada y regla de decisión documentada
- `governance`: warnings contextualizados y cerrados narrativamente
- `protocol_frozen=true`

Lectura:
- ya no hay bloqueos metodológicos abiertos antes de la corrida final
- lo pendiente ya es confirmatorio/editorial, no de diseño del pipeline

## Update incorporado de la mega run `champion-2026-03-12-mega-definitive`

Nota:
- Esta sección describe el resultado histórico de la mega run promovida.
- Si entra en conflicto con la sección `Estado vigente tras validación P0`, tomar la sección P0 como verdad actual.

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
- Calibración elegida en la mega run: `Isotonic Regression`
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
- esa fue la decisión vigente al cierre de la mega run
- hoy la calibración canónica ya fue revalidada y actualizada a `Venn-Abers`

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

- `conformal_policy`: `PASS` en comparación operativa de la mega run
- `conformal_promotion_pass=true` en ese snapshot histórico
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
- esa fue la lectura al cierre de la mega run
- hoy ya no debe tomarse como estado vigente
- tras validación P0 actual:
  - `promotion_pass=false`
  - `overall_pass=false`
  - `checks_passed=9/13`
  - `methodological_justification_pass=true`
  - `non_statistical_checks_pass=true`

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
- tras validación P0 se cerró la deuda estructural del panel, pero no la de promoción de intervalos

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
- calibración oficial en la mega run: `Isotonic Regression`
- calibración canónica vigente hoy: `Venn-Abers`
- policy champion nueva: `risk_tolerance=0.18`, `capped_blended_uncertainty`
- survival RSF mejorado
- fairness y governance promovidos
- LGD/EAD conformal promovido
- PD conformal queda cerrado para paper-grade, pero no promovido operativamente
- time series intervalos ya tienen decisión final documentada (`research_only`)
- causal/CATE ya tiene decisión final documentada

### Tabla rápida de handoff: antes vs ahora

| Área | Antes | Ahora | Estado actual |
|---|---|---|---|
| Baseline oficial | `2026-03-11-C-official-selector-v3-freeze` | `champion-2026-03-12-mega-definitive` | Promovido y activo |
| PD best model | champion anterior | `CatBoost (tuned + calibrated)` | Promovido |
| Calibración oficial en la mega run | baseline anterior | `Isotonic Regression` | Histórico |
| Calibración canónica vigente | `Isotonic Regression` | `Venn-Abers` | Promovido |
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
| PD conformal | baseline anterior operativo | mega run `conformal_promotion_pass=true`; hoy `promotion_pass=false`, `methodological_justification_pass=true` | Cerrado para paper-grade, no promovido operativamente |
| LGD conformal | variantes previas | `direct_adaptive_grade_temporal` | Promovido |
| EAD conformal | baseline previo | cobertura alineada a target | Promovido |
| Time series point forecast | usable | `AutoARIMA` promotable | Se mantiene usable |
| Time series intervals | abiertos | `interval_promotable=false`, `coverage_90=0.8102` | Pendiente crítico |
| Causal ATE | exploratorio | ATE positivo pequeño con CI cruzando cero | Sigue research-only |
| CATE portfolio | exploratorio | `promotion_eligible=false` | Sigue research-only |
| RAPIDS / GPU path | más fragmentado | integrado en `champion_search` para OR/LGD-EAD/A-B/CATE | Mejorado y validado |

### Mini resumen para pegar en otra sesión

- Nuevo baseline oficial: `champion-2026-03-12-mega-definitive`
- PD promovido en la mega run: `CatBoost tuned + calibrated`, calibración `Isotonic Regression`
- Mejoras PD: AUC `0.7116 -> 0.7128`, Gini `0.4233 -> 0.4256`, Brier `0.1548 -> 0.1545`
- HPO extendido: `295` trials, mejor validation AUC `0.7226`
- Portfolio champion promovido: `risk_tolerance=0.18`, `capped_blended_uncertainty`, `gamma=0.05`
- A/B operativo pasa: retorno `221297 -> 227630`, funded `209 -> 233`, significancia aún no cerrada
- Survival RSF mejora fuerte: `0.66341 -> 0.67966`
- Fairness promovido: `6/6` atributos pasan, threshold operativo `0.35`
- Governance promovido: pasa, pero con warnings `c2st` y distribution drift
- PD conformal en la mega run: promotable operativo, no cerrado aún para paper/Q1 por tests estadísticos
- PD conformal vigente post-validación P0: `promotion_pass=false`, `9/13`, `methodological_justification_pass=true`
- LGD/EAD conformal: promovido
- Time series: point forecast usable, intervalos siguen pendientes; panel mensual regularizado ya corregido
- Calibración vigente post-validación P0: `Venn-Abers`
- Causal/CATE: decisión final documentada; se mantiene fuera del camino canónico operativo

## Backlog por pipeline

### 1. `champion_search`

#### 1.1 Estado actual
Objetivo:
- no reabrir `champion_search` todavía
- dejar sembrados en código y bundle todos los cambios ya validados
- reservar este pipeline para la corrida confirmatoria final, no para seguir explorando metodología

Pendientes:
- ninguno metodológico bloqueante
- mantener sembrado:
  - `Venn-Abers`
  - `threshold_semantics`
  - selector conformal endurecido con `winkler_90`
  - banda compensada formal de `winkler_90`
  - diagnostics de rare-event calibration y set prediction
  - benchmark time series con metadata extendida
- ejecutar solo cuando exista `study_name` limpio de PD y se quiera la corrida confirmatoria final

Entregable:
- pipeline listo para heredar el estado metodológico ya cerrado sin reabrir búsqueda

Estado actual:
- cerrado para paper-grade con `methodological_justification_pass=true`
- sigue sin `promotion_pass`, así que no obliga a cambiar la postura operativa conservadora

#### 1.2 Corrida final paper-grade
Objetivo:
- ejecutar una sola corrida confirmatoria limpia, con protocolo ya congelado

Pendientes:
- crear `study_name` nuevo y limpio para PD
- ejecutar la corrida final confirmatoria
- refrescar `paper_grade_protocol_status.json`, `storytelling_snapshot.json` y `champion_search_bundle.json` solo si la corrida mueve artefactos oficiales

Estado actual:
- esta ya es la única pieza realmente bloqueante a nivel pipeline

Entregable:
- evidencia confirmatoria final para paper/Q1

#### 1.3 No reabrir durante esta fase
Objetivo:
- preservar alcance y evitar que la corrida final se mezcle con research extra

Pendientes:
- causal/CATE ya está cerrado metodológicamente
- A/B ya tiene evidencia ampliada suficiente para protocolo
- time series ya tiene decisión final documentada
- dejar extensiones adicionales para `insights_factory`

Entregable:
- corrida final enfocada en confirmación, no en ampliación de alcance

### 2. `canonical_rebuild`

#### 2.1 Estado actual
Objetivo:
- reproducir el estado metodológico ya cerrado sin abrir búsquedas pesadas

Pendientes:
- heredar y congelar:
  - calibración `Venn-Abers`
  - thresholds canónicos
  - `PD conformal` cerrado para paper-grade
  - `time_series` con decisión `research_only`
  - `causal/CATE` con decisión final documentada
  - governance contextualizado

Entregable:
- rebuild barato, reproducible y consistente con el protocolo final

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

#### 2.3 Refresh PD sin retraining completo
Objetivo:
- oficializar el camino barato para recalibrar y refrescar artifacts PD sin reabrir HPO ni reruns pesados

Pendientes:
- mantener `scripts/refresh_pd_calibration_artifacts.py` como workflow oficial
- documentar en runbook cuándo usar:
  - refresh liviano
  - rerun PD completo
- asegurar que `canonical_rebuild` lo pueda invocar cuando el modelo base no cambie

Entregable:
- recalibración PD rápida, reproducible y barata

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

Estado actual:
- cerrado en [paper_grade_protocol_status.json](/home/eigenlinux/projects/lending-club-risk-project/models/paper_grade_protocol_status.json)
- mantener solo sincronización documental si cambian artifacts oficiales

Objetivo:
- congelar la metodología antes de la corrida final paper-grade

Pendientes residuales:
- reflejar el cierre final en todos los docs narrativos si aparece alguna contradicción
- no reabrir metodología salvo evidencia nueva material

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

Nota:
- esta ya es la única pieza realmente bloqueante a nivel pipeline

## Orden recomendado entre sesiones

Sesión 1:
- definir `study_name` limpio de PD
- preparar corrida final confirmatoria

Sesión 2:
- ejecutar corrida final paper-grade

Sesión 3:
- refrescar protocolo/snapshot/bundle si la corrida final cambia artefactos

Sesión 4:
- Quarto / figures / tables publication-grade

Sesión 5+:
- research opcional y writing de papers

## Definición de terminado

Antes de la corrida final paper-grade, deben quedar cerrados estos checks:
- PD conformal sin warning crítico o con justificación metodológica explícita y aceptable
- time series con decisión final documentada
- causal/CATE con decisión final documentada
- A/B con evidencia ampliada
- governance con warnings contextualizados
- protocolo final congelado y versionado

Estado actual:
- todos esos checks ya figuran cerrados en `models/paper_grade_protocol_status.json`
- lo pendiente antes de terminar el proyecto es confirmarlos en la corrida final limpia y luego cerrar la capa editorial/publicación

## Nota de uso

Este archivo es la referencia principal de pendientes entre sesiones. Si una sesión cambia prioridades o descarta una línea de trabajo, actualizar este documento primero y luego ejecutar cambios de código o corrida.
