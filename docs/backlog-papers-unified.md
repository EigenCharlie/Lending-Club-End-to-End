<!-- cspell:disable -->
<!-- markdownlint-disable -->

# Backlog Unificado: Pipeline + Papers + Quarto

Fecha: 2026-03-13
Baseline operativo: `champion-2026-03-12-mega-definitive`
Origen: fusión de `backlog-13-03.md` + estrategia de publicaciones (plan humble-doodling-mountain)

## Prioridad global (orden recomendado)

1. Study limpio de PD + corrida final paper-grade
2. Migración Quarto + companion Streamlit
3. Writing papers con artifacts ya congelados
4. Research opcional post-freeze

---

## Resumen ejecutivo

### Ya promovido / cerrado

- PD core: CatBoost tuned + calibrated, AUC 0.7128, Brier 0.1545
- Calibración oficial vigente: Venn-Abers
- Portfolio champion: risk_tol=0.18, capped_blended_uncertainty
- Survival RSF: c-index 0.6797 (mejora fuerte)
- Fairness: 6/6 PASS, threshold 0.35
- Governance: overall_pass, challenger_promotable
- LGD/EAD conformal: promovido
- PD conformal: cerrado para paper-grade con regla formal de Winkler compensado
- Time series: decisión final documentada como `research_only` para intervalos
- Causal/CATE: decisión final documentada
- A/B: evidencia ampliada y decisión final documentada
- Protocolo paper-grade congelado y versionado
- Baseline operativo completo

### Pendiente de cierre real (pipeline)

- `study_name` limpio de PD
- corrida final paper-grade confirmatoria
- si esa corrida mueve artefactos, refrescar protocolo/snapshot/bundle

### Pendiente nuevo (papers + Quarto)

- empaquetado editorial de variantes CP ya benchmarkeadas para Paper 3
- uncertainty set baselines para Paper Estrella
- bound teórico alpha-Gamma para Paper Estrella
- SICR trigger formalización para Paper 2
- ECL sensitivity a alpha conformal para Paper 2
- Migración Streamlit a Quarto + Streamlit
- Writing de 3 papers + Quarto book

### Pendiente de documentación real

- convertir artifacts finales en tablas/figuras para Paper 3
- reflejar en Quarto la decisión final de `PD conformal`, `time series`, `A/B`, `causal/CATE` y `governance`
- sincronizar narrativa histórica vs narrativa vigente en capítulos, apéndices y companion app
- dejar explícito qué resultados son:
  - baseline histórico de la mega run
  - estado canónico vigente post-validación P0 / paper-grade

### Pendiente real de hardening/policy

Fuente canónica:
- `docs/AUDITORIA_HARDENING_GATES_PAPER_GRADE_2026-03-13.md`

Conclusión de auditoría:
- el stack de `gates` y `policies` ya no necesita rediseño mayor;
- lo pendiente para dejar el protocolo óptimo es normalización contractual y narrativa final.

Estado:
- `P0 contract fix`: cerrado
- `P1 policy normalization`: cerrado
- `P1 test hardening`: cerrado
- `P2 editorial cleanup`: cerrado

Implementado:
- `run_tag` normalizado en artifacts refreshed post-hoc relevantes
- `threshold_semantics` propagado a `champion_registry` y `champion_search_bundle`
- fallback de fairness alineado al threshold operativo vigente `0.35`
- `storytelling_snapshot` refrescado con schema vigente
- tests de coherencia semántica añadidos
- dossier histórico bannerizado como histórico

### Estado consolidado del protocolo

Fuente canónica:
- [paper_grade_protocol_status.json](/home/eigenlinux/projects/lending-club-risk-project/models/paper_grade_protocol_status.json)

Estado:
- `pd_conformal=true`
- `time_series=true`
- `causal_cate=true`
- `ab_evidence=true`
- `governance=true`
- `protocol_frozen=true`

---

## 1. PD conformal estricto

Estado actual:
- cerrado para paper-grade
- no promotable operativamente (`promotion_pass=false`)
- cierre canónico logrado con:
  - `strict_overall_pass=false`
  - `methodological_justification_pass=true`
  - regla formal de `winkler_90` con banda compensada

Qué queda:
- convertir artifacts y tablas actuales en material de Paper 3
- no reabrir el método salvo que la corrida final limpia contradiga el estado actual

Prioridad: **1 (CRÍTICA)**
Bloquea: Paper 3 (COPA), Paper Estrella (MS/OR)
Conexión papers: es el corazón metodológico de Paper 3 y componente clave del Estrella

### 1.1 Benchmark de variantes CP (backlog original + paper)

Estado:
- benchmark ya extendido con variantes relevantes para el carril actual:
  - `global_split`
  - `mondrian_scaled`
  - `mondrian_unscaled`
  - `score_decile_mondrian`
  - `grade_x_scoreband_mondrian`
  - `cross_conformal_score_space`
- artifacts vigentes:
  - `data/processed/conformal_variant_selection_report.parquet`
  - `data/processed/conformal_temporal_diagnostics.parquet`
  - `data/processed/conformal_local_diagnostics.parquet`

Pendiente editorial:
- tablas/figuras para Paper 3
- decidir si vale la pena añadir variantes nuevas solo para publicación, no para canónico

### 1.2 Selector de variante conformal

Estado:
- selector explícito ya existe y quedó endurecido con:
  - cobertura
  - subgroup coverage
  - `winkler_90`
  - estabilidad temporal
- `promotion_pass` sigue en `false`
- `research_closed` ya sí queda resuelto para paper-grade

### 1.3 Posicionamiento académico (nuevo, para Paper 3)

Pendientes:
- posicionamiento narrativo y citas en el paper
- opcional: robustness appendix sobre partición de grades

Entregable:
- material para Paper 3 (tablas, figuras, narrativa)
- variante seleccionada con justificación publicable

---

## 2. Time series intervals

Prioridad: **2**
Bloquea: integración fuerte en Paper 2, no la corrida final
Conexión papers: alimenta escenarios ECL en Paper 2

Estado actual:
- decisión final documentada: `research_only` (reconfirmada en paper-grade run 2026-03-13)
- point forecast `AutoARIMA` sigue promotable
- champion intervalar `AutoARIMA`: coverage_90=81%, coverage_gap=0.090 — supera el target máximo de 0.03
- `EnbPI` también falla gate (coverage~36%) — ambos métodos son diagnóstico, no baseline oficial
- mejora pendiente: ACI/TCP rolling window (ver P3.2 en research-p3-p4-backlog.md)
- no hay bloqueo metodológico abierto; sí queda oportunidad research/editorial

### 2.1 Benchmark TS intervals (backlog original)

Pendientes:

- Mantener point forecast actual (AutoARIMA) como baseline
- Benchmarkear:
  - ACI (Adaptive Conformal Inference)
  - EnbPI
  - OnlineConformal
  - Variantes Nixtla / StatsForecast
- Medir:
  - Cobertura 80/90/95
  - Sharpness
  - Estabilidad rolling
  - Degradación por horizonte
  - Comportamiento en cambio de régimen
- Revisar criterio de selección:
  - Horizonte 12 fijo
  - Selección multi-horizonte
  - Selección a 6 y evaluación a 12
- Determinar si la falla (81% vs 90%) viene de:
  - Forecast base
  - Método conformal
  - Shift temporal

Entregable:
- mantener la decisión formal actual o reabrir solo si aparece mejora material en research posterior

---

## 3. A/B más fuerte

Prioridad: **3**
Bloquea: fortalecimiento del Paper Estrella, no la corrida final
Conexión papers: evidencia económica para Paper Estrella

Estado actual:
- evidencia A/B ya documentada en protocolo final
- escenario `ambiguity_defer` no mejora y no debe promoverse
- la policy actual se mantiene

### 3.1 Ampliar evidencia A/B (backlog original + paper)

Pendientes:

- Aumentar bootstrap y seeds (10K+ replications)
- Sensibilidad por:
  - Cohortes temporales
  - Segmentos de riesgo (grade)
  - Segmentos de monto
  - Segmentos de ingreso
- Ampliar reporte con:
  - Retorno total, retorno por funded
  - Variabilidad, downside risk
  - Robustez del uplift
  - Sharpe-like ratio (retorno / volatilidad)
- Revisar si la policy champion debe optimizar métrica más alineada al A/B final

### 3.2 Decision regret analysis (nuevo, para Paper Estrella)

Pendientes:

- Implementar comparación de decision regret (sensu Elmachtoub & Grigas 2022)
- Comparar regret: robusto conformal vs no-robusto vs SPO+ (ya en spo_integration.py)
- Alpha sweep: {0.01, 0.05, 0.10, 0.15, 0.20} con curvas de Pareto coverage-width-return

Entregable:

- si se hace trabajo extra aquí, ya es para reforzar publicación, no para cerrar backlog operativo

---

## 4. Governance warnings

Prioridad: **4**
Bloquea: narrativa más fuerte, no la corrida final
Conexión papers: contexto regulatorio MRM

Estado actual:
- governance ya quedó contextualizado y cerrado en protocolo
- warnings permanecen visibles, no maquillados

### 4.1 Contextualizar warnings (backlog original)

Pendientes:

- Analizar por qué c2st y distribution tests disparan warning
- Separar drift benigno de drift material
- Construir política de materialidad:
  - PSI por feature
  - Importancia de feature
  - Efecto sobre score
  - Efecto sobre decisión
- Reforzar narrativa de estabilidad:
  - SHAP rank overlap (ya 0.90)
  - Reason codes (ya estabilidad 1.0)
  - Threshold operativo estable
- Disclaimer estándar: drift estadístico esperado por OOT temporal, sin deterioro operativo material

Entregable:

- Governance defendible para tesis, libro Quarto y papers

---

## 5. Causal policy / CATE

Prioridad: **5**
No bloquea papers core. Alimenta insights_factory y Quarto book.
Conexión papers: mención en Paper 2, extensión futura en Paper Estrella

Estado actual:
- decisión final ya documentada
- regla elegida: `high_plus_medium_positive`
- queda como carril cerrado metodológicamente; mejoras adicionales son opcionales

### 5.1 Reforzar evidencia causal (backlog original)

Pendientes:

- Reforzar refutaciones: placebo, random common cause, subset, sensitivity
- Ampliar tuning CausalForestDML: cv, mc_iters, criterion, min_balancedness_tol
- Revisar validez de diseño: tratamiento continuo, overlap, confounders
- Repetir evaluación OOT: valor neto, tail risk, robustez por segmentos
- Fijar criterio binario de promoción: canonical_candidate o insights_only

### 5.2 Causal como insights_factory (backlog original)

Pendientes:

- Separar outputs: exploratorio, candidate-to-canonical, descartado
- Producir figuras y tablas para Quarto book (cap 7)

Entregable:

- Material para Quarto cap 7

---

## 6. Cierre de protocolo paper

Prioridad: **ya cerrado**
Bloquea: nada, salvo contradicciones documentales nuevas

### 6.1 Congelar metodología (backlog original)

Pendientes residuales:
- sincronización documental si la corrida final limpia produce cambios materiales

Entregable:

- Protocolo fijo y versionado para la corrida final

---

## 7. Corrida final paper-grade

Prioridad: **7** (después de protocolo congelado)
Bloquea: evidencia confirmatoria para todos los papers

### 7.1 Study limpio y mega corrida (backlog original)

Pendientes:

- Crear study_name nuevo y limpio para PD
- No mezclar trials históricos
- Reutilizar historia previa solo para rangos, semillas, intuición
- Correr con:
  - Protocolo congelado
  - Conformal shortlist cerrada
  - Time series decidido
  - Causal decidido
  - Promotion rules finales

Entregable:

- Evidencia confirmatoria final para paper/Q1

---

## 8. Migración Quarto + Streamlit

Prioridad: **8** (paralelo, no bloquea corrida)
Bloquea: tesis de maestría final
Conexión: el Quarto book ES la tesis de maestría

### 8.1 Scaffolding Quarto book

Pendientes:

- Crear estructura Quarto project (_quarto.yml)
- 16 capítulos según blueprint existente (docs/QUARTO_BOOK_BLUEPRINT.md)
- Cada capítulo como .qmd con código Python ejecutable
- Papers como capítulos 11-13
- Integrar con DVC para reproducibilidad

### 8.2 Migrar contenido de Streamlit a Quarto

Pendientes:

- Identificar qué contenido de Streamlit migra a Quarto (narrativa detallada, ecuaciones, análisis profundo)
- Identificar qué queda en Streamlit (exploratorio interactivo, dashboards, toggles)
- Migrar: thesis_contribution, thesis_end_to_end, research_landscape, paper_1/2/3 → capítulos Quarto
- Mantener en Streamlit: model_laboratory, portfolio_optimizer, uncertainty_quantification como demos interactivas

### 8.3 Figuras publication-quality

Pendientes:

- Convertir figuras Plotly → matplotlib/seaborn para papers y Quarto
- Estilo consistente para paper (2-column IEEE/Springer format)
- Exportar como PDF/SVG para LaTeX

### 8.4 Streamlit como companion

Pendientes:

- Reorientar Streamlit como "Interactive Companion" del Quarto book
- Agregar links bidireccionales: Quarto → Streamlit demo, Streamlit → Quarto chapter
- Reducir duplicación narrativa (Quarto tiene el detalle, Streamlit tiene la interacción)

Entregable:

- Quarto book funcional como tesis de maestría
- Streamlit como companion interactivo
- Papers embebidos como capítulos

---

## 9. Writing papers

Prioridad: **9** (post corrida final)
Depende de: items 1-7 cerrados

### 9.1 Paper 3: Mondrian CP → COPA 2026

Timeline: abril-mayo 2026
Venue: COPA 2026 (PMLR proceedings)
Formato: ~8-10 páginas PMLR

Pendientes writing:

- Abstract y framing final
- Related work: citar Kandinsky, Gibbs & Cherian, Zhou & Sesia, Angelopoulos
- Methods: ecuaciones finales, notación limpia
- Results: tablas y figuras del benchmark de variantes (item 1)
- Discussion: trade-offs eficiencia vs garantía por grupo
- Threats to validity
- Reproducibility package

### 9.2 Paper 2: IFRS9 E2E → JBF/JORS

Timeline: junio-septiembre 2026
Venue: Journal of Banking & Finance o JORS
Formato: ~25-30 páginas journal

Pendientes writing:

- Formalizar SICR trigger con CP width (threshold optimization)
- ECL sensitivity a alpha conformal
- Comparación con BMA (práctica bancaria actual)
- Citar: ECB 2024, IFRS Board SICR 2024, Annals of OR 2025
- ECL intervals completos (PD x LGD x EAD todos con CP, ya promovidos)
- Integrar TS forecast intervals si se cierran (item 2)
- Cost-of-misclassification: S1 vs S2

### 9.3 Paper Estrella: Predict-then-Optimize → MS/OR/EJOR

Timeline: julio-diciembre 2026
Venue: Management Science > Operations Research > EJOR
Formato: ~30-35 páginas + online appendix (Quarto book)

Pendientes writing:

- Bound teórico alpha-conformal ↔ Gamma-robustez (Bertsimas & Sim)
- Baselines uncertainty sets: ellipsoidal, bootstrap, parametric, Venn-Abers
- CQR como CP alternativo
- Alpha sweep {0.01..0.20} → Pareto frontier
- Decision regret comparison (SPO+)
- Figuras matplotlib publication-quality
- Online companion → Quarto book URL

---

## 10. RAPIDS y GPU (insights_factory)

Prioridad: **10** (no bloquea papers)
Conexión: anexo técnico en Quarto book

### 10.1 Consolidar benchmarks (backlog original)

Pendientes:

- Consolidar CPU vs GPU benchmarks
- IFRS9 Monte Carlo GPU como anexo research
- Tabla: speedup, estabilidad, rol canónico vs research

Entregable:

- Anexo técnico para Quarto book

---

## 11. Notebooks y figuras de evidencia

Prioridad: **11** (paralelo con Quarto migration)
Conexión: atlas de notebooks en Quarto

### 11.1 Clasificar y enlazar notebooks (backlog original)

Pendientes:

- Clasificar en: evidencia reusable, exploración histórica, side projects
- Enlazar con: capítulo Quarto, artefactos de entrada, outputs reutilizables

Entregable:

- Inventario listo para narrativa editorial

---

## Tabla de conexiones: item ↔ paper ↔ Quarto

| Item | Paper 3 | Paper 2 | Estrella | Quarto Cap |
| --- | --- | --- | --- | --- |
| 1. PD conformal | CRÍTICO | Alimenta | CRÍTICO | 5 |
| 2. TS intervals | - | Alimenta ECL | - | 6 |
| 3. A/B fuerte | - | - | IMPORTANTE | 9 |
| 4. Governance | - | IMPORTANTE | Alimenta | 8 |
| 5. Causal/CATE | - | Mención | Ext. futura | 7 |
| 6. Protocolo | Prerequisito | Prerequisito | Prerequisito | - |
| 7. Corrida final | Evidencia | Evidencia | Evidencia | Todo |
| 8. Quarto migration | - | - | Online companion | CRÍTICO |
| 9. Writing | Paper 3 | Paper 2 | Estrella | 11-13 |
| 10. RAPIDS | - | - | - | Anexo |
| 11. Notebooks | Material | Material | Material | Atlas |

---

## Orden recomendado entre sesiones

- Sesión 1: study limpio de PD + preparación de corrida final
- Sesión 2: mega corrida final paper-grade
- Sesión 3: refresh de protocolo/snapshot si cambia algo
- Sesión 4: scaffolding Quarto book
- Sesión 5: writing Paper 3
- Sesión 6: writing Paper 2
- Sesión 7+: writing Paper Estrella y research opcional

---

## Definición de terminado (pre corrida final)

Antes de la corrida final paper-grade:

- PD conformal: cerrado
- Time series: cerrado
- A/B: cerrado
- Governance: cerrado
- Causal/CATE: cerrado
- Protocolo: congelado y versionado
- Quarto scaffolding: estructura lista (no necesita contenido completo)

---

## Nota de uso

Este archivo reemplaza `backlog-13-03.md` como referencia principal de pendientes.
Todos los items del backlog anterior están incluidos aquí con sus conexiones a papers.
Si una sesión cambia prioridades, actualizar este documento primero.

Referencia de papers: `docs/PAPER_REFERENCES_STATE_OF_ART.md` (~80 papers con links directos)
Plan de publicación: `.claude/plans/humble-doodling-mountain.md`
