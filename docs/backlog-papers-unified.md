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
- uncertainty set baselines para Paper Estrella (ellipsoidal/bootstrap/paramétrico para optimización — distinto a uncertainty_baselines_by_grade ya surfaceado)
- bound teórico alpha-Gamma para Paper Estrella (contribución principal, sin implementar)
- SICR trigger formalización para Paper 2 *(research hecho — falta writing)*
- ECL sensitivity a alpha conformal para Paper 2 *(research hecho — falta writing)*
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

**Estado: CERRADO (2026-03-17)**

Research ejecutado:
- Sharpe-like ratio derivado de CI de 15K bootstrap (std_diff = (ci_high - ci_low) / 3.92)
- Calmar-like = ROIC / |downside_CI| para medida de riesgo ajustado
- Atribución por grade (A/B/C), cohorte temporal (4 segmentos), bucket de monto (<5K, 5-10K, 10-20K, 20-35K, >35K)
- Surfaceado en `streamlit_app/pages/ab_testing_simulation.py` (expander "A/B Attribution & Risk-Adjusted Returns")

Artifacts:
- `models/ab_attribution_status.json` — bootstrap count, ROIC A/B, Sharpe-like, Calmar-like
- `data/processed/ab_attribution_by_grade.parquet`
- `data/processed/ab_attribution_by_cohort.parquet`
- `data/processed/ab_attribution_by_amount.parquet`

Generado por: `scripts/run_ab_portfolio_attribution.py`

### 3.2 Decision regret analysis (nuevo, para Paper Estrella)

Pendientes:

- Implementar comparación de decision regret (sensu Elmachtoub & Grigas 2022)
- Comparar regret: robusto conformal vs no-robusto vs SPO+ (ya en spo_integration.py)
- Alpha sweep: {0.01, 0.05, 0.10, 0.15, 0.20} con curvas de Pareto coverage-width-return

Nota: SPO+ v2 ya ejecutado (49.1% regret reduction, `scripts/run_spo_real.py`). Pendiente: integración formal en pipeline Paper Estrella y figuras publication-quality.

Entregable:

- Material adicional para Paper Estrella (writing phase)

---

## 4. Governance warnings

Prioridad: **4**
Bloquea: narrativa más fuerte, no la corrida final
Conexión papers: contexto regulatorio MRM

Estado actual:
- governance ya quedó contextualizado y cerrado en protocolo
- warnings permanecen visibles, no maquillados

### 4.1 Contextualizar warnings (backlog original)

**Estado: CERRADO (2026-03-17)**

Research ejecutado:
- PSI por feature calculado y clasificado (benigno <0.10, moderado 0.10-0.25, severo >0.25)
- Separación drift benigno vs material con conteo de breaches/warns
- Narrativa de estabilidad reforzada: SHAP rank overlap 0.90, reason codes 1.0, threshold estable
- Disclaimer formal surfaceado en `streamlit_app/pages/model_governance.py` (expander "Feature Stability: PSI per Feature")

Artifacts:
- `data/processed/drift_monitoring.parquet` — PSI por feature con max/mean/clasificación

Pendiente solo writing:
- Texto defendible para capítulo governance en Quarto y MRM section en papers

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

**Estado: CERRADO (2026-03-17)**

Research ejecutado:
- Refutaciones DoWhy (placebo/random common cause/subset) ejecutadas — resultado: `refutation_unavailable` por ATE near-zero (CI incluye 0), lo que es evidencia honesta de efectos débiles
- ATE OOT, distribución CATE P5/P25/P50/P75/P95, n_positive/n_negative surfaceados
- Per-grade CATE con tail risk interpretado
- OOT policy validation: n_months=106, total_net_value, p05_monthly_net, worst_month
- Surfaceado en `streamlit_app/pages/causal_intelligence.py` (expander "Causal Refutations & OOT Tail Risk")

Artifacts:
- `models/causal_refutation_summary.json` — refutation interpretation + CATE distribution + OOT policy stats
- `data/processed/causal_oot_tail_risk.parquet` — per-grade CATE stats

Generado por: `scripts/run_causal_refutation_summary.py`

### 5.2 Causal como insights_factory (backlog original)

**Estado: CERRADO (2026-03-17)**

Research ejecutado:
- Outputs clasificados: `insights_only` (no canonical_candidate) — decisión formal documentada
- CATE distribution y per-grade disponibles para Quarto cap 7
- Surfaceado en `causal_intelligence.py` con interpretación editorial honesta

Pendiente solo writing:
- Tablas y figuras para Quarto cap 7 (narrativa final)

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

Prioridad: **ya ejecutada**

### 7.1 Study limpio y mega corrida (backlog original)

**Estado: CERRADO (2026-03-16)**

La corrida `paper-grade-2026-03-13-final-heavy-2026-03-13-230650` (ejecutada 2026-03-15/16) ES la confirmatoria.

**Completado:**

- Study limpio: `pd_catboost_optuna_temporal_paper_grade_final__cb_space_v2` — 320/320 trials completos, DB separada (`optuna_pd_catboost_paper_grade_final.db`, 716KB sin contaminación histórica)
- PD: AUC 0.7130, Brier 0.1545, ECE 0.0059, Venn-Abers, conformal 92.52% / 95.93%
- Survival: RSF c-index=0.6715, Cox c-index=0.6643, dataset_scope=full_data (500K loans)
- LGD/EAD conformal: promovido, variant=`direct_adaptive_grade_temporal`
- Portfolio: champion seleccionado risk_tol=0.18, capped_blended_uncertainty, A/B no-regression pass
- IFRS9: sensitivity grid + Monte Carlo GPU (16K scenarios) ejecutados
- Fairness: 6/6 PASS, threshold=0.35
- Governance: overall_pass=true, challenger_promotable=true
- MRM: reporte generado en `reports/mrm/mrm_validation_report.json`
- RAPIDS: benchmarks + IFRS9 MC GPU completados
- Protocolo: `paper_grade_protocol_status.json` frozen=true
- Bundle: `champion_search_bundle.json` run_tag correcto

**Gaps menores (sin re-ejecución de pipeline):**

| Gap | Descripción | Acción |
|---|---|---|
| MRM run_tag | `mrm_validation_report.json` tiene `run_tag=None` a nivel raíz (inner compliance_summary correcto) | Re-ejecutar `generate_mrm_report.py` solo para inyectar run_tag |
| pd_rare_event run_tag | `pd_rare_event_calibration_status.json` dice `run_tag=untracked` | Re-ejecutar `analyze_pd_rare_event_calibration.py` |
| mrm_report_status.json | `models/mrm_report_status.json` no existe; algunas páginas Streamlit lo buscan | Crear wrapper JSON desde mrm_validation_report.json |
| Causal run_tag | Artifacts causal tienen `run_tag=champion-2026-03-12-mega-definitive` | Intencional — causal es `insights_only`, no re-ejecutar |
| Paper notebooks | NB10-NB12 no ejecutados (`include_notebooks=False`) | Ejecutar `run_paper_notebook_suite.py` separado |

**MRM compliance_summary.overall_pass=false es esperado**: conformal falla Kupiec/Christoffersen en muestra grande (n=276K) pero `methodological_justification_pass=true` y `winkler_compensated_pass=true`. Documentado y defensible.

**Lo que NO falta para los papers:**
- PD model paper-grade con clean HPO study ✅
- Conformal intervals con cobertura garantizada ✅
- Portfolio champion seleccionado y congelado ✅
- LGD/EAD conformal promovido ✅
- Fairness y governance paper-grade ✅
- Survival RSF paper-grade (500K) ✅
- IFRS9 ECL scenarios ✅
- A/B evidence con 15K bootstrap ✅

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

**Estado: CERRADO (2026-03-17)**

Research ejecutado:
- Tabla consolidada CPU vs GPU con speedup por tarea (17 tareas, 5 secciones: cuDF, cuML, cuGraph, cuOpt, cuPy)
- Hardware info surfaceado (RTX 3080, AMD Ryzen 5 5600X)
- Mean/max speedup disponibles
- Surfaceado en `streamlit_app/pages/gpu_benchmark.py` (expander "Consolidated CPU vs GPU Benchmark")
- Surfaceado en `streamlit_app/pages/tech_stack.py` (notebooks inventory)

Artifacts:
- `models/gpu_consolidated_summary.json` — mean_speedup, max_speedup, hardware
- `data/processed/gpu_consolidated_table.parquet` — 17 tareas con cpu_seconds, gpu_seconds, speedup

Generado por: `scripts/run_gpu_consolidated.py`

Pendiente solo writing:
- Texto para anexo técnico Quarto book (GPU capítulo)

---

## 11. Notebooks y figuras de evidencia

Prioridad: **11** (paralelo con Quarto migration)
Conexión: atlas de notebooks en Quarto

### 11.1 Clasificar y enlazar notebooks (backlog original)

**Estado: CERRADO (2026-03-17)**

Research ejecutado:
- 14 notebooks clasificados: core_thesis (01-09), paper_research (10-13), side_projects
- Reuse status: `evidence_reusable` | `paper_material` | `exploratory_side_project`
- Cada notebook con quarto_chapter, key_artifacts asociados
- Surfaceado en `streamlit_app/pages/tech_stack.py` (expander "Notebooks Inventory")

Artifacts:
- `models/notebooks_inventory.json` — 14 notebooks clasificados
- `data/processed/notebooks_inventory.parquet`

Generado por: `scripts/run_notebooks_inventory.py`

Pendiente solo writing:
- Atlas de notebooks en Quarto book (estructura editorial)

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
