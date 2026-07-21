# Paper Estrella Audit - CRPTO

> **HISTORICAL AUDIT — SUPERSEDED 2026-07-20.** This document reconstructs the
> pre-identification champion narrative. It is not evidence for the autonomous
> CRPTO/IJDS paper, which currently selects no learner, comparator or policy.
> Use `docs/research/crpto_evolution_cross_project_audit_2026-07-20.md`.

Fecha: 2026-05-04
Objetivo editorial: paper publicable tipo Management Science / Operations Research / EJOR sobre Conformal Robust Predict-then-Optimize (CRPTO).

## Decision canonica

El champion oficial del Paper Estrella es `paper-thesis-final-economic-2026-04-06`, con la policy `bound_aware_276k_economic_champion`. El comparador `theorem-tight` queda como punto interno para tightness teorica, no como policy oficial.

Artefactos fuente:

| Rol | Fuente canonica | Uso |
|---|---|---|
| Policy oficial y region robusta | `models/final_project_promotion.json` | champion, retorno, bound pass, region `45/45` |
| Policy exportada | `models/champion_portfolio_policy.json` | congelamiento operativo de la policy |
| Resumen final | `data/processed/final_project_summary.parquet` | soporte paper/thesis |
| Bound exacto 276k | `data/processed/portfolio_bound_aware/rank1_alpha01_bound_aware_276k_full_2026-04-05-1734/portfolio_bound_aware_bound_eval.parquet` | checks exactos por alpha/policy |
| Winner conformal | `models/final_project_promotion.json::conformal_upstream.winner_metrics` | cobertura/ancho/min group/winkler del paper |
| PD operativo | `data/processed/pipeline_summary.json`, `reports/dvc/metrics_summary.json` | AUC/Brier/ECE vigentes |
| Historico | `reports/run_comparisons/*`, `reports/mlruns/*`, `models/search_pd/*` | solo comparacion intra-familia |

Material paper-facing regenerado: `reports/paper_material/paper1/tables/*` se actualizo con `scripts/export_paper1_canonical_tables.py` para heredar del cierre `final_project_promotion.json`, del shortlist/bound eval `276k` y del conformal reopen final.

## Metricas canonicas

### Paper Estrella - policy oficial

| Metrica | Valor |
|---|---:|
| Run tag | `paper-thesis-final-economic-2026-04-06` |
| Label | `bound_aware_276k_economic_champion` |
| `risk_tolerance` | 0.175 |
| `policy_mode` | `blended_uncertainty` |
| `gamma` | 0.45 |
| `uncertainty_aversion` | 0.10 |
| `alpha01_exact_pass` | true |
| `alpha03_exact_pass` | true |
| `alpha10_exact_pass` | true |
| `alpha01_weighted_miscoverage_V` | 0.03645 |
| `alpha01_gamma_cp` | 0.18591 |
| `alpha01_violation` | 0.00000 |
| Retorno realizado | 170,464.54 |
| Price of robustness pct | -10.56% |
| Region robusta | 45/45 policies pasan `alpha01` |

### Winner conformal del Paper Estrella

| Metrica | Valor |
|---|---:|
| `coverage_90` | 0.929714 |
| `coverage_95` | 0.966388 |
| `avg_width_90` | 0.784230 |
| `min_group_coverage_90` | 0.918983 |
| `winkler_90` | 1.110742 |

### PD operativo y nota historica

El run operativo vigente `paper1-e2e-all-champions-2026-04-07` reporta AUC 0.712438, Brier 0.154631 y ECE 0.006380 en `data/processed/pipeline_summary.json`; `reports/dvc/metrics_summary.json` conserva la misma AUC/Brier y ECE 0.006248 por su agregacion plana. En la familia PD pura, el candidato historico `models/search_pd/pd-hpo-local-2026-04-03-1325/pd_training_status.json` tiene AUC 0.713852 y Brier 0.154393, por lo que la metrica PD actual no es el mejor valor historico observado. No se promueve en este dossier porque no es el champion paper/thesis y no debe mezclarse con la familia bound-aware.

### Nota v38 sobre coherencia de artifacts

Los reportes amplios con `artifact_coherence_pass=false` mezclan scopes y run
tags de carriles research-only, en especial causal/CATE/time-series. No son una
inconsistencia del champion Paper Estrella. La capa canonical de Paper Estrella
(`models/final_project_promotion.json`, `models/champion_portfolio_policy.json`,
`models/champion_registry.json`, `reports/dvc/metrics_summary.json` y
`paper1_table0_key_metrics.csv`) sigue sincronizada para
`paper-thesis-final-economic-2026-04-06`.

### Regla de comparacion historica

| Familia | Comparar con | No comparar con |
|---|---|---|
| PD | AUC, Brier, ECE, PR-AUC entre runs PD | retorno de portafolio o bound-aware |
| Conformal | coverage, width, min group coverage, Winkler entre variantes conformal | AUC PD o retorno |
| Portfolio | retorno, funded count, price of robustness dentro del mismo selector | bound `V` si cambia la familia |
| Bound-aware | `alpha01_exact_pass`, `V`, `gamma_cp`, `violation`, region pass | tablas legacy de paper1 |
| DVC canonico | snapshot operativo vigente | experimental HPO no promovido |

## Matriz de literatura local - PDFs base

| PDF | Concepto/foco | Formula/figura relevante | Uso actual | Mejora concreta |
|---|---|---|---|---|
| `Conformal_Risk_Control.pdf` | CRC controla esperanza de perdidas monotónicas acotadas | Risk control sobre `E[L(C_lambda(X),Y)]` | Base teorica del Teorema 1 | Mantener como cita central, aclarando que CRPTO instancia la perdida como no cobertura ponderada |
| `Distribution-Free, Risk-Controlling Prediction Sets.pdf` | RCPS y calibracion de riesgo distribution-free | Seleccion de lambda para controlar riesgo con muestra de calibracion | Soporte del lenguaje `risk control` | Usar para defender calibracion post-hoc y separar garantia teorica de evidencia post-seleccion |
| `Learn then test.pdf` | LTT como marco de testeo despues de aprendizaje | Procedimiento learn-then-test con control finito | Soporta narrativa de cierre approval-based | Usar para justificar guardrails y auditoria, no para inflar el bound |
| `Conformal Uncertainty Sets for Robust Optimization.pdf` | CP como conjuntos de incertidumbre para RO | Conformal set -> robust feasibility | Cita puente CP -> RO | Contrastar: ellos no aterrizan funded-set ponderado de credito |
| `Predict-then-Calibrate.pdf` | Calibrar incertidumbre despues de predecir para optimizacion | Garantia de cobertura de vector de costos | Comparador metodologico mas cercano | Explicar diferencia: PtC cubre parametro/costo; CRPTO controla target ponderado del funded set |
| `Smart “Predict, then Optimize”.pdf` | SPO+ y regret decisional | SPO loss / SPO+ surrogate | Comparador DFL principal | Mantener como benchmark: mejor regret, menor auditabilidad formal |
| `Task-based End-to-end Model Learning in Stochastic Optimization.pdf` | DFL end-to-end via diferenciacion de solucion | Gradientes a traves del problema de optimizacion | Fundamento DFL | Usar como antecedente, no como competidor directo de garantia conformal |
| `ONLINE DECISION-FOCUSED LEARNING.pdf` | DFL en ambientes dinamicos/no estacionarios | Static/dynamic regret online | Ausente antes; agregado como `@capitaine2026online` | Future work secuencial junto con SDAM y online conformal |
| `The Price of Robustness.pdf` | Presupuesto de robustez y costo de proteccion | Bertsimas-Sim `Gamma`; price of robustness | Base del lenguaje `Gamma_CP` | Reforzar analogia, aclarando que `Gamma_CP` se hereda de CP |
| `View of Decision-Focused Learning_ Foundations, State of the Art, Benchmark and Future Opportunities.pdf` | Survey DFL | Taxonomia DFL y benchmarks | Contexto SPO+/DFL | Usar para ubicar CRPTO como complemento auditable |

## Refresh de PDFs nuevos descargados

La carpeta `C:\Users\carlos\Documents\Papers_tesis` contiene ahora 17 PDFs. Además de los 10 PDFs base, se re-leyeron los 7 PDFs recientes que se habían propuesto para ampliar el estado del arte. El resultado principal es que no obligan a cambiar el champion ni a reentrenar, pero sí endurecen la agenda editorial: el paper debe declarar que CRPTO hoy es post-hoc/two-stage y que la versión journal puede moverse hacia selección conformal decision-aware, riesgo OCE/CVaR, cobertura multi-distribución y calibración online.

| PDF nuevo | Lectura clave | Formula/figura relevante | Falta o mejora concreta para Paper Estrella |
|---|---|---|---|
| `10_Conformal_Robust_Optimizati.pdf` | CRO/CRS para predictores black-box; satisficing robusto como alternativa a optimizar retorno bajo incertidumbre | `U_alpha(z) = {d: s(z,d) <= eta_alpha}`; equivalencia CRO/CRS; concentración `O(n^-1/2)`; cotas de suboptimalidad/price of robustness | Citar como puente directo a robust satisficing; proponer una métrica futura de fragilidad/satisficing junto a `gamma_cp` y price of robustness |
| `2409.20534v2.pdf` | Calibración conformal end-to-end para optimización; aprende conjuntos convexos informados por decision loss | Figura 1 ETO vs E2E; diferenciación exacta de la calibración; PICNN para sets convexos | Reconocer que CRPTO actual calibra post-hoc; futuro: score conformal entrenado por pérdida de portafolio |
| `2505.13564v3.pdf` | Online DFL en ambientes dinámicos con regret estático/dinámico | DF-FTPL y DF-OGD; regret online | Future work secuencial para originación continua y comparación CRPTO vs DFL bajo drift |
| `2507.04716v2.pdf` | CROMS selecciona modelos de CRO por riesgo decisional bajo robustez objetivo | E-CROMS, F-CROMS y J-CROMS; trade-off robustez finita/costo computacional | Cambiar agenda futura: seleccionar familia conformal por retorno robusto, `V`, `gamma_cp` y violación, no solo coverage/width |
| `2510.08748v1.pdf` | Conformal Risk Training optimiza CRC end-to-end y extiende el riesgo a OCE/CVaR | Conformal OCE risk control; diferenciación de riesgo conformal | Future work matemático: controlar cola de pérdida del funded set con OCE/CVaR, no solo esperanza/Markov |
| `2601.02998v1.pdf` | MDCP garantiza cobertura uniforme si el test proviene de una fuente/grupo desconocido o mezcla de fuentes | Garantía `min_k P(Y in C(X) | P^(k)) >= 1-alpha`; agregación max-p; Figura 1 sobre sets multi-fuente | Relevante para fairness y shift: extender Mondrian a cobertura robusta sin conocer grupo test-time |
| `2602.03168v1.pdf` | Online CP mediante universal portfolio algorithms; conecta regret con cobertura online | UP-OCP parameter-free; pinball loss; cotas finitas de miscoverage | Reemplaza la mención genérica de ACI por una ruta online más moderna para recalibración bajo stream |

## Literatura reciente agregada/verificada

| Key | Fuente | Estado | Uso recomendado |
|---|---|---|---|
| `@capitaine2026online` | Online Decision-Focused Learning, ICLR 2026, arXiv 2505.13564 | peer-reviewed ICLR 2026 | Future work para DFL secuencial |
| `@yang2026multidistribution` | Multi-Distribution Robust Conformal Prediction, arXiv 2601.02998 | preprint arXiv | Cobertura uniforme ante fuentes/grupos multiples sin observar grupo test-time |
| `@bao2025croms` | Optimal Model Selection for Conformalized Robust Optimization, arXiv 2507.04716 | preprint arXiv | Seleccion de modelos/sets CRO por riesgo decisional |
| `@liu2026portfolio` | Online CP via Universal Portfolio Algorithms, arXiv 2602.03168 | preprint arXiv | Online conformal / no estacionariedad |
| `@yeh2026` | End-to-End Conformal Calibration for Optimization Under Uncertainty, TMLR 12/2025 / arXiv v2 2026 | peer-reviewed TMLR | Puente end-to-end conformal + optimizacion |
| `@yeh2025training` | Conformal Risk Training, NeurIPS 2025 | peer-reviewed | OCE/CVaR y riesgo conformal end-to-end |
| `@zhao2025robust` | Conformal Robust Optimization and Satisficing, AISTATS workshop / SSRN 5338354 | workshop/working paper | Satisficing robusto como related/future |

## Auditoria del bound

Estado corregido en `book/chapters/14-paper-estrella/14b-theoretical-framework.qmd`:

- El teorema ahora controla un target acotado `Y_i in [0,1]` y la no cobertura ponderada `V = sum_i w_i 1{Y_i > u_i}`.
- La interpretacion de `Y_i` como PD latente se marca como supuesto adicional, no como observable.
- La policy debe leerse como fijada antes de observar los labels evaluados; el `276k` es validacion empirica post-seleccion, no una garantia conformal mas fuerte.
- Markov queda como resultado distribution-free principal.
- Hoeffding/Bernstein quedan como tightening condicional bajo supuestos adicionales de independencia/estructura, no como reemplazo.

Riesgo matematico que sigue abierto para journal: formalizar si el wrapper conformal debe cubrir default observado, PD latente, perdida esperada o un score de riesgo calibrado. El codigo actual (`scripts/validate_alpha_gamma_bound.py`) usa `y_true`/`default_flag` contra `pd_high`, por lo que el texto no debe afirmar mas que eso sin un lemma adicional.

## Mapa claim -> artefacto

| Claim paper | Artefacto que lo soporta | Estado |
|---|---|---|
| Champion oficial = economic champion bound-aware | `models/final_project_promotion.json` | sincronizado |
| Retorno oficial `$170.5K` | `final_champion.realized_total_return` | sincronizado |
| Pass exacto `alpha01` | `final_champion.alpha01_exact_pass` | sincronizado |
| Region robusta completa `45/45` | `robust_region_summary` | sincronizado |
| `V=0.03645`, `gamma_cp=0.18591`, `violation=0` | `final_champion` | sincronizado |
| Conformal winner coverage/ancho/min group | `conformal_upstream.winner_metrics` | sincronizado |
| PD operativo AUC/Brier/ECE | `pipeline_summary.json`, `reports/dvc/metrics_summary.json` | sincronizado con nota de familia |
| Historico PD mejor que actual | `models/search_pd/pd-hpo-local-2026-04-03-1325` | documentado, no promovido |
| Tablas paper1 | `reports/paper_material/paper1/tables/*`, `scripts/export_paper1_canonical_tables.py` | regeneradas desde fuentes canonicas |

## Cambios Quarto aplicados

- `14c-methodology.qmd`: elimina la contradiccion que decia que el economic champion no era oficial; agrega jerarquia de artefactos canonicos.
- `14b-theoretical-framework.qmd`: acota el bound a target observado/no cobertura ponderada y separa garantia distribution-free de validacion empirica post-seleccion.
- `14a-introduction-motivation.qmd`: agrega Online DFL, CROMS, end-to-end conformal calibration, robust CP multi-distribucion y fuente final de policy.
- `14e-discussion-conclusions.qmd`: actualiza auditoria, reproducibilidad, limitaciones y future work con los papers recientes, incluyendo selección conformal decision-aware y riesgo OCE/CVaR.
- `book/references.bib` y `docs/PAPER_REFERENCES_STATE_OF_ART.md`: corrigen metadatos de los PDFs recientes y documentan Online DFL, MDCP, CROMS, UP-OCP, end-to-end conformal calibration, conformal risk training y CRO/CRS.
- `reports/paper_material/paper1/tables/*`: regeneradas desde la promocion final para eliminar drift legacy.

## Issues separados

1. Paper Mondrian fue retirado como paper local el 2026-06-06; cualquier drift histórico queda absorbido por la capa Mondrian del libro y no bloquea el proyecto CRPTO externo.
2. Paper-facing tables: ya fueron regeneradas desde `final_project_promotion.json`; mantener `scripts/export_paper1_canonical_tables.py` como unica ruta de actualizacion.
3. Tightening teorico: Hoeffding/Bernstein requieren supuestos adicionales y posiblemente nested holdout/post-selection correction.

## Plan de mejora priorizado

1. Crear una tabla journal `claim -> artifact -> test` dentro del apendice reproducible. **Aplicado** en `book/chapters/14-paper-estrella/14e-discussion-conclusions.qmd` como `tbl-p1-claim-artifact-test`.
2. Formalizar un lemma separado para la lectura de PD latente, o retirar esa lectura del theorem principal.
3. Agregar experimento future-work minimo: nested holdout para confirmar que la seleccion bound-aware no usa el mismo OOT como unica evidencia.
4. Agregar un selector conformal decision-aware inspirado en CROMS: comparar familias por retorno robusto, `V`, `gamma_cp`, violación y métricas conformales, manteniendo la selección en holdout separado.
5. Diseñar una extensión OCE/CVaR del funded-set risk inspirada en Conformal Risk Training para estudiar colas de pérdida, no solo esperanza/Markov.
6. Incorporar `@capitaine2026online`, `@liu2026portfolio` y `@yang2026multidistribution` como trabajo futuro secuencial/no estacionario y como robustez multi-fuente.
7. Mantener SPO+ como comparador de regret, no como baseline de cobertura.
8. Abrir issue especifico para el drift del Paper Mondrian y no mezclarlo con el cierre del Paper Estrella.

Backlog operativo creado: `docs/research/paper_estrella_backlog_2026-05-04.md` separa P0/P1/P2/P3 y marca que el champion `paper-thesis-final-economic-2026-04-06` no debe reabrirse sin una busqueda nueva aprobada.
