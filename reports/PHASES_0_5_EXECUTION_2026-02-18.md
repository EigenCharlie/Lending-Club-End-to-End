# Ejecucion Fases 0-5 (Validity Hardening + Panorama de Investigacion)
Fecha de ejecucion: 2026-02-18
Branch: `publication/validity-hardening`

## 1) Contexto y objetivo
La meta de esta corrida fue dejar los papers y el proyecto listos para publicacion futura (sin publicar ahora), con foco en:
- validez metodologica (evitar leakage y alinear datos/decisiones),
- reproducibilidad completa por DVC,
- coherencia narrativa en Streamlit,
- fortalecimiento del panorama de investigacion con literatura reciente.

## 2) Fase 0 — Baseline y control de cambios
Acciones:
- Snapshot inicial en `reports/baseline_2026-02-18.json` con:
  - commit/branch,
  - `dvc status` local/cloud,
  - metricas canonicas,
  - hashes SHA256 de artefactos clave.
- Verificacion de reproducibilidad base (`dvc repro` up-to-date).

Motivo:
- Evitar comparaciones ambiguas y tener trazabilidad before/after defendible.

## 3) Fase 1 — Fixes de validez critica
### 3.1 Conformal sin leakage de test labels
Archivo: `scripts/generate_conformal_intervals.py`
- Se elimino la adaptacion con etiquetas de test.
- La calibracion de pisos por subgrupo ahora se aprende en holdout de calibracion.
- Se agregaron metadatos de split y tuning; se agrega `id` en artefacto para joins robustos.

Motivo:
- La adaptacion con `y_test` rompe validez out-of-sample y sobreestima cobertura.

### 3.2 Alineacion exacta loan-interval en optimizacion
Archivos:
- `scripts/optimize_portfolio.py`
- `scripts/optimize_portfolio_tradeoff.py`

Cambios:
- Join por `id` entre prestamos candidatos e intervalos conformales.
- Fallback posicional explicito (con warning) solo si falta `id`.
- Se agrega `baseline_nonrobust_funded` al resumen de robustez.
- Se extiende frontera strict a `risk_tolerance=0.12`.

Motivo:
- Evitar desalineacion silenciosa entre riesgo estimado y decision de financiamiento.

### 3.3 Orquestacion E2E completa y consistente
Archivo: `scripts/end_to_end_pipeline.py`
- Se incorporaron pasos faltantes: survival, IFRS9 y tradeoff optimization.
- Se corrigio lectura de funded baseline (`baseline_nonrobust_funded` con fallback).

Test asociado:
- `tests/test_scripts/test_end_to_end_pipeline.py` actualizado.

## 4) Fase 2 — Coherencia narrativa + panorama de investigacion
### 4.1 UI sin snapshots obsoletos
Archivos:
- `streamlit_app/pages/thesis_contribution.py`
- `streamlit_app/pages/executive_summary.py`
- `streamlit_app/pages/model_laboratory.py`
- `streamlit_app/pages/survival_analysis.py`
- `streamlit_app/pages/glossary_fundamentals.py`

Cambios:
- Remocion de metricas hardcodeadas.
- Carga dinamica desde artefactos canonicos.

Test asociado:
- `tests/test_docs/test_narrative_consistency.py` endurecido contra snapshots stale.

### 4.2 Panorama de investigacion actualizado con papers recientes
Archivo:
- `streamlit_app/pages/research_landscape.py`

Cambios:
- Se actualizaron listas "Papers clave" por seccion (ML credit scoring, conformal, predict-then-optimize, conformal+OR, survival, time series).
- Se reemplazaron entradas dudosas por referencias recientes verificables.

## 5) Fase 3 — Benchmark de variantes conformales
Nuevo script:
- `scripts/benchmark_conformal_variants.py`

Salidas:
- `data/processed/conformal_variant_benchmark.parquet`
- `data/processed/conformal_variant_benchmark_by_group.parquet`

Motivo:
- Comparar eficiencia/cobertura entre variantes y sustentar decision metodologica de Mondrian.

## 6) Fase 4 — Survival + IFRS9
### 6.1 Survival con tiempos reales cuando existen
Archivo:
- `scripts/run_survival_analysis.py`

Cambio:
- `time_to_event` se calcula desde `issue_d` a `last_pymnt_d` (raw), con fallback proxy solo para faltantes.

Resultado:
- cobertura de tiempos reales: 99.6%; fallback: 0.4%.

### 6.2 SICR por incertidumbre realmente operativo
Archivo:
- `src/evaluation/ifrs9.py`

Cambio:
- Stage 2 ya no "reconfirma" solo SICR absoluto; ahora activa Stage 2 en alta incertidumbre (decil superior) si PD no mejora.

Test nuevo:
- `tests/test_evaluation/test_ifrs9.py::test_assign_stage_uncertainty_trigger_stage2_without_abs_sicr`.

## 7) Fase 5 — Backtest temporal OOT de politica causal
Nuevo script:
- `scripts/backtest_causal_policy_oot.py`

Salidas:
- `data/processed/causal_policy_oot_backtest.parquet`
- `data/processed/causal_policy_oot_backtest_by_grade.parquet`
- `models/causal_policy_oot_status.json`

Resultado:
- 106 meses evaluados, `p05_monthly_net > 0`.

## 8) DVC / pipeline
Se agregaron stages en `dvc.yaml`:
- `benchmark_conformal_variants`
- `backtest_causal_policy_oot`

Reproduccion ejecutada:
- `dvc repro generate_conformal benchmark_conformal_variants backtest_conformal_coverage validate_conformal_policy optimize_portfolio optimize_portfolio_tradeoff run_survival_analysis run_ifrs9_sensitivity simulate_causal_policy validate_causal_policy backtest_causal_policy_oot build_pipeline_results export_streamlit_artifacts export_storytelling_snapshot`

Estado local:
- `uv run dvc status --json` -> `{}`

Nota cloud:
- `uv run dvc status -c --json` muestra objetos nuevos pendientes de `dvc push`.

## 9) Delta de metricas (baseline -> actual)
Fuente baseline: `reports/baseline_2026-02-18.json`
Fuente actual: artefactos regenerados 2026-02-18.

- Conformal coverage 90: `0.9021 -> 0.8887` (esperado tras quitar leakage).
- Conformal coverage 95: `0.9489 -> 0.9480`.
- Conformal avg width 90: `0.7527 -> 0.7459`.
- Policy checks pass: `6/7 -> 1/7` (umbral vigente ahora mas estricto frente a estimacion no filtrada).
- Robust funded (pipeline): `0 -> 9`.
- Non-robust funded (pipeline): `0 -> 150`.
- ECL baseline: `797.9M -> 1.010B`.
- ECL severe: `1.380B -> 1.851B`.

## 10) Validacion final
Comandos ejecutados:
- `uv run pytest -q` -> `201 passed`.
- `uv run python -m py_compile ...` (scripts y paginas modificadas) -> OK.
- `uv run dvc status --json` -> `{}`.

Conclusión operativa:
- Las fases 0-5 quedaron ejecutadas end-to-end, con artefactos regenerados, tests en verde y documentacion de motivos/impacto.

## 11) Referencias recientes verificadas (agregadas al panorama)
Consultadas y usadas para actualizar `research_landscape.py` (verificadas el 2026-02-18):
- Ayari et al. (AI Review): https://link.springer.com/article/10.1007/s10462-025-11152-x
- Soni et al. (Cybernetics and IT): https://sciendo.com/article/10.2478/cait-2024-0013
- Liu et al. (ESWA): https://www.sciencedirect.com/science/article/pii/S095741742600197X
- Plassier et al. (arXiv): https://arxiv.org/abs/2410.21167
- Capitaine et al. (arXiv): https://arxiv.org/abs/2502.07862
- Patel et al. (AISTATS/OpenReview): https://openreview.net/forum?id=Mt9Qud1YNC
- Bao et al. (arXiv): https://arxiv.org/abs/2507.04716
- Kato (arXiv): https://arxiv.org/abs/2405.04127
- Bárcena Saavedra et al. (ESWA): https://www.sciencedirect.com/science/article/pii/S0957417424016434
- Botha & Verster (arXiv): https://arxiv.org/abs/2505.10270
- Ptak-Chmielewska & Kopciuszewski (JCR): https://www.risk.net/journal-of-credit-risk/7959625/random-survival-forests-and-cox-regression-in-loss-given-default-estimation
- Schlembach et al. (Machine Learning): https://link.springer.com/article/10.1007/s10994-025-06742-0
- Wang & Hyndman (arXiv / conformalForecast): https://arxiv.org/abs/2405.16824

## 12) Update UI de investigacion para revision de profesor (2026-02-19)
Objetivo de esta actualizacion:
- pasar de workspaces cortos a drafts tipo paper completo en Streamlit,
- centralizar guias de buenas practicas/herramientas en una sola pagina transversal,
- dejar fases + pendientes de revision al final de cada paper para facilitar feedback experto.

Cambios ejecutados:
- `streamlit_app/pages/paper_2_ifrs9_e2e.py`:
  - estructura completa de paper (0-10),
  - ecuaciones LaTeX, figuras y tablas numeradas,
  - seccion final con tracker de fases + bullets de cambios potenciales por seccion/figura/tabla.
- `streamlit_app/pages/paper_3_mondrian.py`:
  - estructura completa de paper (0-10),
  - ecuaciones LaTeX, figuras y tablas numeradas,
  - seccion final con tracker de fases + bullets de cambios potenciales por seccion/figura/tabla.
- `streamlit_app/pages/paper_1_cp_robust_opt.py`:
  - limpieza para asegurar que el cierre final quede en fases + bullets (sin bloque posterior).
- nueva pagina transversal:
  - `streamlit_app/pages/research_best_practices.py`
  - consolida buenas practicas de redaccion, reproducibilidad, uso de LaTeX, figuras y tablas.
- navegacion:
  - `streamlit_app/app.py` actualizado para incluir la nueva pagina en seccion `Investigación`.
- trazabilidad adicional:
  - `docs/PAPER_DEVELOPMENT_PLAYBOOK_2026.md` actualizado con la nueva organizacion y criterios.
- discoverability:
  - `streamlit_app/pages/research_landscape.py` agrega link directo a la guia transversal.
- testing:
  - `tests/test_streamlit/test_page_imports.py` actualizado a 24 paginas.

Motivo de diseño:
- Reducir friccion en la revision con profesor PhD: cada paper ahora se lee como borrador integral,
  y la guia de practicas queda separada para no repetir contenido en cada pagina.

Validacion de esta actualizacion:
- `python -m py_compile ...` sobre paginas modificadas -> OK.
- `uv run pytest -q tests/test_streamlit/test_page_imports.py` -> `51 passed`.
- `uv run pytest -q` -> `209 passed`.
