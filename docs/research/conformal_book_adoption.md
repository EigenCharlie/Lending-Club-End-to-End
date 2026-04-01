> **RESEARCH NOTE** — Material de referencia y adopción editorial; no describe la política operativa viva.

# Conformal Book Adoption Registry

Fecha: 2026-03-13

Este documento fija qué partes de `Practical Guide to Applied Conformal Prediction in Python`
se adoptan de forma canónica en el proyecto y cuáles quedan descartadas o solo como research.

## Adoptado

- `MAPIE 1.3.x` como librería primaria para:
  - intervalos conformales principales,
  - `classification sets`,
  - benchmark adaptativo en time series (`EnbPI`, `ACI`).
- `venn-abers 1.5.x` como implementación oficial de calibración `Venn-Abers`.
- `crepes 0.9.x` solo para `p-values`, predictive systems y experiments.
- Taxonomía operativa del repo:
  - `split conformal`
  - `cross conformal` en score-space como benchmark liviano
  - `Mondrian/group-conditional`
  - `CQR`
  - `time-series adaptive`

## Research-only

- `classification sets` binarios para abstención/triage.
- `OnlineConformal` en forecasting.
- predictive systems de `crepes`.
- extensiones multiclase futuras para `loan_status` o migración de stage.

## Descartado del carril canónico

- `Nonconformist`
  - legado, mantenimiento insuficiente, API vieja.
- `aws-fortuna`
  - proyecto archivado; no conviene como dependencia productiva.
- `NeuralProphet`
  - fuera del foco tabular credit risk y sin ventaja clara para el carril canónico actual.
- instalaciones `git+...`
  - no se usan en rebuild/productive workflows salvo justificación excepcional.

## Regla de promoción

- Ningún método entra a `canonical_rebuild` por novedad o por aparecer en el libro.
- Primero debe ganar o corregir algo real frente al baseline vivo con artifacts reproducibles.
- `insights_factory` absorbe primero todo método nuevo con costo o riesgo metodológico alto.

## Estado Ejecutado

- `Venn-Abers` quedó confirmado como calibrador oficial actual.
- `classification sets (LAC)` quedó validado como sidecar de abstención, no como policy canónica.
- Nuevas particiones `score_decile_mondrian` y `grade_x_scoreband_mondrian` fueron implementadas y benchmarkeadas.
- `PD conformal` sigue sin promoción canónica:
  - `checks_passed=8/13`
  - falla no estadística principal: `winkler_90`
- `EnbPI/ACI` quedaron formalizados en `time_series_status`, pero siguen `research_only`.
- `LGD/EAD` recibió benchmark corto con `direct_cqr` y `jackknife_after_bootstrap_short_benchmark`; el champion vigente no cambió.
