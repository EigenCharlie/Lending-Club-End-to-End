# Tesis de Especialización (versión evidencia-ejecutable)

## Enfoque
Este documento resume la tesis de especialización con un criterio operativo:
cada afirmación debe estar soportada por artefactos reproducibles en el repositorio.

Pregunta de trabajo:
- ¿Cómo mejorar calibración y cuantificación de incertidumbre en riesgo crediticio (PD/LGD/EAD) con Conformal Prediction, y cómo se compara frente a calibración tradicional?

## Alineación con la propuesta
Objetivos mantenidos:
- Pipeline reproducible sobre Lending Club.
- Comparación de calibración tradicional vs enfoque conformal.
- Conexión a lectura regulatoria IFRS9/Basilea.

Evolución técnica incorporada:
- Modelo PD basado en CatBoost.
- Hardening operativo del pipeline (corridas resumibles, monitoreo, contratos de artefactos).
- Gate de promoción con chequeos de coherencia de artefactos.

## Evidencia canónica (artefactos)
- `data/processed/model_comparison.json`
- `models/conformal_policy_status.json`
- `models/fairness_audit_status.json`
- `models/ab_simulation_status.json`
- `models/conformal_lgd_ead_status.json`
- `reports/run_comparisons/<run_tag>/comparison.json`

## Criterio de promoción técnica de C
Se adopta política por fases:
1. Candidate:
   - `conformal_promotion_pass = true`
   - `artifact_coherence = true`
2. Merge-ready:
   - `pd_quality = true`
   - `ab_no_regression = true`
   - `fairness_relative = true`
   - `overall_pass = true`

## Fairness en especialización
Fairness conformal se mantiene como bloque secundario de gobernanza:
- se reporta estado y gaps;
- no se presenta como claim metodológico principal de novedad.

La profundización (cobertura condicional, trade-offs por grupo, adaptive/online CP) se traslada a:
- `streamlit_app/pages/thesis_end_to_end.py`
- `streamlit_app/pages/thesis_contribution.py`
- `streamlit_app/pages/research_landscape.py`

## Estado LGD/EAD conformal
Se agrega salida ejecutable para intervalos conformales de regresión:
- `data/processed/conformal_intervals_lgd.parquet`
- `data/processed/conformal_intervals_ead.parquet`
- `models/conformal_lgd_ead_status.json`

Si el artefacto no existe en una corrida, la narrativa debe mostrar “no disponible” explícitamente.

## Nota de trazabilidad
La versión oficial debe fijarse por SHA y `run_tag`, con baseline snapshot regenerado al reanudar corridas.
