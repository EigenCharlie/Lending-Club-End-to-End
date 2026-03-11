# Tesis de Especializacion - Guion de Defensa (version sincronizada con pagina Streamlit)

## Resumen ejecutivo
Este documento espejo cuenta la misma historia de `streamlit_app/pages/tesis_especializacion.py`: contexto, teoria, datos, metodologia, resultados y cierre defendible.

- Titulo: Conformal Prediction para la Calibracion y Cuantificacion de Incertidumbre en Modelos de Riesgo Crediticio
- Autor: Carlos Alfredo Vergara Rojas
- Programa: Especializacion en Analitica y Ciencia de Datos Aplicada (UTP)
- Run canonicamente activo: `2026-03-01-C-smart-v2`
- Principio rector: claims solo con evidencia ejecutable de artefactos vigentes

## Capitulo 1. Apertura del problema y objetivos de la tesis
Problema central:
- En riesgo crediticio, una prediccion puntual puede servir para ranking, pero no para decidir con prudencia cupo, pricing, provisiones o capital.

Objetivo narrativo:
- Pasar de "modelo que predice" a "modelo defendible para decidir bajo incertidumbre".

Pregunta de investigacion:
- Como mejorar calibracion y cuantificacion de incertidumbre en modelos PD/LGD/EAD mediante Conformal Prediction y como se compara frente a metodos tradicionales.

Objetivo general:
- Implementar y evaluar tecnicas conformales para calibracion y cuantificacion de incertidumbre en PD/LGD/EAD con Lending Club como caso de estudio.

Respuesta al evaluador (aprobado con ajustes):
- Se acota alcance a evidencia trazable por componente.
- Se prioriza calidad metodologica sobre expansion de alcance.
- Se mantiene continuidad a maestria como agenda futura, sin quitar protagonismo al cierre de especializacion.

## Capitulo 2. Contexto financiero y planteamiento del problema
Problema de negocio:
- Predicciones puntuales sin banda de incertidumbre generan decisiones sobreconfiadas.

Impacto operativo:
- Riesgo de subestimar perdidas (o sobreprovisionar), afectando rentabilidad y asignacion de capital.

Impacto regulatorio:
- Mayor dificultad de defensa tecnica bajo IFRS9/Basilea y marcos de Model Risk Management.

Decision que esta en juego:
- No solo "quien tiene mas riesgo", sino "con que grado de confianza se toma cada decision".

## Capitulo 3. Marco teorico y estado del arte
Ecuaciones eje:
- `ECL = PD * LGD * EAD`
- `P(Y in C(X)) >= 1 - alpha`

Lectura metodologica:
- Calibracion (Platt/Isotonic/Venn-Abers) corrige nivel probabilistico.
- Conformal (Split/Mondrian/CQR) agrega cuantificacion de incertidumbre con cobertura controlada.
- Se usan de forma complementaria, no excluyente.

Cinco papers clave (alineados al anteproyecto):
1. Vovk, Gammerman, Shafer (2022): base formal de Conformal Prediction.
2. Angelopoulos, Bates (2023): sintesis moderna de UQ distribution-free.
3. Romano, Patterson, Candes (2019): CQR para regresion heteroscedastica.
4. Vovk, Petej (2014): Venn-Abers para salida probabilistica intervalar.
5. Bellotti (2017): aplicacion conformal en credit scoring.

## Capitulo 4. Datos y preparacion (Lending Club)
Fuente:
- Lending Club 2007-2020, con particion temporal OOT.

Splits canonicos:
- Train: 1,346,311 filas, 110 columnas.
- Calibration: 237,584 filas, 110 columnas.
- Test: 276,869 filas, 110 columnas.

EDA clave (train):
- Default rate: 18.52%.
- Plazo: 36 meses (1,005,656), 60 meses (340,655).
- Gradiente de riesgo por grade: A=5.63%, B=12.29%, C=20.64%, D=28.31%, E=36.47%, F=43.44%, G=47.71%.

Lectura de calidad:
- El gradiente por grade confirma senal economica de riesgo.
- Splits temporales y contrato de datos reducen leakage y sostienen validez de resultados.
- Diccionario de variables (`dataset_dictionary.json`) refuerza trazabilidad de interpretacion.

## Capitulo 5. Metodologia y diseno experimental
Marco operativo:
- CRISP-DM con trazabilidad objetivo especifico -> evidencia ejecutable.

Trazabilidad por objetivos:
1. Estado del arte: `docs/conformal_prediction_research_2026.md`, `docs/conformal_prediction_quick_reference.md`
2. Datos y preparacion: `src/data/make_dataset.py`, `data/processed/train.parquet`, `data/processed/test.parquet`
3. Implementacion conformal: `src/models/conformal.py`, `scripts/generate_conformal_intervals.py`
4. Comparativa vs calibracion: `data/processed/model_comparison.json`, `data/processed/conformal_variant_benchmark.parquet`
5. Impacto IFRS9/backtesting: `data/processed/ifrs9_scenario_summary.parquet`, `src/evaluation/backtesting.py`
6. Lineamientos de adopcion: `models/conformal_policy_status.json`, `models/fairness_audit_status.json`, `docs/RUNBOOK.md`

Desarrollo detallado por fase CRISP-DM:
- Business Understanding: se acota alcance segun observacion del evaluador y se fija criterio de exito defendible.
- Data Understanding: se valida senal de riesgo, distribuciones y calidad en Lending Club.
- Data Preparation: se consolidan splits OOT, contratos de features y se cierra disponibilidad de target LGD.
- Modeling: se integran PD calibrada + conformal y se compara benchmark de variantes LGD/EAD.
- Evaluation: se revisa discriminacion/calibracion, cobertura/ancho, estabilidad temporal, fairness y sensibilidad IFRS9.
- Deployment: se consolida narrativa reproducible en Streamlit + reporte espejo sincronizado.

## Capitulo 6. Resultados PD
Fuente canonica:
- `data/processed/model_comparison.json`

Metricas OOT (run actual):
- AUC ROC: 0.7117
- Gini: 0.4234
- Brier: 0.1548
- ECE: 0.0072
- KS: 0.3129
- Calibrador seleccionado: Isotonic Regression

Interpretacion:
- AUC/Gini/KS muestran capacidad de separacion.
- Brier/ECE confirman calidad probabilistica para uso en decision.
- Conclusion: PD no solo clasifica; tambien es util para pricing/provision por calibracion defendible.

Evidencia complementaria:
- `data/processed/roc_curve_data.parquet`
- `data/processed/calibration_curve_data.parquet`
- `data/processed/shap_summary.parquet`
- `data/processed/permutation_importance.parquet`

## Capitulo 7. Resultados LGD/EAD
Fuente canonica:
- `models/conformal_lgd_ead_status.json`

Estado por componente:
- PD: disponible
- LGD: disponible
- EAD: disponible

LGD (run actual):
- Variante seleccionada: `direct_adaptive_grade_temporal`
- Coverage 90%: 90.50% (target 90%)
- Coverage 95%: 95.50% (target 95%)
- Ancho medio 90%: 0.4959
- Ancho medio 95%: 0.5744
- Guardrails: `overall_pass=true`

EAD (run actual):
- Coverage 90%: 91.37%
- Coverage 95%: 95.45%
- Ancho medio 90%: 160.88
- Ancho medio 95%: 241.88

Lectura de cierre de brecha:
- La brecha ya no es "no hay LGD", sino sostener estabilidad de cobertura/eficiencia en monitoreo.

Artefactos de soporte:
- `data/processed/conformal_intervals_lgd.parquet`
- `data/processed/lgd_variant_benchmark.parquet`
- `data/processed/lgd_coverage_by_grade.parquet`
- `data/processed/lgd_coverage_by_year.parquet`
- `data/processed/conformal_intervals_ead.parquet`

## Capitulo 8. Conformal prediction y evaluacion comparativa
Fuente canonica:
- `models/conformal_policy_status.json`

KPIs conformal (run actual):
- Coverage 90%: 91.67%
- Coverage 95%: 95.59%
- Min group coverage 90%: 88.40%
- Avg width 90%: 0.7442
- Winkler 90%: 1.2281
- Checks: 8/13
- Alerts: 4
- Estado estricto: `overall_pass=false`

Interpretacion:
- La cobertura global cumple, pero quedan alertas/chequeos por cerrar para gate estricto.
- Conformal se presenta como sistema gobernable (metas + alertas + backtesting), no como claim abstracto.

Evidencia complementaria:
- `data/processed/conformal_policy_checks.parquet`
- `data/processed/conformal_backtest_monthly.parquet`
- `data/processed/conformal_backtest_monthly_grade.parquet`
- `data/processed/conformal_variant_benchmark.parquet`
- `data/processed/conformal_backtest_alerts.parquet`

## Capitulo 9. Impacto regulatorio (IFRS9)
Fuente canonica:
- `data/processed/ifrs9_scenario_summary.parquet`

ECL por escenario (run actual):
- Baseline: 976,657,703.50
- Mild stress: 1,200,039,584.84
- Adverse: 1,462,982,396.67
- Severe: 1,791,015,732.81

Lectura stage en baseline:
- Stage 1: 95,547 (34.51%)
- Stage 2: 119,071 (43.01%)
- Stage 3: 62,251 (22.48%)

Interpretacion:
- El paso baseline -> severe evidencia sensibilidad prudencial relevante.
- El valor de la tesis no es solo estimar ECL, sino explicar su variacion por escenario, stage y grade.

Artefactos de soporte:
- `data/processed/ifrs9_scenario_grade_summary.parquet`
- `data/processed/ifrs9_sensitivity_grid.parquet`
- `data/processed/ifrs9_input_quality.parquet`

## Capitulo 10. Conclusiones y proximos pasos
### Conclusiones sobre resultados obtenidos
1. La narrativa ya integra contexto, teoria, datos, metodologia y resultados con evidencia actual.
2. PD esta consolidado con metrica OOT y calibracion defendible para decision.
3. LGD y EAD disponen de evidencia conformal; el foco pasa a sostenibilidad de calidad en el tiempo.
4. IFRS9 queda conectado a escenarios y stages con lectura tecnica y de negocio.
5. Fairness se reporta de forma transparente como diagnostico de gobernanza.

### Impacto en industria financiera
- Mejora de decisiones de originacion/pricing al evitar sobreconfianza en predicciones puntuales.
- Mayor defendibilidad ante auditoria por trazabilidad, backtesting y monitoreo de cobertura.
- Mejor comunicacion de sensibilidad de provisiones IFRS9 hacia comites de riesgo/finanzas.

### Impacto en academia aplicada
- Integra calibracion + conformal + IFRS9 en un caso reproducible de credito.
- Mantiene equilibrio PD/LGD/EAD en la narrativa, evitando sesgo de reportar solo PD.
- Mueve evaluacion desde metricas aisladas hacia guardrails de decision y gobernanza.

### Proximos pasos (continuidad sin perder protagonista de especializacion)
1. Cierre especializacion: validar con directora narrativa final y preparar paquete para jurados.
2. Continuidad maestria - Paper 1: extender CP hacia robust optimization con uncertainty sets conformales.
3. Continuidad maestria - Paper 2: escalar IFRS9 end-to-end con incertidumbre conformal en staging/ECL.

### Preguntas esperables de directora/jurados
1. Como se controlo el alcance tras observaciones del evaluador?
- Respuesta: estado por componente + evidencia ejecutable + priorizacion de entregables defendibles.
2. Por que calibracion y conformal juntos?
- Respuesta: calibracion corrige nivel probabilistico; conformal agrega cobertura de incertidumbre.
3. Que falta para cierre previo a jurados?
- Respuesta: consolidar monitoreo de estabilidad, cerrar ajustes de narrativa y empaquetar sustentacion.

---

## Matriz minima de trazabilidad (run_tag / generated_at_utc)
| Artefacto | run_tag | generated_at_utc |
|---|---|---|
| `data/processed/model_comparison.json` | n/a | 2026-03-01T17:35:03.767184+00:00 |
| `models/conformal_policy_status.json` | 2026-03-01-C-smart-v2 | 2026-03-01T17:23:42.637129+00:00 |
| `models/conformal_lgd_ead_status.json` | 2026-03-01-C-smart-v2 | 2026-03-02T05:06:31.405406+00:00 |
| `models/fairness_audit_status.json` | 2026-03-01-C-smart-v2 | 2026-03-01T17:34:45.953826+00:00 |
| `reports/dvc/metrics_summary.json` | 2026-03-01-C-smart-v2 | 2026-03-01T17:35:57.001229+00:00 |
| `data/processed/runtime_status.json` | n/a | 2026-03-01T17:35:54.302664+00:00 |
