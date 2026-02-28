# OFFICIAL RERUN MASTER PLAN (2026-02-27)

## 1) Objetivo y alcance oficial

Este documento define la corrida oficial candidata a promocion a `main`, con atribucion correcta de impacto por capa de rama:

- `A = main` (baseline oficial actual)
- `B = research/toboml2-integration-v1` (integracion TOBoML2 no corrida full-data contra `main`)
- `C = experiment/conformal-toboml-integration-rerun` (B + hardening operativo final)

Alcance bloqueante de promocion:

- Core end-to-end (preflight, main_pre, heavy_main, post_core)
- Causal
- Governance/MRM

Alcance anexo (evidencia extendida, no bloqueante):

- RAPIDS
- Notebooks masivos

Regla de interpretacion oficial:

- `main -> experiment` no representa solo cambios recientes; incluye paquete heredado de `research`.
- La decision de promocion se toma con resultados de `C`, pero la causalidad de mejora se explica con `A/B/C`.

---

## 2) Relacion entre ramas (evidencia Git)

Evidencia levantada en esta maquina:

- `A_SHA (main) = dbf5f18fda7a72d3b4c217194c6269fdc6d3b511`
- `B_SHA (research/toboml2-integration-v1) = b8b44b144e3a79fb2e8f1d69a4ee9ce68994fb42`
- `C_SHA (experiment/conformal-toboml-integration-rerun) = b8b44b144e3a79fb2e8f1d69a4ee9ce68994fb42` (antes de commit final de hardening)

Conteo de divergencia:

- `main...research = 0|9`
- `research...experiment = 0|0` (a nivel commit base actual)
- `main...experiment = 0|9`

Relacion con `experiment/overnight-full-rerun-2026-02-26`:

- Es ancestro de `research` (`ancestor=yes`)
- `overnight...research = 0|8`

Nota operativa clave:

- Antes de iniciar corridas oficiales, cerrar el hardening pendiente en `experiment` con commit(es) final(es), volver a capturar `C_SHA` y limpiar working tree.

---

## 3) Metodologia de atribucion (A/B/C)

Se ejecutan tres snapshots y tres comparaciones oficiales:

1. `Delta(B-A)` = impacto de integracion TOBoML2 (`research` vs `main`).
2. `Delta(C-B)` = impacto incremental de hardening final (`experiment` vs `research`).
3. `Delta(C-A)` = impacto total candidato a promocion (`experiment` vs `main`).

Reglas de lectura:

- Si `Delta(C-A)` mejora pero `Delta(C-B)` es pequeno, la mayor parte del valor viene del paquete `research`.
- Si `Delta(C-B)` agrega mejoras en estabilidad/gates/operacion, ese valor se atribuye al hardening final.
- Promocion se decide sobre `C` completo, pero informe ejecutivo debe separar contribuciones por delta.

---

## 4) Plan operativo faseado (full-data)

### 4.1 Preparacion previa obligatoria

1. Congelar SHAs:
   - `A_SHA = main`
   - `B_SHA = research/toboml2-integration-v1`
   - `C_SHA = experiment/conformal-toboml-integration-rerun` (commit final de hardening)
2. Verificar working tree limpio en cada rama antes de correr:
   - `git status --short` debe quedar vacio.
3. Mantener datos/splits congelados:
   - `data/raw/Loan_status_2007-2020Q3.csv`
   - particiones `train/calibration/test` y `_fe` vigentes.
4. Ejecutar analitica full-data:
   - `sample_size = 0` donde aplique.
5. Correr A/B/C de forma secuencial en la misma maquina para reducir ruido de infraestructura.

### 4.2 Fase A - Baseline oficial (`main`)

```bash
git checkout main
git status --short
bash scripts/start_long_run.sh 2026-02-27-A-main-core --no-rapids --no-notebooks --stop-on-optional-failure
bash scripts/monitor_long_run.sh 2026-02-27-A-main-core
uv run python scripts/run_comparison.py snapshot --run-tag 2026-02-27-A-main-final
```

Salida minima esperada:

- `reports/run_logs/2026-02-27-A-main-core/`
- `reports/run_comparisons/2026-02-27-A-main-final/baseline_snapshot.json`

### 4.3 Fase B - Impacto research (`research/toboml2-integration-v1`)

```bash
git checkout research/toboml2-integration-v1
git status --short
bash scripts/start_long_run.sh 2026-02-27-B-research-core --no-rapids --no-notebooks --stop-on-optional-failure
bash scripts/monitor_long_run.sh 2026-02-27-B-research-core
uv run python scripts/run_comparison.py snapshot --run-tag 2026-02-27-B-research-final
```

Salida minima esperada:

- `reports/run_logs/2026-02-27-B-research-core/`
- `reports/run_comparisons/2026-02-27-B-research-final/baseline_snapshot.json`

### 4.4 Fase C - Candidato final (`experiment/conformal-toboml-integration-rerun`)

```bash
git checkout experiment/conformal-toboml-integration-rerun
git status --short
bash scripts/start_long_run.sh 2026-02-27-C-experiment-core --no-rapids --no-notebooks --stop-on-optional-failure
bash scripts/monitor_long_run.sh 2026-02-27-C-experiment-core
uv run python scripts/run_comparison.py snapshot --run-tag 2026-02-27-C-experiment-final
```

Salida minima esperada:

- `reports/run_logs/2026-02-27-C-experiment-core/`
- `reports/run_comparisons/2026-02-27-C-experiment-final/baseline_snapshot.json`

### 4.5 Fase D - Comparaciones cruzadas obligatorias

```bash
# Delta(B-A): research vs main
uv run python scripts/run_comparison.py compare \
  --run-tag 2026-02-27-B-research-core \
  --baseline reports/run_comparisons/2026-02-27-A-main-final/baseline_snapshot.json

# Delta(C-B): experiment vs research
uv run python scripts/run_comparison.py compare \
  --run-tag 2026-02-27-C-experiment-core \
  --baseline reports/run_comparisons/2026-02-27-B-research-final/baseline_snapshot.json

# Delta(C-A): experiment vs main
uv run python scripts/run_comparison.py compare \
  --run-tag 2026-02-27-C-experiment-core \
  --baseline reports/run_comparisons/2026-02-27-A-main-final/baseline_snapshot.json
```

---

## 5) Monitoreo y resiliencia

Componentes oficiales de monitoreo:

- Orquestador: `scripts/run_long_pipeline.py`
- Lanzador desacoplado: `scripts/start_long_run.sh`
- Monitor continuo: `scripts/monitor_long_run.sh`
- Estado/heartbeat: `reports/run_logs/<run_tag>/heartbeat.json`
- Estado por etapa: `reports/run_logs/<run_tag>/status/*.exit`
- Log maestro: `reports/run_logs/<run_tag>/master.log`
- Monitor SQL Optuna: `sql/optuna_live_monitor.sql`

Politica de recuperacion por corte/reinicio:

```bash
bash scripts/start_long_run.sh <run_tag> --resume --no-rapids --no-notebooks --stop-on-optional-failure
```

Controles de resiliencia que deben verificarse:

1. `--resume` no repite etapas ya exitosas (`*.exit = 0`).
2. Heartbeat refleja `state/current_step/last_update_utc/final_exit_code`.
3. Si falla paso opcional y se usa `--stop-on-optional-failure`, salida final debe ser `exit code 1`.

---

## 6) Criterios de cierre oficial (promocion a main)

Para recomendar merge de `C` a `main`:

1. `Delta(C-A)` con `comparison.overall_pass == true`.
2. `conformal_promotion_pass == true` (warnings estadisticos permitidos y documentados).
3. Sin regresion material en PD/Fairness/Survival/OR/IFRS9 bajo umbrales vigentes.
4. Artifacts obligatorios presentes y coherentes (modelos, evaluaciones, comparaciones, governance, mrm).
5. Informe oficial incluye atribucion A/B/C y justificacion por capa.

Regla adicional de gobierno:

- `mrm_validation_report` se usa como evidencia diagnostica de apoyo para esta promocion, no como bloqueo global.

---

## 7) Riesgos y mitigaciones

1. Riesgo: trials Optuna en estado `RUNNING` huerfano.
   - Mitigacion: monitoreo SQL + heartbeat + reanudacion controlada; documentar limpieza si aplica.
2. Riesgo: fairness audit degradado por ausencia de `test_predictions.parquet`.
   - Mitigacion: validar produccion de artifacts antes de fairness stage; fail-fast si falta insumo.
3. Riesgo: drift/gate de governance en limite.
   - Mitigacion: revisar `generate_governance_status.py` previo a MRM en `post_core`.
4. Riesgo: duraciones largas en RAPIDS/notebooks.
   - Mitigacion: ejecutar como anexo no bloqueante y reportar por separado.
5. Riesgo: atribucion incorrecta por cambios sin commit.
   - Mitigacion: exigir tree limpio + SHA congelado antes de cada corrida.

---

## 8) Checklist final ejecutable (go/no-go)

### 8.1 Secuencia de ejecucion

```bash
# 0) Verificacion de entorno/limpieza
git status --short
nproc
free -h
df -h /
nvidia-smi

# 1) Fase A (main)
git checkout main
bash scripts/start_long_run.sh 2026-02-27-A-main-core --no-rapids --no-notebooks --stop-on-optional-failure
bash scripts/monitor_long_run.sh 2026-02-27-A-main-core
uv run python scripts/run_comparison.py snapshot --run-tag 2026-02-27-A-main-final

# 2) Fase B (research)
git checkout research/toboml2-integration-v1
bash scripts/start_long_run.sh 2026-02-27-B-research-core --no-rapids --no-notebooks --stop-on-optional-failure
bash scripts/monitor_long_run.sh 2026-02-27-B-research-core
uv run python scripts/run_comparison.py snapshot --run-tag 2026-02-27-B-research-final

# 3) Fase C (experiment)
git checkout experiment/conformal-toboml-integration-rerun
bash scripts/start_long_run.sh 2026-02-27-C-experiment-core --no-rapids --no-notebooks --stop-on-optional-failure
bash scripts/monitor_long_run.sh 2026-02-27-C-experiment-core
uv run python scripts/run_comparison.py snapshot --run-tag 2026-02-27-C-experiment-final

# 4) Comparaciones cruzadas
uv run python scripts/run_comparison.py compare --run-tag 2026-02-27-B-research-core --baseline reports/run_comparisons/2026-02-27-A-main-final/baseline_snapshot.json
uv run python scripts/run_comparison.py compare --run-tag 2026-02-27-C-experiment-core --baseline reports/run_comparisons/2026-02-27-B-research-final/baseline_snapshot.json
uv run python scripts/run_comparison.py compare --run-tag 2026-02-27-C-experiment-core --baseline reports/run_comparisons/2026-02-27-A-main-final/baseline_snapshot.json
```

### 8.2 Tabla de validacion go/no-go

| Item | Evidencia | Estado |
|---|---|---|
| A snapshot creado | `reports/run_comparisons/2026-02-27-A-main-final/baseline_snapshot.json` | TODO |
| B snapshot creado | `reports/run_comparisons/2026-02-27-B-research-final/baseline_snapshot.json` | TODO |
| C snapshot creado | `reports/run_comparisons/2026-02-27-C-experiment-final/baseline_snapshot.json` | TODO |
| Delta(B-A) generado | carpeta `reports/run_comparisons/2026-02-27-B-research-core/` | TODO |
| Delta(C-B) generado | carpeta `reports/run_comparisons/2026-02-27-C-experiment-core/` | TODO |
| Delta(C-A) generado | carpeta `reports/run_comparisons/2026-02-27-C-experiment-core/` | TODO |
| `comparison.overall_pass` en C-A | `comparison.json` | TODO |
| `conformal_promotion_pass` en C-A | `comparison.json` | TODO |
| Artifacts governance + mrm presentes | `reports/*governance*`, `reports/*mrm*` | TODO |
| Decision promocion documentada | informe ejecutivo final | TODO |

---

## 9) Analisis y graficas obligatorias del reporte final

El cierre ejecutivo debe incluir:

1. Waterfall A->B, B->C, A->C por KPI clave:
   - PD: AUC, ECE, D2
   - Conformal: coverage_90/95, min_group_coverage_90, winkler_90, critical_alerts
   - Fairness: pass count
   - Survival: c-index
   - Optimizacion: retorno robusto y price of robustness
2. Tabla de gates por rama (PASS/FAIL por A/B/C).
3. Radar de madurez por subsistema:
   - PD, Conformal, Fairness, Causal, Governance, OR, IFRS9, Export contracts.
4. Panel temporal de runtime/estabilidad:
   - duracion por etapa
   - eventos de reintento y resume.

---

## 10) Uso maximo de extensiones instaladas (fase anexa)

Ejecutar despues del cierre core bloqueante:

1. RAPIDS:
   - ejecutar corrida equivalente sin `--no-rapids`.
2. Notebooks:
   - `uv run python scripts/run_all_notebooks.py`
   - `uv run python scripts/run_paper_notebook_suite.py`
   - `uv run python scripts/extract_notebook_images.py`
3. MLflow + DVC:
   - `uv run python scripts/log_mlflow_experiment_suite.py`
   - `dvc metrics show`
   - `dvc plots show` (o export de plots)

Regla:

- Core + causal + governance define go/no-go de promocion.
- RAPIDS/notebooks/extra analytics quedan como evidencia extendida.

---

## 11) Cambios en APIs/interfaces/tipos publicos

1. No se modifican endpoints FastAPI ni schemas publicos en este ciclo.
2. Se agrega este artefacto documental oficial:
   - `docs/OFFICIAL_RERUN_MASTER_PLAN_2026-02-27.md`
3. Se institucionaliza contrato de atribucion analitica:
   - comparaciones oficiales `B-A`, `C-B`, `C-A` como parte obligatoria de cierre.

---

## 12) Escenarios de prueba obligatorios

1. Atribucion correcta:
   - demostrar separacion de impacto entre `B-A`, `C-B`, `C-A`.
2. Reanudacion:
   - interrumpir corrida y retomar con `--resume` sin repetir etapas exitosas.
3. Warnings estadisticos conformal:
   - warnings presentes no bloquean promocion si checks de negocio pasan.
4. Falla opcional con `--stop-on-optional-failure`:
   - run termina con `exit code 1`.

---

## 13) Supuestos y defaults

1. Datos y splits congelados para comparabilidad.
2. Misma maquina para A/B/C.
3. Corridas secuenciales para evitar contencion de recursos.
4. Optuna puede tener estados `RUNNING` huerfanos; se controla con SQL + heartbeat.
5. Si surge blocker en data/feature engineering:
   - aplicar fix minimo documentado y volver a congelar SHAs antes de seguir.
