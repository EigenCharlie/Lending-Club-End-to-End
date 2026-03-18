# OFFICIAL RERUN MASTER PLAN (2026-02-27)

> **HISTORICAL** — This plan was executed and completed in February-March 2026. For current state, see `SESSION_STATE.md` and `docs/backlog-papers-unified.md`.

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
- `C_SHA (experiment/conformal-toboml-integration-rerun) = c08187d9ec188fcab4958e9a09e357c29fffe44b` (commit final de hardening v1)

Conteo de divergencia:

- `main...research = 0|9`
- `research...experiment = 0|1`
- `main...experiment = 0|10`

Relacion con `experiment/overnight-full-rerun-2026-02-26`:

- Es ancestro de `research` (`ancestor=yes`)
- `overnight...research = 0|8`

Nota operativa clave:

- Corridas oficiales deben iniciarse con working tree limpio en cada rama y SHAs congelados en el reporte final.

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

## 4) Plan operativo faseado (atribucion rapida + full-data final)

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
   - no usar `--sample_size` (o usar `--full-data` cuando exista ese flag).
5. Para comparaciones rapidas A/B/C:
   - usar `--sample_size 250000` en entrenamiento PD/causal/survival.
   - desactivar HPO y fijar parametros del best trial historico para recortar tiempo.
6. Correr A/B/C de forma secuencial en la misma maquina para reducir ruido de infraestructura.

### 4.2 Fase A - Baseline oficial (`main`) con reuso de artefactos (recomendado)

Nota: en `main` no existe el stack `start_long_run.sh`/`run_long_pipeline.py`/`run_comparison.py`.
Para atribucion rapida, se reutiliza baseline ya generado en `main` y se evita rerun largo.

```bash
mkdir -p reports/run_comparisons/2026-02-27-A-main-final
cp reports/run_comparisons/2026-02-26-long-full-v3/baseline_snapshot.json \
  reports/run_comparisons/2026-02-27-A-main-final/baseline_snapshot.json
```

Salida minima esperada:

- `reports/run_comparisons/2026-02-27-A-main-final/baseline_snapshot.json`

### 4.3 Fase B - Impacto research en modo corto (`research/toboml2-integration-v1`)

```bash
git checkout research/toboml2-integration-v1
git status --short

# Build config temporal: HPO off + best trial historico (855)
cp configs/pd_model.yaml /tmp/pd_model_quick_compare.yaml
python - <<'PY'
import yaml
from pathlib import Path
p = Path('/tmp/pd_model_quick_compare.yaml')
cfg = yaml.safe_load(p.read_text(encoding='utf-8'))
cfg['hpo']['enabled'] = False
cfg['model']['params'].update({
    'bootstrap_type': 'Bernoulli',
    'learning_rate': 0.02762511818970642,
    'depth': 6,
    'l2_leaf_reg': 14.910998969314008,
    'min_data_in_leaf': 195,
    'rsm': 0.7972651915505469,
    'random_strength': 3.276099048942537e-05,
    'border_count': 148,
    'subsample': 0.8270994520471426,
})
p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
PY

uv run python scripts/train_pd_model.py --config /tmp/pd_model_quick_compare.yaml --sample_size 250000
uv run python scripts/generate_conformal_intervals.py
uv run python scripts/backtest_conformal_coverage.py
uv run python scripts/validate_conformal_policy.py
uv run python scripts/run_fairness_audit.py
uv run python scripts/run_survival_analysis.py --sample_size 250000
uv run python scripts/estimate_causal_effects.py --treatment int_rate --sample_size 250000
uv run python scripts/generate_governance_status.py --config configs/mrm_policy.yaml
uv run python scripts/generate_mrm_report.py --config configs/mrm_policy.yaml
uv run python scripts/run_comparison.py snapshot --run-tag 2026-02-27-B-research-fast
```

Salida minima esperada:

- `reports/run_comparisons/2026-02-27-B-research-fast/baseline_snapshot.json`

### 4.4 Fase C - Candidato final en modo corto (`experiment/conformal-toboml-integration-rerun`)

```bash
git checkout experiment/conformal-toboml-integration-rerun
git status --short

cp configs/pd_model.yaml /tmp/pd_model_quick_compare.yaml
python - <<'PY'
import yaml
from pathlib import Path
p = Path('/tmp/pd_model_quick_compare.yaml')
cfg = yaml.safe_load(p.read_text(encoding='utf-8'))
cfg['hpo']['enabled'] = False
cfg['model']['params'].update({
    'bootstrap_type': 'Bernoulli',
    'learning_rate': 0.02762511818970642,
    'depth': 6,
    'l2_leaf_reg': 14.910998969314008,
    'min_data_in_leaf': 195,
    'rsm': 0.7972651915505469,
    'random_strength': 3.276099048942537e-05,
    'border_count': 148,
    'subsample': 0.8270994520471426,
})
p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
PY

uv run python scripts/train_pd_model.py --config /tmp/pd_model_quick_compare.yaml --sample_size 250000
uv run python scripts/generate_conformal_intervals.py
uv run python scripts/backtest_conformal_coverage.py
uv run python scripts/validate_conformal_policy.py
uv run python scripts/run_fairness_audit.py
uv run python scripts/run_survival_analysis.py --sample_size 250000
uv run python scripts/estimate_causal_effects.py --treatment int_rate --sample_size 250000
uv run python scripts/generate_governance_status.py --config configs/mrm_policy.yaml
uv run python scripts/generate_mrm_report.py --config configs/mrm_policy.yaml
uv run python scripts/run_comparison.py snapshot --run-tag 2026-02-27-C-experiment-fast
```

Salida minima esperada:

- `reports/run_comparisons/2026-02-27-C-experiment-fast/baseline_snapshot.json`

### 4.5 Fase D - Comparaciones cruzadas obligatorias (modo corto)

```bash
# Delta(B-A): research vs main
uv run python scripts/run_comparison.py compare \
  --run-tag 2026-02-27-B-research-fast \
  --baseline reports/run_comparisons/2026-02-27-A-main-final/baseline_snapshot.json

# Delta(C-B): experiment vs research
uv run python scripts/run_comparison.py compare \
  --run-tag 2026-02-27-C-experiment-fast \
  --baseline reports/run_comparisons/2026-02-27-B-research-fast/baseline_snapshot.json

# Delta(C-A): experiment vs main
uv run python scripts/run_comparison.py compare \
  --run-tag 2026-02-27-C-experiment-fast \
  --baseline reports/run_comparisons/2026-02-27-A-main-final/baseline_snapshot.json
```

### 4.6 Fase E - Full-data oficial solo para rama ganadora (esperada: C)

Una vez validadas comparaciones rapidas y si `C` gana:

```bash
git checkout experiment/conformal-toboml-integration-rerun
bash scripts/start_long_run.sh 2026-02-28-C-official-full --stop-on-optional-failure
bash scripts/monitor_long_run.sh 2026-02-28-C-official-full
```

Opcional para cierre extendido:

```bash
bash scripts/start_long_run.sh 2026-02-28-C-official-full --resume
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

# 1) Fase A (main, reuso baseline)
mkdir -p reports/run_comparisons/2026-02-27-A-main-final
cp reports/run_comparisons/2026-02-26-long-full-v3/baseline_snapshot.json reports/run_comparisons/2026-02-27-A-main-final/baseline_snapshot.json

# 2) Fase B (research, corrida corta)
git checkout research/toboml2-integration-v1
cp configs/pd_model.yaml /tmp/pd_model_quick_compare.yaml
# aplicar override (HPO off + trial 855 params) y ejecutar stack corto
uv run python scripts/train_pd_model.py --config /tmp/pd_model_quick_compare.yaml --sample_size 250000
uv run python scripts/generate_conformal_intervals.py
uv run python scripts/backtest_conformal_coverage.py
uv run python scripts/validate_conformal_policy.py
uv run python scripts/run_fairness_audit.py
uv run python scripts/run_survival_analysis.py --sample_size 250000
uv run python scripts/estimate_causal_effects.py --treatment int_rate --sample_size 250000
uv run python scripts/generate_governance_status.py --config configs/mrm_policy.yaml
uv run python scripts/generate_mrm_report.py --config configs/mrm_policy.yaml
uv run python scripts/run_comparison.py snapshot --run-tag 2026-02-27-B-research-fast

# 3) Fase C (experiment, corrida corta)
git checkout experiment/conformal-toboml-integration-rerun
cp configs/pd_model.yaml /tmp/pd_model_quick_compare.yaml
# aplicar override (HPO off + trial 855 params) y ejecutar stack corto
uv run python scripts/train_pd_model.py --config /tmp/pd_model_quick_compare.yaml --sample_size 250000
uv run python scripts/generate_conformal_intervals.py
uv run python scripts/backtest_conformal_coverage.py
uv run python scripts/validate_conformal_policy.py
uv run python scripts/run_fairness_audit.py
uv run python scripts/run_survival_analysis.py --sample_size 250000
uv run python scripts/estimate_causal_effects.py --treatment int_rate --sample_size 250000
uv run python scripts/generate_governance_status.py --config configs/mrm_policy.yaml
uv run python scripts/generate_mrm_report.py --config configs/mrm_policy.yaml
uv run python scripts/run_comparison.py snapshot --run-tag 2026-02-27-C-experiment-fast

# 4) Comparaciones cruzadas
uv run python scripts/run_comparison.py compare --run-tag 2026-02-27-B-research-fast --baseline reports/run_comparisons/2026-02-27-A-main-final/baseline_snapshot.json
uv run python scripts/run_comparison.py compare --run-tag 2026-02-27-C-experiment-fast --baseline reports/run_comparisons/2026-02-27-B-research-fast/baseline_snapshot.json
uv run python scripts/run_comparison.py compare --run-tag 2026-02-27-C-experiment-fast --baseline reports/run_comparisons/2026-02-27-A-main-final/baseline_snapshot.json

# 5) Full-data oficial solo en rama ganadora (esperada C)
bash scripts/start_long_run.sh 2026-02-28-C-official-full --stop-on-optional-failure
bash scripts/monitor_long_run.sh 2026-02-28-C-official-full
```

### 8.2 Tabla de validacion go/no-go

| Item | Evidencia | Estado |
|---|---|---|
| A snapshot creado | `reports/run_comparisons/2026-02-27-A-main-final/baseline_snapshot.json` | TODO |
| B snapshot creado | `reports/run_comparisons/2026-02-27-B-research-fast/baseline_snapshot.json` | TODO |
| C snapshot creado | `reports/run_comparisons/2026-02-27-C-experiment-fast/baseline_snapshot.json` | TODO |
| Delta(B-A) generado | carpeta `reports/run_comparisons/2026-02-27-B-research-fast/` | TODO |
| Delta(C-B) generado | carpeta `reports/run_comparisons/2026-02-27-C-experiment-fast/` | TODO |
| Delta(C-A) generado | carpeta `reports/run_comparisons/2026-02-27-C-experiment-fast/` | TODO |
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

---

## 14) Estimacion de duracion (logs reales + estrategia actual)

### 14.1 Evidencia observada en logs

1. En `reports/run_logs/2026-02-26-long-full-v3/main_pre.log`:
   - mejor trial registrado: `Trial 855` con `value: 0.7201277317`.
   - progreso observado hasta `Trial 882`.
2. Ritmo empirico HPO:
   - ventana `805 -> 882`: `77` trials en `1230` minutos.
   - velocidad media: `~15.97 min/trial` (aprox `16 min/trial`).

### 14.2 Proyeccion de tiempo para corrida full-data

Con velocidad `~16 min/trial`:

- `900 -> 1200`: `300` trials, `~80.0 h` (`~3.33 dias`) solo HPO.
- `882 -> 1200`: `318` trials, `~84.8 h` (`~3.53 dias`) solo HPO.

Agregando etapas no-HPO (conformal/fairness/survival/causal/governance/export) + ventanas operativas:

- Core full-data sin notebooks/rapids: `~4.5 a 6.0 dias`.
- Full extendido (core + notebooks + rapids): `~6 a 8 dias`.

Esto es consistente con la expectativa operativa de una semana corrida para cierre total.

### 14.3 Estimacion para comparaciones cortas A/B/C (recomendado)

1. `A (main)` reusando snapshot existente:
   - `~10 a 20 min` (copiar baseline + verificaciones).
2. `B (research)` corrida corta sin HPO (`sample_size=250k`):
   - `~2.5 a 4.5 h`.
3. `C (experiment)` corrida corta sin HPO (`sample_size=250k`):
   - `~2.5 a 5.0 h`.
4. Comparaciones cruzadas `B-A`, `C-B`, `C-A`:
   - `~10 a 20 min`.

Total atribucion rapida A/B/C:

- `~5.5 a 10.0 h` en una jornada larga.

### 14.4 Decision operativa derivada

1. Ejecutar primero atribucion rapida A/B/C para decidir rama ganadora.
2. Correr full-data multi-dia solo en la ganadora (esperada: `C`).
3. Reservar notebooks/RAPIDS para cierre extendido, no para decidir promocion inicial.
