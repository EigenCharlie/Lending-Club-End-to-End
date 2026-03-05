# Long Run Execution Playbook

## Block 1: Core Launch with Official Frozen Baseline

### Launch / Relaunch (recommended defaults)
```bash
bash scripts/start_long_run.sh 2026-03-05-C-core-next \
  --sampling-profile balanced \
  --no-rapids --no-notebooks \
  --stop-on-optional-failure
```

Applied by default (unless overridden):
- `--resume`
- `--sampling-profile full`
- `--env-file .env` (if present)
- `--refresh-baseline-on-resume`
- If run tag is core/official and no baseline flag is provided:
  launcher auto-resolves baseline from `configs/baselines/core_official_baseline.json`.

### Monitor (terminal view)
```bash
bash scripts/monitor_long_run.sh 2026-03-05-C-core-next
```

### Monitor (structured samples + incidents every 15 min)
```bash
uv run python scripts/monitor_pipeline_health.py \
  --run-tag 2026-03-05-C-core-next \
  --interval-seconds 900
```

Outputs:
- `reports/run_logs/<run_tag>/monitoring/health_samples.jsonl`
- `reports/run_logs/<run_tag>/monitoring/incidents.jsonl`
- `reports/run_logs/<run_tag>/monitoring/diagnostics/*.txt`

### Directed rerun by phase window
```bash
uv run python scripts/run_long_pipeline.py \
  --run-tag 2026-03-05-C-core-next \
  --resume \
  --sampling-profile balanced \
  --env-file .env \
  --from-step rapids \
  --until-step notebooks \
  --stop-on-optional-failure
```

## Block 2: Technical Quality Fixes + Targeted Rerun

### Conformal technical recalibration (group + temporal segment)
```bash
uv run python scripts/generate_conformal_intervals.py \
  --temporal_segment_floor_enabled 1 \
  --temporal_segment_freq Q \
  --temporal_segment_min_size 250
```

Then:
```bash
uv run python scripts/benchmark_conformal_variants.py
uv run python scripts/backtest_conformal_coverage.py
uv run python scripts/validate_conformal_policy.py --config configs/conformal_policy.yaml
```

### A/B robust vs non-robust with no-regression gate
```bash
uv run python scripts/simulate_ab_test.py \
  --max_candidates 10000 \
  --n_boot 3000 \
  --no_regression_tolerance_pct 0.05
```

Status contract:
- `models/ab_simulation_status.json`
- gate: `no_regression`
- significance: diagnostic only

### A/B sensitivity sweep
```bash
uv run python scripts/run_ab_sensitivity.py \
  --candidate-grid 10000,0 \
  --boot-grid 1000,3000,5000 \
  --seed-grid 42,52,62
```

### Comparison gate (conformal strict + A/B no-regression)
```bash
uv run python scripts/run_comparison.py compare \
  --run-tag 2026-03-05-C-core-next \
  --baseline reports/run_comparisons/2026-03-04-C-core-balanced-cert2/baseline_snapshot.json
```

### Freeze official baseline
```bash
uv run python scripts/freeze_core_baseline.py \
  --run-tag 2026-03-04-C-core-balanced-cert2 \
  --refresh-snapshot \
  --set-current
```

## Resilience Guards

### Non-interactive MLflow mode
- `scripts/log_mlflow_experiment_suite.py` now uses DagsHub if token exists.
- If token is missing, it falls back to local tracking URI (`reports/mlruns`) without OAuth prompts.

### Stale Optuna trial cleanup
Automatic in `main_pre`:
```bash
uv run python -u scripts/cleanup_optuna_stale_trials.py \
  --db-path models/optuna_pd_catboost.db \
  --min-age-hours 6
```

Manual dry-run:
```bash
uv run python scripts/cleanup_optuna_stale_trials.py \
  --db-path models/optuna_pd_catboost.db \
  --min-age-hours 6 \
  --dry-run
```
