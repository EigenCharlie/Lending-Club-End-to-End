# Artifact Retention Policy

Version: 2026-03-05

## Scope

This policy defines what to keep vs purge for local workspace artifacts after official core closure.

## Official Core Baseline

- Source of truth: `configs/baselines/core_official_baseline.json`
- Current official baseline run tag: `2026-03-04-C-core-balanced-cert2`
- Snapshot artifact: `reports/run_comparisons/<run_tag>/baseline_snapshot.json`

## Retention Profile: `core_closure_6`

### Keep in `reports/run_logs/`

1. `2026-03-04-C-core-balanced-cert2`
2. `2026-03-04-C-core-balanced-e2e-cert1`
3. `2026-03-04-C-core-balanced-pass2`
4. `2026-03-01-C-official-smart`
5. `2026-03-04-C-rapids-annex-pass3`
6. `2026-03-04-C-notebooks-annex-pass2`

### Keep in `reports/run_comparisons/`

1. `2026-03-04-C-core-balanced-cert2`
2. `2026-03-04-C-core-balanced-e2e-cert1`
3. `2026-03-04-C-core-balanced-pass2`
4. `2026-03-03-C-core-balanced-pass1`
5. `2026-03-03-C-core-balanced-ws1-manual`
6. `2026-03-01-C-official-smart`

## Legacy and Temporary Artifact Cleanup

Cleanup removes temporary artifacts such as:

- `models/ab_simulation_status_tmp_custom*.json`
- `models/ab_tmp_sweep.json`
- `data/processed/tmp_*ab*.parquet`
- legacy dual-write leftovers (`*_v2` status artifacts)

## Execution

Dry-run:

```bash
uv run python scripts/cleanup_workspace_artifacts.py --retention-profile core_closure_6
```

Apply:

```bash
uv run python scripts/cleanup_workspace_artifacts.py \
  --retention-profile core_closure_6 \
  --apply
```

Apply + backup/purge local MLflow cache:

```bash
uv run python scripts/cleanup_workspace_artifacts.py \
  --retention-profile core_closure_6 \
  --apply \
  --purge-mlruns-local
```
