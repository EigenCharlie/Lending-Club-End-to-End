# Artifact Retention Policy

Version: 2026-03-13

## Scope

This policy defines what to keep vs purge for local workspace artifacts after canonical operational closure.

## Canonical Operational Baseline

- Source of truth: `configs/baselines/canonical_operational_baseline.json`
- Legacy fallback: `configs/baselines/core_official_baseline.json`
- Current official baseline run tag: `champion-2026-03-12-mega-definitive`
- Snapshot artifact: `reports/run_comparisons/<run_tag>/baseline_snapshot.json`

## Retention Profile: `core_closure_6`

### Keep in `reports/run_logs/`

1. `champion-2026-03-12-mega-definitive`
2. `2026-03-11-C-official-selector-v3-freeze`
3. `2026-03-04-C-core-balanced-cert2`
4. `2026-03-04-C-core-balanced-e2e-cert1`
5. `2026-03-04-C-core-balanced-pass2`
6. `2026-03-01-C-official-smart`
7. `2026-03-04-C-rapids-annex-pass3`
8. `2026-03-04-C-notebooks-annex-pass2`

### Keep in `reports/run_comparisons/`

1. `champion-2026-03-12-mega-definitive`
2. `2026-03-11-C-official-selector-v3-freeze`
3. `2026-03-04-C-core-balanced-cert2`
4. `2026-03-04-C-core-balanced-e2e-cert1`
5. `2026-03-04-C-core-balanced-pass2`
6. `2026-03-03-C-core-balanced-pass1`
7. `2026-03-03-C-core-balanced-ws1-manual`
8. `2026-03-01-C-official-smart`

## Legacy and Temporary Artifact Cleanup

Cleanup removes temporary artifacts such as:

- `models/ab_simulation_status_tmp_custom*.json`
- `models/ab_tmp_sweep.json`
- `data/processed/tmp_*ab*.parquet`
- legacy dual-write leftovers (`*_v2` status artifacts)

## Notebook Output Policy

- Notebooks are reference/analysis assets; they must not overwrite canonical pipeline outputs.
- Canonical targets protected from notebook writes:
  - `data/processed/*`
  - `models/*`
  - `reports/paper_material/*`
  - `reports/figures/*`
- Redirected notebook-generated files are stored under:
  - `reports/notebook_exec/generated/`

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
