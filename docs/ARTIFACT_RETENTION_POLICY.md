# Artifact Retention Policy

Version: 2026-03-31

## Scope

This policy defines the active vs historical vs generated contract for project artifacts after the pipeline-first refactor and the Quarto-first editorial cleanup.

## Canonical Operational Baseline

- Source of truth: `configs/baselines/canonical_operational_baseline.json`
- Legacy fallback: `configs/baselines/core_official_baseline.json`
- Snapshot artifact: `reports/run_comparisons/<run_tag>/baseline_snapshot.json`

## Directory Contract

### Active surfaces

- `docs/` root: active technical and editorial references.
- `reports/` root: editorial outputs still used by Quarto, Streamlit, deploy, MRM, or technical logging.
- `reports/paper_material/`: only publication figures/tables that still feed papers, the book, or review workflows.
- `reports/notebook_images/`: only image families still consumed by Quarto, Streamlit, poster, or thesis document builders.

### Historical surfaces

- `docs/history/`: archived plans, audits, and completed transition notes.
- `docs/research/`: research-only reading, comparative notes, and literature support.
- `reports/history/`: old baselines, recompute snapshots, transition reports, recovery notes, and archived mirrors.

### Generated / disposable

- `reports/run_logs/`, `reports/run_comparisons/`, `reports/notebook_exec/`, and similar runtime-heavy directories remain generated-first and can be refreshed from the pipeline or launchers.
- Large intermediate HTML exports under paper notebook figures are not part of the editorial snapshot; keep them in history or regenerate on demand.
- Runtime checkpoint directories under `models/*_runtime_checkpoints/` are scratch observability artifacts, not canonical deliverables.
- Namespaced conformal-gap experiment directories under `data/processed/conformal_gap/*/` are research scratch unless explicitly promoted into a root summary artifact.

## Notebook Output Policy

- Notebooks are reference and analysis assets; they must not overwrite canonical pipeline outputs.
- Canonical targets protected from notebook writes:
  - `data/processed/*`
  - `models/*`
  - `reports/paper_material/*`
  - `reports/figures/*`
- Redirected notebook-generated files are stored under:
  - `reports/notebook_exec/generated/`

## Cleanup Principles

- Archive before delete when an artifact still has audit or thesis provenance value.
- Keep only editorially used tracked binaries in Git.
- Make deploy/export/MLflow scripts tolerant to missing historical snapshots.
- Do not let `research_only` or `historical_demo` assets masquerade as canonical dependencies.

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
