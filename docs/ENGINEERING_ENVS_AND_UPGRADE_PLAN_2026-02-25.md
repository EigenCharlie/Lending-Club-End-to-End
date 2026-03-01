# Engineering Environments and Upgrade Plan (2026-02-25)

## 1) Environment split (main vs RAPIDS)

### Main project environment (`lending-club-venv`, managed with `uv`)
- Scope: core pipeline, DVC, tests, Streamlit, API, dbt optional extra.
- Python target: `3.12` (project supports `>=3.11,<3.13`; current default `.python-version`).
- Must stay free of RAPIDS/CUDA packages to avoid resolver and `uv pip check` conflicts.
- VS Code + terminal use `UV_PROJECT_ENVIRONMENT=lending-club-venv`.
- Compatibility symlink kept: `.venv -> lending-club-venv` (avoids breaking legacy scripts and `uv` defaults).

### GPU side-project environment (`conda` env `rapids`)
- Scope: RAPIDS benchmarks, GPU experiments, `notebooks/side_projects/10_rapids_gpu_benchmark_lending_club.ipynb`, `reports/gpu_benchmark/tmp_scripts/*`.
- Confirmed env name: `rapids`.
- Keep CUDA/RAPIDS packages here only (`cudf`, `cuml`, `cugraph`, `cuopt`, `cupy`, related `*-cu12` libs).

### Feast environment (`.venv-feast`, managed with `uv`)
- Scope: feature store registry/apply workflows under `feature_repo/`.
- Kept outside the main `pyproject.toml` because `feast` pins `uvicorn<=0.34.0`, which blocks API upgrades in the main lock.
- Install from `requirements/feast-platform.txt`.

### Causal EconML environment (`.venv-causal`, managed with `uv`)
- Scope: EconML-backed causal estimators (`CausalForestDML`) used by `src/models/causal.py`.
- Kept outside the main `pyproject.toml` because `econml==0.16.0` pins `scikit-learn<1.7` and `shap<0.49`.
- Install from `requirements/causal-econml.txt`.
- This is a task-specific overlay env (not a general replacement for `lending-club-venv`).

### Recommended command pattern for GPU side-projects
```bash
conda run -n rapids python reports/gpu_benchmark/tmp_scripts/run_all_benchmarks.py
conda run -n rapids jupyter lab
```

Wrapper script added:
- `scripts/side_projects/run_rapids_benchmarks.sh`

## 2) Upgrade policy (safe-first)

### Baseline decision
- Promote **Python 3.12** as the default project runtime (completed).
- Do **not** upgrade the main project to Python 3.14 yet.
- Python **3.12 pilot passed** in `lending-club-venv-py312` and the **main env was promoted** to 3.12 after validation.

### High-value, lower-risk package updates (within current constraints)
- `catboost`: `1.2.8 -> 1.2.10`
- `fastapi`: `0.128.3 -> 0.133.1`
- `uvicorn`: `0.34.0 -> 0.41.0` (completed after isolating `feast` from the main lock)
- `highspy`: `1.13.0 -> 1.13.1`
- `pyomo`: `6.9.5 -> 6.10.0`
- `dagshub`: `0.6.5 -> 0.6.8`
- `ruff`: `0.15.0 -> 0.15.2`
- `nbstripout`: `0.9.0 -> 0.9.1`

### Medium-risk updates (run full validation)
- Completed in main env:
  - `pyarrow`: `22.0.0 -> 23.0.1`
  - `scikit-survival`: `0.26.0 -> 0.27.0` (after isolating `econml`)
  - `shap`: `0.48.0 -> 0.50.0` (after isolating `econml`)
  - `scikit-learn`: `1.6.1 -> 1.8.0` (after isolating `econml`)
  - `cvxpy`: `1.7.5 -> 1.8.1`
  - `dowhy`: `0.12 -> 0.14`
  - `mlflow`: `3.9.0 -> 3.10.0`
- Constraint note:
  - `econml 0.16.0` remains available in `.venv-causal`; its metadata pins `scikit-learn<1.7` and `shap<0.49`, so it should not live in the main lock.

### Deferred / separate migration
- `pandas 3.x`: blocked by current constraint (`pandas>=2.2,<3`) and should be a dedicated migration.

## 3) Validation matrix for upgrades

Run after each batch:
```bash
uv sync --python 3.12 --extra dev --extra platform
uv run pytest -q
uv run ruff check src/ scripts/ tests/
uv run python -m pytest -q tests/test_streamlit/test_page_imports.py
uv run python scripts/generate_mrm_report.py --config configs/mrm_policy.yaml
```

Named-env tip (`uv` gotcha):
- If you use a named env with a different interpreter than `.python-version`, run tests with that env's interpreter
  (e.g. `lending-club-venv-py312/bin/python -m pytest -q`) or pass `--python` explicitly to `uv`.
- `uv run` alone follows the project's `.python-version` and may recreate the named env with that interpreter.

Feast validation (separate env):
```bash
uv venv .venv-feast
uv pip install --python .venv-feast/bin/python -r requirements/feast-platform.txt
cd feature_repo && ../.venv-feast/bin/feast apply
```

EconML validation (separate env):
```bash
bash scripts/causal/setup_causal_env.sh .venv-causal
./.venv-causal/bin/python scripts/estimate_causal_effects.py
```

Optional regression checks:
- `uv run python scripts/export_streamlit_artifacts.py`
- `uv run python scripts/export_storytelling_snapshot.py`
- `uv run dvc repro export_dvc_metrics`

## 4) Notes on modern features worth adopting

- Python 3.12 promoted: runtime/perf improvements and better ergonomics without leaving the supported range.
- `scikit-learn 1.8`: `d2_brier_score` is now available for probability-model evaluation.
- `pandas 3.x`: Copy-on-Write by default is attractive, but defer until a compatibility pass is planned.
