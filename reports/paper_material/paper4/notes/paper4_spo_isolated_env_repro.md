# Paper 4 SPO Isolated Environment Repro

Generated: 2026-05-15T15:09:47.607541+00:00

The main project environment was not mutated. Formal differentiable SPO+
remains blocked until an isolated cvxpy/cvxpylayers/torch stack validates
and beats the oracle-regret baseline.

Suggested isolated route:

```bash
python -m venv .venv-paper4-spo-v45
.venv-paper4-spo-v45/bin/python -m pip install --upgrade pip
.venv-paper4-spo-v45/bin/python -m pip install 'numpy<2' cvxpy cvxpylayers torch pyomo highspy scikit-learn catboost
.venv-paper4-spo-v45/bin/python - <<'PY'
import cvxpy, torch, cvxpylayers
print(cvxpy.__version__, torch.__version__)
PY
```

Current smoke-test results are stored in
`paper4_v46_spo_isolated_env_smoke_test.csv`.
