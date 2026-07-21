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

## v55-v62 update

Generated: 2026-05-15T16:45:00+00:00

The v57 dependency probe rechecked the main repo environment and kept the
formal differentiable SPO+ claim blocked. The current environment can use
`scipy`, `highspy`, `pyomo`, `catboost`, and `sklearn`, but `cvxpy` fails under
the current NumPy 2.x ABI, while `cvxpylayers` and `torch` are not installed.

Current evidence:

- `reports/paper_material/paper4/tables/paper4_v57_spo_dependency_probe.csv`
- `reports/paper_material/paper4/tables/paper4_v57_spo_oracle_regret_bridge.csv`

Decision:

- Do not mutate the main `.venv` for differentiable SPO.
- Keep using solver-oracle/regret bridges in the main environment.
- Open a separate environment only when we are ready to spend the disk/runtime
  budget on a Torch stack.
- A future isolated environment should pin `numpy<2` until the local `cvxpy`
  wheel stack is compatible with NumPy 2.x.

Claim boundary:

Paper 4 may claim an oracle-regret/SPO-style diagnostic path, but not formal
differentiable SPO+ training until a separate environment validates
`cvxpy + cvxpylayers + torch` and the differentiable candidate beats the
solver-oracle baseline under temporal splits.
