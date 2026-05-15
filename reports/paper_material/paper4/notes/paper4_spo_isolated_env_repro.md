# Paper 4 SPO Isolated Environment Repro

Generated: 2026-05-15T14:29:53.653821+00:00

The main environment is not mutated by v42. The safe route is an isolated
environment, because the main cvxpy/cvxcore stack may be incompatible with
the active NumPy ABI.

Suggested commands:

```bash
python -m venv .venv-spo
.venv-spo/bin/python -m pip install --upgrade pip
.venv-spo/bin/python -m pip install 'numpy<2' cvxpy cvxpylayers torch
.venv-spo/bin/python - <<'PY'
import cvxpy, torch, cvxpylayers
print(cvxpy.__version__, torch.__version__)
PY
```

- cvxpy import clean in current env: `False`
- torch import clean in current env: `False`
- cvxpylayers import clean in current env: `False`

Formal differentiable SPO+ remains a prohibited claim until the isolated
stack runs a validated optimization layer and beats the oracle-regret baseline.
