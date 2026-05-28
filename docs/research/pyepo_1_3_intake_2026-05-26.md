# PyEPO 1.3 Intake Memo - 2026-05-26

> **Research note** - This memo evaluates PyEPO 1.3 for the Lending Club
> CRPTO project, the Quarto book, Paper 4 and Paper Estrella. It is not a
> promotion artifact and does not modify the canonical champion.

## Bottom Line

PyEPO 1.3 is now mature enough to reopen the **SPO/DFL lane as a bounded,
isolated prototype**, especially for Paper 4. It should not replace CRPTO in
Paper Estrella. The strongest near-term contribution is a cleaner comparator
suite:

1. `SPOPlus` as the existing regret-minimizing baseline.
2. `regularizedFrankWolfeFenchelYoung` as the new low-friction differentiable
   loss for the current continuous LP-style credit selection problem.
3. `perturbedFenchelYoungMul` only for positive-cost risk/ECL variants, not for
   raw net-return costs that can be negative.
4. `coneAlignedCosine` / CaVE only if we run a **binary** approval or top-k
   selection prototype with a Gurobi-backed model.

The key update relative to the older Paper 4 v32 blocker is that we do **not**
need `cvxpy/cvxpylayers` to make a formal PyEPO prototype. PyEPO 1.3.7 works in
an isolated OR-Tools environment with PyTorch autograd modules.

## Version Check

Official surfaces checked:

| Source | Finding |
|---|---|
| PyPI | Latest package is `pyepo 1.3.7`, released 2026-05-26. Requires Python `>=3.9` and publishes typed wheels. |
| GitHub releases | 1.3 release notes are attached to the 2026-05-25 series and describe the 1.3 feature block. |
| Docs | Documentation site reports `PyEPO 1.3.7` and has a rebuilt tutorial structure. |
| Repo README | PyEPO is an official implementation of the MPC paper and now asks users to cite CaVE separately when using that loss. |

Sources:

- https://pypi.org/project/pyepo/
- https://github.com/khalil-research/PyEPO/releases
- https://khalil-research.github.io/PyEPO/build/html/index.html
- https://github.com/khalil-research/PyEPO

## What Changed In 1.3

### CaVE / CaVE+

`pyepo.func.coneAlignedCosine` implements cone-aligned vector estimation for
binary linear programs. Instead of differentiating through the solver or
sampling perturbations, it aligns predicted costs with the normal cone at the
true binary optimum. The default `max_iter=3` is the CaVE+ preset, deliberately
using a truncated Clarabel projection.

Project fit:

- Strong for a binary approval/top-k prototype.
- Weak for the current continuous/fractional CRPTO champion.
- Requires `optDatasetConstrs`, which currently needs Gurobi-backed constraint
  extraction.

Decision:

- **Paper 4**: viable only as a small Gurobi-enabled binary lane.
- **Paper Estrella**: do not import into the main claim.

### Regularized Frank-Wolfe

`regularizedFrankWolfeOpt` and `regularizedFrankWolfeFenchelYoung` smooth the LP
oracle with L2 regularization. This is the best new match for our current
credit optimizer because it only needs a linear optimization oracle and works
with our OR-Tools-style `optModel`.

Project fit:

- Good for continuous LP portfolio selection.
- Better than CaVE for the current CRPTO shape.
- Avoids the old `cvxpy/cvxpylayers` dependency blocker.
- Gives Paper 4 a more credible "formal DFL" experiment than the earlier
  oracle-regret surrogate.

Decision:

- **Promote to next experiment**: add RFYL as a third DFL comparator beside
  two-stage and SPO+.

### Multiplicative Perturbation

`perturbedOptMul` and `perturbedFenchelYoungMul` preserve cost signs by using
multiplicative noise. This matters when a solver expects nonnegative costs.

Project fit:

- Good for `PD`, `PD_high`, ECL/loss-only or positive shifted cost variants.
- Dangerous for raw `PD * LGD - int_rate`, because those costs naturally cross
  zero. Shifting costs changes optimization behavior unless carefully justified.

Decision:

- Use only in a risk-cost/ECL experiment, not as the default Paper Estrella
  regret comparator.

### CVRP Models

PyEPO 1.3 adds CVRP models across Gurobi/COPT/Pyomo backends.

Project fit:

- Not directly useful for Lending Club credit selection.
- Useful only as a didactic Quarto sidebar or future servicing/collections
  operations example.

Decision:

- Park. Do not spend project budget here.

### Performance And API Hardening

The release notes report broad 1.0-3.3x speedups for core methods and 6-14x
CaVE+ speedups versus SPO+ on TSP. They also add full public type annotations,
solution-pool refactors, expanded tests, CUDA tests and a docs overhaul.

Project fit:

- Supports upgrading the optional `spo` dependency from a loose old lower bound.
- Makes the DFL lane less fragile in reviewer-facing reproducibility language.

## Local Probe

Created a dedicated environment outside the repo:

```bash
uv venv /tmp/lending-club-pyepo-venv --python 3.12
uv pip install --python /tmp/lending-club-pyepo-venv/bin/python 'pyepo[ortools]==1.3.7'
```

Installed versions:

| Package | Version |
|---|---|
| `pyepo` | 1.3.7 |
| `torch` | 2.12.0+cu130 |
| `ortools` | 9.15.6755 |
| `numpy` | 2.4.6 |
| `pandas` | 3.0.3 |
| `clarabel` | 0.11.1 |

Smoke result on a tiny OR-Tools credit-selection LP:

| Check | Result |
|---|---|
| `optDataset` pre-solve | 48 instances solved |
| `SPOPlus` two-epoch training | loss moved from `1.403071` to `0.618665` |
| `regularizedFrankWolfeFenchelYoung` forward pass | returned finite scalar loss |
| `perturbedFenchelYoungMul` forward pass | returned finite scalar loss on positive shifted costs |

Interpretation:

- PyEPO 1.3.7 is usable in an isolated env with OR-Tools.
- We no longer need to describe the whole DFL lane as blocked by
  `cvxpy/cvxpylayers`.
- The environment is heavy because `pyepo[ortools]` pulls PyTorch and CUDA
  wheels, so it should remain optional/isolated.

## Repo Fit

Relevant local surfaces:

| File | Observation | Action |
|---|---|---|
| `pyproject.toml` | `spo = ["pyepo>=0.5"]` is too loose for the new claim. | Later change to `pyepo[ortools]>=1.3.7,<1.4` in a dedicated dependency PR. |
| `src/optimization/spo_integration.py` | Stale helper: assumes loaders yield `(features, costs)` and calls `SPOPlus(pred, costs)`. PyEPO 1.3 SPO+ expects `pred, costs, sols, objs`. | Deprecate or rewrite around `optDataset`. |
| `scripts/run_spo_real.py` | Already uses the correct `optDataset` tuple and `SPOPlus(c_hat, costs, sols, objs)`. | Make this the canonical DFL prototype entrypoint. |
| `book/chapters/14-paper-estrella/14j-spo-protocol-and-regret.qmd` | Already frames SPO+ as a regret comparator, not a conformal replacement. | Add a short PyEPO 1.3 footnote only after a real rerun. |
| `book/chapters/19-paper-mega-extension/19bu-v32-spo-environment-oracle.qmd` | Historical v32 page says formal SPO+ remains blocked if the differentiable stack is unavailable. | Add a new v39/v40 update page rather than rewriting v32 history. |

## Recommended Experiments

### Experiment A - PyEPO 1.3 SPO+ Repro Rerun

Run the existing `scripts/run_spo_real.py` inside the isolated PyEPO env with
the same `n_items=100`, `budget=30`, `epochs=50`, `seeds=5` protocol.

Gate:

- Reproduces the current 49.1% regret-improvement story within tolerance.
- Runtime is acceptable.
- Artifacts include PyEPO version, Torch version and solver backend.

Sink:

- Paper Estrella appendix and Quarto `14j`.

### Experiment B - RFYL Comparator

Add `regularizedFrankWolfeFenchelYoung` to the same sampled instances. Compare:

- two-stage Ridge,
- SPO+,
- RFYL,
- CRPTO robust costs.

Gate:

- RFYL produces finite losses and stable regret across seeds.
- It improves over two-stage or gives a useful speed/stability trade-off.

Sink:

- Paper 4 DFL lane.
- Optional Paper Estrella appendix if it clarifies why CRPTO keeps the
  auditability role.

### Experiment C - Multiplicative PFYL For Risk-Only Costs

Use positive costs such as `PD`, `PD_high`, ECL or a carefully documented
positive loss proxy. Do not use raw `PD * LGD - int_rate` unless the objective is
reframed.

Gate:

- Sign-preserving perturbations improve stability or runtime.
- The cost definition is not editorially confusing.

Sink:

- Paper 4 only.

### Experiment D - CaVE Binary Prototype

Only run if Gurobi is available. Convert the decision to a binary top-k approval
or fixed-budget funded/not-funded problem, then use `optDatasetConstrs` and
`coneAlignedCosine`.

Gate:

- Binary optimum extraction works on a small top-k sample.
- CaVE training is materially faster than SPO+.
- Regret/return comparison does not conflict with CRPTO's conformal claim.

Sink:

- Paper 4 method appendix.
- Quarto advanced DFL page.

## Claim Boundaries

Allowed:

- "PyEPO 1.3.7 enables a reproducible isolated DFL comparator stack for the
  project."
- "RFYL is now the most natural PyEPO 1.3 method for the current continuous
  credit LP prototype."
- "CaVE is promising for binary approval/top-k variants, but requires a
  Gurobi-backed binary model."

Not allowed:

- "PyEPO replaces CRPTO."
- "SPO+/RFYL/CaVE provide conformal coverage guarantees."
- "CaVE applies to the current fractional champion without reformulating the
  decision problem."
- "Multiplicative perturbation is valid for signed net-return costs without a
  documented transformation."

## Editorial Recommendation

For **Paper Estrella**, keep PyEPO as a comparator appendix. The paper's core
claim remains CRPTO: calibrated PD plus Mondrian conformal uncertainty plus a
robust portfolio policy that is auditable.

For **Paper 4**, reopen the SPO/DFL lane with a new stop rule:

- one SPO+ rerun,
- one RFYL comparator,
- optional multiplicative PFYL risk-only probe,
- optional CaVE binary probe only if Gurobi is available.

If those do not change a manuscript claim or produce a cleaner comparator
table, stop. Do not let this become another open-ended artifact wave.
