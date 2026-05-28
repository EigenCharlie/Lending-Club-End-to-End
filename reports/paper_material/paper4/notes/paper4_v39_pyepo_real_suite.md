# Paper 4 v39 - PyEPO 1.3.7 Real Suite Results

Date: 2026-05-28

## Canonical runner

`scripts/run_pyepo_real_suite.py` is the canonical entrypoint for the formal
PyEPO lane. It pins the executable protocol to `pyepo[ortools]==1.3.7` plus
`gurobipy>=13.0,<14` and validates `pyepo.__version__ == "1.3.7"` before
training unless explicitly overridden for exploratory work.

The final implementation uses two solver backends:

- Standard SPO+, RFYL, PFYL-Mul and pairwise LTR use
  `exact_topk_numpy_lexicographic`, because the credit portfolio subproblem is a
  fixed-budget top-k minimization. This removes the OR-Tools GLOP numerical
  fallback observed in the first PFYL-Mul attempt.
- CaVE uses `gurobi_binary_optDatasetConstrs`, because PyEPO's
  `coneAlignedCosine` requires `optDatasetConstrs` and tight-constraint normals
  from a Gurobi model.

The Gurobi WLS license is active through `/home/eigenlinux/gurobi.lic`, and the
official runs recorded `gurobipy==13.0.2` and academic WLS license `2828337`.

## Gates executed

| gate | run tag | command | result |
| --- | --- | --- | --- |
| Gate 0 | `pyepo137_wls_topk_smoke_20260528` | `uv run --extra spo python scripts/run_pyepo_real_suite.py --mode smoke --torch-num-threads 4` | Passed with SPO+, RFYL, PFYL-Mul, LTR and CaVE |
| Gate 1 | `paper_estrella_pyepo137_wls_topk_paired_20260528` | `uv run --extra spo python scripts/run_pyepo_real_suite.py --mode paired --torch-num-threads 4` | Passed; SPO+ improvement vs two-stage = 48.51% |
| Gate 2 | `paper4_pyepo137_wls_topk_full_20260528` | `uv run --extra spo python scripts/run_pyepo_real_suite.py --mode paper4_full --torch-num-threads 4` | Passed; 10 seeds, 500 test instances |
| Gate 3 | `paper4_pyepo137_wls_topk_temporal_20260528` | `uv run --extra spo python scripts/run_pyepo_real_suite.py --mode temporal --torch-num-threads 4` | Passed; 5 OOT periods |

Additional checks:

- `uv run --extra spo pytest tests/test_scripts/test_run_pyepo_real_suite.py -q`: 9 passed.
- `uv run ruff check scripts/run_pyepo_real_suite.py tests/test_scripts/test_run_pyepo_real_suite.py`: passed.
- All final runs have `nonnegative_regret_tolerance=true` and
  `pyepo_version_ok=true`.

## Paper Estrella paired result

Configuration: `n_items=100`, `budget=30`, `n_train=800`, `n_test=200`,
`epochs=50`, `seeds=5`, historical 10-feature set.

| method | mean regret | std regret | improvement vs two-stage | Wilcoxon p-value | observations |
| --- | ---: | ---: | ---: | ---: | ---: |
| SPO+ | 0.184366 | 0.061471 | 48.51% | 3.80e-163 | 1,000 |
| Two-stage Ridge | 0.358073 | 0.102747 | 0.00% | n/a | 1,000 |
| CRPTO robust | 0.917909 | 0.193864 | -156.35% | 1.00 | 1,000 |

Use in Paper Estrella: update only the SPO+ reproducibility appendix. The main
claim remains unchanged: SPO+ minimizes decision regret; CRPTO remains the
coverage/auditability method.

Archived artifacts:

`reports/paper_material/paper_estrella/pyepo/paper_estrella_pyepo137_wls_topk_paired_20260528/`

## Paper 4 full result

Configuration: `n_items=100`, `budget=30`, `n_train=2000`, `n_test=500`,
`epochs=75`, `seeds=10`, full 15-feature set.

| method | mean regret | std regret | median regret | improvement vs two-stage | observations |
| --- | ---: | ---: | ---: | ---: | ---: |
| SPO+ | 0.122379 | 0.047146 | 0.118561 | 57.66% | 5,000 |
| RFYL | 0.125405 | 0.048365 | 0.120128 | 56.61% | 5,000 |
| CaVE | 0.128109 | 0.048809 | 0.122989 | 55.68% | 5,000 |
| Pairwise LTR | 0.224244 | 0.075218 | 0.217076 | 22.42% | 5,000 |
| Two-stage Ridge | 0.289031 | 0.084595 | 0.281715 | 0.00% | 5,000 |
| PFYL-Mul | 0.729797 | 0.156013 | 0.720062 | -152.50% | 5,000 |
| CRPTO robust | 0.910585 | 0.194092 | 0.906895 | -215.05% | 5,000 |

Use in Paper 4: this is the first formal PyEPO DFL suite in the project. The
results support a modern methods lane: SPO+, RFYL and CaVE form the low-regret
frontier; LTR is a useful middle benchmark; PFYL-Mul is a negative result under
the current positive-shifted fixed-budget formulation.

Archived artifacts:

`reports/paper_material/paper4/status/paper4_pyepo137_wls_topk_full_20260528/`

## Temporal OOT result

Configuration: `n_items=50`, `budget=15`, `n_train=1600`, `n_test=80` per
period, `epochs=75`, `seeds=10`, full 15-feature set.

| method | mean regret | std regret | median regret | improvement vs two-stage | observations |
| --- | ---: | ---: | ---: | ---: | ---: |
| SPO+ | 0.061835 | 0.033803 | 0.058122 | 61.07% | 4,000 |
| RFYL | 0.071448 | 0.038911 | 0.065369 | 55.02% | 4,000 |
| CaVE | 0.072284 | 0.040168 | 0.066383 | 54.49% | 4,000 |
| Pairwise LTR | 0.101961 | 0.050198 | 0.094573 | 35.81% | 4,000 |
| Two-stage Ridge | 0.158845 | 0.069056 | 0.149510 | 0.00% | 4,000 |
| PFYL-Mul | 0.388130 | 0.125369 | 0.374109 | -144.35% | 4,000 |
| CRPTO robust | 0.515646 | 0.174696 | 0.509262 | -224.62% | 4,000 |

Mean regret by period:

| period | SPO+ | RFYL | CaVE | LTR | two-stage | PFYL-Mul | CRPTO robust |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2018H1 | 0.063664 | 0.065835 | 0.065966 | 0.095797 | 0.137667 | 0.322939 | 0.348838 |
| 2018H2 | 0.060940 | 0.062723 | 0.064424 | 0.091375 | 0.143135 | 0.348927 | 0.460519 |
| 2019H1 | 0.063649 | 0.069624 | 0.069244 | 0.100438 | 0.140951 | 0.369469 | 0.516106 |
| 2019H2 | 0.060691 | 0.072268 | 0.074457 | 0.107570 | 0.161256 | 0.418176 | 0.584341 |
| 2020 | 0.060233 | 0.086788 | 0.087332 | 0.114625 | 0.211215 | 0.481140 | 0.668429 |

Coverage diagnostics from the same temporal status:

| period | loans | default rate | coverage 90 | coverage 95 | mean width 90 | min grade coverage 90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2018H1 | 110,839 | 24.46% | 92.41% | 95.67% | 0.7736 | 91.23% |
| 2018H2 | 86,339 | 23.30% | 92.52% | 96.20% | 0.7630 | 88.57% |
| 2019H1 | 50,245 | 20.77% | 92.90% | 96.54% | 0.7552 | 81.25% |
| 2019H2 | 25,160 | 12.44% | 95.63% | 97.64% | 0.7548 | 94.10% |
| 2020 | 4,286 | 1.35% | 99.25% | 99.70% | 0.7025 | 98.87% |

Use in Paper 4: 2020 is the stress OOT period. SPO+ keeps the best temporal
mean regret, while RFYL and CaVE degrade more visibly in 2020. CRPTO is not a
regret winner, but its conformal coverage remains auditable and above target.

Archived artifacts:

`reports/paper_material/paper4/status/paper4_pyepo137_wls_topk_temporal_20260528/`

## Static tables and figures

Static tables for docs and Quarto:

- `reports/paper_material/paper4/tables/pyepo_real_suite_execution_ledger_20260528.csv`
- `reports/paper_material/paper4/tables/pyepo_real_suite_summary_full_20260528.csv`
- `reports/paper_material/paper4/tables/pyepo_real_suite_summary_temporal_20260528.csv`
- `reports/paper_material/paper4/tables/pyepo_real_suite_temporal_by_period_20260528.csv`

Figures with explicit names:

- `reports/paper_material/figures_publication/pyepo_real_suite_full_regret.{pdf,png}`
- `reports/paper_material/figures_publication/pyepo_real_suite_full_regret_auditability.{pdf,png}`
- `reports/paper_material/figures_publication/pyepo_real_suite_temporal_regret.{pdf,png}`
- `reports/paper_material/figures_publication/pyepo_real_suite_temporal_regret_auditability.{pdf,png}`
- `reports/paper_material/figures_publication/pyepo_real_suite_temporal_stability.{pdf,png}`

## Claim boundary

Paper Estrella should cite only the paired SPO+ reproduction and keep CRPTO as
the conformal robust contribution.

Paper 4 can now replace the earlier "oracle-regret only" language for the
trained methods with "formal PyEPO DFL suite". CRPTO remains outside DFL and is
reported as an auditable robust comparator rather than a low-regret learner.
