# Paper 4 Lab 4 Lanes 3, 4, 6, 7, 8 and 9 Execution Memo - 2026-05-18

## Scope

This memo executes the remaining literature-driven Lab 4 lanes over the retained
Paper 4 living-lab artifact surface. The outputs classify evidence as append,
park, or future-prototype material. No Paper Estrella champion search is reopened.

## Lane Results

| lane | decision | key result | boundary |
| --- | --- | --- | --- |
| lane3_e2e_conformal_calibration | park | No promotable E2E conformal training artifact exists in the current Lab 4 surface. | Proxy audits are not end-to-end conformal calibration. |
| lane4_online_multisource_conformal | park_with_appendix_limitation | Best bounded defended min coverage is 0.7788. | Retrospective source governance only. |
| lane6_spo_dfl_comparator | park_integrated_dfl_append_oracle_regret | Toy oracle gap remains 4329509.73. | Oracle-regret/surrogate evidence only; not integrated DFL. |
| lane7_ifrs9_proxy | append | Combined raw SICR lift is 73.6416. | IFRS9-inspired proxy only. |
| lane8_governance_fairness_proxy | append | Legal fair-lending claim allowed = False. | Proxy/source governance, not fair-lending proof. |
| lane9_causal_cate_boundary | park | Overlap share 10-90 is 0.2838. | Observational sensitivity only, not policy value. |

## Details

### Lane 3 - E2E Conformal Calibration

- Decision: `park`.
- Dependency surface: `torch=False;cvxpy=True;cvxpylayers=False;pyepo=False`.
- The existing evidence is post-hoc/source-conformal plus CROMS-lite selector audit,
  not an end-to-end learned uncertainty set.

### Lane 4 - Online / Multi-Source Conformal

- Decision: `park_with_appendix_limitation`.
- Best bounded defended min coverage: `0.7788`.
- Strict holdout pass rows: `2`.
- The lane is useful as source governance, but not as live online validity.

### Lane 6 - SPO / DFL

- Decision: `park_integrated_dfl_append_oracle_regret`.
- Toy oracle gap: `4329509.73`.
- Current dependency probe: `torch=False;cvxpylayers=False;pyepo=False`.
- Keep oracle-regret/surrogate evidence; do not claim formal differentiable SPO+.

### Lane 7 - IFRS9 Proxy

- Decision: `append`.
- Combined raw SICR lift: `73.6416`.
- Missing contractual requirements: `forbearance_hardship;cure_timing;recovery_timing;prepayment_timing;macro_scenarios`.
- Keep IFRS9-inspired ECL/SICR proxy; contractual IFRS9 remains blocked.

### Lane 8 - Governance / Fairness Proxy

- Decision: `append`.
- Legal fair-lending claim allowed: `False`.
- Top dispersion dimension: `grade`.
- Keep source/proxy governance only; do not infer protected attributes.

### Lane 9 - Causal / CATE Boundary

- Decision: `park`.
- Overlap share 10-90: `0.2838`.
- Protocol items not allowing policy value: `8.0`.
- Keep as causal boundary note; no policy-value claim.

## Stop Rules

- Do not mutate the main environment for Lane 3 or Lane 6 prototypes.
- Do not claim live online/source validity from historical replay.
- Do not claim contractual IFRS9, legal fair-lending compliance, or causal policy value.
- Export to Paper Estrella only as limitation, model-risk context, or reviewer defense.
