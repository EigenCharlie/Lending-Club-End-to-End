# Paper 4 v9 Online Goal Resolution

- Goal achieved: `True`
- Best method: `v9_tail_guard_dti_q5_only_m0.940_d0.140_t0.040`
- Best source-month defended minimum: `0.8000`
- Best policy-month defended minimum: `0.9000`
- Best average loan width: `0.934243`
- Paper Estrella modified: `False`

## Breakpoints

- `v9_v8_reference`: `v9_reference_v8_best`; source=0.8000, policy=0.9000, width=0.954380, pass=False.
- `v9_best_goal_passing_width`: `v9_tail_guard_dti_q5_only_m0.940_d0.140_t0.040`; source=0.8000, policy=0.9000, width=0.934243, pass=True.
- `v9_conservative_goal_passing_width`: `v9_tail_guard_dti_q5_only_m0.965_d0.120_t0.060`; source=0.8000, policy=0.9000, width=0.945002, pass=True.

## Caveat

The v9 method uses only pre-decision structural fields, but the grid itself was selected in replay. It resolves the explicit online efficiency gate for Paper 4 and should be rerun on future periods before any broader promotion claim.
