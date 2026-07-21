# Causal Identification Sketch for Paper 4

This note keeps CATE/causal CRPTO gated. It is not a central claim and it does
not change the Paper Estrella champion.

## Current status

- Selected rule: `discount_100_only`
- Overlap pass: `True`
- Sensitivity pass: `False`
- CATE portfolio state: `research_blocked_by_policy_gate`

## Required before promotion

1. Treatment definition: approval, funding, pricing, hardship or intervention.
2. Outcome definition: default, loss, repayment, recovery or net value.
3. Overlap and balance report.
4. Sensitivity analysis that passes predeclared thresholds.
5. Policy value estimator with uncertainty.
6. Decision link to `x_t`, `W_{t+1}(S_t, x_t)`, and `C_t`.

## Current decision

Keep causal signals as `B_t`/future-intervention hypotheses. Do not use them as
an objective or selector in the Paper 4 MVP.
