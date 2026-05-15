# Paper 4 Causal Identification Dossier

This dossier expands the causal lane for Paper 4 without promoting CATE into the
main objective.

## Current gate

- Causal rule: `discount_100_only`
- Rule promotion state: `validated_research_policy`
- Overlap pass: `True`
- Sensitivity pass: `False`
- CATE portfolio state: `research_blocked_by_policy_gate`

## Toy policy value

`paper4_cate_policy_value_toy.csv` computes a diagnostic loss-reduction proxy:

```text
toy_value = -min(CATE, 0) * LGD * funded_exposure
```

This reads negative CATE as a reduction in default probability. It is a toy
calculation, not a causal claim. It can become a policy objective only after a
clean treatment, outcome, overlap report, sensitivity report and policy-value
estimator are accepted.

## Decision

Keep CATE as `B_t`/future-intervention hypothesis. Do not use it as `C_t` or
`X^pi` in the Paper 4 selector yet.
