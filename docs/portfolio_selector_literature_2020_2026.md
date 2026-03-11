# Portfolio Selector Literature 2020-2026

## Purpose

This note tracks recent literature that is directly relevant to the project's
portfolio policy selection problem:

- PD point estimates are available, but portfolio decisions depend on whether
  the uncertainty-adjusted policy is economically viable.
- Conformal prediction provides loan-level uncertainty (`pd_high - pd_point`).
- The key practical problem is not computing a robust frontier, but selecting a
  portfolio policy that preserves return while reacting to uncertainty in a
  locally meaningful way.

The literature below supports three design decisions already implemented in the
codebase:

1. separate frontier generation from canonical policy selection,
2. localize uncertainty treatment instead of applying global worst-case
   penalties,
3. evaluate candidate policies using downstream economic impact on the actual
   decision universe.

## Recent papers that matter most

### 1. Localized Conformal Prediction

- Leying Guan, "Localized Conformal Prediction: A Generalized Inference
  Framework for Conformal Prediction", arXiv:2106.08460, 2021
  Link: https://arxiv.org/abs/2106.08460

Why it matters:
- The paper argues that conformal uncertainty should adapt to the local region
  of the test point rather than remain globally uniform.
- This directly motivates policy families that only penalize uncertainty where
  it is locally high instead of globally pushing every loan toward `pd_high`.

Project implication:
- Supports `tail_blended_uncertainty` and
  `segment_tail_blended_uncertainty`.
- Suggests the right direction is localized or segment-aware uncertainty, not
  uniform worst-case robustification.

### 2. Conformal Contextual Robust Optimization

- Yash Patel, Sahana Rayan, Ambuj Tewari, "Conformal Contextual Robust
  Optimization", arXiv:2310.10003, 2023
  Link: https://arxiv.org/abs/2310.10003

Why it matters:
- The paper studies predict-then-optimize with context-dependent conformal
  uncertainty regions.
- The key lesson is that contextual uncertainty sets can be less conservative
  than global uncertainty sets while retaining distribution-free guarantees.

Project implication:
- Supports policy families that use loan context (`grade`, `term`, potentially
  `verification_status`) to determine which part of the uncertainty should
  matter for optimization.
- Supports the architectural change that policy selection must be downstream of
  uncertainty, not a fixed ex ante ranking detached from the actual decision
  universe.

### 3. Conformal Robustness Control

- "Conformal Robustness Control: Uncertainty Sets for Robust Optimization from
  Contextual Data", arXiv preprint, 2026
  Search entry used during design:
  https://arxiv.org/search/?query=Conformal+Robustness+Control&searchtype=all

Why it matters:
- The framing is almost identical to our problem: use conformalized uncertainty
  to define uncertainty sets for robust optimization, but control the amount of
  robustness.

Project implication:
- Supports explicit policy families where robustness is a tunable design choice
  (`gamma`, caps, tails, contextual segments), not a binary switch.
- Supports maintaining a canonical fallback to non-robust decisions when no
  robust policy survives the economic constraints.

### 4. Optimal Model Selection for Conformalized Robust Optimization

- "Optimal Model Selection for Conformalized Robust Optimization", arXiv
  preprint, 2025
  Search entry used during design:
  https://arxiv.org/search/?query=Optimal+Model+Selection+for+Conformalized+Robust+Optimization&searchtype=all

Why it matters:
- The core message is that predictive models should be selected by downstream
  optimization quality, not only by predictive metrics.

Project implication:
- Supports the shift already made in the project:
  - `tradeoff` produces candidates,
  - `select_economic_portfolio_policy.py` chooses the canonical policy on the
    actual A/B universe,
  - `simulate_ab_test.py` becomes audit-only in the canonical path.
- This is directly aligned with our `economic_actual_ab_v1` selector.

### 5. Group-Weighted Conformal Prediction

- Aabesh Bhattacharyya, Rina Foygel Barber, "Group-Weighted Conformal
  Prediction", arXiv:2401.17452, 2024
  Link: https://arxiv.org/abs/2401.17452

Why it matters:
- Weighted conformal methods are useful when exchangeability or sampling
  assumptions are not fully homogeneous across groups.
- Group-weighted conformal gives a practical way to improve uncertainty quality
  under group-driven shift.

Project implication:
- Suggests future work where conformal intervals are weighted or localized by
  group (`grade`, `term`, `verification_status`) before feeding robust
  optimization.
- This is especially relevant because the portfolio selector is already moving
  toward group/segment-aware robustification.

### 6. Conformal Predictive Systems under Covariate Shift

- Jef Jonkers, Glenn Van Wallendael, Luc Duchateau, Sofie Van Hoecke,
  "Conformal Predictive Systems Under Covariate Shift", arXiv:2404.15018, 2024
  Link: https://arxiv.org/abs/2404.15018

Why it matters:
- This work extends conformal predictive systems to covariate shift using
  weighting.
- While the Lending Club dataset is frozen, this still matters for our
  train/cal/test mismatch and for the interpretation of uncertainty in specific
  subpopulations.

Project implication:
- Supports future work where conformal widths are not treated as equally
  reliable across the entire test population.
- Motivates weighting or local calibration instead of purely global conformal
  widths in the policy selector.

## Related domain papers

### 7. Probability of default for lifetime credit loss for IFRS 9 using machine learning competing risks survival analysis models

- Expert Systems with Applications, 2024
  Link: https://www.sciencedirect.com/science/article/pii/S095741742400472X

Why it matters:
- Shows that lifetime PD modeling for IFRS 9 benefits from survival-type
  approaches rather than static one-period classification alone.

Project implication:
- Reinforces the current project architecture where PD, survival, IFRS9 and
  uncertainty should be connected rather than treated as isolated modules.
- Suggests that future robust policies could potentially use lifetime-tail risk
  signals instead of only one-period conformal width.

### 8. Approaches for modelling the term-structure of default risk under IFRS 9: A tutorial using discrete-time survival analysis

- arXiv:2507.15441, 2025 / journal tutorial in 2026
  Link: https://arxiv.org/abs/2507.15441

Why it matters:
- Reviews term-structure modeling for lifetime PD under IFRS 9.

Project implication:
- Supports keeping the canonical pipeline's survival/lifetime PD path and using
  uncertainty-aware portfolio policies as a separate decision layer rather than
  collapsing everything into a single static PD score.

### 9. Stabilising Lifetime PD Models under Forecast Uncertainty

- arXiv:2509.10586, 2025
  Link: https://arxiv.org/abs/2509.10586

Why it matters:
- Explicitly studies uncertainty in lifetime PD forecasting and stabilization.

Project implication:
- Supports the broader project thesis that uncertainty should be propagated
  into decision-making layers, not merely reported in dashboards.

## What the literature suggests for this project

### A. The current global robust policies are too coarse

Empirical result in the project:
- robust policies with `gamma > 0` often destroy return in A/B on the real
  decision universe.

Literature-backed interpretation:
- global worst-case or globally blended conformal penalties are too
  conservative for portfolio selection.
- localized/contextual uncertainty is the correct direction.

Implemented project response:
- `tail_blended_uncertainty`
- `segment_tail_blended_uncertainty`

### B. Canonical policy selection must be decision-focused

Empirical result in the project:
- selecting a champion directly from the frontier and auditing afterwards was
  structurally wrong.

Literature-backed interpretation:
- downstream decision quality should drive model/policy choice.

Implemented project response:
- `select_economic_portfolio_policy.py`
- `economic_actual_ab_v1`
- `simulate_ab_test.py` in canonical mode as audit-only

### C. If no robust policy survives, the system should say so explicitly

Empirical result in the project:
- several runs end in `fallback_nonrobust`.

Literature-backed interpretation:
- that is not a failure of robust optimization; it is a valid finding that the
  uncertainty treatment is not yet economically viable.

Implemented project response:
- canonical policies now record `selection_outcome`
- non-robust fallback is explicit rather than disguised as a robust champion

## Next research directions supported by the literature

1. Weighted or group-weighted conformal intervals for the portfolio selector
   path.
2. Segment-aware robustification beyond `grade x term`, e.g.
   `grade x term x verification_status`.
3. Decision-focused uncertainty features such as:
   - absolute conformal width,
   - relative width,
   - local width percentile within segment,
   - width combined with expected return floor.
4. Offline research grids, but fixed canonical policy in the heavy main once
   the dataset is considered static and the champion is frozen.
