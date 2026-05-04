# Chapter 3 Memo: Classification Conformal Ideas To Port Into This Repo

Date: 2026-04-03

Source reviewed: local PDF `/home/eigenlinux/documentos/Applied_Conformal_Prediction (2).pdf`, Chapter 3, especially Sections 3.2-3.5 (pp. 63-87 in the PDF pagination shown by the text extract).

## Bottom line

For this repo, Chapter 3 does not point to a radical replacement of the current PD interval stack. It points to three concrete search expansions:

1. Search over calibrated probability spaces, not only over Mondrian partition and residual scaling.
2. Add explicit binary classification-set score families to the research sidecar search, especially hinge/LAC vs margin.
3. Treat Venn-Abers as a first-class conformal search dimension, not just as an upstream calibration artifact.

The highest-value idea for binary credit-risk PD is still:

- probability-first decisions,
- Venn-Abers or similarly strong calibration first,
- conformal as a guardrail around that calibrated score,
- Mondrian/group-aware calibration where subgroup reliability matters.

## Repo observations that matter

Current repo behavior is narrower than the chapter's design space:

- `scripts/generate_conformal_intervals.py` searches only over `alpha`, partition, `scaled_scores`, and `min_group_size`.
- `scripts/benchmark_conformal_variants.py` benchmarks interval variants, but not nonconformity-score families for classification sets.
- `scripts/benchmark_pd_set_prediction.py` benchmarks only one MAPIE classification-set method (`lac` by default).
- `src/models/conformal.py` already contains:
  - interval construction,
  - binary classification sets via MAPIE,
  - Mondrian partition builders,
  - a Venn-Abers interval helper.
- Score-band Mondrian partitions are currently built from raw `predict_proba` outputs, not calibrated probabilities.
- The binary set benchmark currently does not consume the chosen probability calibrator, even though Chapter 3 repeatedly recommends "calibrate probabilities first, then conformalize."

That last point is the biggest implementation mismatch with Chapter 3.

## 1. Best ideas to test in our conformal search

### A. Calibrator-aware conformal search

Why:

- Chapter 3 makes calibration upstream of conformal efficiency, especially for hinge, margin, APS/RAPS, and conditional-coverage behavior.
- Better calibrated probabilities should improve set efficiency and reduce subgroup coverage gaps even when marginal coverage is already valid.

What to test:

- `venn_abers` vs `isotonic` vs `platt` vs `beta` as the probability source feeding conformal.
- For interval search, compare the same Mondrian config under different calibrators.
- For binary set search, compare singleton rate, ambiguity rate, and grade-level coverage under each calibrator.

### B. Partition score space as a search dimension

Why:

- The repo's score-decile and grade x score-band Mondrian partitions currently use raw model scores.
- Chapter 3 recommends calibrated probabilities as the working object for tabular classification.

What to test:

- partition on raw probability,
- partition on calibrated probability,
- partition on logit(raw probability),
- partition on logit(calibrated probability).

Expected value:

- likely better subgroup alignment than raw-score bins,
- likely better match between partitioning and the actual score used downstream for uncertainty.

### C. Binary set nonconformity search: hinge/LAC vs margin

Why:

- Chapter 3's cleanest practical comparison for classification is hinge vs margin.
- For binary credit decisions, this is directly useful for abstention/triage around the approval threshold.

What to test:

- current `lac` baseline,
- custom hinge implementation,
- custom margin implementation,
- optionally calibrated-threshold versions around the economic decision threshold.

Metrics:

- set coverage,
- singleton rate,
- ambiguity rate,
- empty-set rate,
- default rate among ambiguous cases,
- group coverage by `grade`,
- temporal stability.

### D. Mondrian/fair coverage for classification sets, not only intervals

Why:

- Chapter 3 explicitly calls out conditional, group-conditional, and fair coverage as the right extension for lending-like settings.
- The repo already applies Mondrian thinking to intervals, but not to binary set search.

What to test:

- global binary sets,
- `grade`-Mondrian binary sets,
- `score_decile_mondrian` binary sets,
- `grade_x_scoreband_mondrian` binary sets,
- optional fairness slices if minimum support is satisfied.

### E. Venn-Abers width as an uncertainty side signal

Why:

- Chapter 3's Venn-Abers section makes the interval `(p0, p1)` itself an uncertainty object, not just a way to get a midpoint probability.
- In this repo, Venn-Abers width may be a useful side feature for triage, deferral, or conformal partitioning.

What to test:

- compare Venn width against conformal set ambiguity,
- use Venn width deciles as a candidate Mondrian partition,
- use Venn width as a routing feature for human-review or policy override sidecars.

## 2. Highest-value ideas for binary credit-risk PD intervals/sets

### Highest value

#### 1. Calibrated hinge-style binary sets with Mondrian partitions

This is the strongest near-term fit.

Why:

- Binary credit risk is probability-first, not label-list-first.
- Hinge has direct threshold semantics on calibrated `P(default)`.
- The repo already works with calibrated PDs and Mondrian partitions.
- It aligns with the repo's existing abstention/triage sidecar much better than multiclass APS/RAPS-style machinery.

Expected outcome:

- cleaner singleton/ambiguity behavior,
- easier explanation to model risk and governance stakeholders,
- better grade-level reliability than a global binary set benchmark.

#### 2. Venn-Abers as the outer calibration choice for conformal

This is high value because the repo already supports it and the book's logic is explicit: better calibration improves conformal usefulness.

Expected outcome:

- better efficiency for the same coverage,
- better subgroup stability,
- better probability semantics for cost-sensitive lending decisions.

### Medium value

#### 3. Binary margin sets for abstention near the policy boundary

Useful if the business question is not only "is coverage valid?" but "where should we defer?"

Why:

- Margin focuses on closeness to the competing class.
- In binary PD, that means it concentrates ambiguity near the decision boundary.

Best role:

- research sidecar,
- approval-review deferral policy,
- analyst escalation.

Less likely role:

- primary replacement for the current interval stack.

### Lower value for the current binary repo

#### Brier/proper-scoring-rule nonconformity as a separate binary search family

Low priority because in binary classification the Brier NCM collapses to a monotone transform of hinge:

- for class 1, `s_brier = (1 - p)^2`,
- for class 0, `s_brier = p^2`.

That means it is unlikely to produce meaningfully different conformal sets from hinge after quantile calibration.

#### Rank-only top-k

Very low value in binary:

- either top-1 is enough, or
- the method collapses to `{0,1}` very often.

This is too coarse for PD work.

#### APS / RAPS / SAPS as a main binary focus

Low value right now.

Reason:

- their main advantage is richer instance-adaptive set sizing in `K > 2` problems.
- in binary PD they mostly reduce to threshold shifts over singleton vs `{0,1}` behavior.

These become much more interesting only if the repo moves to:

- multiclass delinquency state prediction,
- multi-stage credit status prediction,
- richer action labels beyond default/non-default.

#### Exotic model-specific scores

Examples like embedding-distance NCMs are not a natural next move for CatBoost tabular PD.

If a model-specific score is tested here, the best candidates are simpler:

- raw-logit space,
- calibrated-logit space,
- Venn-width-conditioned partitions,
- score-conditioned Mondrian routing.

## 3. Code and search-space changes implied

### `src/models/conformal.py`

Add explicit binary classification-set builders that do not rely only on MAPIE's built-in score names.

Suggested additions:

- `compute_binary_nonconformity_scores(...)`
- `create_binary_classification_sets_custom(...)`
- `create_binary_classification_sets_mondrian(...)`

Searchable score families:

- `hinge`
- `margin`
- `brier` as a documented sanity-check / expected-near-duplicate of hinge
- `rank_topk` as a negative-control baseline

Also add an option to apply the repo calibrator before set construction.

### `scripts/benchmark_pd_set_prediction.py`

Expand from one-method benchmarking to a small search runner.

Suggested new CLI dimensions:

- `--score_families lac,hinge,margin`
- `--partition_candidates global,grade,score_decile_mondrian,grade_x_scoreband_mondrian`
- `--partition_score_space raw_prob,calibrated_prob,raw_logit,calibrated_logit`
- `--calibrator_override_path ...`

Suggested new outputs:

- by-variant benchmark table,
- by-group conditional coverage table,
- ambiguity concentration near decision-threshold slices,
- fairness slice table when configured.

### `scripts/generate_conformal_intervals.py`

Keep the current interval pipeline, but extend the search space with score-space awareness.

Concrete additions:

- `partition_score_space`
- optionally `residual_score_space` (`prob`, `logit`)
- optionally `calibrator_family` or `calibrator_tag`

Important correction:

- if the final interval is built around calibrated probabilities, the score-band partition should be allowed to use those same calibrated probabilities.

### `scripts/benchmark_conformal_variants.py`

Add calibrated global baseline and score-space diagnostics.

Right now the "global split" reference is not using the calibrator path, while the Mondrian interval variants do.

That should become:

- `global_split_raw`
- `global_split_calibrated`

and the benchmark should record the score space used for partitioning.

### `configs/profiles/search_conformal_exhaustive.yaml`

Extend the conformal search space with:

- `partition_score_spaces`
- `classification_set_score_families`
- `classification_set_partition_candidates`
- `calibrator_candidates_for_conformal`

This is the main config change needed to make the Chapter 3 ideas searchable rather than ad hoc.

## Recommended execution order

1. Fix the score-space mismatch:
   - allow calibrated probabilities to drive score-band Mondrian partitions.
2. Add calibrator-aware binary set benchmarking:
   - hinge/LAC vs margin.
3. Add Mondrian partitions to the binary set benchmark.
4. Only then test lower-priority families:
   - Brier,
   - rank-only,
   - APS/RAPS/SAPS in binary.

## Recommendation

If the goal is to improve the repo's real conformal search rather than just widen the method catalog, the best next experiment is:

- Venn-Abers-calibrated binary hinge sets,
- benchmarked globally and with `grade` / `score_decile_mondrian` / `grade_x_scoreband_mondrian`,
- with partitioning based on calibrated probability rather than raw score,
- compared against a binary margin sidecar.

That is the highest signal-to-effort move suggested by Chapter 3 for this repo's current binary credit-risk setting.
