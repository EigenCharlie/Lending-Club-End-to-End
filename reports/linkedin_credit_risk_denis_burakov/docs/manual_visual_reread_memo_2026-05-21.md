# Manual Visual Reread Memo

Generated: 2026-05-21

## Scope

This memo closes the non-Tesseract visual reread requested after the overnight
backlog. It uses direct visual reading of local images/contact sheets rather
than OCR. The purpose is not bibliography; it is concept intake for the Quarto
book, Paper 4, and Paper Estrella.

Artifacts read:

- Image/carousel posts: POST-001, POST-002, POST-004, POST-005, POST-006,
  POST-007, POST-009, POST-010, POST-013, POST-015, POST-017, POST-019,
  POST-021, POST-024, POST-028, POST-037, POST-038, POST-041, POST-043,
  POST-047, POST-049, POST-050, POST-051, POST-056.
- Parked/low-text PDF visual sheets: POST-008, POST-014, POST-023, POST-027,
  POST-031, POST-039, POST-040, POST-052, POST-053, POST-059.

## Decisions Changed Or Strengthened

### Promote To Book Language

POST-015: economic value of Gini. The visuals make the best project-level point:
Gini matters only through risk appetite, approval volume, and unit economics. A
new note was added to Chapter 09 explaining that Gini gains become valuable only
when they change capital allocation or expected net return under the PD cap.

POST-056: WOE for text classification. The visual analogy is useful even outside
text: WOE can be read as contrastive evidence, where each bin/token/rule pushes
log-odds toward or away from a hypothesis. A new note was added to Chapter 11 to
separate WOE as stable feature engineering from WOE as communicative reason-code
language.

POST-006: per-observation Gini contribution. The visuals show a useful diagnostic
frame: easy non-defaults and defaults contribute positively to ranking, while
overlap-zone observations identify where separation is lost. Chapter 06 now
records this as a backlog diagnostic, not as a champion metric.

### Keep As Existing Append/Backlog

POST-001: Brier decomposition. The images reinforce the existing Brier/Gini/ECE
separation: two models can share Brier while differing in discrimination because
the Yates components offset each other.

POST-002: GBDT fine-tuning. The images clarify the mechanism: new trees absorb a
new bureau-style feature while old trees stay frozen, and score contributions
redistribute. Keep as model-maintenance backlog; do not implement without a true
new-source experiment.

POST-005: WOE recalibration. The images show fixed bins with updated WOE values
and a restored Spiegelhalter calibration test. Keep as bounded maintenance lane;
it already supports the Ch05 WOE recalibration note.

POST-013: Gini under class imbalance. The visual CAP formula and normalization
support the existing Ch06 warning: imbalance alone does not invalidate Gini.

POST-014: probabilistic LGD via quantile-based classification. The PDF visuals
show the bimodal/zero-heavy LGD distribution, quantile probabilities, stress
shifting, and calibration comparison. Keep as Paper 4 limitation/backlog because
the current project lacks true LGD default-state labels.

POST-019: Fisher exact test and loan-offer take-up. The visuals are more useful
than the prior status implied. They show an A/B testing example with acceptance
rate lift and Fisher exact p-value. Park as a future experimental-design sidebar
for policy changes or champion deployment monitoring.

POST-024 and POST-028: xBooster/FastWoe. The visuals reinforce boosted
scorecards, SHAP scorecards, WOE estimator families, and interval scorecard
compression. Keep as prototype backlog only.

POST-031: multiclass WOE. The visuals clarify simple-vs-composite hypotheses and
Bayes-style posterior class inference. Keep as Paper 4 caveat/backlog; do not
convert the current binary default proxy into a multiclass claim.

POST-037: fraud count modeling. The visuals show Poisson underfit under excess
zeros, Negative Binomial fit, and tree-based Poisson regressors for hourly fraud
counts. Useful conceptually for event-count monitoring, but not directly
implementable on the current Lending Club loan-level dataset.

POST-039 and POST-059: robust/focal/multinomial logistic via Fisher scoring. The
visuals strengthen the label-noise/imbalance robustness backlog, but do not
justify adding a dependency or reopening the champion.

POST-047: random forest interval pruning. The visuals show pruning trees whose
predictions are outside an observation-level interval, reducing model size while
tracking Brier/log-loss. Useful for compression/governance ideas, but not aligned
with the current CatBoost champion.

POST-049: WoeBoost. The visuals show iterative WOE boosting decision boundaries
and normalized evidence features. Keep as future benchmark candidate only.

POST-053: boosting beyond trees. The low-text PDF was visually read. It compares
boosted decision trees, boosted linear models, and boosted neural networks across
synthetic boundaries. The concept is pedagogical; no project change.

### Archive / No Project Action

POST-007, POST-011, POST-012, POST-040: book/source-discovery materials. Keep for
bibliography triage, not as claims.

POST-009, POST-010, POST-017, POST-021, POST-023, POST-027, POST-043: AWS,
event, AI/LLM, or certification context. The useful governance boundaries are
already captured in Chapter 10; no new implementation lane.

POST-038, POST-050, POST-051: generative image/patch/GAN examples. Archive.

POST-041 and POST-052: MCMC logistic and bias-variance teaching material. Useful
pedagogy, but no direct claim or implementation for the current artifacts.

## Net Contribution To The Project

Implemented from this reread:

- Chapter 06: added observation-level Gini contribution as a diagnostic backlog.
- Chapter 09: added economic-value framing for Gini through risk appetite and
  expected net return.
- Chapter 11: added WOE as contrastive explanation/reason-code language.

No manuscript claim is promoted from image material alone. The visual reread only
changes project language and backlog classification; it does not add bibliography
or reopen champion experiments.
