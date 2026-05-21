# Image Batch 1 Execution Memo

Generated: 2026-05-21

Scope: high-relevance image/carousel posts that were blocked by missing OCR but
could be read manually from local contact sheets.

## POST-002: GBDT Fine-Tuning for New Credit Features

- Visual content read: CatBoost-style continuation training where a base tree
  model remains frozen and later trees absorb a new bureau feature; waterfall
  points redistribute after fine-tuning.
- Post text read: fine-tuning with `init_model` across XGBoost, LightGBM, and
  CatBoost; SHAP-based score tracking across base and fine-tuned trees.
- Project value: useful for model-maintenance governance; not directly
  implementable because the Lending Club data contract does not include new
  bureau/alternative data sources.
- Decision: append to governance/backlog, no experiment.
- Stop condition: closed after recording that feature-source expansion requires
  challenger review and explanation-drift checks, not automatic champion update.

## POST-007: On Credit Book Availability

- Visual content read: book availability/marketing image.
- Post text read: book covers credit-scoring foundations from Turing/Bletchley
  through modern decision systems.
- Project value: source discovery only.
- Decision: archive.
- Stop condition: closed after external links are resolved as book/vendor links;
  no direct project implementation.

## POST-013: Gini Under Class Imbalance

- Visual content read: CAP diagram with default-rate-constrained denominator and
  Gini/accuracy-ratio calculation.
- Post text read: argues that Gini is appropriate for naturally imbalanced credit
  portfolios because CAP/Gini normalizes by the maximum achievable area given the
  default rate.
- Project value: strong metric-governance caveat for Ch06.
- Decision: append to Ch06 language.
- Stop condition: closed after Ch06 distinguishes rank metrics from threshold
  metrics and explains why imbalance alone does not invalidate Gini.

## POST-021: Credit Risk Modeling on AWS

- Visual content read: zero-shot classification with Ollama/Strands and feature
  flags architecture with DynamoDB streams, S3 config, and Docker Lambda.
- Post text read: local/free AWS-style credit-risk guide with Docker,
  LocalStack, SageMaker, and GitHub repository.
- Project value: MLOps/context. Useful only as implementation inspiration.
- Decision: append as context already covered by Ch10 governance note; no new
  implementation.
- Stop condition: closed after GitHub repo is resolved and no AWS lane is opened.

## POST-024: Boosted Scorecards in Python

- Visual content read: xBooster release supports XGBoost, CatBoost, and LightGBM;
  converts boosted trees into scorecards and points toward SHAP scorecards for
  deeper trees.
- Post text read: boosted scorecards/LightGBM support; deeper-tree interactions
  still challenging.
- Project value: useful for Ch05/Ch11 backlog, but not enough to add dependency
  or benchmark.
- Decision: park as prototype candidate.
- Stop condition: closed unless a bounded boosted-scorecard appendix can change
  a claim or reviewer response.

## POST-028: xBooster and FastWoe Updates

- Visual content read: comparison of WOE estimation approaches (GMM, GAM,
  histogram, FastWoe tree, monotonic tree, FAISS k-means) and interval scorecard
  compression for XGBoost.
- Post text read: xBooster interval method, FastWoe monotonic binning,
  multiclass inference, and Somers D implementation.
- Project value: strong for source/backlog; overlaps with Ch05 WOE governance and
  Ch06 metric notes.
- Decision: append to backlog, no dependency addition.
- Stop condition: closed after methods are recorded as optional implementation
  candidates with evidence-gated future use.

## POST-049: WoeBoost

- Visual content read: WoeBoost decision-boundary iterations and normalized
  evidence feature ranking.
- Post text read: WOE gradient boosting combines independent feature evidence
  summaries with boosting; links to GitHub, visual explanation, notebook, and
  related WOE posts.
- Project value: interesting but high claim risk. It would require a real
  benchmark against existing LR/CatBoost, and the current champion should not be
  reopened.
- Decision: park as future prototype candidate.
- Stop condition: closed for current goal; reopen only if a bounded benchmark can
  change an appendix table or reviewer response.

## Cross-Cutting Updates

- Ch06 should state explicitly that class imbalance does not by itself invalidate
  Gini/CAP, while threshold metrics like precision/F1 remain prevalence-sensitive.
- Ch10/Ch11 already receive the necessary governance caveats: feature-source
  expansion, model fine-tuning, and SHAP-derived scorecards require lineage and
  explanation-drift monitoring.
