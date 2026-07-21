# PDF/Text Batch 1 Execution Memo

Generated: 2026-05-21

Scope: first non-P1 set of posts whose primary asset already has text extracted
or whose public post text is sufficient for a decision. This memo closes the
first ready queue for implementation triage; image-only posts remain in the
OCR/manual-reading queue.

## Decisions By Post

### POST-016: MLOps Credit Risk on AWS

- Source status: LinkedIn document PDF plus GitHub repository link.
- Content read: MLflow tracking server with SageMaker, SAM/IAM roles, S3
  artifact storage, PostgreSQL metadata, LocalStack/Docker local stack.
- Project value: strong for Book Ch10/14/19 governance and reproducibility
  language; not a novel empirical claim.
- Implementable: add MRM note that experiment tracking must preserve model,
  calibration, artifact, and decision lineage across local and cloud modes.
- Decision: append to book governance/backlog.
- Stop condition: closed once MRM text receives a small operational note and
  GitHub source remains classified as implementation evidence, not scholarly
  evidence.

### POST-018: Batch Scoring and Credit Limit Management

- Source status: LinkedIn document PDF plus GitHub repository link.
- Content read: batch scoring architecture, EventBridge schedule, SageMaker
  Pipelines, Redshift/PostgreSQL scoring table, Lambda business rules, limit
  INCREASE/DECREASE/KEEP decisions.
- Project value: very useful for separating model scoring from business rules,
  decision audit tables, and governance ownership.
- Implementable: strengthen MRM text on separation of concerns: data science
  produces scores; credit risk owns policy rules; both need independent tests.
- Decision: append to Book Ch10/14/19; no Paper 4 claim unless converted into a
  decision-registry artifact.
- Stop condition: closed after governance text includes the score/rule split and
  the backlog records no need to rerun the champion.

### POST-023: LLM and AI Notes

- Source status: LinkedIn document PDF; mostly broad AI/MLOps notes.
- Content read: tokenization, embeddings, LLM training loop, FTI pipelines, data
  centric ML product structure, MLOps duties.
- Project value: context for AI/LLM architecture, not core to credit-risk papers.
- Implementable: archive as context for future AI companion; no current Quarto or
  Paper 4 change.
- Decision: park.
- Stop condition: closed with no implementation because it does not change a
  current claim, appendix, or reviewer response.

### POST-026: Real-Time Application Credit Scoring on AWS

- Source status: LinkedIn document PDF plus GitHub repository link.
- Content read: WOE logistic scorecard, scorecard formula, Kinesis/Lambda/DynamoDB
  inference path, cutoff decision, LocalStack/SageMaker local mode.
- Project value: useful bridge between WOE chapter and deployment/governance
  chapter.
- Implementable: mention that a scorecard deployment has two separately audited
  objects: the model artifact and the decision threshold/policy.
- Decision: append to Book Ch05 and Ch10 as an operational caveat.
- Stop condition: closed after this split is represented in governance text; no
  need to implement AWS infrastructure in the Lending Club project.

### POST-027: Deep Learning in Credit Risk

- Source status: LinkedIn document PDF; broad historical AI/deep-learning chapter.
- Content read: neural-network history, AI waves, use cases in financial services,
  transformer-era framing.
- Project value: background only; not needed for current claims.
- Implementable: none now. Possible future research-agenda context if a deep
  learning section is revisited.
- Decision: park/archive.
- Stop condition: closed once parked; no current evidence gap changes.

### POST-030: Logistic Regression Foundations

- Source status: LinkedIn document PDF with historical/statistical treatment.
- Content read: maximum likelihood, logistic/probit origins, WOE equivalence,
  multinomial WOE, confidence intervals, conformalized Pearson-residual style
  intervals.
- Project value: strong for Book Ch05/06/07 foundations, but manuscript claims
  require canonical citations rather than LinkedIn text.
- Implementable: add WOE caveat that WOE can be read as centered log-odds /
  log-likelihood-ratio evidence, not merely a convenience encoding; add backlog
  note for Pearson residual intervals.
- Decision: append to book foundations and Paper Estrella related-work contrast.
- Stop condition: closed when book text has the conceptual bridge and any
  conformal interval claim remains parked until canonical source verification.

### POST-034: WOE Foundations

- Source status: LinkedIn document PDF; source trail includes scorecard book
  references but not yet independently verified.
- Content read: decibans, Turing/Bletchley weight of evidence, log Bayes factors,
  WOE sign conventions, scorecard interpretability.
- Project value: useful for narrative and teaching in Ch05.
- Implementable: strengthen Ch05 with a short warning about sign convention and
  bin stability/recalibration, without adding historical claims as formal
  manuscript evidence.
- Decision: append to Ch05.
- Stop condition: closed when Ch05 has a practical WOE governance note and
  historical claims remain source-discovery only.

### POST-039: Robust Logistic Regression

- Source status: LinkedIn document PDF; references Murphy 2022 and fisher-scoring
  package/GitHub.
- Content read: robust logistic mixture likelihood for label noise/outliers,
  Fisher-scoring implementation sketch, robust/focal/multinomial package variants.
- Project value: useful as robustness caveat, not core to the champion.
- Implementable: create Paper 4/book backlog item for a bounded "label-noise
  sensitivity" appendix only if it can be run without reopening champion.
- Decision: append to research backlog; no immediate experiment.
- Stop condition: closed when parked as bounded robustness candidate with
  rejection rule: run only if it changes a robustness table or reviewer response.

### POST-042: Precision, Recall, F1, and Prevalence

- Source status: LinkedIn document PDF; conceptual, non-credit example.
- Content read: precision as prevalence-sensitive posterior probability; F1 can
  look poor under rare events even with high sensitivity/specificity.
- Project value: useful for metric governance in imbalanced default settings.
- Implementable: add Ch06 note that threshold metrics must be interpreted with
  prevalence and decision target; do not use F1 alone as PD model selector.
- Decision: append to Ch06 metric-governance language.
- Stop condition: closed when Ch06 includes prevalence caveat.

### POST-044: SHAP Scoring

- Source status: LinkedIn document PDF; code-style demonstration.
- Content read: relationship between WOE/logistic coefficients and linear SHAP,
  additive SHAP log-odds converted to PDO points, tree/linear SHAP scorecards,
  warning that SHAP explanations shift with data distribution.
- Project value: high for explainability governance and scorecard-style reason
  code language.
- Implementable: add Ch11 caveat that SHAP-scorecard conversion is a governance
  convenience, not a stable causal decomposition; monitor explanation drift.
- Decision: append to Ch11.
- Stop condition: closed when Ch11 has SHAP-scorecard caveat and no causal claim
  is implied.

### POST-046: WOE to Logistic Regression

- Source status: LinkedIn document PDF plus nbviewer notebook link.
- Content read: univariate WOE/log-odds/logistic coefficient equivalence,
  category-level target encoding to log-odds to WOE bridge.
- Project value: teaching bridge for Ch05.
- Implementable: combine with posts 30/34 in a Ch05 WOE caveat.
- Decision: append to Ch05.
- Stop condition: closed once Ch05 note covers WOE-logit relation and sign
  convention.

### POST-048: Logistic Regression Inference

- Source status: LinkedIn document PDF plus fisher-scoring package link.
- Content read: coefficient/prediction/mean-response intervals, statsmodels-style
  logistic inference, FisherScoringLogisticRegression examples.
- Project value: useful contrast for conformal intervals and LR baseline
  inference.
- Implementable: add related-work/backlog note; no change to official champion.
- Decision: append to Book Ch07/Paper Estrella related-work contrast.
- Stop condition: closed when interval type is clearly distinguished from
  conformal prediction intervals.

### POST-053: Boosting Beyond Trees

- Source status: LinkedIn document PDF but extracted text is mostly figure labels.
- Content read: decision-boundary comparison across decision trees, linear models,
  and neural networks under boosting.
- Project value: low without OCR/full notebook; source-discovery only.
- Implementable: none now.
- Decision: park pending image/notebook review.
- Stop condition: closed for this batch as insufficient-text; reopen only if the
  notebook or images become necessary for a boosting explanation section.

### POST-054: Binary and Multiclass Credit Default

- Source status: LinkedIn document PDF plus linked posts.
- Content read: binary logistic vs two-class softmax equivalence, multiclass
  default decomposition into DPD/UTP/no default, multinomial logistic limits,
  GBDT multiclass-to-binary framing.
- Project value: useful for Paper 4/LGD/default-state limitations and Ch07
  uncertainty framing.
- Implementable: add Paper 4 note: Lending Club target is a binary proxy and does
  not identify regulatory default subtypes such as DPD vs UTP.
- Decision: append to Paper 4 caveat/backlog, not experiment.
- Stop condition: closed when caveat is documented; no claim of true multiclass
  regulatory default without data.

### POST-059: Fisher Scoring Package

- Source status: LinkedIn document PDF plus GitHub package link.
- Content read: Fisher-scoring implementations for logistic, multinomial, and
  focal loss logistic regression; WOE pipeline example with OptBinning.
- Project value: implementation reference for possible LR inference/robustness
  companion, not core champion.
- Implementable: park as optional dependency candidate; avoid adding package to
  project unless a concrete test/appendix requires it.
- Decision: park/append to implementation backlog.
- Stop condition: closed when no dependency is added and optional use case is
  recorded.

## Cross-Cutting Project Changes To Apply

1. Ch05: add WOE governance note covering log-odds/log-likelihood-ratio reading,
   sign convention, and WOE recalibration as a maintenance candidate.
2. Ch06: add metric-governance caveat that precision/F1 depend strongly on
   prevalence and should not drive PD model selection alone.
3. Ch10: add governance note separating score production from policy/business
   rules and preserving lineage across model, calibrator, threshold, and decision.
4. Ch11: add SHAP-scorecard caveat about distribution dependence and explanation
   drift.
5. Paper 4/Paper Estrella: queue caveats, not champion changes, for robust
   logistic, Fisher-scoring inference, multiclass default-state decomposition,
   and Pearson/conformal interval contrast.
