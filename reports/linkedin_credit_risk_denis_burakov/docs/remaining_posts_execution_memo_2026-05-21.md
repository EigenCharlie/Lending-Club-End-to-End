# Remaining Posts Execution Memo

Generated: 2026-05-21

Scope: posts outside P1, PDF/Text Batch 1, and Image Batch 1. These were read
from extracted PDFs, public post text, and manual contact-sheet inspection. Most
close as source discovery, implementation context, or parked future candidates.

## PDF/Text Posts

### POST-040: Lending, Credit Risk & Data Science Resource Collection

- Content read: broad bibliography/resource collection covering model ops, risk
  management, profit scoring, loan default, credit scoring software, regulatory
  materials, alternative data, P2P lending, and portfolio economics.
- Project value: source-discovery only. It may help later bibliography triage,
  but it is not itself evidence.
- Decision: park as source-discovery index.
- Stop condition: closed after external/source links are already represented in
  `external_link_backlog.csv`; no direct book/paper claim.

### POST-052: Bias-Variance Tradeoff

- Content read: bias/variance decomposition, underfit/overfit visual examples,
  bootstrapping for classification loss visualization, note that MSE equals Brier
  for binary outcomes.
- Project value: teaching context for ML foundations; current book already covers
  enough model comparison/calibration.
- Decision: park.
- Stop condition: closed because it does not change a current claim or artifact.

## Public Text-Only Posts

### POST-011 and POST-012: On Credit Book Release and Code

- Content read: book announcement and code repository for scoring foundations.
- Project value: source discovery; the code repository is relevant but not
  canonical academic evidence.
- Decision: archive/source-discovery.
- Stop condition: closed after links resolved; no book/paper implementation.

### POST-025, POST-029, POST-033: AWS/SageMaker/MLflow/LocalStack Articles

- Content read: orchestration posts for SageMaker Pipelines, CatBoost, MLflow,
  LocalStack, AWS SAM, and SHAP/batch explainability.
- Project value: implementation context already absorbed by the Ch10 governance
  note on score/calibrator/policy/rule lineage.
- Decision: append as context, no further code.
- Stop condition: closed because no AWS implementation lane is opened.

### POST-036: Fisher-Yates Shuffle

- Content read: short historical note.
- Project value: none for current credit-risk claims.
- Decision: archive.
- Stop condition: closed with no action.

### POST-057: Visualizing Gradient Boosting

- Content read: LogitBoost/gradient boosting visualization, pseudo-residuals, and
  links to notebooks/related posts.
- Project value: teaching context for boosting; not needed for current manuscript
  claims.
- Decision: park as optional visual-teaching source.
- Stop condition: closed unless a boosting pedagogy section needs an example.

## Image Posts

### POST-009 and POST-043: AWS Certifications

- Content read: certification images and text about cloud/GenAI/Data Engineer
  credentials.
- Project value: contextual only.
- Decision: archive.
- Stop condition: closed with no implementation.

### POST-010 and POST-017: Events / AI Adoption

- Content read: event photos and text on analytics adoption, data quality, cloud
  sovereignty, Cursor/subagents/context-management.
- Project value: light context for research workflow, not for credit-risk claims.
- Decision: archive.
- Stop condition: closed with no implementation.

### POST-019: Fisher Exact Test and A/B Loan Take-Up

- Content read: Fisher's exact test visualization and loan-offer take-up A/B
  example.
- Project value: possible teaching note for small-sample A/B tests, but not
  material to current papers.
- Decision: park.
- Stop condition: closed unless the A/B simulation chapter is expanded.

### POST-037: Fraud Count Modeling

- Content read: rare fraud modeling via Poisson/negative-binomial counts and
  hourly/daily monitoring.
- Project value: adjacent risk-modeling idea, not Lending Club PD/ECL.
- Decision: park.
- Stop condition: closed; no implementation without a fraud-count dataset.

### POST-038 and POST-050: Visual Generation / GANs

- Content read: PatchGPT/autoregressive image generation and GAN image examples.
- Project value: outside credit-risk scope.
- Decision: archive.
- Stop condition: closed with no implementation.

### POST-041: MCMC Logistic Regression

- Content read: Metropolis-Hastings visualization for logistic-regression
  coefficients and posterior exploration.
- Project value: optional uncertainty/teaching context; not needed for current
  conformal or champion claims.
- Decision: park.
- Stop condition: closed without Bayesian/MCMC implementation.

### POST-047: Random Forest Interval Pruning

- Content read: pruning random-forest trees by interval bounds around prediction
  distributions; possible model compression without accuracy loss.
- Project value: interesting but not champion-relevant.
- Decision: park as future model-compression candidate.
- Stop condition: closed unless a reviewer asks for model-size/interpretable-RF
  analysis.

### POST-051 and POST-056: WOE for Image/Text Classification

- Content read: WOE/Naive Bayes outside credit scoring; WOE for transparent
  banking-intent text classification, token-level evidence bars, and WOE image
  generation examples.
- Project value: expands WOE intuition but does not alter current feature
  engineering or papers.
- Decision: park/archive. POST-056 stays as optional explainability context;
  POST-051 archives as low relevance.
- Stop condition: closed after Ch05 WOE governance note already absorbs the
  useful evidence-reading idea.

## Outcome

All indexed posts now have a closed analysis decision. Remaining open work is
not post reading; it is optional child-post capture and deeper reading of
selected GitHub/official/preprint links if they become necessary for a future
claim or appendix.
