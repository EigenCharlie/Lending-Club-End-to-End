# P1 LinkedIn Reading Memo - Denis Burakov Credit Risk Corpus

Generated: 2026-05-21

## Scope

This memo covers the first high-priority batch of 16 posts from the Denis
Burakov LinkedIn corpus:

`1, 3, 4, 5, 6, 8, 14, 15, 20, 22, 31, 32, 35, 45, 55, 58`.

Public permalinks were captured without browser cookies or credentials. The
capture includes full public post text, raw HTML, LinkedIn document manifests,
document transcripts, PDFs where exposed by the public media manifest, and
feedshare images. Image-only material without machine OCR was read visually for
this memo and remains marked as manually read in
`data/attachment_manifest.csv`.

## Capture Status

- Posts captured: 16/16.
- LinkedIn documents captured from public manifests: 11 primary documents.
- Feedshare image posts manually read: posts 1, 4, 5, 6, and 15.
- Image-only calibration deck manually read: post 55 after PDF-to-JPG conversion.
- Main extracted artifacts live under `attachments/raw/{activity_id}/`.
- PDF text extracted where possible under `attachments/extracted/`.

This is still source intake, not manuscript-grade evidence. LinkedIn material
can motivate project work, but peer-reviewed, official, preprint, code, or book
sources must be verified before claims enter Paper Estrella or the formal book
bibliography.

## Concept Intake

### 1. Metric Governance: Brier, Gini, Calibration, Somers D

Posts: 1, 3, 35, 55, 58.

The useful thesis is not "use metric X", but "separate metric families by the
decision they answer." The captured material distinguishes:

- ranking/discrimination: AUC, Gini, CAP, Somers D/Dxy;
- probability accuracy: Brier, log loss, calibration diagrams, cumulative
  calibration profile, ECE/MCE/CE;
- threshold/action utility: approval rate, bad-rate appetite, cost-weighted
  decision curves.

Post 1 shows the key pitfall for this project: equal Brier scores can hide very
different PD range and ranking power. The Yates-style decomposition frames this
as base-rate uncertainty, calibration/mean error, prediction spread, and a
discrimination covariance term. For CRPTO, that means Brier cannot be the sole
champion selector because the portfolio optimizer needs both reliable absolute
PDs and useful risk ordering.

Post 3 is a source binder, not one standalone argument. The main value is the
map of metric literature: accuracy is weak for inference and utility; rank-based
metrics answer ordering; proper scoring rules answer probability quality; and
model choice should be tied to the use case. This supports a short Ch06
literature note plus a source-discovery trail for bibliography verification.

Post 35 reinforces a rare-event warning: training on a balanced 50/50 sample can
produce probabilities that require prior correction or recalibration before use
as real-world PDs. This directly supports the existing calibration section and
argues against treating resampling metrics as decision-ready risk estimates.

Post 55 adds a practical calibration toolkit view: calibration diagrams,
logistic/isotonic/Venn-Abers/proper scoring rules, and cumulative calibration
profiles. The image deck makes one point especially relevant to this project:
post-hoc calibration can improve good raw probability outputs, but cannot rescue
weak model expressivity or narrow score range.

Post 58 extends the ranking family beyond binary default. Somers D/Dxy is the
natural bridge for LGD or ordinal/continuous outcomes, and belongs as a small
metric extension in Ch06 and possibly the LGD appendix, not as a new main
contribution.

Decision: **append to Book Ch06 and Paper 4 metric-governance appendix; use in
Paper Estrella only as reviewer-defense framing.**

### 2. Economic Meaning of Gini

Posts: 4 and 15.

The captured images and post text translate Gini into business decision space.
The material connects separation of goods/bads to acceptance-rate curves, bad
rate under fixed risk appetite, and expected profit under a simple unit-economics
formula. This is very useful for Ch09 and the Paper Estrella introduction because
it explains why better ranking can change approved volume or profit even when
the statistical metric itself is abstract.

The strongest project link is to the CRPTO selector: a point of Gini is valuable
only through a downstream policy, such as approving more loans at the same bad
rate, lowering losses at a fixed approval rate, or increasing expected profit
under calibrated PD and margin assumptions. This lines up with the project's
economic-champion framing.

Decision: **append to Book Ch09 and Paper Estrella motivation/discussion. Do not
claim a universal dollar value of Gini; tie any value statement to the project's
own funded-set artifacts.**

### 3. WOE, Scorecards, And Interpretable ML

Posts: 5, 8, 20, 31.

Post 5 proposes WOE recalibration as a middle path between full redevelopment
and intercept-only recalibration. The image example keeps bin boundaries fixed
for a risk driver, updates WOE values on new data, and shows improved
calibration without rebuilding the scorecard. This is a strong Paper 4 prototype
candidate because it can be tested under temporal drift with existing WOE bins.

Post 8 is a compact "GBDT leaf interactions to WOE" idea. The procedure is:
train a boosted tree model, extract leaf indices as joint/interacted regions,
WOE-encode those leaves, and feed them to a logistic scorecard. The value is
capturing interactions while preserving a linear downstream model. This is a
promising benchmark idea, but dependency and scope risk are high.

Post 20 introduces FastWoe as a credit-risk-focused toolkit: WOE encoding with
uncertainty, marginal Somers D feature screening, CAP curves, EAD-weighted CAP,
and styled audit outputs. For this repo, the immediate value is not adopting the
package blindly; it is harvesting missing diagnostics and comparing them with
OptBinning artifacts.

Post 31 extends WOE to multiclass outcomes using likelihood-ratio thinking for
multiple default states. This is conceptually useful for DPD default vs UTP
default, LGD bins, or default reason categories, but current Lending Club labels
limit how far this can be pushed.

Decision: **append WOE recalibration and multiclass WOE to Book Ch05; open a
bounded Paper 4 prototype only for WOE recalibration under drift. Park full
GBDT-leaf/boosted-scorecard benchmark until a reviewer or clear table gate
requires it.**

### 4. Probabilistic LGD And Distributional Risk Parameters

Post: 14.

The LGD document is one of the most actionable captures. It reframes LGD
regression as multiclass classification over quantile bins, producing a full
conditional distribution instead of a single conditional mean. From that
distribution, one can compute a point estimate, percentiles, expected shortfall,
and stress scenarios by shifting probability mass toward severe bins.

The strongest connection to this project is Paper 4's IFRS9/SICR and LGD/EAD
appendix boundary. The idea fits the current direction because conformal and
robust optimization already emphasize uncertainty-aware decisions. But it should
not be sold as full IFRS9 compliance in this project because contractual
servicing, monthly DPD, macro scenario infrastructure, and recovery timing remain
limited.

Decision: **append to Book Ch07/Ch10 and Paper 4 LGD/IFRS proxy appendix. Reopen
as an experiment only if it yields one compact comparison table against the
existing LGD baseline and respects the proxy boundary.**

### 5. SHAP Distillation And Explainable Scorecards

Posts: 22 and 32.

Post 22 gives background on Breiman, CART, forests, boosting, information gain,
and TreeSHAP as the bridge that made tree ensembles more usable in high-stakes
credit settings. The useful project angle is the connection between tree
splitting, information gain, WOE/IV language, and model-risk explainability.

Post 32 is more directly implementable: train a teacher model, compute SHAP
values, then distill behavior into either binned SHAP features with logistic
regression or GAM-style smooth transformations. The deck claims strong retention
of teacher ranking power in a credit-risk example while producing a more
transparent student model.

For this project, the idea belongs in governance and explainability, not causal
interpretation. SHAP explanations describe predictive decomposition; they do not
prove mechanisms or fair-lending compliance.

Decision: **append to Book Ch06/Ch10 and Paper 4 governance appendix. Prototype
only if scoped as "student-model auditability" with no causal claim.**

### 6. Probability / Classification Intervals

Post: 45.

The Pearsonify material is close to the project's conformal thread. It argues
that binary classification sets like `{true, false}` are often unhelpful for
credit scoring, and explores intervals around predicted probabilities using a
Pearson-residual style conformity score.

This is useful as a contrast, not a replacement. CRPTO already turns calibrated
PD intervals into decision constraints. Pearsonify gives language for why
probability intervals matter in classification, but the project should keep its
main claim around conformal PD intervals and robust portfolio decisions.

Decision: **append to Book Ch07 and Paper Estrella related-work contrast.**

## Immediate Project Actions

1. Update Book Ch06 with a short metric-governance paragraph: rank, probability,
   calibration, and utility metrics answer different questions.
2. Add a Ch09 note connecting Gini to approval/risk-appetite/profit rather than
   treating Gini as a purely abstract model score.
3. Add a bounded Paper 4 candidate memo for WOE recalibration under temporal
   drift. Evidence gate: same bins, updated WOE, improved calibration, no full
   redevelopment.
4. Add probabilistic LGD as an appendix candidate, with explicit "proxy only"
   boundary.
5. Add SHAP distillation to the governance backlog as a student-model
   auditability route.
6. Keep Pearsonify as related work for probability intervals, not as a CRPTO
   replacement.

## Remaining Gaps

- External links and source papers inside the binders still need canonical
  bibliographic verification.
- The 43 non-P1 posts remain prior-index only.
- Tesseract is not installed in the WSL environment, so image OCR is unavailable;
  P1 image material was read visually instead.
- Some LinkedIn PDFs are image-based and need visual reading or OCR; post 55 was
  handled manually for this memo.
