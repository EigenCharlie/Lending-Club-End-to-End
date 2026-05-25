# LinkedIn Backlog Intake for Paper 4 and Paper Estrella

Generated: 2026-05-21

This memo records manuscript-safe intake from the Denis Burakov LinkedIn corpus.
It does not promote LinkedIn material as scholarly evidence. Items below are
paper/backlog decisions after reading the local post/PDF assets and resolving
external links where available.

## Paper 4

### Append As Caveat: Binary Target vs Regulatory Default States

Source posts: POST-014, POST-031, POST-054.

Decision: append as caveat, not as a new claim. The Lending Club target remains a
binary default proxy. It does not identify regulatory default subtypes such as
days-past-due (DPD) versus unlikely-to-pay (UTP), nor does it provide the data
needed for a true multinomial regulatory default model. The safe Paper 4 use is a
limitations paragraph: distributional or multiclass risk-parameter methods are
conceptually relevant but outside the current data contract.

Stop rule: do not open a multiclass experiment unless a data artifact exists with
separable default states and a claim/artifact table would change.

### Park As Bounded Robustness Candidate: Robust/Focal Logistic Regression

Source posts: POST-039 and POST-059.

Decision: park. Robust logistic and focal-loss logistic variants are plausible
stress baselines for label noise or severe imbalance, but they do not justify
reopening the official champion. The only acceptable future use is a bounded
appendix experiment with a pre-registered rejection rule: include only if it
changes a robustness table or directly answers reviewer feedback.

Stop rule: no dependency (`fisher-scoring`) is added unless a testable appendix
lane is created.

### Append As Governance Language: Score/Rule Separation

Source posts: POST-016, POST-018, POST-026.

Decision: append to governance language. Paper 4 can use this as reviewer-defense
framing: the decision system should distinguish model scores, calibrator,
threshold/policy, and downstream business rules. This strengthens traceability
without changing the optimization champion.

Stop rule: close after the book/governance text captures the distinction; do not
start an AWS implementation lane.

## Paper Estrella

### Append As Related-Work Contrast: Classical Intervals vs Conformal Intervals

Source posts: POST-030, POST-045, POST-048.

Decision: append as framing only. Classical logistic confidence intervals,
Pearson-residual style probability intervals, Venn-Abers probability intervals,
and conformal prediction sets answer different uncertainty questions. Paper
Estrella should keep its champion unchanged and use this only to sharpen
related-work language.

Stop rule: no new experiment unless it changes a table already used in the
journal package.

### Append As Metric-Governance Reviewer Defense

Source posts: POST-001, POST-003, POST-004, POST-006, POST-015, POST-042,
POST-055, POST-058. External source status: POST-058 includes a readable arXiv
preprint snapshot on AUROC/AUPRC under class imbalance; it is useful as a
preprint-labeled caveat, not as peer-reviewed proof.

Decision: append as language only. The paper should keep the distinction between
ranking, calibration, uncertainty, and portfolio value explicit. Brier, ECE,
Gini/Somers D, precision-recall, and economic value are complementary; none is a
universal selector.

Stop rule: no additional benchmark unless the source can change a reviewer
response or already-promoted claim.

### Visual Reread Addendum

Source posts: POST-006, POST-015, POST-019, POST-047, POST-056.

Decision: append only the language that strengthens existing claims. POST-006
adds an observation-level Gini diagnostic idea; POST-015 sharpens the bridge from
Gini to risk appetite and net return; POST-056 frames WOE as contrastive
reason-code language. POST-019 and POST-047 remain parked as optional
experimental-design/model-compression sidebars.

Stop rule: no new empirical lane from visual material alone. These items can
reopen only if they change an existing chapter table, appendix, or reviewer
response.

## Second Ingest Addendum

Source pack: `reports/linkedin_credit_risk_denis_burakov/second_ingest/`.

The second ingest added 15 post-level rows (13 newly discovered posts plus two
LinkedIn child posts) and 12 public LinkedIn articles. All discovered PDFs,
decks, high/medium-priority article visuals, and relevant external links were
processed into the second-ingest backlog. The key manuscript-safe decisions are:

- **Append to book, not papers as evidence**: WOE uncertainty, WOE as
  likelihood-ratio language, log loss as a probability-quality metric, and PD as
  probability estimation rather than mere classification.
- **Append to governance**: the ECB internal-model guide is the strongest new
  source-status item. It supports language on seeds, observation ordering,
  explainability, conceptual soundness, and validation reproducibility.
- **Append to portfolio framing**: profit scoring reinforces that PD and Gini
  do not equal profitability. The current LP remains a valid expected-loss
  bridge, while full profit scoring is future work because the project lacks
  realized revenue, recovery, write-off, and operating-cost targets.
- **Park as bounded prototypes**: FastWOE uncertainty, xBooster/CatBoost
  scorecard extraction, tree-level validation via contingency tests, trended
  sequence scoring, and LGD LR-CR decomposition.
- **Archive as pedagogy/context**: MLE-to-LLM framing, Poisson count models for
  binary PD, broad CatBoost deployment tips, and AI-underwriter workflow ideas.

Stop rule: the second ingest is closed for discovered material. Do not reopen
the official champion or Paper Estrella experiments from LinkedIn material.
Only reopen a parked lane if it can change an existing appendix table, reviewer
response, or verified source-backed claim.

## Logged-In Comment/Link Addendum

Source pack:
`reports/linkedin_credit_risk_denis_burakov/logged_in_review/`.

The logged-in pass re-opened 80 posts through the user's authenticated visible
browser session, captured 503 visible comments across 67 posts, and deduped 234
comment/post link rows. It attempted 42 high-priority unique sources, extracted
26 directly readable sources, recovered 6 additional readable alternate sources,
and left the remaining blockers explicitly logged.

The project-safe decisions are:

- **Promote to the book**: WOE as auditable evidence rather than only encoding,
  IV caution, Naive-Bayes-as-evidence analogy, Brier/reliability caveats,
  Gini/Somers/rare-events clarification, PSI threshold caution, score versus
  calibrator versus policy separation, and explainable prototypes via RuleFit
  or GBDT-leaf linearization.
- **Use in Paper Estrella only as reviewer-defense framing**: calibration
  diagrams, Brier decomposition, rare-event probability correction,
  explainability-cost language, and metric-governance language. The champion
  remains closed.
- **Park as Paper 4 backlog lanes**: WOE recalibration or Bayesian/Good WOE
  under drift, RuleFit/GBDT-leaf scorecards, PSI uncertainty, rare-event
  sampling/calibration, and the Gini-to-economic-value bridge.
- **Archive as blocked or redundant**: LinkedIn-only comments, login-walled
  tutorials, social mirrors, and blocked pages where a stronger independent
  source was recovered.

Stop rule: the logged-in review is closed for the current corpus. Reopen only
for new posts/comments after 2026-05-21 or if a reviewer asks directly about one
of the parked lanes above.

## Andrija Djurovic / ADSFCR Addendum

Source pack:
`reports/linkedin_credit_risk_andrija_djurovic/`.

The Andrija pass applies the improved Denis workflow to a second credit-risk
profile. It indexed 21 public posts, captured 4 public LinkedIn articles,
resolved 58 external links, extracted 50 high-value readable source rows, read
9 LinkedIn document/deck PDFs, and triaged 52 visual rows. Two public activity
URLs only returned generic LinkedIn text and were archived as blocked rather
than inferred from snippets.

The project-safe decisions are:

- **Promote to CRPTO reviewer-defense language**: risk differentiation versus
  risk quantification, PD calibration before portfolio optimization,
  multi-period PD backtesting caveats, conformal inference as uncertainty
  framing, and decision value beyond ranking metrics.
- **Append to the mini-book as governed caveats**: multi-period average PD
  testing, reliability limits of normal multi-period tests, model-based
  discrete PD rating-scale calibration, and WoE encoding instability.
- **Park as Paper 4 prototypes**: tree-based risk-factor interactions under
  monotonic/business constraints, model-based PD calibration, WoE instability
  diagnostics, and optional MoC Type C simulation. None reopens the official
  champion unless it changes a claim, appendix table, or reviewer response.
- **Route to thesis rather than IJDS body**: IFRS9 supervised macro index,
  PCA/ADF/OLS/dynamic/recursive regression material, LGD/EAD Somers D under
  conservatism, and scorecard model shift/MRM.
- **Archive as tooling/source discovery**: PDtoolkit, monobinpy, R IRB toolkit,
  ADSFCR repository announcements, book-release posts, and LinkedIn-only
  material without independent source status.

Stop rule: the Andrija intake is closed for the current public corpus. Reopen
only for newly visible logged-in comments/links, canonical papers, or local
experiments that can change a paper claim, appendix artifact, reviewer response,
or thesis chapter. LinkedIn material remains intake evidence only.

## Andrija Logged-In P0/P1 Addendum

Source pack:
`reports/linkedin_credit_risk_andrija_djurovic/logged_in_review/`.

The logged-in pass was completed through Windows Opera GX, using a
non-destructive copy of the user's visible authenticated browser profile. It
captured 37 P0/P1 rows, with 34 rendered captures, 2 checkpoint/signup
surfaces, 1 capture error, 121 visible comments and 72 external link rows.

Project-safe logged-in deltas:

- **Promote to CRPTO metric governance**: an arXiv preprint on statistical
  hypothesis testing for IV supports the caveat that fixed IV thresholds are
  inherited heuristics, especially fragile under class imbalance. This goes to
  language and optional Paper 4 prototype only, not to the IJDS empirical core.
- **Promote to WoE stability caveat**: the WoE instability thread sharpened the
  point that iterative replacement using model-predicted outcomes can change a
  scorecard even when the first model replicates perfectly. This is a Paper 4
  candidate only under a local experiment gate.
- **Park as residual-tree validation prototype**: tree-based residual/segment
  validation may identify omitted risk factors or over/underestimation pockets,
  but it must preserve monotonicity, minimum-leaf and auditability constraints.
- **Promote to PD backtesting caveat**: autocorrelation and common macro factors
  affect effective sample size; multi-period PD tests should not be narrated as
  mechanical pass/fail evidence.
- **Append to thesis/MRM**: model shift remains useful for specification-risk
  monitoring and systematic scorecard validation, but not as a new IJDS claim.
- **Archive**: non-credit-risk dense threads, hybrid-threat financial stability
  material and financial-crime-compliance AI sources are out of CRPTO scope.

Stop rule: the logged-in Andrija pass is closed for P0/P1. Reopen only if a
new independent source or local experiment can change a claim, appendix table,
reviewer response or thesis chapter. No LinkedIn comment is public evidence by
itself.
