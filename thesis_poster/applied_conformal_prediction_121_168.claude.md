5. Probability Calibration, Venn Predictors,
and Conformal Efficiency
"A probability that cannot be trusted is merelyascore; a calibrated
probability isastatement about reality."
— Author's maxim.
5.1 Why Classification Needs Calibration
Chapter 3 established conform al prediction for classification: givenanybaseclassifieranda
calibrationset,thesplitconformalprocedureconstructspredictionsets C α (x)⊆Y satisfying
P(Y ∈C α (X))≥1−α under exchangeability. That chapter answered whether the true label
is covered. This chapter asksadifferent question: how trustworthy are the probabilities
themselves?
Modern classifiers begin with scores, not probabilities. Given an input x, a classifier
produces
(cid:0) (cid:1)
s(x)= s (x),...,s (x) ,
1 K
where s k (x)∈R isanumerical score associated with class k. These scores may arise from
margins (support vector machines), logits (neural networks), boosted tree ensembles, or
other internal mechanisms.
Scores are not probabilities.
Scores Versus Probabilities
1) Deterministic outputs. Forafixed trained model, the classifier score s k (x) is a
deterministic function of the input x. However, across training samples, the score is a
random variable through the randomness of the training data.
2) Frequentist interpretation. In the frequentist framework, probability is defined
asalong-run relative frequency. Among events assigned probability p, approximately
a fractionpshould occur.

3) Softmax is not calibration. Transforming scores via softmax
exp(s (x))
k
pˆ (x)=
k PK exp(s (x))
j=1 j
produces normalised outputs satisfying P k pˆ k (x) = 1, but normalisation does not
guarantee P(Y =k|pˆ k (X)=p)=p [32].
4) Calibration is empirical. Calibration concerns whether predicted probabilities
match observed frequencies [15, 18].
5) Why this matters. Uncalibrated probabilities lead to systematically distorted risk
estimates, incorrect expected-loss calculations, and unreliable decision thresholds.
5.1.1 The Universal Miscalibration Problem
A persistent misconception inmachine learning is that certain classifiersproduce calibrated
probabilities "out of the box." This is false. In practical settings, all widely used classifiers
exhibit measurable miscalibration, and the belief otherwise has led to decades of silently
unreliable probability estimates in production systems.
Deep networks. Guo et al. [32] demonstrated that modern neural networks—despite
achieving steadily improving accuracy—have become progressively worse at calibration.
Deep architectures with batch normalisation, residual connections, and overparameterised
capacity produce confidently wrong probability estimates. Minderer et al. [57] confirmed
this at scale: the relationship between accuracy and calibration is architecture-dependent
and cannot be assumed.
Shallowneuralnetworks. Niculescu-Miziland Caruana[61]claimedthatneuralnetworks
with one or two hidden layers are reasonably well calibrated, a finding that became widely
cited as evidence that atleast some architectures produce trustworthy probabilities without
post-hoc correction. Johansson and Gabrielsson [36] overturned this claim: across 25
datasets, both single multilayer perceptrons and neural network ensembles were found to
be often poorly calibrated, benefiting significantly from post-hoc calibration via Venn–Abers
predictors. Theapparentcalibrationobservedin 2005 wasanartefactofthelow-dimensional
benchmarks of that era.
Tree ensembles. Random forests and gradient-boosted trees are no exception. Niculescu-
Mizil and Caruana [61] showed that boosted trees systematically push probabilities toward
0 and 1, while random forests compress them toward 0.5. Neitherproducesfrequenciesthat
match the claimed probabilities.
Support vector machines. SV Ms output margins, not probabilities. The standard Platt-
scaling wrapper [66] retrofitsasigmoid to the margin, but the resulting class scores inherit
the margin distribution distortion. Despite this, SVC(probability=True) is routinely used
in production as though it returned calibrated probabilities.
Naïve Bayes. The conditional independence assumption produces posterior estimates that
are systematically miscalibrated—typically overconfident for predictions near 0 and 1 and
underconfident near 0.5 [61].
Large-scale empirical evaluation. Manokhin and Grønhaug [54] evaluated seven model
families across 72 tabular datasets from the Tab Arena-v 0.1 benchmark. Notasingle base
classifier –logistic regression, random forest, gradient-boosted trees, neural network, SVM,
naïve Bayes, or k-nearest neighbors—produced acceptably calibrated probabilities without

5.1 Why Classification Needs Calibration 123
post-hoc correction. The degree of miscalibration varied by architecture, but the direction
was universal: every class score produced by predict_proba required recalibration. Post–
hoc Venn-Aberscalibrationreducedlog-lossbyanaverageof 14.17%acrossallmodel–dataset
pairs, a gap that would not exist if any classifier were "calibrated by default."
5.1.2 Logistic Regression: The Myth of Natural Calibration
Perhaps the most entrenched misconception in applied machine learning is that logistic
regression is "naturally calibrated" because it optimizes log-loss and outputs probabilities
through the sigmoid function. This claim appeared in textbooks for decades and was
reinforced by the most widely used software library in the field. It is wrong.
Three distinct mechanisms guarantee that logistic regression is miscalibrated in practice.
1) Structural overconfidence: the Θ(d/n) bias. Bai, Lee, and Liang[5] proved thateven
under ideal conditions–well-specified model, independent identically distributed data, no
noise–maximum-likelihood logistic regression is inherently overconfident. The calibration
error is Θ(d/n), wheredis the number of features andnthe sample size. The sigmoid link
function structurally pushes predicted probabilities away from 0.5 toward the extremes.
For any predicted probability above 50%, the expected true conditional probability is lower
than the predicted value. The overconfidence does not vanish; it shrinks at rate O(d/n) but
persists due to the geometry of the sigmoid. In practical terms, noticeable calibration bias
typically emerges when d/n is non-negligible—a common regime in genomics, NLP, clinical
risk models, and high-dimensional tabular data—producing probability estimates that are
materially unreliable.
2) Model misspecification and link function incompatibility. The "calibration equa-
tions" from generalized linear model (GLM) theory guarantee only marginal calibration: the
average predicted probability equals the average observed outcome. Theydonot guarantee
auto-calibration—that E[Y |pˆ(X)=p]=p for every p [53]. As mechanism (1) established,
even when the sigmoid link is perfectly specified, LR is overconfident by Θ(d/n). When the
true data-generating process does not followasigmoid link—and it rarely does in real-world
data—a second layer of miscalibration compounds on top of the structural bias. The two
sources are additive: finite-sample overconfidence is irreducible regardless of the link
function, and misspecification introduces additional distortion at every probability level. No
amount of correct model specification eliminates the first; no amount of data eliminates the
second.
3) Regularisation bias. Modern implementations of logistic regression use L 2 regular-
isation by default (scikit-learn sets C=1.0). Shrinkage introduces bias in the predicted
probabilities: it pushes coefficients toward zero, which flattens probabilities toward 0.5.
Heavy regularisation induces underconfidence; weak regularisation in high dimensions
permits the Θ(d/n) overconfidence to dominate. There is no "default" setting that ensures
calibration—the claim that "properly regularised logistic regression is well calibrated" is
circular [5].
Empirical confirmation across domains. Ojeda et al. [63] conductedalarge simulation
andreal-datastudyinbiomedicalsettingsandfoundthatbetacalibrationandlogistic(Platt)
calibration consistently outperform raw logistic regression in log-loss and Brier score, with
rawLRshowinginterceptandslopedeviationsinreliabilitydiagrams. Vanden Goorberghet
al.[28]showedthatclass-imbalancecorrections—commoninmedicalriskprediction—cause
strong miscalibration in logistic regression, with predicted probabilities severely biased
even when discrimination improves.

A controlled experiment testing scikit-learn's calibration claim. We designed a
controlled experiment to test the "calibrated by default" claim directly. Onasynthetic
binary classification task with 200 features and 2,000 samples (d/n=0.10, class imbalance
95:5), raw logistic regression yieldsaBrier score of 0.078 andaSpiegelhalter Z-statistic of
+19.5—far exceeding the ±1.96 threshold for the null hypothesis of perfect calibration at
the 5% significance level. If the scikit-learn documentation were correct, Z should cluster
near zero. Applying Venn–Abers calibration to the same predictions reduces the Brier score
to 0.050 (a 36% improvement) and bringsZto −1.8—within the acceptance region. The
corresponding log-loss improvement is 66%. Contrary to what scikit-learn's documentation
suggests, logistic regression is not inherently calibrated. The apparent calibration comes
from the post-hoc logistic correction itself—that step is the entire source of the calibration
effect.
Crucially, onalow-dimensional version of the same task (d=50, n=20,000, d/n=0.0025),
the same logistic regression yields Z =−0.63—well within the acceptance region. This is
the regime that scikit-learn's documentation examples use. The documentation incorrectly
generalizes from d/n≈0 to all of machine learning without warning practitioners about the
high-dimensional case.
The full Colab notebook with reproducible code for all five experiments and interactive
plots is included in the Pro edition of this book:
https://valeman.gumroad.com/l/applied_conformal_prediction_pro
Figures 5.1–5.3 visualise the key findings.
Figure 5.1: Side-by-side reliability diagrams for logistic regression in two regimes. Left:
d/n=0.0025(d=50,n=20,000);Spiegelhalter Z=−0.63(withinacceptanceregion). Right:
d/n=0.10 (d=200, n=2,000); Z =+19.5 (far exceeding the ±1.96 threshold). The docu-
mentation regime (left) hides the overconfidence that dominates modern high-dimensional
settings (right).
The dual reliability diagram and ECE values for this experiment are available in the Pro edi-
tion notebook (https://valeman.gumroad.com/l/applied_conformal_prediction_pro).
5.1.3 The fallacy of predict_proba
The universal miscalibration problem is compounded by software interfaces that obscure it.

5.1 Why Classification Needs Calibration 125
Figure 5.2: Reliability diagrams for raw logistic regression versus Venn–Abers calibration
on the high-dimensional task (d/n=0.10). Raw LR (red) shows systematic overconfidence:
the curve bows below the diagonal, with Z =+19.5. After Venn–Abers calibration (green),
thecurvetracksthediagonalmoreclosely,with Z =−1.8,Brierscoreimprovedby 36%,and
log-loss improved by 66%.
Figure 5.3: Spiegelhalter Z-statistic and Brier score asafunction of d/n ratio. As d/n
increases from 0.005 to 0.50, Z grows monotonically from the acceptance region into catas-
trophic miscalibration, confirming the Θ(d/n) overconfidence predicted by [5]. The Brier
score degrades in parallel. The dashed horizontal lines mark the ±1.96 significance thresh-
old.

Warning: The fallacy of predict_proba
The most widely used machine learning library in the world has, for overadecade,
given practitioners incomplete guidance on the calibration of logistic regression
probabilities.
Historical claim (scikit-learn v 0.17–v 0.22, circa 2015–2020):
"Logistic Regression returns well calibrated predictions by default as
it directly optimizes log-loss."
Current claim (scikit-learn v 1.3.2–v 1.8.0, 2024–2026):
"Logistic Regression returns well calibrated predictions by default as
it directly optimizes log-loss. In addition it hasacanonical link
function for its loss, the so-called balance property..."
The "balance property" cited in the documentation is real but insufficient: it guar-
antees only marginal calibration—that the average predicted probability equals the
average observed frequency—not auto-calibration at every probability level. Formally,
the balance property ensures
E[pˆ(X)]=E[Y],
but provides no guardrails against the
Θ(d/n) shrinkage of individual probabilities toward the extremes. The required con-
dition for trustworthy probabilities is the strictly stronger E[Y |pˆ(X)=p]=p for all
p. The documentation conflates the two. It also heavily cites [61], a study conducted
on low-dimensional datasets (d≪n) where the Θ(d/n) bias is negligible. Modern ML
operates inadifferent regime.
The method name predict_proba itself is grossly misleading. It suggests the
output isaprobability in the calibrated, frequentist sense. In reality, it re-
turns normalised scores: numbers in [0,1] that sum to one but bear no guar-
anteed correspondence to observed event frequencies. This is equally true
for predict_proba in Random Forest Classifier, Gradient Boosting Classifier,
SVC(probability=True), and every other scikit-learn estimator.
Rule: treat every predict_proba output as an uncalibrated score until proven
otherwise through rigorous calibration diagnostics on held-out data.
5.1.3.1 Persistence of Incomplete Guidance: scikit-learn v 1.3.2 – v 1.8.0
A systematic audit of scikit-learn's calibration documentation across six major releases
reveals that the incomplete guidance is notarelic of an earlier era—it has persisted
unchanged through every version. Table 5.1 documents the persistence.
Beyond the headline claim, the documentation contains four categories of incomplete or
misleading guidance:
1. The Proper Scoring Rule Fallacy. The documentation implies that optimising log-
loss (a strictly proper scoring rule) isasufficient condition for calibration. This
conflates two issues. First, proper scoring rules guarantee calibration only in the limit
of infinite data foraperfectly specified model; in finite samples, the Θ(d/n) bias from
[5] ensures structural overconfidence regardless of the loss function. Second, the
Brier score decomposes into reliability (calibration), resolution (discrimination), and
uncertainty. Amodelwithhighresolutionbutpoorreliabilitycanachievealower Brier
score thanawell-calibrated model with moderate resolution. Proper scoring rules are
therefore insufficient for assessing calibration without explicit decomposition [15].
2. The Regularisation Blind Spot. scikit-learn's Logistic Regression applies L 2 regu-

5.1 Why Classification Needs Calibration 127
Table 5.1: Persistence of the "well calibrated by default" claim across scikit-learn releases.
Status: the claim has remained active and substantively unchanged across every audited
version.
Version LR claim Notes
v 1.3.2 Active "Balance property" and "canonical link function" cited as
justification;heavyrelianceon[61].
v 1.4.2 Active Unchanged. "Balanceproperty"claimstillcentral.
v 1.5.2 Active Minorformatting;"balanceproperty"narrativepersists.
v 1.6.1 Active "Balanceproperty"stillciteddespite[5]counter-evidence.
v 1.7.2 Active Brierscoredetailsadded,butLRstillusedasthe"calibrated"
anchorinallvisualisations.
v 1.8.0 Active Temperaturescalingintroduced,butnowarningthatitdoes
notfixthestructuralΘ(d/n)biasof LR.
larisation by default (C=1.0), yet the calibration documentation discusses probabilities
without acknowledging that regularisation introduces systematic bias. Shrinkage
pushes probabilities toward 0.5; the very architecture of the default estimator is
designed to be miscalibrated.
3. The One-vs-Rest Multi-class Failure. Formulti-classcalibration,thedocumentation
suggests calibrating each class separately inaone-vs-rest scheme and normalising
so the probabilities sum to one. This ignores the interdependence of class scores.
Pairwise coupling (PKPD) [51] and Venn–Abers extensions provide significantly more
reliable multi-class probability vectors than naive post-hoc normalisation.
4. The Anchor Bias in Visualisations. Every version of the calibration documentation
includesa"Calibration Curve"exampleinwhichlogisticregressionappearsasanearly
perfect 45◦ diagonal. This visual anchoring creates confirmation bias: practitioners
who see their model's reliability curve resembling the LR curve in the documentation
conclude they are "safe." On high-dimensional tabular data, the LR curve itself is
bowed—a fact the documentation's low-dimensional example cannot reveal.
Table 5.2 summarisesthegapbetweenscikit-learn'sdocumentationclaimsandthescientific
evidence.
5.1.4 Three Distinct Objects Practitioners Conflate
The confusion above stems from conflating three distinct objects that arise in classification:
1. Probabilistic prediction. A full vector pˆ(x)=(pˆ 1 (x),...,pˆ K (x)) of class probabilities.
This is what predict_proba claims to return.
2. Calibration. A property of those probabilities, measured by proper scoring rules and
calibration diagnostics. Calibration is notamethod—it isastatement about empirical
reliability: among all instances where the model predicts pˆ k (x)=0.7, approximately
70% should belong to class k.
3. Conformal classification. A set-valued prediction C α (x)⊆Y withauser-chosen
error rate α, guaranteed under exchangeability by construction—not by assumption.
Calibration improves the meaning of probabilities. Conformal prediction guarantees the
validity of uncertainty statements. Combiningthemimprovesdecisionqualityandefficiency:
calibrated probabilities yield smaller prediction sets at the same target coverage, clearer

Table 5.2: Audit of scikit-learn documentation claims against the calibration literature.
scikit-learn claim Verdict Reason
"LR is well-calibrated by Incorrect Structural bias Θ(d/n) makes LR over-
default" confidentinfiniteandhigh-dimensional
samples[5].
"Optimising log-loss en- Misleading Log-lossisaproperscoringrule,butmin-
surescalibration" imisingitontrainingdata̸=calibration
ontestdata. Finite-samplebiasandmis-
specificationpersist.
"Properly regularised LR Incorrect L 2 regular is ation injects shrinkage bias
iscalibrated" toward 0.5. There is no regularisation
strength that simultaneously eliminates
overconfidence and avoids underconfi-
denceacrossallregimes.
"L Risthecalibrationbase- Overstated Niculescu-Miziland Caruana[61]showed
line" L Rislessmiscalibrated than SV Msand
Naïve Bayes—not that it is calibrated.
Thestudyusedlow-dimensionaldatasets
whered/n≈0.
risk control, and better model selection. This chapter develops each thread and shows how
they connect.
5.1.5 Chapter Roadmap
This chapter develops three ideas that together form the modern reliability framework for
classification:
1. Probability calibration (Sections 5.2–5.3). We formalise what it means for predicted
probabilities to be trustworthy, introduce proper scoring rules as the principled
evaluation metric, and examine the principal post-hoc calibration operators—Platt
scaling, isotonic regression, beta calibration, and temperature scaling—alongside
recent empirical evidence that calibration can degrade strong models when applied
blindly [54].
2. Venn predictors and Venn–Abers calibration (Section 5.4). We introduce Venn
predictors—probabilisticanaloguesofconformalpredictors—whichproducecalibrated
multiprobability intervals with provable long-run calibration guarantees under ex-
changeability, valid for any calibration set size. Venn–Abers predictors, their most
practical implementation, combine isotonic regression with the conformal machin-
ery to yield calibrated probabilities that provably outperform classical calibrators on
modern benchmarks.
The chapter concludes withacomplete evaluation protocol (Section 5.6), an end-to-end
workedexample(Section 5.7),andapracticaldecisionguideforwhentocalibrateandwhen
to proceed directly to conformal prediction (Section 5.8).
Notationconvention. Throughoutthischapter,pˆ k (x)denotestheraw predictedprobability
of classkfrom the base model. p˜ k (x) denotes the calibrated probability after post-hoc
adjustment. Nonconformity scores are denoted s i (not α i, to avoid collision with the
significance level α). The calibration set size is denotedmthroughout; n denotes the
training set size or, in the Venn prediction framework, the length of the prediction sequence.

5.2 A Brief History of Probability Calibration 129
5.2 A Brief History of Probability Calibration
Theideathatpredictedprobabilitiesshouldmatchobservedfrequenciesdidnotoriginatein
machinelearning. Itemergedfrommeteorology,wasformalisedbystatisticians,laydormant
during the accuracy-obsessed era of the 1990 s and 2000 s, and was rediscovered—with
urgency—when deep learning produced confidently wrong predictions at scale.
This section traces that arc, from weather forecasts to modern neural networks, and
identifies the key milestones that define the calibration problem as we understand it today.
5.2.1 Meteorological Origins (1950–1977)
Calibration was first studied quantitatively in the context of weather forecasting, where
staking decisions on probability estimates has immediate, measurable consequences.
The Brier score (1950). Brier [14] introduced the proper scoring rule for probability
forecasts:
BS= 1 X N (pˆ i −y i )2, (5.1)
N
i=1
where pˆ i is the forecast probability and y i ∈{0,1} is the observed outcome. The Brier score
penalises both overconfidence and underconfidence quadratically, rewarding forecasters
whose probabilities closely match event frequencies. It remains the standard calibration
metric for binary classification seven decades later.
Reliability diagrams and decomposition (1973–1977). Murphy [59] and Murphy and
Winkler [60] developed the reliability diagram—a plot of observed frequencies against
predicted probabilities—and the fundamental decomposition of the Brier score into three
orthogonal components:
BS= REL − RES + UNC . (5.2)
|{z} |{z} | {z }
reliability resolution uncertainty
Forapartition of the predictions into bins k=1,...,K, with n k observations in bin k, total
sample size N, observed event frequency o k in bin k, mean forecast probability p k in bin k,
and overall event rate
1 X N
o¯= y ,
i
N
i=1
the three terms are
REL= X K n k (p −o )2,
k k
N
k=1
RES= X K n k (o −o¯)2,
k
N
k=1
UNC=o¯(1−o¯).
The interpretation of RES must be stated carefully. Resolution does not measure how much
the predicted probabilities themselves differ from the base rate. Rather, it measures how
much the observed outcome frequencies in the prediction groups differ from the overall
eventrate. Equivalently,resolutionquantifieshowwelltheforecastseparatesthepopulation
into groups with different empirical event frequencies.
Thus:

• REL measures lack of calibration, that is, the discrepancy between forecast probabili-
ties and observed frequencies within bins; smaller values are better.
• RES measures discriminatory power in the sense of separating observations into
groups with different observed event frequencies; larger values are better.
• UNC measures the intrinsic variability of the outcome and depends only on the
marginal event rate.
This decomposition is critical because it reveals thatamodel can achievealow Brier
score through high resolution despite poor calibration—the exact failure mode that plagues
modern neural networks. A model with excellent discrimination but terrible reliability will
rank instances correctly while assigning meaningless probability values.
Why the Brier Decomposition Matters
Consider two models onadataset with base rate 10%:
Model A: High resolution, poor reliability. Assigns probabilities concentrated near
0.0 and 0.8, with good separation between positives and negatives, but the "0.8"
predictions have true frequency 0.5. BS=0.12.
Model B: Moderate resolution, excellent reliability. Assigns probabilities spread
across [0,0.4] that closely match observed frequencies. BS=0.14.
ModelAhas the better Brier score, yet its probabilities are unreliable—decisions
based on "80% chance" when the true chance is 50% will systematically misallocate
resources. ModelBis the safer basis for decision-making. This is why calibration
diagnostics must go beyond aggregate scores.
The Spiegelhalter calibration score (1986). While the Brier score measures overall
predictive accuracy, it does not directly test whether forecast probabilities are statistically
consistent with the observed outcomes. Spiegelhalter [76] proposedadiagnostic score
that evaluates whether the observed outcomes deviate systematically from the predicted
probabilities.
Let pˆ i denote the forecast probability for observationiand y i ∈{0,1} the observed outcome.
Under perfect calibration, we should have
E[y |pˆ]=pˆ.
i i i
Spiegelhalter derivedastandardised statistic from the derivative of the expected Brier
score under the null hypothesis of perfect calibration:
PN (y −pˆ)(1−2 pˆ)
Z = i=1 i i i . (5.3)
q
PN (1−2 pˆ)2 pˆ(1−pˆ)
i=1 i i i
Under H 0, Z is asymptotically standard normal; rejecting at the 5% level requires |Z|>1.96.
The (1−2 pˆ i ) weighting arises because the Brier score is more sensitive to miscalibration at
extreme probabilities than near 0.5.
Unlikethe Brierscore, whichconflatescalibrationandresolution,the Spiegelhalterstatistic
isolates calibration error directly by testing whether the predicted probabilities match
observed frequencies.

5.2 A Brief History of Probability Calibration 131
Limitations. The Spiegelhalter test evaluates calibration only in the aggregate. A model
may pass the test while still being poorly calibrated in specific probability regions. Conse-
quently, graphical diagnostics such as reliability diagrams remain essential complements to
statistical tests.
In modern machine learning practice this statistic is rarely used, despite being one of the
few formal hypothesis tests of probabilistic calibration.
5.2.2 Formalisation and Statistical Foundations (1982–1983)
Dawid's calibration framework (1982). Dawid [18] provided the first rigorous fre-
quentist definition of calibration forasequence of probability forecasts. A forecaster is
well-calibrated if, among all instances where the forecast is p, the long-run frequency of the
event converges to p. Formally:
E[Y |pˆ(X)=p]=p for all p∈[0,1]. (5.4)
This is auto-calibration—the strong form of calibration that Section 5.1.2 showed logistic
regression fails to achieve. Dawid distinguished this from marginal calibration (
E[pˆ]=E[Y]),
which is the weaker condition that the "balance property" of GL Ms actually guarantees.
De Groot and Fienberg (1983). De Groot and Fienberg [19] formalised the comparison of
probability forecasters, introducing the notion of calibration refinement: forecaster Aisbet-
tercalibratedthanBif A'sreliabilitydiagramliesclosertothediagonalforeveryprobability
level. Theyprovedthatcalibrationandrefinement(resolution)areorthogonal properties—a
result that anticipated the modern understanding that accuracy and calibration are largely
independent.
Proper scoring rules: the theoretical foundation. Gneiting and Raftery [27] provided
the definitive mathematical treatment of proper scoring rules, proving thatascoring rule
is strictly proper if and only if the forecaster's expected score is uniquely minimised by
reporting the true probabilities. Their characterisation theorem unifies the Brier score,
log-loss, and the continuous ranked probability score (CRPS) underasingle framework and
provides the theoretical backbone for every calibration metric used in this chapter.
5.2.3 The Machine Learning Era: Post-Hoc Calibration (1999–2005)
As machine learning models moved from research to production in the late 1990 s, practi-
tioners discovered that classifier scores were not probabilities—and needed to be fixed.
Platt scaling (1999). Platt [66] proposed fittingasigmoid function to the outputs of a
trained SVM to produce calibrated probabilities:
p˜(x)= , (5.5)
1+exp(A·s(x)+B)
where s(x) is the SVM decision score and A,B are learned onaheld-out calibration set by
minimising negative log-likelihood. Platt scaling isaparametric calibrator: it assumes that
the uncalibrated scores are related to probabilities throughasigmoid transformation. This
assumption is reasonable for SV Ms (whose margins are approximately sigmoid-distributed),
butitfailswhenthescoredistributionisnon-sigmoidal—whichisthecasefortreeensembles,
neural networks, and most other classifiers.
1 import numpy as np
2 from sklearn.svm import SVC
3 from sklearn.calibration import Calibrated Classifier CV
4 from sklearn.datasets import make_classification

5 from sklearn.model_selection import train_test_split
7 # Synthetic data
8 X, y = make_classification(n_samples=5000, n_features=20,
n_informative=10, random_state=42)
10 X_tr, X_test, y_tr, y_test = train_test_split(
11 X, y, test_size=0.3, random_state=42)
13 # Uncalibrated SVM
14 svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_tr, y_tr)
16 raw_scores = svm.decision_function(X_test)
18 # Platt scaling via Calibrated Classifier CV
19 platt_svm = Calibrated Classifier CV(svm, method='sigmoid', cv=5)
platt_svm.fit(X_tr, y_tr)
21 platt_probs = platt_svm.predict_proba(X_test)[:, 1]
23 print(f"Raw score range: [{raw_scores.min():.2 f}, {raw_scores.max():.2 f}]")
24 print(f"Platt prob range: [{platt_probs.min():.3 f}, {platt_probs.max():.3 f}]")
Listing 5.1: Platt scaling: fittingasigmoid to raw scores
Histogram binning and isotonic regression (2001–2002). Zadrozny and Elkan intro-
duced two nonparametric alternatives. Histogram binning [93] partitions the score range
into bins and replaces each score with the empirical frequency within its bin. Isotonic
regression [91] fitsamonotonically non-decreasing step function to the scores, preserving
their ranking while remapping them to observed frequencies.
Isotonic regression is more flexible than Platt scaling because it makes no parametric
assumption about the score distribution. However, it requires more calibration data to
avoid overfitting (a step function withksteps haskfree parameters, whereas Platt scaling
has exactly two). The choice between parametric and nonparametric calibration remains a
practical trade-off that depends on the calibration set size and the score distribution.
1 from sklearn.calibration import Calibrated Classifier CV
2 from sklearn.ensemble import Random Forest Classifier
4 # Uncalibrated Random Forest
5 rf = Random Forest Classifier(n_estimators=200, random_state=42)
rf.fit(X_tr, y_tr)
7 raw_probs = rf.predict_proba(X_test)[:, 1]
9 # Isotonic calibration
10 iso_rf = Calibrated Classifier CV(rf, method='isotonic', cv=5)
iso_rf.fit(X_tr, y_tr)
12 iso_probs = iso_rf.predict_proba(X_test)[:, 1]
14 print(f"Raw predict_proba range: [{raw_probs.min():.3 f}, {raw_probs.max():.3 f}]")
15 print(f"Isotonic prob range: [{iso_probs.min():.3 f}, {iso_probs.max():.3 f}]")
Listing 5.2: Isotonic regression calibration
The Niculescu-Mizil and Caruana survey (2005). Niculescu-Mizil and Caruana [61]
conducted the first large-scale empirical comparison of classifier calibration. Their key
findings—that boosted trees push probabilities toward the extremes, random forests com-
pressthemtoward 0.5,SV Msaresigmoid-shaped,andnaïve Bayesproducesacharacteristic
"push-to-the-sides" distortion—became the canonical reference for calibration behaviour
across model families.

5.2 A Brief History of Probability Calibration 133
Two findings from this study have been widely misinterpreted. First, the claim that logistic
regression is "well calibrated" was based on low-dimensional datasets where the Θ(d/n)
bias from Bai et al. [5] is negligible (see Section 5.1.2). Second, the claim that shallow
neural networks are "reasonably calibrated" was overturned by Johansson and Gabrielsson
[36], who showed that onabroader set of 25 datasets, both ML Ps and neural network
ensembles were often poorly calibrated.
The study's lasting contribution is not its specific conclusions about individual classifiers—
which were artefacts of the low-dimensional benchmarks of 2005—but the methodology:
reliability diagrams paired with proper scoring rules as the standard evaluation protocol for
calibration.
5.2.4 Modern Calibration Methods (2017–Present)
Thedeeplearningerabroughtcalibrationbackintofocus. Asmodelsgrewdeeper,wider,and
more overparameterised, their probability estimates became progressively less trustworthy.
Beta calibration (2017). Kull et al. [42] proposed beta calibration, which models the
mapping from raw scores to calibrated probabilities usingabeta distribution:
p˜(x)= , (5.6)
(cid:16) (cid:17)−(a−b) (cid:16) (cid:17)a
1+ 1 · pˆ(x) · 1
exp(c) 1−pˆ(x) pˆ(x)
with three parameters a, b, and c. Beta calibration generalises Platt scaling (which is
the special case a=b) and can handle asymmetric distortions that the sigmoid cannot. It
is particularly effective for classifiers that produce scores concentrated near 0 and 1—a
common pattern in boosted trees and deep networks.
The deep calibration crisis (2017). Guo et al. [32] demonstrated that modern deep
neural networks are systematically overconfident: as depth, width, batch normalisation,
and weight decay increase, accuracy improves but calibration degrades. They introduced
temperature scaling, a single-parameter variant of Platt scaling:
exp(s (x)/T)
k
p˜ k (x)= PK exp(s (x)/T) , (5.7)
j=1 j
where T > 0 isascalar temperature learned on the calibration set. T > 1 softens the
distribution (reducing overconfidence); T <1 sharpens it. Temperature scaling is attractive
because of its simplicity—one parameter, preserves the top-1 prediction—but it can only
correct global miscalibration. If different regions of the input space are miscalibrated in
different directions, a single temperature cannot fix them all.
Beyond expected calibration error. The standard evaluation metric during this period—
expected calibration error (ECE)—was shown to be deeply flawed. Vaicenavicius et al. [81]
proved that binning-based ECE estimators are biased and inconsistent, depending heavily
on the number of bins and the binning scheme. Kumar et al. [43] demonstrated that ECE
canbemadearbitrarilysmallbyadversarialbinning,renderingitunreliableasastandalone
diagnostic. This motivated the use of multiple diagnostics in combination: reliability
diagrams, the Spiegelhalter Z-statistic [77], proper scoring rules with decomposition, and
calibration tests with well-defined statistical properties.
5.2.5 Experiment: Comparing Classical Calibrators Head-to-Head
To make the historical progression concrete, we compare five post-hoc calibrators—Platt
scaling, isotonic regression, beta calibration, temperature scaling, and Venn–Abers—on a
single controlled task.

Experimental design. We generateasynthetic binary classification problem with d=200
features (10 informative, 10 redundant), n=5,000 samples, and 95:5 class imbalance. The
d/n=0.04 ratio is lower than the Section 5.1.2 experiment (d/n=0.10), yet—as the results
confirm—raw logistic regression remains significantly overconfident even in this milder
regime. We use n = 5,000 (rather than n = 2,000) to ensure that every calibration split
contains at least ∼50 positive instances, avoiding the small-sample artefacts that plague
calibration comparisons under heavy class imbalance.
Fair data budgets. A three-way stratified split allocates 60% to training (3,000 samples),
20% to calibration (1,000), and 20% to testing (1,000). A single logistic regression—the
shared base model—is trained on the training set. Platt scaling and isotonic regression
use cv='prefit' mode: they take the already-fitted LR and calibrate its outputs on the
calibration set. Beta calibration and temperature scaling calibrate on the shared LR's
scores on the calibration set. Venn–Abers receives the combined train+cal data (4,000
samples) with cal_size=0.25, so its internal split produces ∼3,000 for training and ∼1,000
for calibration—exactly matching the data budget of the other methods.
1 import numpy as np
2 from sklearn.linear_model import Logistic Regression
3 from sklearn.calibration import Calibrated Classifier CV
4 from sklearn.datasets import make_classification
5 from sklearn.model_selection import train_test_split
6 from sklearn.metrics import brier_score_loss, log_loss
7 from venn_abers import Venn Abers Calibrator
8 from betacal import Beta Calibration
10 # Data: d=200, n=5000, d/n=0.04, 95:5 imbalance
11 X, y = make_classification(n_samples=5000, n_features=200,
n_informative=10, n_redundant=10,
13 weights=[0.95, 0.05], random_state=42)
15 # Three-way split: train (60%) / cal (20%) / test (20%)
16 X_train, X_tmp, y_train, y_tmp = train_test_split(
17 X, y, test_size=0.4, random_state=42, stratify=y)
18 X_cal, X_test, y_cal, y_test = train_test_split(
X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
21 # Shared base model -- all calibrators use this LR
22 lr = Logistic Regression(max_iter=2000, random_state=42)
lr.fit(X_train, y_train)
25 # Platt and Isotonic: cv='prefit' with shared LR
26 platt = Calibrated Classifier CV(lr, method='sigmoid', cv='prefit')
platt.fit(X_cal, y_cal)
29 iso = Calibrated Classifier CV(lr, method='isotonic', cv='prefit')
iso.fit(X_cal, y_cal)
32 # Venn-Abers: matched data budget (train+cal, cal_size=0.25)
33 X_full = np.vstack([X_train, X_cal])
34 y_full = np.concatenate([y_train, y_cal])
35 va = Venn Abers Calibrator(
estimator=Logistic Regression(max_iter=2000, random_state=42),
37 inductive=True, cal_size=0.25, random_state=42)
va.fit(X_full, y_full)
Listing 5.3: Fair head-to-head calibrator comparison (d=200, n=5,000, d/n=0.04)

5.2 A Brief History of Probability Calibration 135
The companion Colab notebook for this section contains the complete code for all six
methods—including beta calibration [42], temperature scaling [32], and the Spiegelhalter
Z-statistic computation—together with reliability diagrams for each method. The notebook
is available in the Pro edition:
https://valeman.gumroad.com/l/applied_conformal_prediction_pro
Figure 5.4: Reliability diagrams for five post-hoc calibrators and the raw baseline on
the high-dimensional task (d=200, n=5,000, d/n=0.04). Raw logistic regression (red,
Z =+3.12) shows systematic overconfidence. Platt scaling (orange) and beta calibration
(purple) track the diagonal closely. Isotonic regression (blue) achieves the best Z-statistic
but overfits in sparse bins. Temperature scaling (brown) is the weakest calibrator. Venn–
Abers (green) provides calibration with provable long-run sequential guarantees.
Table 5.3 summarises the quantitative results.
The results confirm the central claim of Section 5.1.2: raw logistic regression is not
calibrated even at the modest ratio d/n=0.04 (Z =+3.12, rejecting the null of perfect
calibration at the 5% level). All five post-hoc calibrators bring the Z-statistic inside the
±1.96 acceptance region.
Beta calibration achieves the lowest Brier score (0.029) and log-loss (0.114), making it
the strongest classical calibrator in this regime. Its three-parameter family handles the
asymmetric distortions thatatwo-parameter sigmoid cannot. Isotonic regression shows
the highest log-loss (0.180) despiteanear-perfect Z-statistic (+0.20). This isaknown
artefact: with only ∼50 positives in the calibration set, the step function overfits isolated
bins,producingextremeprobabilityvaluesthatarepenalisedheavilybythelogarithmicloss.
Temperature scaling, constrained toasingle global parameter, is the weakest calibrator in
this comparison; it cannot correct local distortions that the nonparametric methods handle
naturally.

Table 5.3: Head-to-head comparison of post-hoc calibrators on the high-dimensional task
(d=200,n=5,000,d/n=0.04). Lower Brierscoreandlog-lossarebetter;|Z|<1.96 indicates
calibration at the 5% significance level.
Method Brier Log-loss Z-stat Calibrated?
Raw LR 0.031 0.129 +3.12 No
Plattscaling 0.029 0.116 −1.01 Yes
Isotonicregression 0.030 0.180 +0.20 Yes
Betacalibration 0.029 0.114 −0.41 Yes
Temperaturescaling 0.031 0.121 −1.72 Yes
Venn–Abers 0.034 0.131 −0.94 Yes
Venn–Abers calibration reaches Z =−0.94 with competitive Brier and log-loss. Its slightly
higher Brierscore(0.034 vs.0.029 forbeta)isthepriceofitsuniqueadvantage: itistheonly
method in this table with provable long-run calibration guarantees under exchangeability.
No other calibrator offers this safety profile. Section 5.4 develops the theory behind this
guarantee.
5.2.6 The Conformal Turn: Venn Predictors (2003–2005)
While the machine learning community was developing post-hoc calibrators, an entirely
different approach to probability estimation was taking shape in the conformal prediction
framework.
From conformal sets to probability estimates. Vovk, Gammerman, and Shafer [86]
introduced Venn predictors—probabilistic analogues of conformal predictors—which pro-
duce multiprobability predictions: notasingle probability vector, butaset of probability
distributions, each provably well-calibrated in the long run under exchangeability.
The key insight is that conformal prediction's finite-sample validity guarantees can be
transferred from set-valued predictions to probability estimates. Where classical calibrators
are asymptotically valid (requiring infinite calibration data), Venn predictors are finitely
valid: the calibration guarantee holds for any sample size, not just in the limit.
Venn–Abers predictors (2014). Vovk and Petej [87] introduced Venn–Abers predictors,
which combine isotonic regression with the Vennpredictionframeworktoproducecalibrated
probability intervals. For each test instance, the predictor outputsapair [p 0 ,p 1 ] where:
• p 0 is the calibrated probability assuming the true label is 0,
• p 1 is the calibrated probability assuming the true label is 1.
The interval [p 0 ,p 1 ] is typically narrow (indicating confident calibration) or wide (indicating
uncertainty about the true probability). The final point prediction is obtained by interpola-
tion.
Venn–Abers predictors have two properties that no classical calibrator can match:
1. Sequential calibration. The calibration guarantee holds for the entire prediction
sequence, not just asymptotically.
2. Automatic adaptation. The isotonic regression component adapts to the local score
distribution, handling asymmetric and non-monotonic distortions without parametric
assumptions.
Section 5.4 develops the theory and practice of Venn–Abers predictors in full detail.

5.3 Post-Hoc Calibration Operators 137
5.2.7 The Current Frontier
The calibration landscape in 2025 is defined by three converging threads:
1) Calibration at scale. Manokhin and Grønhaug [54] evaluated calibration across 72
datasetsandsevenmodelfamilies,demonstratingthatno baseclassifierproducescalibrated
probabilities and that Venn–Abers calibration provides consistent improvements across all
architectures. This study shifted the conversation from "which classifiers are calibrated?"
(answer: none) to "which calibration method is most reliable?" (answer: Venn–Abers).
2) Conditional and multivalid calibration. Gopalan et al. [29] and Gupta et al. [33]
developed notions of multivalid and multi-group calibration that require calibration to
hold not just marginally but across all subgroups defined byarich collection of tests.
This addressesafundamental limitation of classical calibration: a model can be perfectly
calibrated overall while being catastrophically miscalibrated for minority subgroups.
3) Integration with conformal prediction. The recognition that calibration and con-
formal prediction are complementary—not competing—has driven recent work on using
calibrated probabilities as the basis for adaptive conformal methods (APS [72], RAPS [4]).
Better-calibrated probabilities yield smaller prediction sets at the same coverage level,
creatingadirect pathway from calibration quality to decision efficiency.
The remainder of this chapter develops each of these threads: the formal machinery of
calibration and proper scoring rules (Section 5.3) and the theory of Venn and Venn–Abers
predictors (Section 5.4).
5.3 Post-Hoc Calibration Operators
Section 5.2 traced the historical development of calibration methods. This section treats
themasmathematicaloperators—mapsfromuncalibratedscorestocalibratedprobabilities—
and examines their formal properties, guarantees, failure modes, and the conditions under
which post-hoc calibration can actually degradeamodel's performance.
5.3.1 Calibration: Formal Definitions
A binary classifier producesapredicted probability pˆ(X)∈[0,1] for the positive class given
input X. The three levels of calibration formastrict hierarchy.
Marginal calibration. The weakest requirement is that the average prediction matches
the average outcome:
E[pˆ(X)]=E[Y].
(5.8)
This is the "balance property" that generalised linear models satisfy by construction [53]. It
says nothing about the reliability of individual predictions.
Calibration (weak). A predictor is calibrated if, for each predicted probability level p, the
conditional expectation of the outcome matches p:
E[Y |pˆ(X)=p]=p for all p∈[0,1]. (5.9)
This is the condition that reliability diagrams visualise. A perfectly calibrated predictor
producesareliability curve that lies exactly on the 45◦ diagonal.
Auto-calibration (strong). A predictor pˆis auto-calibrated if
E[Y |pˆ(X)]=pˆ(X) almost surely. (5.10)
Auto-calibration implies weak calibration but is strictly stronger: it requires that the
calibration property holds not just at each probability level but asapointwise identity. No

finite-sample estimator achieves auto-calibration exactly; the goal of post-hoc calibration is
to approximate it as closely as possible.
The hierarchy. The three conditions formastrict chain:
Auto-calibration =⇒ Calibration =⇒ Marginal calibration.
Neither implication reverses. A marginally calibrated predictor can be catastrophically
wrong at every individual probability level—as the logistic regression experiments in Sec-
tion 5.1.2 demonstrated.
5.3.2 Proper Scoring Rules and Their Decompositions
A scoring rule S(pˆ,y) assignsanumerical penalty toaprobability forecast pˆ given the
observed outcome y∈{0,1}.
Properness. A scoring rule is proper if the expected score is minimised (for negatively
oriented rules) or maximised (for positively oriented rules) when the forecast equals the
true probability:
E[S(p∗,Y)]≤E[S(q,Y)]
for all
q̸=p∗,
(5.11)
where p∗ =P(Y =1|X) is the true conditional probability. The rule is strictly proper if
equality holds only when q=p∗. Strictly proper scoring rules incentivise honest reporting—
any deviation from the true probability increases the expected penalty.
The Brier score. The Brier score (Equation 5.1) is the canonical proper scoring rule for
binary outcomes:
BS= 1 X N (pˆ
i
−y
i
)2.
N
i=1
Its Murphy decomposition (Equation 5.2) separates calibration from discrimination:
BS= 1 X B n
b
(p¯
b
−y¯
b
)2− 1 X B n
b
(y¯
b
−y¯)2+ y¯(1−y¯) ,
N N
| {z }
b=1 b=1
| {z } | {z } Uncertainty(UNC)
Reliability(REL) Resolution(RES)
where the sum is overBbins, n b is the bin count, p¯ b is the mean predicted probability in bin
b, y¯ b is the observed frequency in bin b, and y¯is the overall positive rate.
A post-hoc calibrator that only improves calibration reduces the reliability term without
affecting resolution. A calibrator that distorts the score ordering can reduce resolution,
potentially increasing the overall Brier score despite improving reliability. This is the
mechanism by which calibration can hurt.
The log-loss. The log-loss (cross-entropy) is the other workhorse scoring rule:
1 X N (cid:2) (cid:3)
LL=− y i logpˆ i +(1−y i )log(1−pˆ i ) . (5.12)
N
i=1
Log-loss is strictly proper and penalises confident wrong predictions exponentially. It
decomposes into calibration and refinement components [15], but the decomposition is less
intuitive than the Brier decomposition because the terms are not orthogonal in the same
additive sense.
The Spiegelhalter Z-statistic. Spiegelhalter[77]proposedastatisticaltestforcalibration
based on the Brier score:
PN (y −pˆ)(1−2 pˆ)
Z = i=1 i i i . (5.13)
q
PN (1−2 pˆ)2 pˆ(1−pˆ)
i=1 i i i

5.3 Post-Hoc Calibration Operators 139
Under the null hypothesis of perfect calibration, Z is asymptotically standard normal.
Rejecting H 0 at the 5% level requires |Z|>1.96. This test has the advantage of producing a
single scalar summary withawell-defined statistical interpretation, unlike ECE which lacks
a known sampling distribution [81].
5.3.3 The Calibration Operator Framework
Every post-hoc calibrator can be viewed asacalibration operatorTthat maps uncalibrated
scores to calibrated probabilities:
T :[0,1]→[0,1], pˆ7→p˜=T(pˆ). (5.14)
The operator is learned onaheld-out calibration setDcal = {(pˆ i ,y i )}m i=1 and applied to
test-time predictions. Different calibrators correspond to different function classes for T.
Table 5.4 summarises the five principal calibration operators.
Table 5.4: Properties of the five principal post-hoc calibration operators. "Parameters"
counts the free parameters learned on the calibration set. "Preserves ranking" indicates
whether pˆ a >pˆ b =⇒ p˜ a ≥p˜ b. "Validity" describes the theoretical calibration guarantee.
Operator Parameters Preserves Validity Key
ranking? weakness
Plattscaling 2(A,B) Yes Asymptotic Assumessigmoid
Isotonicregression O(m) Yes Asymptotic Overfitssmallm
Betacalibration 3(a,b,c) Yes Asymptotic Parametric
Temperaturescaling 1(T) Yes Asymptotic Globalonly
Venn–Abers O(m) Yes Sequential Needsbaseestimator
5.3.4 Platt Scaling as an Operator
Platt scaling [66] definesTasalogistic sigmoid:
T (s)= , (5.15)
Platt 1+exp(A·s+B)
wheresis the uncalibrated score (or logit) and A,B are learned by minimising negative
log-likelihood on D .
cal
When Platt scaling works. The sigmoid assumption is appropriate when the uncalibrated
scores are approximately linearly related to the log-odds of the true probability. This
holds for SV Ms (whose margins have sigmoid-shaped score distributions) and for logistic
regression with moderate miscalibration.
When Platt scaling fails. If the true calibration map is non-sigmoidal—asymmetric, non-
monotonic,ormultimodal—thetwo-parametersigmoidcannotcapturethedistortion. Thisis
common for tree ensembles (whose score histograms have characteristic peaks near 0 and
1) and deep networks (whose logit distributions can be irregular). Platt scaling applied to a
random forest typically undercorrects near the extremes and overcorrects near the centre.
5.3.5 Isotonic Regression as an Operator
Isotonic regression [91] definesTas the best monotonically non-decreasing step function:
m
T iso =argmin X (g(pˆ i )−y i )2, (5.16)
g∈G
↑i=1

where G ↑ is the class of non-decreasing functions [0,1]→[0,1]. The Pool Adjacent Violators
(PAV) algorithm solves this in O(m) time.
Strengths. Isotonic regression makes no parametric assumptions about the shape of the
calibration map. It can capture any monotonic distortion, however irregular. With sufficient
calibration data, it converges to the true calibration function at rate O(m−1/3) [9].
Weaknesses. The step function has O(m) free parameters in the worst case. On small
calibrationsets(m<500),thestepfunctionoverfits: bins in the tails of the score distribution
contain few observations, and the empirical frequency in those bins is noisy. This manifests
as jagged reliability diagrams with high variance in the extreme probability regions.
1 import numpy as np
2 from sklearn.isotonic import Isotonic Regression
3 from sklearn.metrics import brier_score_loss
5 np.random.seed(42)
6 # True calibration function: mild non-linear distortion
7 true_p = np.linspace(0.01, 0.99, 1000)
8 predicted_p = 1 / (1 + np.exp(-2.5 * (np.log(true_p / (1 - true_p)))))
10 formin [100, 500, 2000, 10000]:
11 # Simulate calibration set of size m
12 idx = np.random.choice(len(true_p), size=m, replace=True)
13 cal_scores = predicted_p[idx]
14 cal_labels = np.random.binomial(1, true_p[idx])
16 # Fit isotonic regression
17 iso = Isotonic Regression(out_of_bounds='clip')
iso.fit(cal_scores, cal_labels)
19 calibrated = iso.predict(predicted_p)
21 # Evaluate on full grid
22 test_labels = np.random.binomial(1, true_p)
23 bs = brier_score_loss(test_labels, calibrated)
24 print(f"m={m:>5 d} Brier={bs:.4 f}")
Listing 5.4: Comparing calibration set size effects on isotonic regression
The listing above illustratesakey practical concern: isotonic regression requires at least
500–1,000 calibration instances to reliably outperform parametric alternatives. Below this
threshold, Platt scaling or beta calibration may be preferable despite their parametric
constraints.
5.3.6 Beta Calibration as an Operator
Beta calibration [42] definesTusing the beta distribution's CDF structure:
T (pˆ)= , (5.17)
beta (cid:16) (cid:17)−(a−b) (cid:16) (cid:17)a
1+ 1 · pˆ · 1
ec 1−pˆ pˆ
with three parameters a, b, c learned by maximum likelihood. The family includes Platt
scaling as the special case a=b.
Why three parameters matter. Two-parameter Platt scaling can shift and scale the log-
odds but cannot correct asymmetric distortions—caseswherethemodelisoverconfidentfor
high probabilities but underconfident for low probabilities, or vice versa. Beta calibration's
third parameter c (the location shift) allows independent adjustment of the two tails.

5.3 Post-Hoc Calibration Operators 141
Practical guidance. Kull et al. [42] recommend the "abm" parameterisation (all three
parameters) as the default, with the "am" variant (a=b, reducing to two parameters) as a
fallback when the calibration set is small. The betacal Python package provides both.
1 import numpy as np
2 from betacal import Beta Calibration
3 from sklearn.metrics import brier_score_loss
5 np.random.seed(42)
6 # Simulated uncalibrated scores with asymmetric distortion
7 n = 2000
8 raw_scores = np.random.beta(0.5, 2.0, size=n)
9 true_probs = 1 / (1 + np.exp(-3 * (raw_scores - 0.3)))
10 labels = np.random.binomial(1, true_probs)
12 # Split into calibration and test
13 cal_scores, test_scores = raw_scores[:1000], raw_scores[1000:]
14 cal_labels, test_labels = labels[:1000], labels[1000:]
15 test_true = true_probs[1000:]
17 # Fit beta calibration (3-parameter)
18 bc = Beta Calibration(parameters='abm')
bc.fit(cal_scores, cal_labels)
20 calibrated = bc.predict(test_scores)
22 print(f"Raw Brier: {brier_score_loss(test_labels, test_scores):.4 f}")
23 print(f"Beta Brier: {brier_score_loss(test_labels, calibrated):.4 f}")
Listing 5.5: Beta calibration: fitting and evaluating the three-parameter model
5.3.7 Temperature Scaling as an Operator
Temperature scaling [32] is the simplest calibration operator: a single scalar parameter
applied to the logit space. For binary classification:
T T (s)=σ(s/T), (5.18)
wheresisthepre-sigmoidlogitand T >0 is the temperature learned by minim ising negative
log-likelihood on D .
cal
Interpretation. Temperature scaling isaglobal correction: it applies the same multi-
plicative factor 1/T to every logit. When T >1, the operator softens predictions (reducing
overconfidence); when T <1, it sharpens them (reducing underconfidence). It preserves
the ranking of all predictions and the top-1 class assignment.
Limitations. Temperature scaling cannot correct local miscalibration—it assumes that
the entire logit range is miscalibrated by the same multiplicative factor. If the model
is overconfident for high-probability predictions but underconfident for low-probability
predictions, no singleTcan fix both regions simultaneously. This is the fundamental
limitation that motivates richer operators like beta calibration and isotonic regression.
1 import numpy as np
2 from scipy.optimize import minimize_scalar
4 def nll_temperature(T, logits, labels):
5 """Negative log-likelihood for temperature scaling."""
6 scaled = 1.0 / (1.0 + np.exp(-logits / T))
7 scaled = np.clip(scaled, 1 e-10, 1 - 1 e-10)
8 return -np.mean(labels * np.log(scaled)

9 + (1 - labels) * np.log(1 - scaled))
11 # Example: learnTon calibration logits
12 np.random.seed(42)
13 cal_logits = np.random.randn(500) * 2 # Simulated logits
14 cal_labels = (np.random.rand(500) < 0.3).astype(float)
16 result = minimize_scalar(nll_temperature, bounds=(0.01, 20.0),
args=(cal_logits, cal_labels),
18 method='bounded')
19 T_opt = result.x
20 print(f"Optimal temperature: T = {T_opt:.3 f}")
22 # Apply to test logits
23 test_logits = np.random.randn(1000) * 2
24 calibrated = 1.0 / (1.0 + np.exp(-test_logits / T_opt))
25 print(f"Calibrated prob range: [{calibrated.min():.3 f}, "
26 f"{calibrated.max():.3 f}]")
Listing 5.6: Temperature scaling: learning the optimal temperature
5.3.8 When Calibration Hurts: The Degradation Problem
The premise of post-hoc calibration is that it can only help: apply it to any model's output,
and the resulting probabilities will be more reliable. This premise is false.
Manokhin and Grønhaug [54] evaluated Platt scaling, isotonic regression, and Venn–Abers
calibration across 72 tabular datasets and seven model families. Their findings challenge
the "always calibrate" assumption:
Finding 1: Isotonic regression can degrade strong models. On well-calibrated base
models (those with small initial calibration error), isotonic regression increased log-loss
in 18% of model–dataset pairs. The mechanism is resolution destruction: the step function
introduces quantisation artefacts that reduce the effective discrimination of the probability
estimates.
Finding 2: Platt scaling fails on non-sigmoid distortions. Platt scaling improved
calibration for SV Ms and logistic regression (where the score distribution is approximately
sigmoid) but degraded calibration for random forests and gradient-boosted trees in 23% of
cases. The parametric sigmoid cannot capture the bimodal score distribution characteristic
of tree ensembles.
Finding 3: Venn–Abers is the safest operator. Venn–Aberscalibrationimprovedlog-loss
in 96% of model–dataset pairs, with an average improvement of 14.17%. Crucially, in the
4% of cases where it did not improve, the degradation was small (mean <1%). No other
calibrator achieved this safety profile.
Why Calibration Can Destroy Resolution
The Brier decomposition BS=REL−RES+UNC reveals the trade-off. A calibrator
that reduces REL (reliability) by ∆ but also reduces RES (resolution) by ∆ will
REL RES
increase the Brier score whenever ∆ >∆ .
RES REL
This happens when the calibrator maps distinct uncalibrated scores to the same
calibrated probability. Isotonic regression's step function is the canonical example: a
range of scores [pˆ a ,pˆ b ] is collapsed toasingle value p¯, destroying the model's ability

5.3 Post-Hoc Calibration Operators 143
to distinguish instances within that range.
The practical implication is clear: never calibrate blindly. Always compare the
calibrated model against the uncalibrated baseline using proper scoring rules with
decomposition, not just aggregate metrics.
5.3.9 Sample Complexity of Calibration
A calibration operator is only as good as the calibration set on which it is trained. The
relationship between calibration set sizemand calibration quality is governed by each
operator's effective complexity.
Parametric operators. Plattscaling(2 parameters)andtemperaturescaling(1 parameter)
require relatively small calibration sets. With m≥100, both estimators typically converge
to stable calibration maps [32, 66]. Beta calibration (3 parameters) requires slightly more:
m≥200 isapractical minimum.
Nonparametric operators. Isotonic regression has O(m) effective parameters and con-
verges at rate O(m−1/3)—slower than the O(m−1/2) parametric rate. In practice, m≥1,000
is needed for reliable isotonic calibration in the tails of the score distribution.
Venn–Abers. Venn–Abers calibration uses an internal isotonic regression step but derives
itscalibrationguaranteefromadifferentmechanism—theconformalpredictionframework—
whichprovideslong-runsequentialcalibrationregardlessofm. Thewidthofthe Venn–Abers
interval [p 0 ,p 1 ] decreases asmgrows, but the calibration guarantee—that the sequence of
predictions is well-calibrated in the long run—holds for any m. This makes Venn–Abers the
only calibration operator whose guarantees do not requireaminimum calibration set size.
Table 5.5: Practical minimum calibration set sizes for reliable post-hoc calibration, based on
published guidelines and empirical evidence.
Operator Min. m Rationale
Temperaturescaling 100 1 parameter;convergesrapidly
Plattscaling 100 2 parameters;sigmoidassumptionhelps
Betacalibration 200 3 parameters;needsbothtailsrepresented
Isotonicregression 1,000 O(m)parameters;tailsneeddensity
Venn–Abers Anym Validityguaranteed;width↓asm↑
5.3.10 The Calibration Operator Decision Tree
Given the properties and failure modes above, the choice of calibration operator follows a
structured decision process.
Step 1: Is the base model already well-calibrated? Compute the Spiegelhalter Z-
statistic onaheld-out test set. If |Z|<1.96, the model is statistically calibrated at the 5%
level. Post-hoccalibrationmaystillimprovethe Brierscore,butitcarriesariskofresolution
degradation. Proceed with caution and always compare against the uncalibrated baseline.
Step 2: How large is the calibration set? If m<200, use temperature scaling (if logits
are available) or Platt scaling. If 200≤m<1,000, beta calibration is the best parametric
option. If m≥1,000, isotonic regression isastrong nonparametric choice.
Step 3: Are provable calibration guarantees required? If the application demands
provable calibration guarantees—medical risk prediction, autonomous driving, financial
regulation—use Venn–Abers calibration regardless of m. It is the only operator with non-
asymptotic validity guarantees.

Step 4: Validate. After applying the chosen operator, evaluate using multiple diagnostics:
the reliability diagram, the Brier score with decomposition, the Spiegelhalter Z-statistic,
and log-loss. If the calibrated model hasahigher Brier score than the uncalibrated baseline,
the calibrator has destroyed more resolution than it improved reliability. In that case, either
switch toadifferent operator or use the uncalibrated model.
The companion Colab notebook for Section 5.2.5 demonstrates this decision process on the
high-dimensional task from Section 5.1.2, with all five operators compared side by side. The
notebook is available in the Pro edition:
https://valeman.gumroad.com/l/applied_conformal_prediction_pro
5.3.11 Summary
Post-hoc calibration operators are powerful tools, but they are not magic. The key lessons
from this section are:
1. Calibration has three levels—marginal, weak, and auto-calibration—formingastrict
hierarchy. GL Ms satisfy only the weakest level.
2. Proper scoring rules (Brier score, log-loss) are the correct evaluation metrics, but
they must be decomposed to distinguish calibration improvements from resolution
destruction.
3. Each calibration operator imposes assumptions on the shape of the calibration map.
Parametric operators (Platt, beta, temperature) are efficient but inflexible; nonpara-
metric operators (isotonic) are flexible but data-hungry.
4. Calibration can degrade strong models. The "always calibrate" advice is wrong.
Always compare against the uncalibrated baseline using proper scoring rules with
decomposition.
5. Venn–Abers is the only operator with long-run sequential calibration guarantees that
hold for any calibration set size. Section 5.4 develops its theory in full.
5.4 Venn Predictors and Venn–Abers Calibration
The calibration operators of Section 5.3 shareacommon limitation: their guarantees are
asymptotic. Platt scaling, isotonic regression, beta calibration, and temperature scaling
all converge to the true calibration map as the calibration set grows without bound, but
they offer no formal statement about reliability at any finite sample size. In practice, the
calibration set is always finite, often small, and the gap between asymptotic theory and
finite-sample reality is exactly the gap in which bad decisions are made.
Venn predictors close this gap. Introduced by Vovk, Gammerman, and Shafer [86] as the
probabilistic counterpart of conformal predictors, Venn predictors produce probability
estimates with provable long-run calibration guarantees under the sole assumption of
exchangeability—the same assumption that powers conformal prediction itself. Venn–Abers
predictors, the most practical member of the family, were introduced by Vovk and Petej
[87] and combine isotonic regression with the conformal machinery to yield calibrated
probabilities that provably outperform classical calibrators on modern benchmarks [54].
This section develops the theory from first principles, derives the key guarantees, and
demonstrates the practical implementation.

5.4 Venn Predictors and Venn–Abers Calibration 145
5.4.1 From Conformal Sets to Probability Estimates
Recall the conformal prediction framework from Chapter 3. Givenanonconformity measure
andacalibration set, conformal prediction constructs set-valued predictions C α (x)⊆Y with
guaranteed coverage:
P(cid:0)
Y ∈C (X )
(cid:1)
≥1−α.
n+1 α n+1
This guarantee is distribution-free and holds at any sample size under exchangeability.
Venn predictors askadifferent question: instead of producingaset of labels with coverage
guarantees, can we produce probability estimates with calibration guarantees—using the
same distribution-free machinery?
Theanswerisyes,butwithatwist. Conformalpredictionexploitsthesymmetryofexchange-
able sequences to bound the rank ofanew observation among the calibration scores. Venn
predictionexploitsthesamesymmetrytoproducemultiprobabilitypredictions—notasingle
probability vector, butaset of probability distributions, each provably well-calibrated in the
long run.
5.4.2 Venn Predictors: The General Framework
A Venn predictor is defined byaVenn taxonomy: a rule that partitions the combined
sequence of calibration examples and the new test example into categories.
Definition. Let z 1 ,...,z n ,z n+1 be an exchangeable sequence of examples, where z i =(x i ,y i )
for i≤n and z n+1 =(x n+1 ,y) for an unknown label y. A Venn taxonomy isameasurable
function
τ :(z 1 ,...,z n+1 )7→{κ 1 ,...,κ K }, (5.19)
that assigns each example to one ofKcategories. The taxonomy is equivariant: it depends
on the examples only through their multiset (i.e., it is invariant to permutations).
Multiprobability prediction. For each possible label y∈Y of the test example, the Venn
predictor:
1. Tentatively assigns z n+1 =(x n+1 ,y).
2. Applies the taxonomy τ to the augmented sequence z 1 ,...,z n ,z n+1.
3. Computes the empirical frequency of labelywithin the category κ to which z n+1 is
assigned.
This producesaprobability estimate pˆ(y) for each hypothesised label y. The collection
{pˆ(y):y∈Y} is the multiprobability prediction.
In the binary case (Y ={0,1}), the multiprobability prediction isapair [p 0 ,p 1 ] where:
• p 0 is the estimated probability of Y =1 assuming Y =0,
• p 1 is the estimated probability of Y =1 assuming Y =1.
Since assuming Y = 1 adds one more positive example to the category, p 1 ≥ p 0 always
holds. The interval [p 0 ,p 1 ] quantifies the predictor's epistemic uncertainty about the true
probability.
Why Multiprobability, NotaSingle Probability?
Classical calibrators produceasingle point estimate p˜(x). If the calibration set is
small or the test example falls inasparse region of the score space, this estimate can
be arbitrarily unreliable—but the calibrator provides no indication of how unreliable

it is.
Venn predictors produce an interval [p 0 ,p 1 ]. A narrow interval (e.g., [0.72,0.74])
indicates confident calibration: the probability estimate is stable regardless of the
true label. A wide interval (e.g., [0.3,0.8]) signals that the calibration set provides
insufficient information to pin down the probability—a valuable diagnostic that no
classical calibrator offers.
√
The width of the interval shrinks as O(1/ m) with calibration set size m, converging
toapoint prediction in the limit.
5.4.3 The Validity Theorem
The central theoretical result for Venn predictors is that the sequence of multiprobability
predictions is automatically well-calibrated in the long run under exchangeability.
Theorem 5.1 — Validity of Venn predictors.
(Vovk et al. [86].) Let z 1 ,z 2 ,... be an exchangeable sequence and let τ be any equivariant
Venn taxonomy. For each n, let pˆ n be the Venn predictor's probability estimate for the
true label y n, computed using z 1 ,...,z n−1 as the calibration set and z n as the test example.
Then:
1. Sequential calibration. Along any subsequence where the predicted probability
lies in an interval [a,b], the empirical frequency of the positive label converges to a
value in [a,b], almost surely.
2. Equivalently (Dawid's formulation):
1 X N
lim (y n −pˆ n )=0 almost surely. (5.20)
N→∞N
n=1
The guarantee is about the sequence of predictions, not about any individual prediction
at finite N.
The key distinction from classical calibration guarantees:
• Classical calibrators (Platt, isotonic, beta, temperature) are calibrated asymptotically
in the calibration set size m. Forafixed m, no formal guarantee holds.
• Venn predictors are calibrated asymptotically in the number of predictions N, for any
fixed calibration set size m. The guarantee is about the sequence of predictions, not
about the calibration procedure.
This isafundamentally different kind of guarantee. It says: even if your calibration set
has only 50 examples, the long-run frequency of outcomes among predictions assigned
probabilitypwill converge to p. No classical calibrator can make this claim.
What Venn Predictors Guarantee (and What They Do Not)
Guaranteed:
• Sequential calibration. The sequence of predicted probabilities pˆ 1 ,pˆ 2 ,...
is well-calibrated in the long-run frequency sense of Dawid [18]: among all
predictions where pˆ n ≈p, the empirical frequency of y n =1 converges to p.
• Any calibration set size. Unlike classical calibrators, the guarantee holds
for any m—including m=50. With small m, the interval [p 0 ,p 1 ] is wider (more

5.4 Venn Predictors and Venn–Abers Calibration 147
honest uncertainty), but the calibration property is not compromised.
• Interval width as epistemic uncertainty. Thewidthp 1 −p 0 isanhonestsignal
of the predictor's uncertainty: it shrinks as data grows.
Not guaranteed:
• Foranysingle testpointatfinite N,theprobabilityestimatepˆ nisnot guaranteed
to equal the true conditional probability P(Y =1|X =x n ). The guarantee is a
sequence-level property, notapoint-level property.
5.4.4 Venn–Abers Predictors
The general Venn prediction framework requires specifyingataxonomy τ. Vovk and Petej
[87] introduced Venn–Abers predictors, which useaspecific, powerful taxonomy based on
isotonic regression.
Construction. GivenacalibrationsetDcal ={(x i ,y i )}m i=1 andabaseclassifierthatproduces
scores s(x i ), the Venn–Abers predictor works as follows:
For each hypothesised label y∈{0,1} of the test example x n+1:
1. Augment. Add (s(x n+1 ),y) to the calibration scores to form the augmented set
{(s
,y
),...,(s
m
,y
m
),(s
n+1
,y)}.
2. Fit isotonic regression. Apply the Pool Adjacent Violators (PAV) algorithm to the
augmented set, fittinganon-decreasing step function g y :R→[0,1] that maps scores
to calibrated probabilities.
3. Predict. Setp y =g y (s(x n+1 ))—thecalibratedprobabilityof Y =1 underthehypothesis
that the true label is y.
The output is the pair [p 0 ,p 1 ], where:
p 0 =g 0 (s(x n+1 )), p 1 =g 1 (s(x n+1 )). (5.21)
Since addingapositive example (y=1) shifts the isotonic fit upward at the score s(x n+1 ),
we always have p 0 ≤p 1.
Point prediction. The pair [p 0 ,p 1 ] is converted toasingle probability estimate by solving
forpsuch that:
p
p= , (5.22)
1−p +p
0 1
which is the unique fixed point of the mixture under the assumption thatpis the true
probability of Y =1. Geometrically, p is the intersection of the line from (0,p 0 ) to (1,p 1 )
with the diagonal y=x. Equivalently, p minimises the logarithmic loss under the two-point
distribution {p 0 ,p 1 }, which is why the fixed-point formula produces well-calibrated point
predictions despite collapsing the interval toascalar.
Why isotonic regression? The choice of isotonic regression as the Venn taxonomy is not
arbitrary. It has three critical properties:
1. Nonparametric. Unlike Platt scaling, it makes no assumption about the shape of the
calibration map.
2. Monotone. The PAV algorithm preserves the score ranking, so the calibrated proba-
bilities respect the base classifier's discrimination.
3. Minimax optimal. Among all monotone calibration maps, isotonic regression min-
imises the sum of squared residuals. Combined with the Venn framework, this yields
the tightest possible calibration intervals.

5.4.5 Transductive vs. Inductive Venn–Abers
The construction above is transductive: for each test example, two isotonic regressions
must be fitted (one for each hypothesised label). Withmcalibration examples, each PAV
fit costs O(mlogm) (sorting) +O(m) (PAV), giving O(mlogm) per test prediction. For batch
prediction onTtest examples, the total cost is O(T ·mlogm)—potentially expensive for
large m.
Inductive Venn–Abers. Vovk, Petej, and Fedorova [87] introduced the inductive variant,
which splits the calibration set intoaproper training set andacalibration set:
1. Train the base classifier on the proper training set.
2. Compute scores s(x i ) on the calibration set.
3. For each test example, augment only the calibration scores (not the training set) with
the hypothesised test score and fit PAV.
The inductive variant sacrificesasmall amount of statistical efficiency (the base classifier
sees fewer training examples) in exchange for computational efficiency: the base classifier
is trained once, and only the PAV step is repeated per test example. This is the variant
implemented in the venn-abers Python package and used throughout this chapter.
1 import numpy as np
2 from sklearn.linear_model import Logistic Regression
3 from sklearn.datasets import make_classification
4 from sklearn.model_selection import train_test_split
5 from sklearn.metrics import brier_score_loss
6 from venn_abers import Venn Abers Calibrator
8 # Synthetic data
9 X, y = make_classification(n_samples=5000, n_features=200,
n_informative=10, n_redundant=10,
11 weights=[0.95, 0.05], random_state=42)
12 X_train, X_test, y_train, y_test = train_test_split(
13 X, y, test_size=0.2, random_state=42, stratify=y)
15 # Inductive Venn-Abers: cal_size controls the internal split
16 va = Venn Abers Calibrator(
estimator=Logistic Regression(max_iter=2000, random_state=42),
18 inductive=True,
19 cal_size=0.25, # 25% of training data reserved for PAV
random_state=42
21 )
va.fit(X_train, y_train)
24 # Predict: returns calibrated probabilities
25 va_probs = va.predict_proba(X_test)[:, 1]
26 print(f"Brier score: {brier_score_loss(y_test, va_probs):.4 f}")
28 # Access the multiprobability interval [p 0, p 1]
29 p 0 p 1 = va.predict_proba(X_test) # columns: [p(y=0), p(y=1)]
30 print(f"Mean interval width: {np.mean(p 0 p 1[:, 1] - p 0 p 1[:, 0]):.4 f}")
Listing 5.7: Inductive Venn–Abers calibration in practice
Multi-class Calibration via Pairwise Coupling (PKPD)
Thebinary Venn–Abersframeworkcanbeextendedto K >2 classes through pairwise
coupling. Manokhin [51] proposed the PKPD (Pairwise Kriging–Probability Distribu-

5.4 Venn Predictors and Venn–Abers Calibration 149
(cid:0)K(cid:1)
tion) approach: fit binary Venn–Abers calibrators—one for each class pair—and
recover the full K-class probability vector by solving the pairwise coupling equations
of Wu, Lin, and Weng.
Pipeline: (1) Train any multi-class classifier producing pairwise scores. (2) For
each pair (i,j), calibrate the binary scores using Venn–Abers on the calibration set
restricted to classesiand j. (3) Solverij ≈p i /(p i +p j ) for the K-vector (p 1 ,...,p K )
via iterative normalisation.
Thisapproachpreservesthesequentialcalibrationguaranteeofeachpairwisecalibra-
tor and produces probability vectors that sum to one and respect the interdependence
between classes—unlike the naïve one-vs-rest normalisation used in scikit-learn's
Calibrated Classifier CV.
5.4.6 Theoretical Properties
Venn–Abers predictors inherit the validity guarantees of the general Venn framework but
add properties specific to the isotonic regression taxonomy.
Property 1: Sequential calibration for any m. The multiprobability prediction [p 0 ,p 1 ] is
valid for any calibration set size: the sequence of predictions is well-calibrated in the long
run under exchangeability. There is no minimummbelow which the guarantee breaks. This
contrasts sharply with isotonic regression alone, which requires m≥1,000 for reliable tail
calibration (Table 5.5).
Property 2: Automatic monotonicity. Because the PAV algorithm producesanon-
decreasingfunction,Venn–Aberspredictionsrespectthescoreranking: ifs(x a )>s(x b ),then
the calibrated probability of x a is at least as large as that of x b. This is not guaranteed by
histogram binning or Platt scaling applied to non-sigmoid score distributions.
Property 3: Interval width as uncertainty. Thewidthp 1 −p 0 providesabuilt-inmeasure
of epistemic uncertainty. Wideintervalsindicatethatthecalibrationsetprovidesinsufficient
evidence to pin down the probability—a signal that the prediction should be treated with
caution. No classical calibrator produces this diagnostic automatically.
Property 4: Consistency. As the calibration set grows (m→∞), the interval [p 0 ,p 1 ] col-
lapsestoapoint,andthe Venn–Aberspredictorconvergestothesamecalibratedprobability
as standard isotonic regression. The sequential calibration guarantee is thereforeastrict
addition to the asymptotic guarantees of isotonic regression—notareplacement.
Property 5: Distribution-free. The only assumption is exchangeability of the data. No
assumption is made about the score distribution, the true calibration map, or the data-
generating process. This makes Venn–Abers the most assumption-lean calibrator available.
5.4.7 Empirical Evidence: The 72-Dataset Study
The theoretical properties of Venn–Abers predictors translate into practical superiority.
Manokhin and Grønhaug [54] conducted the largest systematic evaluation of calibration
methods to date, comparing Venn–Abers calibration against Platt scaling and isotonic
regression across 72 tabular datasets from the Tab Arena-v 0.1 benchmark and seven model
families (logistic regression, random forest, gradient-boosted trees, neural networks, SV Ms,
naïve Bayes, and k-nearest neighbours).
Key findings.
1. Universal improvement. Venn–Abers calibration improved log-loss in 96% of all

model–dataset pairs, with an average improvement of 14.17%. No other calibrator
achieved this breadth.
2. Safety. Inthe 4%ofcaseswhere Venn–Abersdidnotimprovelog-loss,thedegradation
was small (mean <1%). By contrast, isotonic regression degraded log-loss in 18% of
pairs, and Platt scaling degraded it in 23% of pairs (Section 5.3.8).
3. Model-agnostic. The improvement was consistent across all seven model families.
Venn–Abers is not tuned toaspecific classifier architecture—it adapts to whatever
score distribution the base model produces.
4. No classifier is calibrated by default. Across all 504 model–dataset pairs (72
datasets × 7 model families), notasingle base classifier produced acceptably cali-
brated probabilities without post-hoc correction. This finding generalises the logistic
regression result from Section 5.1.2 to all major classifier architectures.
The head-to-head comparison in Section 5.2.5 confirmed these findings onacontrolled
synthetic task: Venn–Abers reached Z =−0.94 (well within the acceptance region) while
maintaining competitive Brier score and log-loss (Table 5.3).
5.4.8 When to Use Venn–Abers
Venn–Abers calibration is the recommended default in three scenarios:
1) Provable calibration guarantees are required. In regulated domains—clinical risk
prediction, autonomous systems, financial credit scoring—stakeholders may demand prov-
able calibration guarantees, not just empirical evidence. Venn–Abers is the only calibrator
that provides sequential calibration guarantees at any sample size.
2) The calibration set is small. When m < 500, classical nonparametric calibrators
(isotonic regression) overfit and parametric calibrators (Platt, beta) may be misspecified.
Venn–Abers provides valid probabilities regardless of m, with the interval width [p 0 ,p 1 ]
honestly communicating the remaining uncertainty.
3) Safety is the priority. When the cost of degradingamodel's probability estimates
exceeds the benefit of improving them, Venn–Abers's 96% improvement rate with <1%
worst-case degradation makes it the safest choice.
Table 5.6: Decision guide: choosingacalibration operator. Entries in bold indicate the
recommended choice for each scenario.
Scenario Platt Iso. Beta Temp. VA
Formalguaranteesneeded ✓
Smallcal.set(m<500) ✓ ✓ ✓ ✓
Safety-criticalapplication ✓
Largem,knownscoredist. ✓ ✓ ✓ ✓
Multi-classdeepnetwork ✓
Asymmetricdistortion ✓ ✓
5.4.9 Venn–Abers and Conformal Prediction
Venn–Abers calibration and conformal prediction are complementary, not competing. They
share the same theoretical foundation—exchangeability and the symmetry of ranks—but
address different questions:
• Conformal prediction answers: "Which labels are consistent with the data at signifi-

5.4 Venn Predictors and Venn–Abers Calibration 151
cance level α?" The output isaset C α (x)⊆Y.
• Venn–Abers calibration answers: "What is the calibrated probability of each label?"
The output isaprobability vector p˜(x)∈[0,1]K.
The connection runs deeper than shared assumptions. Calibrated probabilities improve
conformal prediction in two ways:
1) Smaller prediction sets. Adaptive conformal methods such as APS [72] and RAPS [4]
use the ranked probability vector to construct prediction sets. Better-calibratedprobabilities
yield smaller prediction sets under these adaptive methods, as later chapters demonstrate
in detail.
2) Conditional coverage. Marginal coverage ( P(Y ∈C α (X))≥1−α) does not guarantee
coverage for subgroups. Calibrated probabilities reduce the gap between marginal and
conditional coverage, because calibration ensures that the score distribution is faithful
to the true class probabilities across all regions of the input space. The concrete set-size
comparisons and side-by-side APS/RAPS benchmarks appear in the dedicated adaptive-sets
chapter later in this book.
5.4.10 Summary
Venn–Abers predictors occupyaunique position in the calibration landscape. They are the
only method that simultaneously provides:
1. Sequential calibration. The calibration guarantee holds for the entire prediction
sequence, not just asymptotically.
2. Built-in uncertainty quantification. The multiprobability interval [p 0 ,p 1 ] provides
an honest measure of calibration confidence that no classical calibrator offers.
3. Nonparametricflexibility. Theisotonicregressionbackboneadaptstoanymonotonic
calibration map without parametric assumptions.
4. Empirical dominance. Across 72 datasets and seven model families, Venn–Abers
improved log-loss in 96% of cases with negligible worst-case degradation [54].
5. Conformal compatibility. Sharing the same exchangeability foundation as con-
formal prediction, Venn–Abers calibration integrates naturally into the conformal
pipeline, improving the efficiency of downstream conformal procedures (detailed in
later chapters).
Theexperimentin Section 5.2.5 demonstrated these properties concretely: onthecontrolled
d/n=0.04 task, Venn–Abers reached Z =−0.94 with competitive Brier and log-loss while
being the only method with provable sequential calibration guarantees (Table 5.3).
Theremainderofthischaptercompletesthereliabilityframeworkwithconditionalcoverage
and the full evaluation protocol.
Pro Edition
The Pro edition includes the full production pipeline used in the 2026 Tab Arena study
(72 datasets, 21 classifiers), advanced multi-class extensions via pairwise coupling
(PKPD), and additional Jupyter notebooks with ready-to-run experiments, plus the
full APS/RAPS efficiency benchmark notebooks. https://valeman.gumroad.com/l/
applied_conformal_prediction_pro

5.5 Conditional Coverage
Split conformal prediction guarantees marginal coverage:
P(cid:0)
Y ∈C (X )
(cid:1)
≥1−α.
n+1 α n+1
This probability is taken over both the calibration set and the test point, averaged across all
possible inputs. Marginal coverage isapowerful guarantee—it holds without distributional
assumptions—but it can hide serious disparities.
5.5.1 The Limitation of Marginal Coverage
Aconformalpredictorwith 90%marginalcoveragemaycover 99%ofeasyexamplesandonly
60%ofhardexamples. Averagedtogether,the 90%targetismet,butthehardsubpopulation
is severely under-covered.
Formally, the ideal guarantee would be conditional coverage:
P(cid:0)
Y n+1 ∈C α (X n+1 )|X n+1 =x
(cid:1)
≥1−α for all x. (5.23)
Vovk [84], Lei and Wasserman [47], and Barber et al. [7] proved that exact conditional cov-
erage is impossible without distributional assumptions: for any distribution-free conformal
predictor, there exist distributions under which the conditional coverage at some pointxis
arbitrarily far from 1−α. Barber et al. [7] further established the finite-sample theory of
distribution-free predictive inference, providing tight coverage guarantees and impossibility
results that underpin much of the modern conformal prediction framework.
This impossibility result does not mean that conditional coverage is hopeless. It means that
practitioners must choose between:
1. Exact coverage for predefined groups (group-conditional coverage), at the cost of
larger prediction sets.
2. Approximateconditionalcoverageeverywhere,usingcalibratedscoresthatreduce
the gap between marginal and conditional coverage.
5.5.2 Group-Conditional Coverage
The most direct approach to conditional coverage is Mondrian conformal prediction [86]:
partition the input space into predefined groups G 1 ,...,G M and run separate conformal
procedures within each group.
Procedure.
(g)
1. Partition the calibration set into groups: D cal ={(x i ,y i ):x i ∈G g }.
2. For each group g, compute the group-specific conformal quantile qˆ g from the noncon-
(g)
formity scores in D .
cal
3. Foranew test point x n+1 ∈G g, use qˆ g instead of the global qˆ.
Mondrian conformal prediction guarantees group-conditional coverage:
P(cid:0)
Y n+1 ∈C α (X n+1 )|X n+1 ∈G g
(cid:1)
≥1−α for each g=1,...,M. (5.24)
The cost. Each group uses only its own calibration examples, so the effective calibration
(g)
set size per group is n g =|D cal |. Smaller calibration sets yield looser quantiles and larger
predictionsets. Ifthetotal calibration set hasn examples split acrossMequalgroups, each
group has approximately n/M examples. For the conformal quantile to be meaningful, each
group needs at least 50–100 calibration examples.

5.5 Conditional Coverage 153
When to Use Mondrian Conformal Prediction
Use Mondrian conformal prediction when:
• The groups are predefined and scientifically meaningful (e.g., demographic
categories, disease subtypes, product lines).
• The calibration set is large enough to support ≥100 examples per group.
• Regulatory or fairness requirements mandate group-level coverage guarantees.
Avoid Mondrian when the groups are data-driven (e.g., clusters), because the group
boundaries are random and the coverage guarantee no longer holds exactly.
5.5.3 Multivalid Conformal Prediction
Gopalan, Jiang, Lei, Mironov, Ramdas, and Zhang [30] introduced multivalid conformal
prediction,whichprovidescoverageguaranteessimultaneouslyforallgroupsinacollection
G—including overlapping groups.
Standard Mondrian conformal prediction handles disjoint partitions. Multivalid conformal
prediction handles overlapping groups such as "female patients over 65" and "patients with
diabetes," which share members.
Key idea. Instead of running separate conformal procedures per group, multivalid confor-
mal prediction iteratively adjusts the conformal quantile to satisfy coverage for all groups
simultaneously. The adjustment usesaboosting-like algorithm: in each round, the quantile
is shifted upward for under-covered groups and downward for over-covered groups, until
convergence.
Guarantee. AfterTrounds, multivalid conformal prediction achieves:
(cid:12) (cid:12)
(cid:12) (cid:12) P(cid:0) Y n+1 ∈C α (X n+1 )|X n+1 ∈G (cid:1) −(1−α)(cid:12) (cid:12) ≤ϵ for all G∈G, (5.25)
where ϵ decreases with the number of calibration examples and rounds.
Trade-off. Multivalid coverage is strictly stronger than marginal coverage but weaker than
exact conditional coverage. It requiresalarger calibration set than standard conformal
prediction (because coverage must hold for all groups simultaneously) but avoids the
impossibility barrier of exact conditional coverage.
5.5.4 How Calibration Reduces the Coverage Gap
Even without explicit Mondrian or multivalid procedures, calibration alone reduces the gap
between marginal and conditional coverage.
Mechanism. The gap between marginal and conditional coverage arises when the non-
conformity scores have different distributions across subpopulations. If the base classifier
assignssystematicallyhigherprobabilitiestoonesubgroup(e.g.,becauseofclassimbalance
or distribution shift), the nonconformity scores for that subgroup will be systematically
lower, leading to under-coverage in the complementary subgroup.
Calibration corrects this: by aligning the predicted probabilities with the true class frequen-
cies within each region of the score space, calibration ensures that the nonconformity score
distribution is more uniform across subpopulations. The result is smaller discrepancies
between group-level and marginal coverage.
Empirical evidence. Guo, Pleiss, Sun, and Weinberger [32] demonstrated that tempera-
ture scaling reduces the expected calibration error of deep networks from >10% to <2%.
Because calibration error directly drives the gap between the score distributions of dif-
ferent subgroups, this reduction translates into more uniform conditional coverage under

conformal prediction.
Venn–Abers calibration (Section 5.4) provides an even stronger correction: its sequential
calibration guarantee ensures that the calibrated probabilities are well-aligned with true
outcomes for any subset of predictions, not just globally. This makes Venn–Abersanatural
pre-processing step for conformal prediction when approximate conditional coverage is
desired.
5.5.5 Summary
The key lessons from this section are:
1. Exact conditional coverage is impossible without distributional assumptions. This
isafundamental limitation, notafailure of any particular method.
2. Group-conditional coverage is achievable via Mondrian conformal prediction, at
the cost of splitting the calibration set across groups.
3. Multivalid conformal predictionextendsgroup-conditionalcoveragetooverlapping
groups [30].
4. Calibration narrows the gap between marginal and conditional coverage by aligning
score distributions across subpopulations. This isafurther reason to calibrate before
applying conformal prediction.
5.5.6 Calibration vs. Conformal Coverage Under Dataset Shift
Allguaranteesinthischapter—calibration,conformalcoverage,and Vennpredictorvalidity—
rest on the assumption of exchangeability between the calibration (or training) data and
the test data. In deployment, dataset shift is the norm, not the exception. Calibration and
conformal coverage fail in different ways under shift, and practitioners must monitor both.
Calibration failure. Calibration isastatement about the conditional relationship P(Y =1|
pˆ(X)=p)=p. Under covariate shift or concept drift, this relationship can break drastically:
a model calibrated on the source distribution may become severely overconfident or under-
confident on the target distribution. Reliability diagrams computed on fresh target data will
reveal the shift.
Conformal coverage failure. Conformal prediction's marginal coverage guarantee P(Y ∈
C α (X))≥1−α depends on exchangeability between the calibration set and the test point.
Undershift,thecoveragecandropbelow 1−α. However,repairstrategiesexist: importance-
weighted conformal prediction reweights the calibration scores by the likelihood ratio
between source and target [80], and adaptive conformal inference methods track coverage
drift online.
Practitioner takeaway. Calibration and coverage are different failure modes under shift.
A model can remain well-calibrated while its conformal prediction sets lose coverage (if the
shift affects different regions of the input space), or vice versa. In production:
• Monitor calibration drift with rolling reliability diagrams and the Spiegelhalter Z-
statistic.
• Monitor coverage drift by tracking empirical coverage of prediction sets on incoming
data.
• These are distinct dashboards; neither subsumes the other.
The next section synthesises the calibration and conformal methods intoacomplete evalua-
tion protocol (Section 5.6).

5.6 Evaluation Protocol 155
5.6 Evaluation Protocol
The previous sections developed the theoretical and practical tools for calibration and
conformal prediction. This section assembles them intoacomplete evaluation protocol: a
step-by-step procedure for assessing whetheraclassifier's probability estimates are trust-
worthy and for measuring the impact of calibration on conformal prediction set efficiency.
5.6.1 Step 1: Data Splitting
The single most important methodological decision is the data split. A fair evaluation
requires three disjoint sets:
1. Training set. Used to fit the base classifier. The classifier must never see calibration
or test data during training.
2. Calibration set. Used to fit the calibration operator (Platt, isotonic, Venn–Abers, etc.)
and to compute the conformal quantile. The calibration operator must see only the
base classifier's held-out scores on this set, not the training scores.
3. Test set. Used to evaluate calibration quality and prediction set efficiency. No model
or calibrator is fitted on this data.
Why Three Sets, Not Two?
A common mistake is to train the base classifier on the full training set, then calibrate
and evaluate on the same held-out set. This conflates the calibration set with the test
set, producing optimistically biased calibration metrics.
The calibrator adapts to the calibration set. If the same data is used for evaluation,
the calibrator appears better than it truly is on unseen data. The three-way split
ensures that evaluation is honest.
When using Calibrated Classifier CV from scikit-learn, always use cv='prefit' to
enforce this separation: the base model is fitted on the training set first, then the
calibrator is fitted on the calibration set, and evaluation uses the test set exclusively.
5.6.2 Step 2: Calibration Metrics
Calibration quality is measured by three complementary metrics, each capturingadifferent
aspect.
Brier score. For binary classification with true labels y i ∈{0,1} and predicted probabilities
pˆ i:
BS= 1 X N (y i −pˆ i )2. (5.26)
N
i=1
The Brier score isaproper scoring rule: it is minimised when pˆ i equals the true conditional
probability P(Y =1|X =x i ). It decomposes into calibration, refinement, and uncertainty
components—the decomposition from Section 5.3.
Log-loss.
1 X N (cid:2) (cid:3)
LL=− y i logpˆ i +(1−y i )log(1−pˆ i ) . (5.27)
N
i=1
Log-lossisalsoaproperscoringrulebutpenalisesconfidentwrongpredictionsmoreheavily
than Brier score. A model that assigns pˆ=0.01 to an example with y=1 receivesalog-loss
penalty of −log(0.01)≈4.6, whereas the Brier penalty is only (1−0.01)2≈0.98.

Spiegelhalter Z-statistic. The Z-statistic tests the null hypothesis H 0: the model is
perfectly calibrated.
PN (y −pˆ)(1−2 pˆ)
Z = i=1 i i i . (5.28)
q
PN (1−2 pˆ)2 pˆ(1−pˆ)
i=1 i i i
Under H 0, Z ∼N(0,1) asymptotically. Reject H 0 at the 5% level if |Z|>1.96.
Calibration tests beyond Z. Dimitriadis, Gneiting, and Jordan [21] developedamodern
framework of calibration tests based on e-values and kernel-based statistics, providing
alternatives to Spiegelhalter's Z that are valid under weaker assumptions and can detect
subtler forms of miscalibration such as localised deviations that do not affect the global
mean. Practitioners working with large datasets or requiring non-asymptotic guarantees
should consider these tests as complements to the Z-statistic.
A note on the Hosmer–Lemeshow test. The Hosmer–Lemeshow goodness-of-fit test
isaclassical calibration test used extensively in biostatistics. It bins predictions into g
groups(typicallyg=10)andcomparesobservedandexpectedfrequenciesviaachi-squared
statistic. While widely cited, its power depends on the choice of g, and it can miss smooth
miscalibration patterns. The Spiegelhalter Z-statistic is preferred in this chapter because it
is bin-free and providesacontinuous measure of calibration deviation.
Recommended practice. Report all three metrics. Brier score and log-loss measure
overall predictive quality; the Z-statistic providesaformal hypothesis test of calibration.
A model can haveagood Brier score (because of strong discrimination) while failing the
Z-test (because the probabilities are biased). The Z-test catches systematic miscalibration
that Brier score and log-loss may absorb into the discrimination component.
5.6.3 Step 3: Reliability Diagrams
Reliability diagrams provide the visual counterpart to the numerical metrics.
Construction.
1. Bin the predicted probabilities pˆ i intoBequally spaced bins (typically B = 10 or
B=20).
2. For each bin b, compute the mean predicted probability p¯ b = |B 1 b | P i∈B b pˆ i and the
observed frequency y¯ b = |B 1 b | P i∈B b y i.
3. Plot (p¯ b ,y¯ b ) for each bin. A perfectly calibrated model lies on the diagonal y¯=p¯.
Interpretation. Points above the diagonal indicate under-confidence (the model predicts
lower probabilities than the observed frequency); points below indicate overconfidence (the
model predicts higher probabilities than observed). The pattern of deviations reveals the
shape of the miscalibration, which is invisible in scalar metrics:
• Sigmoid distortion: the raw probabilities are compressed toward 0.5. This is the
signature of logistic regression on high-dimensional data (Section 5.1.2).
• Overconfidence in the tails: themodelassignspˆnear 0 or 1 toofrequently. Common
in deep networks [32].
• Non-monotonic distortion: the calibration map is not monotone, indicating that
higher scores do not always correspond to higher true probabilities. This violates the
assumptions of isotonic regression and Platt scaling.
Murphy diagrams. Reliability diagrams bin predicted probabilities and plot observed
frequencies, but the choice of binning scheme introduces artefacts. Murphy diagrams [22]
provideacomplementary, bin-free diagnostic: they plot the elementary score S θ (pˆ,y)=
(1{y≤θ}−1{pˆ≤θ})2 asafunctionofthethresholdθ. Aperfectlycalibratedmodelproduces
a flat Murphy curve. Regions where one model's curve lies below another's indicate

5.7 End-to-End Worked Example 157
superiority at that threshold, providingamore granular comparison thanasingle Brier
score.
5.6.4 Step 4: Prediction Set Evaluation
If the classifier's probabilities are used for conformal prediction, the evaluation protocol
must also assess the prediction sets.
Coverage. Compute the empirical coverage on the test set:
N
1 Xtest
Cdov= 1[y i ∈C α (x i )]. (5.29)
N
test i=1
The empirical coverage should be at or above the target 1−α.
Average set size. The average number of classes in the prediction set:
N
1 Xtest
|C|= |C α (x i )|. (5.30)
N
test i=1
Smallersetsaremoreinformative. Thegoalofcalibrationistoreduce|C|whilemaintaining
coverage.
Conditionalcoveragegap. Forpredefinedsubgroups G 1 ,...,G M,computethegroup-level
coverage and report the worst-case gap:
(cid:12) (cid:12)
∆
cov
= max (cid:12)(1−α)−Cdovg(cid:12). (5.31)
g=1,...,M
A small ∆ indicates that the prediction sets provide uniform coverage across groups.
cov
5.6.5 Step 5: Reporting
The evaluation protocol producesastructured report containing:
1. Data split summary: sizes of training, calibration, and test sets; stratification
strategy; random seed for reproducibility.
2. Calibration metrics table: Brier score, log-loss, and Z-statistic for the uncalibrated
model and each calibrated variant, as in Table 5.3.
3. Reliability diagrams: side-by-side comparison of uncalibrated and calibrated models.
4. Prediction set metrics (if applicable): empirical coverage, average set size, and
conditional coverage gap for each calibration method.
5. Decision: which calibrator (if any) to deploy, with justification based on the metrics
above.
Thedecisionshouldfollowthehierarchyfrom Table 5.6: ifformalguaranteesareneeded,use
Venn–Abers;ifthecalibrationsetislargeandthescoredistributioniswell-understood,Platt
or beta calibration may suffice; if the Z-test passes without calibration, consider deploying
the raw model. The next section demonstrates this protocol end to end (Section 5.7).
5.7 End-to-End Worked Example
This section walks through the complete pipeline—from raw classifier to calibrated con-
formal prediction sets—on the same synthetic task used in the calibrator comparison
(Section 5.2.5). The purpose is to demonstrate every step of the evaluation protocol from
Section 5.6 inasingle, reproducible workflow.

5.7.1 Setup
The task parameters are identical to the calibrator comparison:
• n = 5,000 samples, d = 200 features (10 informative, 10 redundant), class weights
[0.95,0.05].
• Three-way split: 60% training (n =3,000), 20% calibration (n =1,000), 20% test
train cal
(n =1,000).
test
• Base classifier: logistic regression with default scikit-learn regularisation (C =1.0, L 2
penalty).
• d/n =200/3,000≈0.067: the moderate-to-high regime where logistic regression is
train
known to produce overconfident probabilities (Section 5.1.2).
5.7.2 Step 1: Train and Score
Thebaseclassifierisfittedonthetrainingset. Onthecalibrationandtestsets,theclassifier
produces raw probability estimates pˆ(x)=σ(x⊤βˆ), where σ is the sigmoid function and βˆ
are the fitted coefficients.
At this stage, the raw probabilities exhibit the overconfidence pattern predicted by the
theory of Bai, Lee, and Liang [5]: the d/n ratio pushes the estimated coefficients away from
zero, inflating the logit magnitudes and producing probabilities that are too close to 0 or 1.
1 import numpy as np
2 from sklearn.datasets import make_classification
3 from sklearn.linear_model import Logistic Regression
4 from sklearn.metrics import brier_score_loss, log_loss
5 from venn_abers import Venn Abers Calibrator
7 # --- Step 1: Generate data and three-way split ---
8 X, y = make_classification(n_samples=5000, n_features=200,
n_informative=10, n_redundant=10,
10 weights=[0.95, 0.05], random_state=42)
11 idx = np.random.Random State(42).permutation(len(y))
12 X_train, y_train = X[idx[:3000]], y[idx[:3000]]
13 X_cal, y_cal = X[idx[3000:4000]], y[idx[3000:4000]]
14 X_test, y_test = X[idx[4000:]], y[idx[4000:]]
16 # --- Step 2: Train base classifier ---
17 lr = Logistic Regression(max_iter=2000, random_state=42)
lr.fit(X_train, y_train)
19 raw_probs = lr.predict_proba(X_test)[:, 1]
21 # --- Step 3: Evaluate raw model ---
22 print(f"Raw Brier: {brier_score_loss(y_test, raw_probs):.4 f}")
23 print(f"Raw Log Loss: {log_loss(y_test, raw_probs):.4 f}")
25 # --- Step 4: Calibrate with Venn-Abers ---
26 va = Venn Abers Calibrator(estimator=lr, inductive=True,
cal_size=0.25, random_state=42)
va.fit(X_train, y_train)
29 va_probs = va.predict_proba(X_test)[:, 1]
30 print(f"VA Brier: {brier_score_loss(y_test, va_probs):.4 f}")
31 print(f"VA Log Loss: {log_loss(y_test, va_probs):.4 f}")
Listing 5.8: End-to-end pipeline: data generation, three-way split, and baseline evaluation
The complete notebook with APS construction, reliability diagrams, and interactive visuali-
sations is available in the Pro edition:

5.7 End-to-End Worked Example 159
https://valeman.gumroad.com/l/applied_conformal_prediction_pro
5.7.3 Step 2: Calibrate
Five calibration operators are applied to the calibration set scores, each using cv='prefit'
to ensureafair comparison:
1. Platt scaling: fitsatwo-parameter sigmoid to the calibration scores.
2. Isotonic regression: fitsanon-decreasing step function via the PAV algorithm.
3. Beta calibration: fitsathree-parameter beta family to the calibration scores.
4. Temperature scaling: divides the logits byalearned temperature T >0.
5. Venn–Abers: fits inductive Venn–Abers with cal_size=0.25.
5.7.4 Step 3: Evaluate Calibration
The calibration metrics on the test set reproduce the results from Table 5.3:
• The raw logistic regression fails the Z-test (Z =+3.12, |Z|>1.96), confirming system-
atic overconfidence.
• All five calibrators pass the Z-test (|Z|<1.96).
• Beta calibration achieves the best Brier score (0.0288); Venn–Abers achieves competi-
tive Brier (0.0337) while being the only method with sequential calibration guarantees.
The reliability diagram (Figure 5.4) shows the raw model's sigmoid distortion corrected by
each calibrator, confirming the numerical metrics visually.
5.7.5 Step 4: Conformal Prediction Sets
With the calibrated probabilities in hand, the Adaptive Prediction Sets (APS) method [72]—
developed fully inalater chapter—is previewed here (full treatment appears in later
chapters) to construct prediction sets at coverage level 1−α=0.90.
Procedure.
1. For each calibrated model, compute the APS nonconformity scores on the calibration
set.
2. Find the conformal quantile qˆ=⌈0.90·(n +1)⌉/n quantile of the calibration scores.
cal cal
3. On the test set, construct prediction sets by accumulating classes in decreasing
probability order until the cumulative sum exceeds qˆ.
Results. In this binary task (K =2), the prediction set is either {0}, {1}, or {0,1}. The key
metric is the fraction of singleton prediction sets (sets containing exactly one class):
• Raw LR: the overconfident probabilities produce many singleton sets that miss the
true class. The empirical coverage is at the target (≥ 90%) by construction, but
the singletons are concentrated on easy examples, leaving hard examples with the
uninformative set {0,1}.
• Venn–Abers: the calibrated probabilities produceahigher fraction of correct single-
tons, because the probability ranking is faithful to the true class likelihoods.
This demonstrates the calibration-efficiency connection from Section 5.4.9: better calibra-
tion yields more informative prediction sets at the same coverage level.
5.7.6 Step 5: Report and Decide
The complete evaluation produces the following decision chain:
1. Is the raw model calibrated? No—the Z-test rejects (Z =+3.12). Calibration is

needed.
2. Which calibrator? All five pass the Z-test. If formal guarantees are required, use
Venn–Abers. If minimal Brier score is the priority and the calibration set is large, beta
calibration is optimal.
3. Prediction set efficiency. Calibrated models produce more informative prediction
sets (higher singleton fraction) than the raw model.
4. Deployment recommendation. For this task: deploy Venn–Abers as the calibration
backbone, followed by APS (detailed in later chapters) for conformal prediction sets.
This combination provides sequential calibration guarantees, competitive discrimina-
tion, and efficient prediction sets.
The companion notebooks, available in the Pro upgrade at
https://valeman.gumroad.com/l/applied_conformal_prediction_pro
contain the full executable code for this worked example, including the reliability diagrams,
calibration metrics, and prediction set analysis.
5.8 Synthesis
This chapter has developed the theory and practice of probability calibration from first
principles, connected it to conform al prediction, and demonstrated the complete pipeline on
controlled experiments. This final section distils the key results into actionable guidance.
5.8.1 The Five Core Results
1. No classifier is calibrated by default. Logistic regression, despite optimising log-loss
and outputting probabilities through the sigmoid function, is not automatically calibrated.
The overconfidence is structural, driven by the d/n ratio (Section 5.1.2). Across 504 model–
dataset pairs in the 72-dataset benchmark, notasingle base classifier produced acceptably
calibrated probabilities without post-hoc correction [54].
2. Calibration can degrade strong models. Post-hoc calibration is not universally
beneficial. Isotonicregressiondegradedlog-lossin 18%ofcases,and Plattscalingdegraded
it in 23% (Section 5.3.8). The "always calibrate" advice is wrong. Every calibration decision
must be justified by proper scoring rules with decomposition.
3. Venn–Abers is the safest calibrator. Venn–Abers calibration improved log-loss in 96%
of model–dataset pairs with worst-case degradation below 1% [54]. It is the only calibrator
with sequential calibration guarantees that hold for any calibration set size (Section 5.4),
making it the recommended default for safety-critical applications and small calibration
sets.
4. Calibration improves conformal efficiency. Better-calibrated probabilities yield
smaller prediction sets under the adaptive conformal methods presented in later chapters
(Section 5.4.9). Calibration is not just about probability quality—it directly affects the
informativeness of conformal prediction.
5. Conditional coverage requires explicit effort. Marginal coverage can hide subgroup
disparities. Mondrian conformal prediction provides exact group-conditional coverage at
the cost of splitting the calibration set; multivalid conformal prediction handles overlapping
groups (Section 5.5). Calibration narrows the gap between marginal and conditional
coverage even without explicit Mondrian procedures.

5.8 Synthesis 161
5.8.2 Decision Flowchart
The practitioner's decision process is:
1. Evaluate the raw model. Compute Brier score, log-loss, the Spiegelhalter Z-statistic,
and produceareliability diagram onaheld-out test set (Section 5.6).
2. If |Z|≤1.96: the raw model passes the calibration test. Consider deploying without
post-hoc calibration, but verify withareliability diagram that there are no localised
distortions.
3. If |Z|>1.96: calibration is needed. Chooseacalibrator based on the decision guide
(Table 5.6):
• Formal guarantees needed ⇒ Venn–Abers.
• Small calibration set (m<500) ⇒ Platt, beta, temperature, or Venn–Abers (not
isotonic).
• Large calibration set, known score distribution ⇒ beta calibration.
• Multi-class deep network ⇒ temperature scaling.
4. If conformal prediction sets are required: apply the adaptive conformal methods
(APS/RAPS) presented in later chapters to the calibrated probabilities. For group-level
coverage, use Mondrian conformal prediction (Section 5.5.2).
5. Report. Follow the reporting protocol from Section 5.6.5: data split summary, cal-
ibration metrics table, reliability diagrams, prediction set metrics, and deployment
recommendation.
5.8.3 Looking Ahead
This chapter has focused on classification calibration and its interaction with conformal
prediction. Several extensions are active areas of research:
• Regression calibration. Calibrating prediction intervals and quantile estimates is
the regression analogue of the classification problem treated here. The subsequent
chapter on regression develops this connection in full, extending the evaluation
protocol and Venn-prediction framework to continuous responses. Conformalised
quantile regression [69] provides one solution; calibrating the base quantile estimator
before conformalisation is an open direction.
• Multi-class Venn–Abers. The current Venn–Abers framework handles binary classifi-
cation. Extending it to multi-class problems with finite-sample guarantees is an area
of ongoing work.
• Online calibration. In streaming settings, the exchangeability assumption may
weaken. Adaptive conformal inference methods that track distribution shift while
maintaining calibration guarantees areanatural next step.
The tools developed in this chapter—proper scoring rules, the Spiegelhalter Z-test, Venn–
Aberscalibration,andtheevaluationprotocol—providearigorousfoundationforallofthese
extensions.
The companion Jupyter notebooks for all experiments, reliability diagrams, and worked
examples in this chapter are available in the Pro upgrade:
https://valeman.gumroad.com/l/applied_conformal_prediction_pro

[1] M. A. Aizerman, E. M. Braverman, and L. I. Rozonoer. "Theoretical Foundations of
the Potential Function Method in Pattern Recognition Learning". In: Automation and
Remote Control 6 (1964), pages 82–98 (cited on page 18).
[2] Marharyta Aleksandrova and Oleg Chertov. "Impact of model-agnostic nonconformity
functions on efficiency of conformal classifiers: an extensive study". In: Proceedings
of the Tenth Symposium on Conformal and Probabilistic Prediction and Applica-
tions (COPA). Volume 152. Proceedings of Machine Learning Research. PMLR, 2021,
pages 151–170. URL: https://proceedings.mlr.press/v 152/aleksandrova 21 a.
html (cited on pages 72, 77, 82).
[3] AnastasiosNAngelopoulos and Stephen Bates. "Conformal prediction: A gentle intro-
duction". In: Foundations and Trends in Machine Learning 16.4 (2023), pages 494–
591 (cited on pages 40, 45, 49, 52, 71).
[4] AnastasiosNAngelopoulos et al. "Uncertainty sets for image classifiers using con-
formal prediction". In: International Conference on Learning Representations. 2021
(cited on pages 71, 72, 74–76, 137, 151).
[5] Yu Bai, Jason D. Lee, and Tengyu Liang. Don't Just Blame Over-Parametrization for
Over-Confidence: Theoretical Analysis of Calibration in Binary Classification. ar Xiv
preprint ar Xiv:2102.07856. Available at https://arxiv.org/abs/2102.07856. 2021
(cited on pages 123, 125–128, 133, 158).
[6] Jie Baoetal."A Reviewand Comparative Analysisof Univariate Conformal Regression
Methods". In: Proceedings of the Fourteenth Symposium on Conformal and Proba-
bilistic Prediction with Applications. Edited by Khuong An Nguyen et al. Volume 266.
Proceedings of Machine Learning Research. PMLR, Oct. 2025, pages 282–304. URL:
https://proceedings.mlr.press/v 266/bao 25 a.html (cited on pages 115, 119).
[7] Rina Foygel Barber, Emmanuel Candès, and Ryan Tibshirani. "Conformal Prediction
under Covariate Shift".In:Advancesin Neural Information Processing Systems (2021)
(cited on pages 25, 152).
[8] Rina Foygel Barberetal."Predictiveinferencewiththejackknife+".In:The Annalsof
Statistics 49.1 (2021), pages 486–507 (cited on pages 45, 51, 67, 91, 101, 102, 108).
[9] Richard E. Barlow et al. Statistical Inference Under Order Restrictions: The Theory
and Application of Isotonic Regression. New York: John Wiley & Sons, 1972 (cited on
page 140).
[10] Bernhard E. Boser, Isabelle M. Guyon, and Vladimir N. Vapnik. "A Training Algo-
rithm for Optimal Margin Classifiers". In: Proceedings of the Fifth Annual Workshop
on Computational Learning Theory (COLT). ACM, 1992, pages 144–152 (cited on
page 19).
[11] Henrik Bostrom, Ulf Johansson, and Tuve Lofstrom. "Accelerating Conformal Regres-
sors Using Inductive Confidence Machines". In: Proceedings of the International
Joint Conference on Neural Networks (IJCNN). IEEE. 2017, pages 1159–1166. DOI:
10.1109/IJCNN.2017.7965960 (cited on pages 102–104).
[12] Henrik Bostrom,Ulf Johansson,and Tuve Lofstrom."Mondrian Conformal Predictors".
In:Machine Learning 109.9-10(2020),pages 1909–1939.DOI:10.1007/s 10994-020-
05918-1 (cited on pages 103, 104).
[13] Henrik Bostrom, Ulf Johansson, and Tuve Lofstrom. "Mondrian Predictive Distribu-
tions". In: Annals of Mathematics and Artificial Intelligence 89.2-4 (2021), pages 199–
223. DOI: 10.1007/s 10472-020-09729-2 (cited on pages 103, 104).

5.8 Synthesis 163
[14] GlennWBrier. "Verification of forecasts expressed in terms of probability". In:
Monthly Weather Review 78.1 (1950), pages 1–3 (cited on page 129).
[15] Jochen Broecker. "Reliability, sufficiency, and the decomposition of proper scores". In:
Quarterly Journal of the Royal Meteorological Society 135.643 (2009), pages 1512–
1519. DOI: 10.1002/qj.456 (cited on pages 122, 126, 138).
[16] Gregory J. Chaitin. "A Theory of Program Size Formally Identical to Information
Theory". In: Journal of the ACM 22.3 (1975), pages 329–340 (cited on page 15).
[17] Victor Chernozhukov, Kaspar Wüthrich, and Yinchu Zhu. "Exact and robust confor-
mal inference methods for predictive machine learning with dependent data". In:
Conference on Learning Theory. PMLR. 2018, pages 732–749 (cited on page 40).
[18] A.Philip Dawid."Thewell-calibrated Bayesian".In:Journalofthe American Statistical
Association 77.379 (1982), pages 605–610 (cited on pages 122, 131, 146).
[19] MorrisHDe Groot and StephenEFienberg. "The comparison and evaluation of
forecasters". In: Journal of the Royal Statistical Society: Series D (The Statistician)
32.1-2 (1983), pages 12–22 (cited on page 131).
[20] Paramveer Dhillon,Anastasios N.Angelopoulos,and Stephen Bates."Onthe Expected
Size of Conformal Prediction Sets". In: International Conference on Artificial Intel-
ligence and Statistics (AISTATS). 2024. URL: https://arxiv.org/abs/2306.07254
(cited on page 50).
[21] Timo Dimitriadis, Tilmann Gneiting, and Alexander Jordan. "Evaluating Probabilistic
Classifiers: The Triptych". In: ar Xiv preprint ar Xiv:2301.10803 (2023). URL: https:
//arxiv.org/abs/2301.10803 (cited on pages 91, 156).
[22] Werner Ehm et al. "Of Quantiles and Expectiles: Consistent Scoring Functions,
Choquet Representations and Forecast Rankings". In: Journal of the Royal Statis-
tical Society: Series B (Statistical Methodology) 78.3 (2016), pages 505–562. DOI:
10.1111/rssb.12154 (cited on page 156).
[23] Matteo Fontana, Giacomo Zeni, and Simone Vantini. "Conformal prediction: A unified
review of theory and new challenges". In: Bernoulli 29.1 (2023), pages 1–28 (cited on
pages 40, 59).
[24] Alexander Gammerman, Vladimir Vovk, and Vladimir Vapnik. "Learning by Trans-
duction". In: Proceedings of the Fourteenth Conference on Uncertainty in Artificial
Intelligence (UAI 1998). San Francisco, CA: Morgan Kaufmann, 1998, pages 148–155.
URL: https://arxiv.org/abs/1106.0722 (cited on pages 52, 53, 56).
[25] Noé Gazin, Gilles Blanchard, and Etienne Roquain. "Transductive Conformal Pre-
diction with Multiple Test Points". In: Proceedings of the 27 th International Con-
ference on Artificial Intelligence and Statistics (AISTATS). Volume 238. Proceed-
ings of Machine Learning Research. PMLR, 2024, pages 11441–11465. URL: https:
//proceedings.mlr.press/v 238/gazin 24 a.html (cited on page 101).
[26] Tilmann Gneiting and Matthias Katzfuss. "Probabilistic forecasting". In: Annual Re-
view of Statistics and Its Application 1.1 (2014), pages 125–151. DOI: 10.1146/
annurev-statistics-062713-085831 (cited on pages 40, 42).
[27] Tilmann Gneiting and Adrian E. Raftery. "Strictly proper scoring rules, prediction,
and estimation". In: Journal of the American Statistical Association 102.477 (2007),
pages 359–378 (cited on page 131).

[28] Ruben van den Goorbergh et al. "The Harm of Class Imbalance Corrections for Risk
Prediction Models: Illustration and Simulation Using Logistic Regression". In: Journal
of the American Medical Informatics Association 29.9 (2022), pages 1525–1534. DOI:
10.1093/jamia/ocac 093 (cited on page 123).
[29] Parikshit Gopalan et al. "Multivalid Conformal Prediction". In: Advances in Neural
Information Processing Systems. 2022 (cited on page 137).
[30] Parikshit Gopalan et al. "Multivalid Conformal Prediction". In: Advances in Neural In-
formation Processing Systems.Volume 35.Curran Associates,Inc.,2022,pages 16995–
17007.URL:https://proceedings.neurips.cc/paper/2022/file/2 b 9227 aa 64 c 6 c 740249 b 0 f 707 cbfda 0 c-
Paper-Conference.pdf (cited on pages 81, 153, 154).
[31] Ananya Guha et al. "Conformal Prediction via Regression-as-Classification". In: Inter-
national Conference on Learning Representations (ICLR). 2024 (cited on page 117).
[32] Chuan Guo et al. "On calibration of modern neural networks". In: International
conference on machine learning. PMLR. 2017, pages 1321–1330 (cited on pages 68,
77, 122, 133, 135, 141, 143, 153, 156).
[33] Chirag Gupta, Danijel Kivaranovic, and Aaditya Ramdas. "Distribution-free Binary
Classification: Prediction Sets, Confidence Intervals and Calibration". In: Advances in
Neural Information Processing Systems. Volume 34. Curran Associates, Inc., 2021,
pages 17410–17421. URL: https://proceedings.neurips.cc/paper/2021/file/
2 ab 56412 f 5 a 7 b 86 ed 66376 e 3 a 1 ee 9 b 4 f-Paper.pdf (cited on pages 81, 137).
[34] Jianguo Huang et al. "Conformal Predictionfor Deep Classifier via Label Ranking". In:
ar Xiv preprint ar Xiv:2310.06430 (2024). URL: https://arxiv.org/abs/2310.06430
(cited on pages 72, 75–77).
[35] Jianguo Huang et al. "Conformal Prediction for Deep Classifier via Label Ranking".
In: Proceedings of the 41 st International Conference on Machine Learning (ICML).
Volume 235. Proceedings of Machine Learning Research. 2024. URL: https://
proceedings.mlr.press/v 235/huang 24 aa.html (cited on page 77).
[36] Ulf Johansson and Patrick Gabrielsson. "Are Traditional Neural Networks Well-
Calibrated?" In: 2019 International Joint Conference on Neural Networks (IJCNN).
IEEE, 2019, pages 1–8 (cited on pages 122, 133).
[37] Ulf Johanssonand Per Gabrielsson."Are Traditional Neural Networks Well-Calibrated?"
In: Proceedings of the International Joint Conference on Neural Networks (IJCNN).
IEEE, 2019, pages 1–8. DOI: 10.1109/IJCNN.2019.8852053 (cited on page 68).
[38] Ulf Johansson,Tuve Lofstrom,and Henrik Bostrom."Regression Conformal Prediction
with Random Forests". In: Machine Learning 97.1-2 (2014), pages 155–176. DOI:
10.1007/s 10994-014-5453-0 (cited on pages 102–104).
[39] Ulf Johansson et al. "Model-agnostic nonconformity functions for conformal classifi-
cation". In: 2017 International Joint Conference on Neural Networks (IJCNN). IEEE,
2017, pages 2072–2079. DOI: 10.1109/IJCNN.2017.7966105 (cited on page 82).
[40] Ulf Johanssonetal."Interpretableand Specialized Conformal Predictors".In:Proceed-
ings of the Eighth Symposium on Conformal and Probabilistic Prediction and Applica-
tions (COPA). Volume 105. Proceedings of Machine Learning Research. PMLR, 2019,
pages 3–22. URL: https://proceedings.mlr.press/v 105/johansson 19 a.html
(cited on pages 72, 77).

5.8 Synthesis 165
[41] Andrey N. Kolmogorov. "Three Approaches to the Definition of the Quantity of Infor-
mation". In: Problems of Information Transmission 1 (1965), pages 3–11 (cited on
page 14).
[42] Meelis Kull, Telmo Silva Filho, and Peter Flach. "Beta calibration: a well-founded and
easily implemented improvement on logistic calibration for binary classifiers". In:
Artificial Intelligence and Statistics. PMLR. 2017, pages 623–631 (cited on pages 68,
133, 135, 140, 141).
[43] Ananya Kumar, Percy Liang, and Tengyu Ma. "Verified Uncertainty Calibration". In:
ar Xiv preprint ar Xiv:1909.10155 (2019) (cited on page 133).
[44] Kiljae Lee and Yuan Zhang. "Leave-One-Out Stable Conformal Prediction". In: ar Xiv
preprintar Xiv:2504.12189 (2025).URL:https://arxiv.org/abs/2504.12189(cited
on page 101).
[45] Jing Lei and Larry Wasserman. "Distribution-free prediction bands for non-parametric
regression". In: Journal of the Royal Statistical Society: Series B (Statistical Method-
ology) 76.1 (2014), pages 71–96 (cited on pages 105, 106).
[46] Jing Leiand Larry Wasserman."Distribution-Free Predictive Inferencefor Regression".
In: Journal of the American Statistical Association 113.523 (2018), pages 1094–1111
(cited on pages 25, 101, 105, 109).
[47] Jing Lei et al. Distribution-Free Predictive Inference For Regression. 2017. ar Xiv:
1604.04173 [stat.ME]. URL: https://arxiv.org/abs/1604.04173 (cited on
pages 38, 40, 41, 44, 46, 59, 67, 91, 152).
[48] Ming Li and Paul Vitányi. "An Introduction to Kolmogorov Complexity and Its Applica-
tions". In: (2008). 3 rd Edition (cited on pages 14, 15).
[49] Henrik Linusson et al. "Model-Agnostic Nonconformity Functions for Conformal
Classification". In: IEEE Proceedings (see IEEE Xplore Document 7966105). 2017.
URL: https://ieeexplore.ieee.org/abstract/document/7966105 (cited on
page 70).
[50] Rui Luo and Zhixin Zhou. "Conformal Thresholded Intervals for Efficient Regression".
In: ar Xiv preprint ar Xiv:2407.14495 (2025) (cited on pages 101, 106, 115, 116, 119).
[51] Valery Manokhin. "Multi-class probabilistic classification using inductive and cross
Venn–Abers predictors". In: Conformal and Probabilistic Prediction and Applications.
PMLR. 2017, pages 228–240 (cited on pages 83, 85–89, 127, 148).
[52] Valery Manokhin. Practical Guide to Applied Conformal Prediction in Python. Packt
Publishing, 2023. ISBN: 9781805120919 (cited on pages 41, 44, 55, 68).
[53] Valery Manokhin. Logistic Regression: The Myth of Natural Calibration. Substack /
Medium. Argues that GLM calibration equations guarantee only marginal calibration,
notauto-calibration;thesigmoidlinkstructurallybakesinoverconfidence.2025.URL:
https://valeman.substack.com (cited on pages 123, 137).
[54] Valery Manokhin and Daniel Grønhaug. "Classifier Calibration at Scale: An Empirical
Study of Model-Agnostic Post-Hoc Methods". In: ar Xiv preprint ar Xiv:2601.19944
(2026) (cited on pages 122, 128, 137, 142, 144, 149, 151, 160).
[55] Paulo Marques. "A Universal Distribution for the Coverage of Split Conformal Predic-
tion". In: Statistics & Probability Letters 219 (2025), page 110350. DOI: 10.1016/j.
spl.2024.110350 (cited on page 50).

[56] Adil Messoudietal."Distributional Conformal Prediction".In:ar Xivpreprintar Xiv:2306.07254
(2023). URL: https://arxiv.org/abs/2306.07254 (cited on page 91).
[57] Matthias Minderer et al. "Revisiting the Calibration of Modern Neural Networks". In:
Advances in Neural Information Processing Systems. Volume 34. 2021, pages 15682–
15694.URL:https://proceedings.neurips.cc/paper/2021/hash/fb 0 a 3 b 9 d 1 b 7 c 7 b 7 cd 4 e 3 e 7 c 6 f 1 d 7 a 4 a 2-
Abstract.html (cited on page 122).
[58] Richardvon Mises.Probability,Statisticsand Truth.Englishtranslationoftheoriginal
1928 German edition. London: Macmillan, 1928 (cited on page 14).
[59] Allan H. Murphy. "A New Vector Partition of the Probability Score". In: Journal of
Applied Meteorology 12.4 (1973), pages 595–600. DOI: 10.1175/1520-0450(1973)
012<0595:ANVPOT>2.0.CO;2. URL: https://doi.org/10.1175/1520-0450(1973)
012%3 C 0595:ANVPOT%3 E 2.0.CO;2 (cited on page 129).
[60] Allan H.Murphyand Robert L.Winkler."Reliabilityof Subjective Probability Forecasts
of Precipitation and Temperature". In: Journal of the Royal Statistical Society. Series
C (Applied Statistics) 26.1 (1977), pages 41–47 (cited on page 129).
[61] Alexandru Niculescu-Mizil and Rich Caruana. "Predicting good probabilities with
supervisedlearning".In:Proceedingsofthe 22 nd international conference on Machine
learning. 2005, pages 625–632 (cited on pages 63, 68, 70, 122, 126–128, 132).
[62] Sangdon Ohn and Juhyun Park. "Fast and Efficient Conformal Prediction with Kernel
Ridge Regression". In: AIMS Mathematics 10.3 (2025), pages 6024–6046. DOI: 10.
3934/math.2025236. URL: https://www.aimspress.com/article/doi/10.3934/
math.2025236 (cited on page 101).
[63] Fabian M. Ojeda et al. "Calibrating Machine Learning Approaches for Probability
Estimation: A Comprehensive Comparison". In: Statistics in Medicine 42.28 (2023),
pages 5065–5097. DOI: 10.1002/sim.9899 (cited on page 123).
[64] Harris Papadopoulos. "Normalized nonconformity measures for regression confor-
mal prediction". In: Proceedings of the 2008 International Conference on Machine
Learning and Applications. IEEE. 2008, pages 64–69 (cited on pages 99, 102, 103).
[65] Harris Papadopoulos, Alexander Gammerman, and Vladimir Vovk. "Inductive Confor-
mal Prediction: Theory and Application to Neural Networks". In: Machine Learning:
ECML 2002. Volume 2430. Lecture Notes in Artificial Intelligence. Springer, 2002,
pages 345–356 (cited on pages 23, 44, 99, 106).
[66] JohnCPlatt. "Probabilistic Outputs for Support Vector Machines and Comparisons to
Regularized Likelihood Methods". In: Advances in Large Margin Classifiers (1999),
pages 61–74 (cited on pages 77, 89, 122, 131, 139, 143).
[67] Aleksandr Podkopaev and Aaditya Ramdas. "Distribution-free uncertainty quantifica-
tion for classification under label shift". In: Proceedings of the 37 th Conference on
Uncertainty in Artificial Intelligence. 2021 (cited on page 40).
[68] Edward Prinster et al. "Conformal Validity Guarantees Exist for Any Data Distribution
(and How to Find Them)". In: ar Xiv preprint (2024). ar Xiv: 2404.04795 (cited on
pages 25, 38, 64).
[69] Yaniv Romano, Evan Patterson, and Emmanuel J. Candes. "Conformalized Quantile
Regression".In:Advancesin Neural Information Processing Systems(Neur IPS).2019.
URL: https://arxiv.org/abs/1905.03222 (cited on pages 44, 101, 106, 111, 112,
119, 161).

5.8 Synthesis 167
[70] Yaniv Romano, Matteo Sesia, and Emmanuel J. Candes. "Classification with Valid
and Adaptive Coverage". In: Advances in Neural Information Processing Systems
(Neur IPS). 2020. ar Xiv: 2006.02544. URL: https://proceedings.neurips.cc/
paper/2020/hash/244 edd 7 e 85 dc 81602 b 7615 cd 705545 f 5-Abstract.html (cited on
pages 71–74, 76).
[71] Yaniv Romano, Matteo Sesia, and Emmanuel J Candès. "Classification with valid
and adaptive coverage". In: Advances in Neural Information Processing Systems 33
(2020), pages 3581–3591 (cited on pages 44, 59, 77).
[72] Yaniv Romano, Matteo Sesia, and Emmanuel J. Candès. "Classification with Valid and
Adaptive Coverage". In: Advances in Neural Information Processing Systems. 2020
(cited on pages 137, 151, 159).
[73] Glenn Shafer and Vladimir Vovk. "A Tutorial on Conformal Prediction". In: Journal of
Machine Learning Research 9 (2008), pages 371–421 (cited on pages 37, 72, 81).
[74] Glenn Shafer and Vladimir Vovk. "A tutorial on conformal prediction". In: Journal of
Machine Learning Research 9 (2008), pages 371–421 (cited on pages 37, 38, 43).
[75] Ray J. Solomonoff. "A Formal Theory of Inductive Inference: PartsIand II". In:
Information and Control 7.1-2 (1964), pages 1–22, 224–254 (cited on page 15).
[76] David J. Spiegelhalter. "Probabilistic Prediction in Patient Management and Clinical
Trials". In: Statistics in Medicine 5.5 (1986), pages 421–433. DOI: 10.1002/sim.
4780050506 (cited on page 130).
[77] David J. Spiegelhalter. "Probabilistic prediction in patient management and clinical
trials". In: Statistics in Medicine 5.5 (Oct. 1986), pages 421–433. DOI: 10.1002/
sim.4780050506. URL: https://pubmed.ncbi.nlm.nih.gov/3786996/ (cited on
pages 133, 138).
[78] Jiaye Teng et al. "Predictive Inference with Feature Conformal Prediction". In: ar Xiv
preprintar Xiv:2210.00173 (2022).URL:https://arxiv.org/abs/2210.00173(cited
on page 72).
[79] Ryan Tibshiranietal."Conformal Predictionunder Covariate Shift".In:ar Xivpreprint
(2019). ar Xiv: 1904.06019 (cited on page 25).
[80] RyanJTibshirani et al. "Conformal prediction under covariate shift". In: Advances in
Neural Information Processing Systems. Volume 32. 2019 (cited on pages 40, 154).
[81] Juozas Vaicenavicius et al. "Evaluating Model Calibration in Classification". In: Pro-
ceedings of the 22 nd International Conference on Artificial Intelligence and Statistics
(AISTATS). Volume 89. PMLR, 2019, pages 3459–3467. URL: https://proceedings.
mlr.press/v 89/vaicenavicius 19 a.html (cited on pages 133, 139).
[82] V. N. Vapnik and A. Y. Chervonenkis. "A Note on One Class of Algorithms for Pattern
Recognition Based on the Generalized Portrait Method". In: Automation and Remote
Control 25(1964).Originallypublishedin Russian,pages 821–837(citedonpages 17,
18).
[83] Vladimir N. Vapnik. Statistical Learning Theory. New York: Wiley, 1998 (cited on
pages 17, 52).
[84] Vladimir Vovk. "Conditional Validity of Inductive Conformal Predictors". In: Ma-
chine Learning 92.2–3 (2013). Proves impossibility of exact conditional coverage for
distribution-free conformal predictors, pages 349–376. DOI: 10.1007/s 10994-013-
5355-6 (cited on page 152).

[85] Vladimir Vovk. "Cross-conformal predictors". In: Annals of Mathematics and Artificial
Intelligence 74.1-2 (2015), pages 9–28 (cited on pages 45, 51, 101).
[86] Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. Algorithmic learning in a
random world. Springer Science & Business Media, 2005 (cited on pages 21, 37, 38,
40, 41, 46, 51, 67, 99–101, 136, 144, 146, 152).
[87] Vladimir Vovkand Ivan Petej."Venn–Aberspredictors".In:Proceedingsofthe Thirtieth
Conferenceon Uncertaintyin Artificial Intelligence.AUAI Press.2014,pages 829–838
(cited on pages 69, 83, 136, 144, 147, 148).
[88] Chengrun Yang, Caleb Fannjiang, and Mihaela van der Schaar. "On the Validity of
Conformal Prediction under Feedback Covariate Shift". In: ar Xiv preprint (2022).
ar Xiv: 2205.08809 (cited on page 25).
[89] Bianca Zadrozny and Charles Elkan. "Obtaining calibrated probability estimates from
decision trees and naive Bayesian classifiers". In: Proceedings of the Eighteenth
International Conference on Machine Learning (2001), pages 609–616 (cited on
pages 83, 89).
[90] Bianca Zadrozny and Charles Elkan. "Transforming Classifier Scores into Accurate
Multiclass Probability Estimates". In: Proceedings of the 8 th ACM SIGKDD Interna-
tional Conference on Knowledge Discovery and Data Mining (KDD 2002). ACM, 2002,
pages 694–699 (cited on pages 83, 89).
[91] Bianca Zadrozny and Charles Elkan. "Transforming Classifier Scores into Accurate
Multiclass Probability Estimates". In: Proceedings of the Eighth ACM SIGKDD Inter-
national Conference on Knowledge Discovery and Data Mining (KDD). ACM, 2002,
pages 694–699 (cited on pages 132, 139).
[92] Bianca Zadrozny and Charles Elkan. "Transforming classifier scores into accurate
multiclassprobabilityestimates".In:Proceedingsofthe Eighth ACMSIGKDD Interna-
tional Conference on Knowledge Discovery and Data Mining (2002), pages 694–699
(cited on page 77).
[93] Bianca Zadroznyand Charles Elkan."Obtaining Calibrated Probability Estimatesfrom
Decision Trees and Naive Bayesian Classifiers". In: Proceedings of the Eighteenth
International Conference on Machine Learning (ICML), pages 609–616 (cited on
page 132).
[94] Julien Zaffranetal."Adaptive Conformal Inferenceunder Distribution Shift".In:ar Xiv
preprint (2022). ar Xiv: 2202.13415 (cited on page 25).
[95] Margaux Zaffran et al. "Adaptive conformal predictions for time series". In: Interna-
tional Conference on Machine Learning. PMLR. 2022, pages 25836–25854 (cited on
pages 40, 60).
