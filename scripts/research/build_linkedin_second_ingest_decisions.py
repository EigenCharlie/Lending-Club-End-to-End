#!/usr/bin/env python3
"""Write decision tables and memos for the second Denis Burakov LinkedIn ingest."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("reports/linkedin_credit_risk_denis_burakov")
PACK = ROOT / "second_ingest"
DATA = PACK / "data"
DOCS = PACK / "docs"


BACKLOG_FIELDS = [
    "item_id",
    "item_kind",
    "activity_or_article_id",
    "title",
    "source_status",
    "content_read",
    "attachment_or_visual_status",
    "external_source_status",
    "project_destination",
    "possible_executable_or_implementable",
    "decision",
    "stop_condition",
    "closure_status",
]


CONCEPT_FIELDS = [
    "concept",
    "source_items",
    "method_family",
    "novelty_for_project",
    "evidence_strength",
    "project_destination",
    "implementation_difficulty",
    "claim_risk",
    "decision",
    "stop_condition",
]


VISUAL_FIELDS = [
    "visual_item_id",
    "source_id",
    "visual_count",
    "visual_read_status",
    "project_signal",
]


BACKLOG_ROWS = [
    {
        "item_id": "S2-POST-001",
        "item_kind": "linkedin_post",
        "activity_or_article_id": "7447901245949792256",
        "title": "You already understand LLMs better than you think",
        "source_status": "linkedin_post_plus_medium_link_blocked_for_full_read",
        "content_read": "Post text read; external Medium article resolved but direct fetch returned target-resolution-only blocker.",
        "attachment_or_visual_status": "No substantive local visual beyond public post text.",
        "external_source_status": "Medium URL resolved; not promoted as evidence.",
        "project_destination": "Book chapter 06 pedagogy only.",
        "possible_executable_or_implementable": "Short framing note connecting MLE, logit, scorecards, and sequence models.",
        "decision": "archive_as_framing",
        "stop_condition": "Stop because it does not change a credit-risk claim, experiment, or table.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-POST-002",
        "item_kind": "linkedin_post",
        "activity_or_article_id": "7376148352284901376",
        "title": "Explainable Credit Risk Models",
        "source_status": "linkedin_post_with_70_page_public_document_pdf",
        "content_read": "Post and 70-page deck read; themes include WOE origins, logistic regression, boosting, pitfalls, and explainability.",
        "attachment_or_visual_status": "PDF text extracted; deck is readable.",
        "external_source_status": "No external source required.",
        "project_destination": "Book chapters 05 and 11; Paper 4 related-work framing.",
        "possible_executable_or_implementable": "Use as conceptual checklist for WOE, sign flips, boosted scorecards, and reason-code governance.",
        "decision": "append_language_no_new_experiment",
        "stop_condition": "Stop after language is incorporated; do not promote LinkedIn deck as scholarly evidence.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-POST-003",
        "item_kind": "linkedin_post",
        "activity_or_article_id": "7211748798618824704",
        "title": "Credit Scoring with Trended Data",
        "source_status": "linkedin_post_with_three_public_images",
        "content_read": "Post and three slides read; slides compare snapshot data to sequence models over transaction histories.",
        "attachment_or_visual_status": "Manual visual read completed; RNN/LSTM/Transformer Gini and Brier examples recorded.",
        "external_source_status": "Linktree resolved and archived as aggregator.",
        "project_destination": "Paper 4 limitations/future work; book feature-engineering caveat.",
        "possible_executable_or_implementable": "Future sequence-data lane if actual transaction histories become available.",
        "decision": "park_data_unavailable",
        "stop_condition": "Stop because Lending Club snapshot data cannot support trended-sequence modeling.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-POST-004",
        "item_kind": "linkedin_post",
        "activity_or_article_id": "7328316894258577408",
        "title": "Poisson Models",
        "source_status": "linkedin_post_with_9_page_public_document_pdf",
        "content_read": "Post and deck read; count-model use cases, Fisher scoring, overdispersion, zero inflation, and negative binomial alternatives reviewed.",
        "attachment_or_visual_status": "PDF text extracted.",
        "external_source_status": "Schema.org link resolved and archived as low-evidence external context.",
        "project_destination": "Book GLM sidebar only.",
        "possible_executable_or_implementable": "No PD implementation; possible future collections/count event module.",
        "decision": "archive_outside_current_pd_target",
        "stop_condition": "Stop because the project target is binary default, not event counts.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-POST-005",
        "item_kind": "linkedin_post",
        "activity_or_article_id": "7381584253375025152",
        "title": "Enhancing Explainability in Credit Scoring",
        "source_status": "linkedin_post_with_13_page_public_document_pdf_and_github",
        "content_read": "Post, paper-style deck, and GitHub README read; constrained trees, responsible ML, calibration, interpretability, and policy testing reviewed.",
        "attachment_or_visual_status": "PDF text extracted.",
        "external_source_status": "GitHub xml-scoring snapshot readable; code context only.",
        "project_destination": "Book chapter 11; Paper 4 related-work/prototype parking.",
        "possible_executable_or_implementable": "Optional future constrained-tree or XML scorecard comparison, only with local benchmark.",
        "decision": "park_prototype",
        "stop_condition": "Stop because it cannot change the promoted champion without a bounded benchmark.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-POST-006",
        "item_kind": "linkedin_post",
        "activity_or_article_id": "7346062624972754944",
        "title": "WOE, Log Odds, and Standard Errors",
        "source_status": "linkedin_post_with_9_page_public_document_pdf_and_fastwoe_github",
        "content_read": "Post, deck, FastWOE README, and WoeBoost article metadata read; WOE as centered log odds and uncertainty quantification reviewed.",
        "attachment_or_visual_status": "PDF text extracted.",
        "external_source_status": "FastWOE GitHub readable; Medium WoeBoost readable as non-peer context.",
        "project_destination": "Book chapter 05; Paper 4 appendix/prototype parking.",
        "possible_executable_or_implementable": "Document WOE standard-error idea; optional WOE uncertainty prototype later.",
        "decision": "append_language_and_park_prototype",
        "stop_condition": "Stop after conceptual language; no dependency added without local benchmark and tests.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-POST-007",
        "item_kind": "linkedin_post",
        "activity_or_article_id": "7320341844326797319",
        "title": "Build explainable scorecards with CatBoost",
        "source_status": "linkedin_post_with_medium_link_blocked_and_xbooster_github",
        "content_read": "Post and xBooster README read; CatBoost scorecard extraction, SHAP scoring, SQL generation, and limitations reviewed.",
        "attachment_or_visual_status": "Text-only post captured.",
        "external_source_status": "xBooster GitHub readable; Medium article fetch blocked but article candidate covered through LinkedIn article capture.",
        "project_destination": "Book chapter 11; Paper 4 prototype parking.",
        "possible_executable_or_implementable": "Future xBooster/CatBoost-to-scorecard comparison if it changes an appendix table.",
        "decision": "park_prototype",
        "stop_condition": "Stop because current champion already has SHAP/explanation governance and no new table is justified.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-POST-008",
        "item_kind": "linkedin_post",
        "activity_or_article_id": "7155476795474030592",
        "title": "Credit Risk Modeling with Naive Bayes",
        "source_status": "linkedin_post_with_3_page_public_document_pdf_and_article",
        "content_read": "Post, short deck, and long Naive Bayes article read; WOE as log-likelihood ratio and NB scorecard workflow reviewed.",
        "attachment_or_visual_status": "PDF text extracted.",
        "external_source_status": "Linktree/schema links archived; associated article captured separately.",
        "project_destination": "Book chapter 05.",
        "possible_executable_or_implementable": "Clarify WOE as evidence/likelihood-ratio language, not only encoding.",
        "decision": "append_language_no_new_model",
        "stop_condition": "Stop because NB scorecard is pedagogically useful but not needed as a new empirical lane.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-POST-009",
        "item_kind": "linkedin_post",
        "activity_or_article_id": "7366001503611924480",
        "title": "Machine Learning in Internal Credit Risk Models",
        "source_status": "linkedin_post_with_23_page_public_document_pdf_and_official_ecb_pdf",
        "content_read": "Post, deck, and ECB guide text read; ML in IRB, reproducibility, observation ordering, explainability, and conceptual soundness reviewed.",
        "attachment_or_visual_status": "PDF text extracted.",
        "external_source_status": "Official ECB guide extracted from 370-page PDF and labeled as supervisory guidance.",
        "project_destination": "Book chapters 10/14/19; Paper Estrella reviewer-defense language.",
        "possible_executable_or_implementable": "Add governance controls for seeds, row ordering, model lineage, explainability, and validation scope.",
        "decision": "append_official_governance_language",
        "stop_condition": "Stop after governance text; do not reopen champion.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-POST-010",
        "item_kind": "linkedin_post",
        "activity_or_article_id": "7372524468570689537",
        "title": "Information-Theoretic Framework for Credit Risk Modeling",
        "source_status": "linkedin_post_with_22_page_preprint_pdf_and_ssrn_link_blocked",
        "content_read": "Post and preprint PDF read; IV, PSI, Jeffreys divergence, standard errors, and fairness/performance tradeoffs reviewed.",
        "attachment_or_visual_status": "PDF text extracted.",
        "external_source_status": "SSRN URL resolved but direct fetch blocked; local LinkedIn PDF treated as preprint artifact.",
        "project_destination": "Book chapters 05 and 10; Paper 4 source-discovery only.",
        "possible_executable_or_implementable": "Mention IV/PSI uncertainty as future governance instrumentation.",
        "decision": "append_as_preprint_labeled_context",
        "stop_condition": "Stop because claim promotion requires stable preprint/citation handling or peer-reviewed source.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-POST-011",
        "item_kind": "linkedin_post",
        "activity_or_article_id": "7181176150688223232",
        "title": "Logistic Regression with Fisher Scoring",
        "source_status": "linkedin_post_text_only_with_child_post_links",
        "content_read": "Post read; child posts for logistic regression and log loss captured and read.",
        "attachment_or_visual_status": "No primary media detected on public permalink.",
        "external_source_status": "Child LinkedIn posts 7152948849597132801 and 7168870006380740608 captured.",
        "project_destination": "Book chapter 06 pedagogy.",
        "possible_executable_or_implementable": "Short explanation of MLE/Fisher scoring if needed; no package dependency.",
        "decision": "archive_as_pedagogy",
        "stop_condition": "Stop because it does not alter the model comparison or champion.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-POST-012",
        "item_kind": "linkedin_post",
        "activity_or_article_id": "7173943545752383488",
        "title": "Additive Logistic Regression",
        "source_status": "linkedin_post_with_13_page_public_document_pdf_and_child_links",
        "content_read": "Post and deck read; LogitBoost as additive logistic regression and bridge to boosting reviewed.",
        "attachment_or_visual_status": "PDF text extracted.",
        "external_source_status": "Log-loss and logistic-regression child posts captured.",
        "project_destination": "Book chapter 06/11 related-work sidebar.",
        "possible_executable_or_implementable": "No implementation; retain as conceptual bridge between logistic regression and boosting.",
        "decision": "park_related_work",
        "stop_condition": "Stop because CatBoost champion already covers the boosted-tree lane.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-POST-013",
        "item_kind": "linkedin_post",
        "activity_or_article_id": "7368538277517156352",
        "title": "Hidden Tricks in CatBoost You Should Know",
        "source_status": "linkedin_post_with_external_article_link",
        "content_read": "Post read; external article resolved as non-peer web context.",
        "attachment_or_visual_status": "Text-only post captured.",
        "external_source_status": "PlainEnglish article reachable but kept as low-evidence implementation context.",
        "project_destination": "Book chapters 10/11 context only.",
        "possible_executable_or_implementable": "None now; MLflow/SageMaker ideas remain outside the current local stack.",
        "decision": "archive_low_evidence_context",
        "stop_condition": "Stop because article does not change local model governance artifacts.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-POST-014",
        "item_kind": "linkedin_child_post",
        "activity_or_article_id": "7152948849597132801",
        "title": "Logistic Regression: The Two Cultures",
        "source_status": "linkedin_child_post_with_12_page_public_document_pdf",
        "content_read": "Post and deck read; regression/classification framing, probability estimation, and log loss objective reviewed.",
        "attachment_or_visual_status": "PDF text extracted.",
        "external_source_status": "No additional source required.",
        "project_destination": "Book chapter 06.",
        "possible_executable_or_implementable": "Clarify that PD models are probability estimators used for decisions, not just classifiers.",
        "decision": "append_language_no_experiment",
        "stop_condition": "Stop after metric-governance language is updated.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-POST-015",
        "item_kind": "linkedin_child_post",
        "activity_or_article_id": "7168870006380740608",
        "title": "Log Loss",
        "source_status": "linkedin_child_post_with_36_page_public_document_pdf",
        "content_read": "Post and deck read; likelihood, binomial/Bernoulli log-likelihood, GBDT log loss, and cumulative log loss reviewed.",
        "attachment_or_visual_status": "PDF text extracted.",
        "external_source_status": "No additional source required.",
        "project_destination": "Book chapter 06.",
        "possible_executable_or_implementable": "Add log loss as complementary probability-quality metric beside Brier/ECE/AUC.",
        "decision": "append_metric_language",
        "stop_condition": "Stop because no new metric gate is needed for the champion.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-ART-001",
        "item_kind": "linkedin_article",
        "activity_or_article_id": "article_designing_ai_underwriters",
        "title": "Designing AI Underwriters",
        "source_status": "public_linkedin_article",
        "content_read": "Article and high-priority visuals read; multi-agent/human-in-the-loop underwriting workflow reviewed.",
        "attachment_or_visual_status": "3 inline figures and cover visually reviewed.",
        "external_source_status": "LinkedIn-only article; not bibliographic evidence.",
        "project_destination": "Governance backlog only.",
        "possible_executable_or_implementable": "Future AI-underwriter assistant concept after model-governance scope is stable.",
        "decision": "park_outside_current_deliverable",
        "stop_condition": "Stop because current deliverable is PD/LGD/portfolio governance, not LLM underwriting automation.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-ART-002",
        "item_kind": "linkedin_article",
        "activity_or_article_id": "article_measuring_calibration_accuracy",
        "title": "Measuring Calibration Accuracy of Modern PD Models",
        "source_status": "public_linkedin_article",
        "content_read": "Article and visuals read; Brier, log loss, ECE, MCE, robust logit and calibration diagrams reviewed.",
        "attachment_or_visual_status": "3 inline figures and cover visually reviewed.",
        "external_source_status": "Article contains source trail; use only as context unless references are separately cited.",
        "project_destination": "Book chapter 06; Paper Estrella reviewer-defense language.",
        "possible_executable_or_implementable": "Strengthen distinction between rank metrics and probability calibration.",
        "decision": "append_language_no_new_experiment",
        "stop_condition": "Stop after chapter 06 metric language is updated.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-ART-003",
        "item_kind": "linkedin_article",
        "activity_or_article_id": "article_designing_credit_scoring_systems_ml_components",
        "title": "Designing Credit Scoring Systems with ML Components",
        "source_status": "public_linkedin_article",
        "content_read": "Article and visuals read; modular scoring, submodels, neutral logits, integration model, ECE, and validation complexity reviewed.",
        "attachment_or_visual_status": "4 inline figures and cover visually reviewed.",
        "external_source_status": "LinkedIn-only article; references should be checked before formal citation.",
        "project_destination": "Book chapter 10; Paper 4 governance language.",
        "possible_executable_or_implementable": "Map score/calibrator/policy/submodel separation to governance text.",
        "decision": "append_governance_language",
        "stop_condition": "Stop after architecture language is captured.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-ART-004",
        "item_kind": "linkedin_article",
        "activity_or_article_id": "article_understanding_lgd_risk",
        "title": "Understanding LGD Risk",
        "source_status": "public_linkedin_article",
        "content_read": "Article and visuals read; workout LGD, direct/indirect LR-CR decomposition, CLAR, and LGD distribution shape reviewed.",
        "attachment_or_visual_status": "7 inline figures and cover visually reviewed.",
        "external_source_status": "References include ECB/EBA material; use separately verified official sources for claims.",
        "project_destination": "Paper 4 LGD appendix/future work; book scope clarification.",
        "possible_executable_or_implementable": "Future probabilistic LGD bins or LR-CR model if recovery/cure data exist.",
        "decision": "park_data_unavailable",
        "stop_condition": "Stop because Lending Club project lacks recovery cash-flow/workout data.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-ART-005",
        "item_kind": "linkedin_article",
        "activity_or_article_id": "article_validating_tree_based_risk_models",
        "title": "Validating Tree-Based Risk Models",
        "source_status": "public_linkedin_article",
        "content_read": "Article and visuals read; validation scope, tree-leaf contingency tables, chi-square stability, and complexity penalty reviewed.",
        "attachment_or_visual_status": "6 inline figures and cover visually reviewed.",
        "external_source_status": "LinkedIn-only article; use as idea intake.",
        "project_destination": "Book chapter 10/11; Paper 4 appendix parking.",
        "possible_executable_or_implementable": "Future tree-level stability diagnostic for GBDT models.",
        "decision": "park_bounded_diagnostic",
        "stop_condition": "Stop until a reviewer asks for tree-level validation or a table changes.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-ART-006",
        "item_kind": "linkedin_article",
        "activity_or_article_id": "article_validating_new_generation_credit_risk_models",
        "title": "Validating New Generation Credit Risk Models",
        "source_status": "public_linkedin_article",
        "content_read": "Article and visuals read; model choice, parsimony, complexity, challenger tests, and Optuna validation reviewed.",
        "attachment_or_visual_status": "10 inline figures and cover visually reviewed.",
        "external_source_status": "Article references official model-risk sources; verify before citation.",
        "project_destination": "Book chapter 10; Paper Estrella reviewer-defense language.",
        "possible_executable_or_implementable": "Use model choice/parsimony/complexity checklist in validation language.",
        "decision": "append_governance_language",
        "stop_condition": "Stop after chapter governance language is updated.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-ART-007",
        "item_kind": "linkedin_article",
        "activity_or_article_id": "article_scorecarding_naive_bayes",
        "title": "Scorecarding with Naive Bayes",
        "source_status": "public_linkedin_article",
        "content_read": "Article and visuals read; WOE/NB workflow, Bayes factor framing, score formula, Gini and Brier examples reviewed.",
        "attachment_or_visual_status": "22 inline figures and cover visually reviewed.",
        "external_source_status": "LinkedIn-only article with references; context only.",
        "project_destination": "Book chapter 05.",
        "possible_executable_or_implementable": "Strengthen WOE as log-likelihood-ratio explanation; no NB model lane.",
        "decision": "append_language_no_new_model",
        "stop_condition": "Stop once WOE/NB caution is in chapter 05.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-ART-008",
        "item_kind": "linkedin_article",
        "activity_or_article_id": "article_leveraging_profit_scoring",
        "title": "Leveraging Profit Scoring in Digital Loan Underwriting",
        "source_status": "public_linkedin_article",
        "content_read": "Article and visuals read; ARR, write-offs, ECL reserve adjustment, Bondora example, SHAP differences, and downturn effects reviewed.",
        "attachment_or_visual_status": "7 inline figures and cover visually reviewed.",
        "external_source_status": "LinkedIn-only article; academic claims require direct source verification.",
        "project_destination": "Book chapter 09; Paper 4 future work.",
        "possible_executable_or_implementable": "Add profit-scoring caveat: PD is not profit and pricing/cost-of-risk requires different data.",
        "decision": "append_economic_caveat",
        "stop_condition": "Stop because current dataset lacks realized revenue, cost, and recovery target needed for profit scoring.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-ART-009",
        "item_kind": "linkedin_article",
        "activity_or_article_id": "article_balancing_risk_and_profit",
        "title": "Balancing Risk and Profit",
        "source_status": "public_linkedin_article",
        "content_read": "Article and visuals read; risk grades, profit rates, standalone profit model, partial dependence, and power curves reviewed.",
        "attachment_or_visual_status": "4 inline figures and cover visually reviewed.",
        "external_source_status": "LinkedIn-only article; context only.",
        "project_destination": "Book chapter 09.",
        "possible_executable_or_implementable": "Use to explain why risk ranking must be converted into net-return decisions.",
        "decision": "append_economic_caveat",
        "stop_condition": "Stop after chapter 09 notes the separation between PD, loss, revenue, and profit.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-ART-010",
        "item_kind": "linkedin_article",
        "activity_or_article_id": "article_benchmarking_pd_models",
        "title": "Benchmarking PD Models",
        "source_status": "public_linkedin_article",
        "content_read": "Article and visuals read; same-Gini model comparison, cutoff pitfalls, log loss, loss uplift, and payout curves reviewed.",
        "attachment_or_visual_status": "8 inline figures and cover visually reviewed.",
        "external_source_status": "LinkedIn-only article; context only.",
        "project_destination": "Book chapters 06 and 09.",
        "possible_executable_or_implementable": "Add metric governance language: AUC/Gini alone cannot decide thresholds or value.",
        "decision": "append_metric_language",
        "stop_condition": "Stop after chapters 06/09 capture the caveat.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-ART-011",
        "item_kind": "linkedin_article",
        "activity_or_article_id": "article_exploring_scorecard_boosting",
        "title": "Exploring Interpretable Scorecard Boosting",
        "source_status": "public_linkedin_article",
        "content_read": "Article and visuals read; boosted scorecards, monotonic stumps, scorecard point extraction, feature importance, and SHAP reviewed.",
        "attachment_or_visual_status": "7 inline figures and cover visually reviewed.",
        "external_source_status": "Associated xBooster GitHub readable.",
        "project_destination": "Book chapter 11; Paper 4 prototype parking.",
        "possible_executable_or_implementable": "Future boosted-scorecard benchmark against CatBoost champion.",
        "decision": "park_prototype",
        "stop_condition": "Stop until a bounded experiment can change an appendix table.",
        "closure_status": "closed",
    },
    {
        "item_id": "S2-ART-012",
        "item_kind": "linkedin_article",
        "activity_or_article_id": "article_unlocking_lending_profitability",
        "title": "Unlocking Lending Profitability with Risk Modeling",
        "source_status": "public_linkedin_article",
        "content_read": "Article and visuals read; profitability, technical debt, risk-based pricing, PD by product, and application/behavioral scoring flow reviewed.",
        "attachment_or_visual_status": "6 inline figures and cover visually reviewed.",
        "external_source_status": "LinkedIn-only article; context only.",
        "project_destination": "Book chapter 09; governance backlog.",
        "possible_executable_or_implementable": "Use as caution that lending profitability depends on operating model and feedback loops, not only model AUC.",
        "decision": "append_economic_caveat",
        "stop_condition": "Stop because it is strategy/governance context, not a current data artifact.",
        "closure_status": "closed",
    },
]


CONCEPT_ROWS = [
    {
        "concept": "WOE as centered log odds with uncertainty",
        "source_items": "S2-POST-002; S2-POST-006; S2-ART-007",
        "method_family": "scorecards / WOE / information theory",
        "novelty_for_project": "High for explanation language; medium for implementation.",
        "evidence_strength": "LinkedIn deck + GitHub tooling; needs primary citation for paper claims.",
        "project_destination": "Book chapter 05; Paper 4 appendix parking.",
        "implementation_difficulty": "Medium",
        "claim_risk": "Medium",
        "decision": "append_language_and_park_prototype",
        "stop_condition": "No dependency or claim promotion without local benchmark and verified source.",
    },
    {
        "concept": "WOE as Naive Bayes likelihood-ratio scorecard",
        "source_items": "S2-POST-008; S2-ART-007",
        "method_family": "Bayesian scoring / WOE",
        "novelty_for_project": "Medium; strengthens pedagogy.",
        "evidence_strength": "LinkedIn article/deck only.",
        "project_destination": "Book chapter 05.",
        "implementation_difficulty": "Low",
        "claim_risk": "Medium",
        "decision": "append_language_no_new_model",
        "stop_condition": "Do not create NB challenger unless a reviewer asks for a simple probabilistic baseline.",
    },
    {
        "concept": "Rank metrics versus probability scoring rules",
        "source_items": "S2-ART-002; S2-ART-010; S2-POST-015",
        "method_family": "model evaluation / calibration",
        "novelty_for_project": "Medium; sharpens existing champion defense.",
        "evidence_strength": "Local project metrics plus LinkedIn context; formal claims already rely on project artifacts and citations.",
        "project_destination": "Book chapter 06; Paper Estrella reviewer defense.",
        "implementation_difficulty": "Low",
        "claim_risk": "Low if framed as metric governance.",
        "decision": "append_metric_language",
        "stop_condition": "No new model benchmark unless it changes an existing table.",
    },
    {
        "concept": "PD model as probability estimator, not only classifier",
        "source_items": "S2-POST-014; S2-POST-011; S2-POST-015",
        "method_family": "logistic regression / MLE pedagogy",
        "novelty_for_project": "Medium.",
        "evidence_strength": "Pedagogical LinkedIn decks; no claim risk when used as explanatory prose.",
        "project_destination": "Book chapter 06.",
        "implementation_difficulty": "Low",
        "claim_risk": "Low",
        "decision": "append_language_no_experiment",
        "stop_condition": "Stop after prose clarifies threshold metrics are downstream policy choices.",
    },
    {
        "concept": "Trended credit data and sequence models",
        "source_items": "S2-POST-003",
        "method_family": "sequence modeling / feature engineering",
        "novelty_for_project": "High but data-infeasible.",
        "evidence_strength": "LinkedIn carousel only.",
        "project_destination": "Paper 4 limitations/future work.",
        "implementation_difficulty": "High",
        "claim_risk": "Medium",
        "decision": "park_data_unavailable",
        "stop_condition": "Requires transaction histories; Lending Club snapshot cannot support it.",
    },
    {
        "concept": "Profit scoring beyond PD",
        "source_items": "S2-ART-008; S2-ART-009; S2-ART-012; S2-ART-010",
        "method_family": "portfolio economics / risk-based pricing",
        "novelty_for_project": "High for chapter 09 framing; low for immediate implementation.",
        "evidence_strength": "LinkedIn articles with source trails; local LP already covers expected loss but not full profit target.",
        "project_destination": "Book chapter 09; Paper 4 future work.",
        "implementation_difficulty": "High",
        "claim_risk": "Medium",
        "decision": "append_economic_caveat",
        "stop_condition": "Requires realized revenue, write-offs, costs, and pricing policy data.",
    },
    {
        "concept": "Official ML governance for internal credit risk models",
        "source_items": "S2-POST-009; ECB official PDF",
        "method_family": "model risk management / IRB governance",
        "novelty_for_project": "High.",
        "evidence_strength": "Official supervisory guidance extracted locally.",
        "project_destination": "Book chapter 10; Paper Estrella reviewer defense.",
        "implementation_difficulty": "Low to medium",
        "claim_risk": "Low when labeled as official guidance.",
        "decision": "append_official_governance_language",
        "stop_condition": "Stop after governance controls are documented; no champion reopen.",
    },
    {
        "concept": "Information-theoretic IV/PSI/Jeffreys framework",
        "source_items": "S2-POST-010",
        "method_family": "information theory / monitoring / fairness",
        "novelty_for_project": "Medium.",
        "evidence_strength": "Preprint PDF captured; SSRN web fetch blocked.",
        "project_destination": "Book chapters 05/10; source discovery for papers.",
        "implementation_difficulty": "Medium",
        "claim_risk": "Medium to high until citation status stabilizes.",
        "decision": "append_as_preprint_labeled_context",
        "stop_condition": "No paper claim promotion without verified preprint/citation trail.",
    },
    {
        "concept": "Tree-based validation with leaf contingency tests",
        "source_items": "S2-ART-005",
        "method_family": "GBDT validation / model monitoring",
        "novelty_for_project": "Medium.",
        "evidence_strength": "LinkedIn article idea intake only.",
        "project_destination": "Book chapter 10/11; Paper 4 appendix parking.",
        "implementation_difficulty": "Medium",
        "claim_risk": "Medium",
        "decision": "park_bounded_diagnostic",
        "stop_condition": "Implement only if it changes a validation table or reviewer response.",
    },
    {
        "concept": "xBooster / boosted scorecard extraction",
        "source_items": "S2-POST-007; S2-ART-011",
        "method_family": "interpretable boosting / scorecard distillation",
        "novelty_for_project": "Medium to high.",
        "evidence_strength": "Readable GitHub README and LinkedIn article; no local benchmark yet.",
        "project_destination": "Book chapter 11; Paper 4 prototype parking.",
        "implementation_difficulty": "Medium",
        "claim_risk": "Medium",
        "decision": "park_prototype",
        "stop_condition": "No dependency until benchmark scope and artifact table are defined.",
    },
    {
        "concept": "LGD direct/indirect decomposition and CLAR",
        "source_items": "S2-ART-004",
        "method_family": "LGD modeling / recovery risk",
        "novelty_for_project": "High for Paper 4 scope, low for current data.",
        "evidence_strength": "LinkedIn article plus official-source trail; claims need direct official/paper citations.",
        "project_destination": "Paper 4 LGD appendix/future work.",
        "implementation_difficulty": "High",
        "claim_risk": "Medium",
        "decision": "park_data_unavailable",
        "stop_condition": "Requires workout/recovery/cure/debt-sale data.",
    },
    {
        "concept": "Poisson/count models for risk event counts",
        "source_items": "S2-POST-004",
        "method_family": "GLM / count modeling",
        "novelty_for_project": "Low for current PD target.",
        "evidence_strength": "LinkedIn deck with textbook source trail.",
        "project_destination": "Book GLM sidebar only.",
        "implementation_difficulty": "Medium",
        "claim_risk": "Low if framed as out-of-scope.",
        "decision": "archive_outside_current_pd_target",
        "stop_condition": "Do not implement for binary PD.",
    },
]


VISUAL_ROWS = [
    {
        "visual_item_id": "S2-VIS-POST-003",
        "source_id": "7211748798618824704",
        "visual_count": "3",
        "visual_read_status": "manual_visual_read_completed",
        "project_signal": "Trended data slides: sequence features, RNN/LSTM/Transformer comparison, and calibration chart.",
    },
    {
        "visual_item_id": "S2-VIS-ARTICLES",
        "source_id": "12_public_linkedin_articles",
        "visual_count": "99_high_or_medium_priority_images",
        "visual_read_status": "manual_visual_read_completed_from_contact_sheets",
        "project_signal": "All high/medium article figures reviewed; archive-only related covers excluded from analysis.",
    },
]


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_memo() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    closed = sum(row["closure_status"] == "closed" for row in BACKLOG_ROWS)
    memo = f"""# Second Ingest Execution Memo - 2026-05-21

## Scope Closed

- LinkedIn posts captured/read: 15, including 2 child posts discovered from external links.
- Public LinkedIn articles captured/read: 12.
- High/medium priority article visuals manually read: 99.
- High-value external source snapshots: 4, including the 370-page official ECB guide extracted with `pdftotext`.
- Backlog items closed: {closed}/{len(BACKLOG_ROWS)}.

## Decisions

The second ingest produced three immediate project edits and several parked lanes.

1. Append language to the book where it strengthens existing claims:
   WOE as centered evidence with uncertainty, log loss as probability-quality metric, profit scoring as a separate economic target, and official ML governance controls.
2. Park prototypes that could become bounded appendix work:
   FastWOE uncertainty, xBooster/CatBoost scorecards, tree-level chi-square validation, and sequence/trended scoring.
3. Archive context that is useful pedagogically but does not change a claim:
   MLE-to-LLM framing, Poisson count models for binary PD, and broad CatBoost deployment tips.

## Source Governance

LinkedIn posts, decks, and articles are treated as private research intake. They can shape
project language and backlog design, but they do not promote public-facing paper claims by
themselves. The ECB guide is separated as official supervisory guidance. GitHub sources are
implementation context only unless locally benchmarked. SSRN/preprint material remains
preprint-labeled until a stable citation trail is available.

## Remaining Profile Gap

Public profile snippets indicate more total posts than the 67 first-ingest rows plus the 15
second-ingest rows captured here. The public recent-activity page is gated, so the remaining
unknown posts cannot be enumerated exhaustively from public HTML alone. The practical
state is: all newly discoverable high-signal credit-risk posts found through public search,
external links, and article discovery on 2026-05-21 are captured/read; exhaustive coverage of
the advertised profile total would require a logged-in visible-browser export/review.
"""
    (DOCS / "second_ingest_execution_memo_2026-05-21.md").write_text(memo, encoding="utf-8")


def write_gap_report() -> None:
    report = """# Second Ingest Profile Gap Report - 2026-05-21

## Captured In This Pass

- 13 newly discovered public LinkedIn posts not present in the first ingest.
- 2 linked child posts that appeared during external-link resolution.
- 12 public LinkedIn articles associated with the same author/topic cluster.
- 21 external links resolved from the second post set.

## Still Potentially Missing

The public profile/search surface suggests a larger total corpus than we can enumerate through
public pages alone. The remaining gap is not a processing failure inside the saved material; it
is a discovery limitation caused by LinkedIn gating recent activity. Items not reachable through
public permalinks, public search snippets, public articles, or already-resolved child links remain
unknown.

## Closure Rule For This Pass

This second pass is closed for all material it discovered. A third pass should only start if one
of these inputs appears:

1. a logged-in visible-browser list/export of additional activity permalinks;
2. a new batch of LinkedIn activity IDs from search;
3. a paper/book need that specifically requires more Denis Burakov profile material.
"""
    (DOCS / "second_ingest_profile_gap_report_2026-05-21.md").write_text(report, encoding="utf-8")


def main() -> None:
    write_csv(DATA / "second_ingest_execution_backlog.csv", BACKLOG_ROWS, BACKLOG_FIELDS)
    write_csv(DATA / "second_ingest_concept_atlas.csv", CONCEPT_ROWS, CONCEPT_FIELDS)
    write_csv(DATA / "second_ingest_visual_read_log.csv", VISUAL_ROWS, VISUAL_FIELDS)
    write_memo()
    write_gap_report()
    print(
        "Second ingest decisions written: "
        f"{len(BACKLOG_ROWS)} backlog rows, {len(CONCEPT_ROWS)} concepts, {len(VISUAL_ROWS)} visual rows"
    )


if __name__ == "__main__":
    main()
