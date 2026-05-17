#!/usr/bin/env python3
"""Build Paper 4 v381 verified literature/source-log artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from scripts.papers.paper4_one_swap_living_lab import (
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    now,
    read_csv,
    write_csv,
    write_json,
)

VERSION = 381
PRIOR_SCAFFOLD_VERSION = 380
NEXT_VERSION = 382
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_global_solver_scope_decision.md"
SOURCE_LOG_MD = NOTEBOOK.parent / "paper4_v381_verified_literature_source_log.md"


def _source_log() -> pd.DataFrame:
    rows = [
        {
            "source_id_v381": "rockafellar_uryasev_cvar_2000",
            "citation_key_v381": "rockafellar2000optimization",
            "title_v381": "Optimization of Conditional Value-at-Risk",
            "authors_v381": "R. Tyrrell Rockafellar; Stanislav Uryasev",
            "year_v381": 2000,
            "source_type_v381": "journal_article",
            "venue_or_publisher_v381": "Journal of Risk 2(3):21-41",
            "doi_v381": "10.21314/JOR.2000.038",
            "canonical_url_v381": "https://doi.org/10.21314/JOR.2000.038",
            "verification_source_v381": "Risk.net DOI landing page",
            "verified_v381": True,
            "paper4_use_v381": "CVaR tail-risk optimization reference for solver methods",
            "supported_bounded_claim_v381": (
                "CVaR portfolio optimization can be represented with optimization machinery."
            ),
            "prohibited_overclaim_v381": (
                "Does not prove Paper 4 full-v55 optimality, integer optimality or promotion."
            ),
            "claim_boundary_v381": "method citation only",
        },
        {
            "source_id_v381": "rockafellar_uryasev_cvar_2002",
            "citation_key_v381": "rockafellar2002conditional",
            "title_v381": "Conditional value-at-risk for general loss distributions",
            "authors_v381": "R. Tyrrell Rockafellar; Stanislav Uryasev",
            "year_v381": 2002,
            "source_type_v381": "journal_article",
            "venue_or_publisher_v381": "Journal of Banking & Finance 26(7):1443-1471",
            "doi_v381": "10.1016/S0378-4266(02)00271-6",
            "canonical_url_v381": "https://www.sciencedirect.com/science/article/abs/pii/S0378426602002716",
            "verification_source_v381": "ScienceDirect article landing page",
            "verified_v381": True,
            "paper4_use_v381": "general/discrete-loss CVaR caveat for scenario evidence",
            "supported_bounded_claim_v381": (
                "CVaR foundations cover general loss distributions and scenario-style losses."
            ),
            "prohibited_overclaim_v381": (
                "Does not validate Paper 4 data, source caps or contractual loss labels."
            ),
            "claim_boundary_v381": "risk-measure foundation only",
        },
        {
            "source_id_v381": "vovk_gammerman_shafer_2005",
            "citation_key_v381": "vovk2005algorithmic",
            "title_v381": "Algorithmic Learning in a Random World",
            "authors_v381": "Vladimir Vovk; Alexander Gammerman; Glenn Shafer",
            "year_v381": 2005,
            "source_type_v381": "book",
            "venue_or_publisher_v381": "Springer New York",
            "doi_v381": "10.1007/b106715",
            "canonical_url_v381": "https://link.springer.com/book/10.1007/b106715",
            "verification_source_v381": "Springer book landing page",
            "verified_v381": True,
            "paper4_use_v381": "foundational conformal prediction reference",
            "supported_bounded_claim_v381": (
                "Conformal prediction provides uncertainty statements under explicit assumptions."
            ),
            "prohibited_overclaim_v381": (
                "Does not make Paper 4 live-deployable under future distribution shift."
            ),
            "claim_boundary_v381": "foundational citation only",
        },
        {
            "source_id_v381": "romano_patterson_candes_cqr_2019",
            "citation_key_v381": "romano2019conformalized",
            "title_v381": "Conformalized Quantile Regression",
            "authors_v381": "Yaniv Romano; Evan Patterson; Emmanuel Candes",
            "year_v381": 2019,
            "source_type_v381": "conference_paper",
            "venue_or_publisher_v381": "Advances in Neural Information Processing Systems 32",
            "doi_v381": "",
            "canonical_url_v381": (
                "https://proceedings.neurips.cc/paper/2019/hash/"
                "5103c3584b063c431bd1268e9b5e76fb-Abstract.html"
            ),
            "verification_source_v381": "NeurIPS proceedings landing page",
            "verified_v381": True,
            "paper4_use_v381": "heteroscedastic conformal interval reference",
            "supported_bounded_claim_v381": (
                "CQR combines conformal prediction with quantile regression for adaptive intervals."
            ),
            "prohibited_overclaim_v381": (
                "Does not prove Paper 4 MAPIE configuration, grade coverage or source governance."
            ),
            "claim_boundary_v381": "related-work citation only",
        },
        {
            "source_id_v381": "gibbs_candes_aci_2021",
            "citation_key_v381": "gibbs2021adaptive",
            "title_v381": "Adaptive Conformal Inference Under Distribution Shift",
            "authors_v381": "Isaac Gibbs; Emmanuel Candes",
            "year_v381": 2021,
            "source_type_v381": "conference_paper",
            "venue_or_publisher_v381": "Advances in Neural Information Processing Systems 34",
            "doi_v381": "",
            "canonical_url_v381": (
                "https://proceedings.neurips.cc/paper/2021/hash/"
                "0d441de75945e5acbc865406fc9a2559-Abstract.html"
            ),
            "verification_source_v381": "NeurIPS proceedings landing page",
            "verified_v381": True,
            "paper4_use_v381": "online/adaptive conformal context for historical replay",
            "supported_bounded_claim_v381": (
                "ACI is a published online conformal approach for varying distributions."
            ),
            "prohibited_overclaim_v381": (
                "Does not validate Paper 4 external holdout or strict live deployment."
            ),
            "claim_boundary_v381": "online conformal context only",
        },
        {
            "source_id_v381": "angelopoulos_bates_fisch_lei_schuster_crc_2024",
            "citation_key_v381": "angelopoulos2024conformal",
            "title_v381": "Conformal Risk Control",
            "authors_v381": (
                "Anastasios Nikolas Angelopoulos; Stephen Bates; Adam Fisch; "
                "Lihua Lei; Tal Schuster"
            ),
            "year_v381": 2024,
            "source_type_v381": "conference_paper",
            "venue_or_publisher_v381": "ICLR 2024 spotlight",
            "doi_v381": "",
            "canonical_url_v381": "https://openreview.net/forum?id=33XGfHLtZg",
            "verification_source_v381": "OpenReview ICLR landing page",
            "verified_v381": True,
            "paper4_use_v381": "future decision-risk-control framing",
            "supported_bounded_claim_v381": (
                "CRC extends conformal prediction toward monotone loss-function risk control."
            ),
            "prohibited_overclaim_v381": (
                "Does not show Paper 4 implements a formal CRC guarantee."
            ),
            "claim_boundary_v381": "future-work/method context only",
        },
        {
            "source_id_v381": "elmachtoub_grigas_spo_2021",
            "citation_key_v381": "elmachtoub2021smart",
            "title_v381": "Smart Predict, then Optimize",
            "authors_v381": "Adam N. Elmachtoub; Paul Grigas",
            "year_v381": 2021,
            "source_type_v381": "journal_article",
            "venue_or_publisher_v381": "Management Science 68(1):9-26",
            "doi_v381": "10.1287/mnsc.2020.3922",
            "canonical_url_v381": "https://doi.org/10.1287/mnsc.2020.3922",
            "verification_source_v381": "INFORMS DOI landing page",
            "verified_v381": True,
            "paper4_use_v381": "predict-then-optimize and SPO/SPO+ boundary reference",
            "supported_bounded_claim_v381": (
                "SPO provides a decision-aware predict-then-optimize framework."
            ),
            "prohibited_overclaim_v381": (
                "Does not prove Paper 4 implements differentiable SPO+ training."
            ),
            "claim_boundary_v381": "method boundary citation only",
        },
        {
            "source_id_v381": "ifrs_foundation_ifrs9_2026",
            "citation_key_v381": "ifrs2026ifrs9",
            "title_v381": "IFRS 9 Financial Instruments",
            "authors_v381": "IFRS Foundation; International Accounting Standards Board",
            "year_v381": 2026,
            "source_type_v381": "official_accounting_standard",
            "venue_or_publisher_v381": "IFRS Accounting Standards Navigator",
            "doi_v381": "",
            "canonical_url_v381": (
                "https://www.ifrs.org/issued-standards/list-of-standards/"
                "ifrs-9-financial-instruments/"
            ),
            "verification_source_v381": "IFRS Foundation official standard page",
            "verified_v381": True,
            "paper4_use_v381": "IFRS9 context and proxy/contractual limitation boundary",
            "supported_bounded_claim_v381": (
                "IFRS 9 covers financial-instrument classification, measurement and impairment."
            ),
            "prohibited_overclaim_v381": (
                "Does not make Paper 4 contractual IFRS9 lifetime ECL compliant."
            ),
            "claim_boundary_v381": "official standard context only",
        },
        {
            "source_id_v381": "cfpb_regulation_b_1002_current",
            "citation_key_v381": "cfpb2026regulationb",
            "title_v381": "12 CFR Part 1002 - Equal Credit Opportunity Act (Regulation B)",
            "authors_v381": "Consumer Financial Protection Bureau",
            "year_v381": 2026,
            "source_type_v381": "official_regulatory_standard",
            "venue_or_publisher_v381": "Consumer Financial Protection Bureau regulations",
            "doi_v381": "",
            "canonical_url_v381": "https://www.consumerfinance.gov/rules-policy/regulations/1002/",
            "verification_source_v381": "CFPB official Regulation B page",
            "verified_v381": True,
            "paper4_use_v381": "fair-lending limitation and legal-review boundary",
            "supported_bounded_claim_v381": (
                "Regulation B is the official ECOA implementing regulation resource."
            ),
            "prohibited_overclaim_v381": (
                "Does not certify Paper 4 legal fair-lending compliance."
            ),
            "claim_boundary_v381": "regulatory context only",
        },
    ]
    return pd.DataFrame(rows)


def _citation_use_matrix(source_log: pd.DataFrame) -> pd.DataFrame:
    section_map = {
        "rockafellar_uryasev_cvar_2000": (
            "Methods: Solver Frontier",
            "CVaR objective/constraint reference; cite beside linear tail-risk formulation.",
        ),
        "rockafellar_uryasev_cvar_2002": (
            "Methods: Solver Frontier",
            "General-loss CVaR context; cite beside scenario/discrete-loss caveats.",
        ),
        "vovk_gammerman_shafer_2005": (
            "Related Work: Conformal Foundations",
            "Foundational conformal prediction reference; cite before implementation details.",
        ),
        "romano_patterson_candes_cqr_2019": (
            "Related Work: Conformal Calibration",
            "Adaptive interval related work; do not imply CQR was implemented here.",
        ),
        "gibbs_candes_aci_2021": (
            "Related Work: Online Conformal",
            "ACI context for online shift; distinguish from Paper 4 historical replay.",
        ),
        "angelopoulos_bates_fisch_lei_schuster_crc_2024": (
            "Future Work: Decision Risk Control",
            "CRC bridge; label as future/formal-extension route.",
        ),
        "elmachtoub_grigas_spo_2021": (
            "Related Work: Predict-then-Optimize",
            "SPO/SPO+ boundary citation; keep current implementation as proxy/surrogate.",
        ),
        "ifrs_foundation_ifrs9_2026": (
            "Limitations: IFRS9 Boundary",
            "Official standard context; state that contractual ECL coverage is not complete.",
        ),
        "cfpb_regulation_b_1002_current": (
            "Limitations: Legal Fairness Boundary",
            "Official regulation context; state that legal review remains required.",
        ),
    }
    citation_key_by_source = dict(
        zip(source_log["source_id_v381"], source_log["citation_key_v381"], strict=False)
    )
    return pd.DataFrame(
        [
            {
                "source_id_v381": source_id,
                "citation_key_v381": citation_key_by_source[source_id],
                "paper_section_v381": section,
                "use_instruction_v381": instruction,
                "allowed_v381": True,
                "prohibited_use_v381": "Do not convert this source into live/legal/global/final claims.",
                "claim_boundary_v381": "bounded citation placement only",
            }
            for source_id, (section, instruction) in section_map.items()
        ]
    )


def _source_gap_register() -> pd.DataFrame:
    rows = [
        {
            "gap_id_v381": "venue_target_bibliography_style_not_selected",
            "blocking_v381": True,
            "evidence_count_v381": 0,
            "required_next_step_v381": "select venue target and reference style",
            "claim_boundary_v381": "source log is not formatted final bibliography",
        },
        {
            "gap_id_v381": "recent_credit_portfolio_literature_not_systematically_reviewed",
            "blocking_v381": True,
            "evidence_count_v381": 0,
            "required_next_step_v381": "run a separate recent related-work search if venue scope requires it",
            "claim_boundary_v381": "anchor-source log only",
        },
        {
            "gap_id_v381": "ifrs9_contractual_application_not_verified",
            "blocking_v381": True,
            "evidence_count_v381": 1,
            "required_next_step_v381": "obtain contractual coverage and accounting review",
            "claim_boundary_v381": "IFRS9 proxy diagnostics only",
        },
        {
            "gap_id_v381": "legal_fairness_review_not_approved",
            "blocking_v381": True,
            "evidence_count_v381": 1,
            "required_next_step_v381": "obtain approved legal/protected-attribute review",
            "claim_boundary_v381": "fairness proxy governance only",
        },
    ]
    return pd.DataFrame(rows)


def _claim_blockers(source_log: pd.DataFrame, source_gaps: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v381": "source_log_is_not_full_literature_review",
            "blocking_v381": True,
            "evidence_count_v381": int(len(source_log)),
            "required_next_artifact_v381": "venue_targeted_related_work_review",
            "claim_boundary_v381": "verified anchor sources only",
        },
        {
            "blocker_id_v381": "source_gap_register_has_open_rows",
            "blocking_v381": True,
            "evidence_count_v381": int(source_gaps["blocking_v381"].astype(bool).sum()),
            "required_next_artifact_v381": NEXT_ARTIFACT,
            "claim_boundary_v381": "open source and review gaps remain",
        },
        {
            "blocker_id_v381": "submission_ready_claim_still_blocked",
            "blocking_v381": True,
            "evidence_count_v381": 0,
            "required_next_artifact_v381": NEXT_ARTIFACT,
            "claim_boundary_v381": "v381 verifies sources but does not close all v378 gaps",
        },
        {
            "blocker_id_v381": "legal_ifrs_live_claims_still_blocked",
            "blocking_v381": True,
            "evidence_count_v381": 2,
            "required_next_artifact_v381": "external_reviews_and_live_holdout_panel",
            "claim_boundary_v381": "official sources are context, not approval",
        },
        {
            "blocker_id_v381": "paper4_final_promotion_forbidden",
            "blocking_v381": True,
            "evidence_count_v381": 1,
            "required_next_artifact_v381": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v381": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    return pd.DataFrame(rows)


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v381_verified_external_source_log_created",
                "allowed": True,
                "artifact": "paper4_v381_verified_literature_source_log.csv",
                "boundary": "verified source metadata only",
            },
            {
                "claim_id": "v381_citation_use_matrix_created",
                "allowed": True,
                "artifact": "paper4_v381_citation_use_matrix.csv",
                "boundary": "bounded section-placement guidance",
            },
            {
                "claim_id": "v381_bounded_related_work_citation_language_allowed",
                "allowed": True,
                "artifact": "paper4_v381_verified_literature_source_log.csv",
                "boundary": "cite verified source facts without overclaiming implementation",
            },
            {
                "claim_id": "v381_submission_ready_or_quarto_promotion",
                "allowed": False,
                "artifact": "paper4_v381_claim_blockers.csv",
                "boundary": "source verification is not manuscript completion",
            },
            {
                "claim_id": "v381_live_legal_ifrs9_compliance_claim",
                "allowed": False,
                "artifact": "paper4_v381_source_gap_register.csv",
                "boundary": "official sources do not replace external approval",
            },
            {
                "claim_id": "v381_global_optimality_or_champion_claim",
                "allowed": False,
                "artifact": "paper4_v381_claim_blockers.csv",
                "boundary": "global solver proof and promotion gates remain separate",
            },
            {
                "claim_id": "v381_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "final promotion remains forbidden",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v381 verifies external literature/source facts for bounded related-work use.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v381_verified_literature_source_log.csv"
                ),
                "boundary": "Verified source metadata only; no implementation or venue claim.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v381 maps verified citations to Paper 4 manuscript sections.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v381_citation_use_matrix.csv"
                ),
                "boundary": "Section-placement guidance only; curated Quarto pages unchanged.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v381 makes Paper 4 submission-ready or bibliography-complete.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v381_claim_blockers.csv"
                ),
                "boundary": "A verified anchor-source log is not a full manuscript or final bibliography.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v381 authorizes live, legal, IFRS9, global or final-promotion claims.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v381_source_gap_register.csv"
                ),
                "boundary": "Official sources are context only; external reviews and solver/live gates remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v381 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v381_claim_blockers.csv"
                ),
                "boundary": "No final promotion artifact, champion replacement or deployment gate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Source Governance/Global",
                "executable_item": (
                    "v381 verifies external anchor sources for Paper 4 related work and "
                    "keeps source-derived claims bounded."
                ),
                "status": "verified_source_log_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v382 decides the global-solver claim scope without enabling global optimality"
                ),
                "last_wave": "v381",
                "execution_result": "source_log_verified_with_open_claim_blockers",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v381")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _source_log_markdown(status: dict[str, Any], source_log: pd.DataFrame) -> str:
    source_lines = "\n".join(
        (
            f"- `{row['citation_key_v381']}`: {row['title_v381']} "
            f"({row['venue_or_publisher_v381']}); boundary: {row['claim_boundary_v381']}."
        )
        for _, row in source_log.iterrows()
    )
    return f"""# Paper 4 Verified Literature Source Log v381

Generated: {status["generated_at_utc"]}

v381 verifies anchor source metadata before any related-work or citation claim is
expanded. This note is not a full literature review, not a final bibliography and
not a Quarto promotion.

## Verified Anchors

{source_lines}

## Required Caveat

The source log may support bounded related-work placement and method context. It
must not be used to claim submission readiness, strict live deployment,
contractual IFRS9 compliance, legal fair-lending compliance, full-v55 global
optimality, a new working champion, Paper Estrella replacement or final Paper 4
promotion.

## Next Executable Wave

Build `{status["next_artifact_v381"]}` to decide the solver/global-claim scope
while keeping global optimality prohibited unless a separate certificate is
generated.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V381_VERIFIED_LITERATURE_SOURCE_LOG_START -->"
    end = "<!-- V381_VERIFIED_LITERATURE_SOURCE_LOG_END -->"
    block = f"""
{start}

## Wave v381: Verified Literature Source Log

Generated: {status["generated_at_utc"]}

### Objective

v381 executes the citation work order from v379/v380: verify external anchor
sources before expanding related-work language, while preserving all live,
legal, IFRS9, global, submission and final-promotion blockers.

### Results

- Source log rows:
  `{status["source_log_rows_v381"]}`.
- Verified source rows:
  `{status["verified_source_rows_v381"]}`.
- Citation-use rows:
  `{status["citation_use_rows_v381"]}`.
- Open source gap rows:
  `{status["open_source_gap_rows_v381"]}`.
- Bounded related-work language allowed:
  `{status["bounded_related_work_language_allowed_v381"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v381"]}`.
- Quarto promotion allowed:
  `{status["quarto_promotion_allowed_v381"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v381"]}`.

### Interpretation

Paper 4 now has a small, verified anchor-source set for methods and limitations:
CVaR, conformal prediction, ACI/CRC, SPO, IFRS9 and ECOA/Reg B. The useful
claim is narrower than a full literature review: these sources can support
bounded citation placement, not stronger paper, legal, live or solver claims.

### Claim Impact

- Allowed: verified source metadata, citation-use mapping and bounded
  related-work placement.
- Still prohibited: submission-ready, Quarto promotion, live/legal/IFRS9,
  global optimality, champion replacement and final promotion.

### Quarto Promotion Decision

Keep v381 in the living notebook. v382 should decide the global-solver claim
scope without enabling global-optimality language.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v380_status = json.loads((STATUS_DIR / "paper4_v380_status.json").read_text(encoding="utf-8"))
    if v380_status["next_artifact_v380"] != "paper4_v381_verified_literature_source_log.csv":
        raise RuntimeError("v381 expects v380 to route to the verified literature source log.")
    scaffold = read_csv("paper4_v380_section_scaffold.csv")
    if scaffold.empty:
        raise RuntimeError("Missing v380 manuscript section scaffold.")

    source_log = _source_log()
    if not source_log["verified_v381"].astype(bool).all():
        raise RuntimeError("v381 source log may not contain unverified rows.")
    if source_log["citation_key_v381"].duplicated().any():
        raise RuntimeError("v381 citation keys must be unique.")
    if source_log["canonical_url_v381"].str.len().min() <= 0:
        raise RuntimeError("v381 sources require canonical URLs.")

    citation_use = _citation_use_matrix(source_log)
    source_gaps = _source_gap_register()
    blockers = _claim_blockers(source_log, source_gaps)
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v381_verified_literature_source_log.csv", source_log)
    write_csv(TABLE_DIR / "paper4_v381_citation_use_matrix.csv", citation_use)
    write_csv(TABLE_DIR / "paper4_v381_source_gap_register.csv", source_gaps)
    write_csv(TABLE_DIR / "paper4_v381_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v381_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    type_counts = source_log["source_type_v381"].value_counts().sort_index().to_dict()
    status = {
        "phase": "v381_verified_literature_source_log",
        "schema_version": "2026-05-17.381",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_scaffold_version_v381": PRIOR_SCAFFOLD_VERSION,
        "prior_v380_section_scaffold_rows_v381": int(v380_status["section_scaffold_rows_v380"]),
        "prior_v380_open_todo_rows_v381": int(v380_status["open_todo_rows_v380"]),
        "source_log_rows_v381": int(len(source_log)),
        "verified_source_rows_v381": int(source_log["verified_v381"].astype(bool).sum()),
        "source_type_counts_v381": type_counts,
        "citation_use_rows_v381": int(len(citation_use)),
        "source_gap_rows_v381": int(len(source_gaps)),
        "open_source_gap_rows_v381": int(source_gaps["blocking_v381"].astype(bool).sum()),
        "claim_blocker_rows_v381": int(len(blockers)),
        "claim_matrix_rows_v381": int(len(claim_matrix)),
        "external_literature_source_log_missing_v381": False,
        "submission_gaps_closed_v381": False,
        "bounded_related_work_language_allowed_v381": True,
        "verified_citation_language_allowed_v381": True,
        "submission_ready_claim_allowed_v381": False,
        "quarto_promotion_allowed_v381": False,
        "strict_live_deployment_language_allowed_v381": False,
        "contractual_or_legal_language_allowed_v381": False,
        "ifrs9_contractual_claim_allowed_v381": False,
        "global_optimality_language_allowed_v381": False,
        "working_champion_claim_allowed_v381": False,
        "paper1_promotion_allowed_v381": False,
        "paper4_working_champion_changed_v381": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "source_log_artifact_v381": (
            "reports/paper_material/paper4/tables/"
            "paper4_v381_verified_literature_source_log.csv"
        ),
        "source_note_artifact_v381": (
            "reports/paper_material/paper4/notes/"
            "paper4_v381_verified_literature_source_log.md"
        ),
        "next_artifact_v381": NEXT_ARTIFACT,
        "claim_boundary": (
            "v381 verifies anchor sources for bounded related-work use; submission-ready, "
            "Quarto, live/legal/IFRS9/global/final claims remain blocked"
        ),
    }
    SOURCE_LOG_MD.write_text(_source_log_markdown(status, source_log), encoding="utf-8")
    write_json(STATUS_DIR / "paper4_v381_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v381": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
