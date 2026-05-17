#!/usr/bin/env python3
"""Build Paper 4 v374 paper-claim language draft artifacts."""

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

VERSION = 374
PRIOR_SCOPE_VERSION = 368
PRIOR_GATE_VERSION = 369
PRIOR_STOP_RULE_VERSION = 373
NEXT_VERSION = 375
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_live_gate_data_contract.csv"
DRAFT = NOTEBOOK.parent / "paper4_v374_paper4_claim_language_section_draft.md"


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v374 drafts bounded Paper 4 manuscript language from v361-v373 evidence.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v374_paper4_claim_language_section_draft.md"
                ),
                "boundary": "Manuscript draft only; no final paper promotion.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v374 provides citable results and limitations wording for the living lab.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v374_claim_language_section_draft.csv"
                ),
                "boundary": "Bounded wording must cite v361-v373 artifacts.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v374 authorizes global optimality, live deployment or contractual/legal claims.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v374_prohibited_language_register.csv"
                ),
                "boundary": "The draft explicitly forbids these phrases.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v374 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v374_claim_blockers.csv"
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
                "lane": "Proxy/Live Separation",
                "executable_item": (
                    "v374 turns v361-v373 evidence into bounded manuscript language and "
                    "routes the next wave to the live-gate data contract."
                ),
                "status": "bounded_manuscript_language_draft_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v375 specifies exact data inputs required before live, contractual "
                    "or deployment claims"
                ),
                "last_wave": "v374",
                "execution_result": "results_limitations_language_drafted_without_promotion",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v374")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _draft_markdown(status: dict[str, Any], sections: pd.DataFrame) -> str:
    section_text = "\n\n".join(
        f"## {row['section_title_v374']}\n\n{row['draft_text_v374']}"
        for _, row in sections.iterrows()
    )
    return f"""# Paper 4 Claim Language Draft v374

Generated: {status["generated_at_utc"]}

{section_text}

## Required Caveat

This language is living-lab manuscript text. It does not promote Paper 4, does
not replace Paper Estrella, and does not claim a working champion, full-v55
optimality, strict live deployment, contractual IFRS9, legal fairness
compliance or production monitoring readiness.

## Next Executable Wave

Build `{status["next_artifact_v374"]}` to specify the exact evidence required
before any live, contractual/legal, deployment or final-promotion language can
be considered.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V374_PAPER_CLAIM_LANGUAGE_DRAFT_START -->"
    end = "<!-- V374_PAPER_CLAIM_LANGUAGE_DRAFT_END -->"
    block = f"""
{start}

## Wave v374: Paper Claim Language Section Draft

Generated: {status["generated_at_utc"]}

### Objective

v373 stopped blind chunking as the next action. v374 turns the evidence frontier
from v361-v373 into bounded manuscript language: results, limitations and
claim boundaries that can be reused without promoting Paper 4.

### Results

- Draft section rows:
  `{status["draft_section_rows_v374"]}`.
- Evidence citation rows:
  `{status["evidence_citation_rows_v374"]}`.
- Prohibited language rows:
  `{status["prohibited_language_rows_v374"]}`.
- Draft artifact:
  `{status["draft_artifact_v374"]}`.
- Next artifact:
  `{status["next_artifact_v374"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.

### Interpretation

The paper-facing result is now text, not another solver claim. The language says
Paper 4 is a reproducible living lab with bounded evidence and transparent
blockers: source governance, global dual-bound gaps and live/proxy separation.

### Claim Impact

- Allowed: manuscript language for bounded evidence and limitations.
- Still prohibited: global optimality, working champion, Paper Estrella
  replacement, strict live deployment, contractual/legal claims and final
  promotion.

### Quarto Promotion Decision

Keep v374 in the living notebook. v375 should define the live-gate data contract.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v361_status = json.loads((STATUS_DIR / "paper4_v361_status.json").read_text(encoding="utf-8"))
    v363_status = json.loads((STATUS_DIR / "paper4_v363_status.json").read_text(encoding="utf-8"))
    v368_status = json.loads((STATUS_DIR / "paper4_v368_status.json").read_text(encoding="utf-8"))
    v369_status = json.loads((STATUS_DIR / "paper4_v369_status.json").read_text(encoding="utf-8"))
    v371_status = json.loads((STATUS_DIR / "paper4_v371_status.json").read_text(encoding="utf-8"))
    v372_status = json.loads((STATUS_DIR / "paper4_v372_status.json").read_text(encoding="utf-8"))
    v373_status = json.loads((STATUS_DIR / "paper4_v373_status.json").read_text(encoding="utf-8"))
    if v373_status["recommended_decision_v373"] != "stop_blind_chunking_after_sampled_source_blocker":
        raise RuntimeError("v374 expects v373 to stop blind chunking.")

    v366_ordered_rows = int(v372_status["ordered_one_swap_rows_v372"])

    sections = pd.DataFrame(
        [
            {
                f"section_id_v{VERSION}": "abstract_claim",
                f"section_title_v{VERSION}": "Abstract Claim",
                f"allowed_v{VERSION}": True,
                f"draft_text_v{VERSION}": (
                    "We present Paper 4 as a reproducible living-lab protocol for "
                    "auditing candidate policy improvements around the protected Paper "
                    "Estrella economic champion, combining bounded source-governance "
                    "screens, tail-risk diagnostics and explicit claim-boundary controls."
                ),
                f"citation_keys_v{VERSION}": "v368;v369;v373",
                f"claim_boundary_v{VERSION}": "method/framing claim only",
            },
            {
                f"section_id_v{VERSION}": "results_solver_frontier",
                f"section_title_v{VERSION}": "Results: Solver Frontier",
                f"allowed_v{VERSION}": True,
                f"draft_text_v{VERSION}": (
                    "The bounded v361 fourth-order source-tight screen evaluated "
                    f"{v361_status['ordered_fourth_order_rows_screened_v361']} ordered rows, "
                    f"found {v361_status['source_exact_fourth_order_rows_v361']} source-exact "
                    "rows and found zero CVaR-feasible entering rows. This supports a "
                    "bounded no-entry statement only; it is not a full-v55 termination proof."
                ),
                f"citation_keys_v{VERSION}": "v361;v363",
                f"claim_boundary_v{VERSION}": "bounded fourth-order scope only",
            },
            {
                f"section_id_v{VERSION}": "results_source_governance",
                f"section_title_v{VERSION}": "Results: Source Governance",
                f"allowed_v{VERSION}": True,
                f"draft_text_v{VERSION}": (
                    "The v366-v373 chunk diagnostics show that source governance is a "
                    "first-order blocker in the full-v55 frontier: v366 chunk 0001 had "
                    f"{v366_ordered_rows} ordered one-swaps "
                    "and zero source-exact rows; v371 identified grade=A as the primary "
                    f"bottleneck with {v371_status['primary_blocker_pass_rows_v371']} passing "
                    "rows; v372 showed grade-A relief was return-negative in the same chunk; "
                    f"and v373 sampled {v373_status['sampled_chunk_count_v373']} chunks with "
                    f"{v373_status['sampled_total_source_exact_rows_v373']} total source-exact rows."
                ),
                f"citation_keys_v{VERSION}": "v371;v372;v373",
                f"claim_boundary_v{VERSION}": "source-governance diagnostic only",
            },
            {
                f"section_id_v{VERSION}": "limitations_global_live",
                f"section_title_v{VERSION}": "Limitations: Global And Live Claims",
                f"allowed_v{VERSION}": True,
                f"draft_text_v{VERSION}": (
                    "The current evidence does not authorize a new working champion, "
                    "Paper Estrella replacement, strict live deployment or full-v55 global "
                    f"optimality claim. v363 still reports {v363_status['v71_improving_omitted_columns_v363']} "
                    "improving omitted columns, and v369 reports only "
                    f"{v369_status['gate_requirements_met_v369']} of "
                    f"{v369_status['gate_requirement_rows_v369']} proxy/live/final gate requirements met."
                ),
                f"citation_keys_v{VERSION}": "v363;v369",
                f"claim_boundary_v{VERSION}": "limitations language only",
            },
            {
                f"section_id_v{VERSION}": "discussion_next_steps",
                f"section_title_v{VERSION}": "Discussion: Next Steps",
                f"allowed_v{VERSION}": True,
                f"draft_text_v{VERSION}": (
                    "The next useful work is to specify live-gate data requirements and "
                    "publication integration, rather than spending more compute on blind "
                    "chunks whose sampled source screens repeatedly collapse before CVaR."
                ),
                f"citation_keys_v{VERSION}": "v369;v373",
                f"claim_boundary_v{VERSION}": "execution roadmap only",
            },
        ]
    )
    citations = pd.DataFrame(
        [
            {
                f"citation_key_v{VERSION}": "v361",
                f"source_artifact_v{VERSION}": "paper4_v361_v353_fourth_order_or_full_dual_bound.csv",
                f"evidence_statement_v{VERSION}": "bounded fourth-order no-entry screen",
                f"evidence_count_v{VERSION}": int(
                    v361_status["ordered_fourth_order_rows_screened_v361"]
                ),
            },
            {
                f"citation_key_v{VERSION}": "v363",
                f"source_artifact_v{VERSION}": "paper4_v363_v353_full_dual_bound_or_gap_certificate.csv",
                f"evidence_statement_v{VERSION}": "full-v55 dual-bound gap remains open",
                f"evidence_count_v{VERSION}": int(
                    v363_status["v71_improving_omitted_columns_v363"]
                ),
            },
            {
                f"citation_key_v{VERSION}": "v368",
                f"source_artifact_v{VERSION}": "paper4_v368_publishable_claim_scope_update.md",
                f"evidence_statement_v{VERSION}": "bounded living-lab framing selected",
                f"evidence_count_v{VERSION}": int(
                    v368_status["allowed_publishable_claim_rows_v368"]
                ),
            },
            {
                f"citation_key_v{VERSION}": "v369",
                f"source_artifact_v{VERSION}": "paper4_v369_proxy_live_gate_separation.csv",
                f"evidence_statement_v{VERSION}": "proxy/live/final gates separated",
                f"evidence_count_v{VERSION}": int(v369_status["gate_requirement_rows_v369"]),
            },
            {
                f"citation_key_v{VERSION}": "v371",
                f"source_artifact_v{VERSION}": "paper4_v371_source_governance_blocker_diagnostic.csv",
                f"evidence_statement_v{VERSION}": "grade=A primary source blocker",
                f"evidence_count_v{VERSION}": int(v371_status["primary_blocker_pass_rows_v371"]),
            },
            {
                f"citation_key_v{VERSION}": "v372",
                f"source_artifact_v{VERSION}": "paper4_v372_grade_a_source_relief_prefilter.csv",
                f"evidence_statement_v{VERSION}": "grade-A relief has no return-improving rows",
                f"evidence_count_v{VERSION}": int(
                    v372_status["grade_a_relief_return_improving_rows_v372"]
                ),
            },
            {
                f"citation_key_v{VERSION}": "v373",
                f"source_artifact_v{VERSION}": "paper4_v373_full_v55_chunk_002_or_stop_rule.csv",
                f"evidence_statement_v{VERSION}": "sampled chunk stop rule selected",
                f"evidence_count_v{VERSION}": int(
                    v373_status["sampled_total_source_exact_rows_v373"]
                ),
            },
        ]
    )
    citations[f"claim_boundary_v{VERSION}"] = "citation evidence only"
    prohibited = pd.DataFrame(
        [
            {
                f"phrase_id_v{VERSION}": "new_working_champion",
                f"allowed_v{VERSION}": False,
                f"replacement_language_v{VERSION}": "protected Paper Estrella champion",
            },
            {
                f"phrase_id_v{VERSION}": "paper_estrella_replacement",
                f"allowed_v{VERSION}": False,
                f"replacement_language_v{VERSION}": "living-lab audit around Paper Estrella",
            },
            {
                f"phrase_id_v{VERSION}": "full_v55_global_optimality",
                f"allowed_v{VERSION}": False,
                f"replacement_language_v{VERSION}": "bounded/gap evidence only",
            },
            {
                f"phrase_id_v{VERSION}": "strict_live_deployment",
                f"allowed_v{VERSION}": False,
                f"replacement_language_v{VERSION}": "offline/proxy evidence with live gates blocked",
            },
            {
                f"phrase_id_v{VERSION}": "contractual_ifrs9_or_legal_fairness",
                f"allowed_v{VERSION}": False,
                f"replacement_language_v{VERSION}": "IFRS9/fairness proxy diagnostics",
            },
            {
                f"phrase_id_v{VERSION}": "final_paper4_promotion",
                f"allowed_v{VERSION}": False,
                f"replacement_language_v{VERSION}": "living notebook evidence only",
            },
        ]
    )
    prohibited[f"claim_boundary_v{VERSION}"] = "prohibited manuscript wording"
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "language_is_draft_not_promotion",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(sections)),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "draft needs live-gate data contract before broader claims",
            },
            {
                f"blocker_id_v{VERSION}": "prohibited_language_register_active",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(prohibited)),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "prohibited phrases remain blocked",
            },
            {
                f"blocker_id_v{VERSION}": "paper4_final_promotion_forbidden",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "paper4_final_promotion_gate_not_created",
                f"claim_boundary_v{VERSION}": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v374_manuscript_language_draft_created",
                "allowed": True,
                "artifact": "paper4_v374_paper4_claim_language_section_draft.md",
                "boundary": "draft language only",
            },
            {
                "claim_id": "v374_evidence_citation_map_created",
                "allowed": True,
                "artifact": "paper4_v374_evidence_citation_map.csv",
                "boundary": "citation support only",
            },
            {
                "claim_id": "v374_global_live_or_contractual_claim",
                "allowed": False,
                "artifact": "paper4_v374_prohibited_language_register.csv",
                "boundary": "prohibited wording",
            },
            {
                "claim_id": "v374_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v374_claim_blockers.csv",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v374_claim_language_section_draft.csv", sections)
    write_csv(TABLE_DIR / "paper4_v374_evidence_citation_map.csv", citations)
    write_csv(TABLE_DIR / "paper4_v374_prohibited_language_register.csv", prohibited)
    write_csv(TABLE_DIR / "paper4_v374_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v374_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v374_paper4_claim_language_section_draft",
        "schema_version": "2026-05-17.374",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_scope_version_v374": PRIOR_SCOPE_VERSION,
        "prior_gate_version_v374": PRIOR_GATE_VERSION,
        "prior_stop_rule_version_v374": PRIOR_STOP_RULE_VERSION,
        "draft_section_rows_v374": int(len(sections)),
        "evidence_citation_rows_v374": int(len(citations)),
        "prohibited_language_rows_v374": int(len(prohibited)),
        "claim_blocker_rows_v374": int(len(blockers)),
        "claim_matrix_rows_v374": int(len(claim_matrix)),
        "bounded_living_lab_language_allowed_v374": True,
        "global_optimality_language_allowed_v374": False,
        "strict_live_deployment_language_allowed_v374": False,
        "contractual_or_legal_language_allowed_v374": False,
        "working_champion_claim_allowed_v374": False,
        "paper1_promotion_allowed_v374": False,
        "paper4_working_champion_changed_v374": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "draft_artifact_v374": (
            "reports/paper_material/paper4/notes/"
            "paper4_v374_paper4_claim_language_section_draft.md"
        ),
        "next_artifact_v374": NEXT_ARTIFACT,
        "claim_boundary": (
            "v374 creates bounded manuscript language; live/legal/global/final claims remain blocked"
        ),
    }
    DRAFT.write_text(_draft_markdown(status, sections), encoding="utf-8")
    write_json(STATUS_DIR / "paper4_v374_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v374": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
