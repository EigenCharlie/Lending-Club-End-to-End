#!/usr/bin/env python3
"""Build Paper 4 v376 publication-integration patch artifacts."""

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

VERSION = 376
PRIOR_CLAIM_LANGUAGE_VERSION = 374
PRIOR_DATA_CONTRACT_VERSION = 375
NEXT_VERSION = 377
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_reproducibility_bundle_manifest.csv"
PATCH = NOTEBOOK.parent / "paper4_v376_publication_integration_patch.md"


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": (
                    "v376 maps bounded Paper 4 evidence into a publication integration patch."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v376_publication_integration_patch.md"
                ),
                "boundary": "Draft integration only; no Quarto promotion or final claim change.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v376 creates section and table plans for future Paper 4 writing.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v376_section_integration_map.csv"
                ),
                "boundary": "Planning artifact only; manuscript edits remain future work.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v376 promotes Paper 4 Quarto pages or finalizes the paper.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v376_claim_blockers.csv"
                ),
                "boundary": "No curated Quarto page or final promotion artifact is modified.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v376 authorizes live, contractual/legal, global or champion claims.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v376_prohibited_sentence_bank.csv"
                ),
                "boundary": "v375 gates remain unmet for those claims.",
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
                "lane": "Reproducibility/Packaging",
                "executable_item": (
                    "v376 maps v374 language and v375 gates into a publication patch, "
                    "then routes the next wave to package reproducibility evidence."
                ),
                "status": "publication_integration_patch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v377 inventories all citable artifacts, statuses and guardrails needed "
                    "for a reproducible Paper 4 appendix"
                ),
                "last_wave": "v376",
                "execution_result": "bounded_publication_patch_without_quarto_or_promotion",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v376")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _patch_markdown(
    status: dict[str, Any],
    sections: pd.DataFrame,
    allowed: pd.DataFrame,
    prohibited: pd.DataFrame,
    plan: pd.DataFrame,
) -> str:
    section_lines = "\n".join(
        (
            f"- **{row['paper_section_v376']}**: {row['integration_action_v376']} "
            f"(source: `{row['source_artifact_v376']}`)."
        )
        for _, row in sections.iterrows()
    )
    allowed_lines = "\n".join(
        f"- {row['allowed_sentence_v376']}" for _, row in allowed.iterrows()
    )
    prohibited_lines = "\n".join(
        f"- `{row['prohibited_phrase_v376']}` -> {row['replacement_language_v376']}"
        for _, row in prohibited.iterrows()
    )
    plan_lines = "\n".join(
        (
            f"- **{row['display_id_v376']}**: {row['display_title_v376']} "
            f"from `{row['source_artifact_v376']}`."
        )
        for _, row in plan.iterrows()
    )
    return f"""# Paper 4 Publication Integration Patch v376

Generated: {status["generated_at_utc"]}

## Purpose

This patch tells the future manuscript where to place the v374 bounded language
and the v375 live-gate data contract. It is not a Quarto promotion and it does
not alter claim permissions.

## Section Integration Map

{section_lines}

## Allowed Sentence Bank

{allowed_lines}

## Table And Figure Plan

{plan_lines}

## Prohibited Language

{prohibited_lines}

## Required Caveat

The future paper may describe Paper 4 as a reproducible living lab with bounded
offline/proxy evidence. It must not claim strict live deployment, contractual
IFRS9, legal fairness compliance, full-v55 global optimality, a new working
champion, Paper Estrella replacement or final Paper 4 promotion.

## Next Executable Wave

Build `{status["next_artifact_v376"]}` to inventory the citable artifacts,
status JSON files and guardrails needed for a reproducible appendix.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V376_PUBLICATION_INTEGRATION_PATCH_START -->"
    end = "<!-- V376_PUBLICATION_INTEGRATION_PATCH_END -->"
    block = f"""
{start}

## Wave v376: Publication Integration Patch

Generated: {status["generated_at_utc"]}

### Objective

v376 converts the v374 claim language and v375 data contract into a future
manuscript integration patch while keeping the work in the living notebook.

### Results

- Section integration rows:
  `{status["section_integration_rows_v376"]}`.
- Allowed sentence rows:
  `{status["allowed_sentence_rows_v376"]}`.
- Prohibited sentence rows:
  `{status["prohibited_sentence_rows_v376"]}`.
- Figure/table plan rows:
  `{status["figure_table_plan_rows_v376"]}`.
- Quarto pages modified:
  `{status["quarto_pages_modified_v376"]}`.
- Strict live deployment language allowed:
  `{status["strict_live_deployment_language_allowed_v376"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v376"]}`.

### Interpretation

The living lab now has paper-facing integration instructions, not just raw
execution logs. This preserves the strongest true claim: bounded, reproducible
offline/proxy evidence with explicit live/legal/global/final blockers.

### Claim Impact

- Allowed: future manuscript placement of bounded results, limitations and
  data-gate language.
- Still prohibited: Quarto promotion, live/legal/global/champion/final claims.

### Quarto Promotion Decision

Keep v376 in the living notebook. v377 should package a reproducibility bundle
manifest for the citable appendix.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v374_status = json.loads((STATUS_DIR / "paper4_v374_status.json").read_text(encoding="utf-8"))
    v375_status = json.loads((STATUS_DIR / "paper4_v375_status.json").read_text(encoding="utf-8"))
    if v375_status["next_artifact_v375"] != "paper4_v376_publication_integration_patch.md":
        raise RuntimeError("v376 expects v375 to route to the publication patch.")

    v374_sections = read_csv("paper4_v374_claim_language_section_draft.csv")
    v374_citations = read_csv("paper4_v374_evidence_citation_map.csv")
    v375_contract = read_csv("paper4_v375_live_gate_data_contract.csv")
    v375_permissions = read_csv("paper4_v375_claim_permission_register.csv")
    if (
        v374_sections.empty
        or v374_citations.empty
        or v375_contract.empty
        or v375_permissions.empty
    ):
        raise RuntimeError("Missing v374/v375 integration inputs.")

    section_map = pd.DataFrame(
        [
            {
                f"paper_section_v{VERSION}": "Abstract",
                f"integration_action_v{VERSION}": (
                    "Use the bounded living-lab framing from v374 without champion language."
                ),
                f"source_artifact_v{VERSION}": "paper4_v374_claim_language_section_draft.csv",
                f"citation_keys_v{VERSION}": "v368;v369;v373",
                f"claim_permission_v{VERSION}": "bounded_living_lab_manuscript_language",
                f"quarto_edit_allowed_v{VERSION}": False,
                f"claim_boundary_v{VERSION}": "abstract framing only",
            },
            {
                f"paper_section_v{VERSION}": "Introduction",
                f"integration_action_v{VERSION}": (
                    "State the contribution as a reproducible audit protocol around Paper Estrella."
                ),
                f"source_artifact_v{VERSION}": "paper4_v374_paper4_claim_language_section_draft.md",
                f"citation_keys_v{VERSION}": "v368;v374",
                f"claim_permission_v{VERSION}": "bounded_living_lab_manuscript_language",
                f"quarto_edit_allowed_v{VERSION}": False,
                f"claim_boundary_v{VERSION}": "motivation and scope only",
            },
            {
                f"paper_section_v{VERSION}": "Methods: Claim Governance",
                f"integration_action_v{VERSION}": (
                    "Insert the v375 data contract as the governance method for stronger claims."
                ),
                f"source_artifact_v{VERSION}": "paper4_v375_live_gate_data_contract.csv",
                f"citation_keys_v{VERSION}": "v369;v375",
                f"claim_permission_v{VERSION}": "bounded_living_lab_manuscript_language",
                f"quarto_edit_allowed_v{VERSION}": False,
                f"claim_boundary_v{VERSION}": "governance method only",
            },
            {
                f"paper_section_v{VERSION}": "Results: Solver Frontier",
                f"integration_action_v{VERSION}": (
                    "Report v361/v363 bounded no-entry and gap evidence as non-global results."
                ),
                f"source_artifact_v{VERSION}": "paper4_v374_claim_language_section_draft.csv",
                f"citation_keys_v{VERSION}": "v361;v363",
                f"claim_permission_v{VERSION}": "bounded_living_lab_manuscript_language",
                f"quarto_edit_allowed_v{VERSION}": False,
                f"claim_boundary_v{VERSION}": "bounded solver evidence only",
            },
            {
                f"paper_section_v{VERSION}": "Results: Source Governance",
                f"integration_action_v{VERSION}": (
                    "Report v371-v373 source-governance blockers and blind-chunk stop rule."
                ),
                f"source_artifact_v{VERSION}": "paper4_v373_sampled_chunk_source_screen.csv",
                f"citation_keys_v{VERSION}": "v371;v372;v373",
                f"claim_permission_v{VERSION}": "bounded_living_lab_manuscript_language",
                f"quarto_edit_allowed_v{VERSION}": False,
                f"claim_boundary_v{VERSION}": "diagnostic result only",
            },
            {
                f"paper_section_v{VERSION}": "Limitations",
                f"integration_action_v{VERSION}": (
                    "Use the v375 gate readiness summary to state live/legal/global blockers."
                ),
                f"source_artifact_v{VERSION}": "paper4_v375_gate_readiness_summary.csv",
                f"citation_keys_v{VERSION}": "v369;v375",
                f"claim_permission_v{VERSION}": "strict_live_deployment_language:blocked",
                f"quarto_edit_allowed_v{VERSION}": False,
                f"claim_boundary_v{VERSION}": "limitations only",
            },
            {
                f"paper_section_v{VERSION}": "Discussion And Appendix",
                f"integration_action_v{VERSION}": (
                    "Route reproducibility artifacts to a future appendix manifest."
                ),
                f"source_artifact_v{VERSION}": "paper4_living_lab_backlog.csv",
                f"citation_keys_v{VERSION}": "v376",
                f"claim_permission_v{VERSION}": "publication_planning_only",
                f"quarto_edit_allowed_v{VERSION}": False,
                f"claim_boundary_v{VERSION}": "future-work plan only",
            },
        ]
    )
    allowed_sentences = pd.DataFrame(
        [
            {
                f"sentence_id_v{VERSION}": "bounded_living_lab_protocol",
                f"allowed_sentence_v{VERSION}": (
                    "Paper 4 is a reproducible living-lab protocol for auditing bounded "
                    "offline/proxy evidence around the protected Paper Estrella champion."
                ),
                f"source_artifact_v{VERSION}": "paper4_v374_paper4_claim_language_section_draft.md",
                f"claim_boundary_v{VERSION}": "framing claim only",
            },
            {
                f"sentence_id_v{VERSION}": "fourth_order_no_entry",
                f"allowed_sentence_v{VERSION}": (
                    "The v361 fourth-order screen supports a bounded no-entry statement, "
                    "not a full-v55 termination proof."
                ),
                f"source_artifact_v{VERSION}": "paper4_v374_claim_language_section_draft.md",
                f"claim_boundary_v{VERSION}": "bounded solver result only",
            },
            {
                f"sentence_id_v{VERSION}": "source_governance_blocker",
                f"allowed_sentence_v{VERSION}": (
                    "The v371-v373 diagnostics identify source governance as the active "
                    "blocker before additional CVaR-heavy probes."
                ),
                f"source_artifact_v{VERSION}": "paper4_v373_full_v55_chunk_002_or_stop_rule.csv",
                f"claim_boundary_v{VERSION}": "diagnostic result only",
            },
            {
                f"sentence_id_v{VERSION}": "offline_proxy_available",
                f"allowed_sentence_v{VERSION}": (
                    "Offline proxy replay evidence is available, but it is not live "
                    "deployment evidence."
                ),
                f"source_artifact_v{VERSION}": "paper4_v375_live_gate_data_contract.csv",
                f"claim_boundary_v{VERSION}": "proxy-only statement",
            },
            {
                f"sentence_id_v{VERSION}": "live_gate_zero",
                f"allowed_sentence_v{VERSION}": (
                    "The v375 data contract records zero live-deployment gates met."
                ),
                f"source_artifact_v{VERSION}": "paper4_v375_status.json",
                f"claim_boundary_v{VERSION}": "limitation statement",
            },
            {
                f"sentence_id_v{VERSION}": "ifrs9_contractual_blocked",
                f"allowed_sentence_v{VERSION}": (
                    "IFRS9 language remains proxy-inspired and not contractual because the "
                    "contractual coverage gate is unmet."
                ),
                f"source_artifact_v{VERSION}": "paper4_v375_claim_permission_register.csv",
                f"claim_boundary_v{VERSION}": "contractual limitation",
            },
            {
                f"sentence_id_v{VERSION}": "global_solver_blocked",
                f"allowed_sentence_v{VERSION}": (
                    "Full-v55 global optimality remains blocked by the missing dual-bound "
                    "certificate."
                ),
                f"source_artifact_v{VERSION}": "paper4_v375_claim_blockers.csv",
                f"claim_boundary_v{VERSION}": "global limitation",
            },
            {
                f"sentence_id_v{VERSION}": "final_promotion_absent",
                f"allowed_sentence_v{VERSION}": (
                    "No Paper 4 final-promotion artifact is created; Paper Estrella remains protected."
                ),
                f"source_artifact_v{VERSION}": "paper4_v375_claim_blockers.csv",
                f"claim_boundary_v{VERSION}": "promotion limitation",
            },
        ]
    )
    prohibited_sentences = pd.DataFrame(
        [
            {
                f"prohibited_phrase_v{VERSION}": "new working champion",
                f"replacement_language_v{VERSION}": "protected Paper Estrella champion",
                f"blocking_artifact_v{VERSION}": "paper4_v374_prohibited_language_register.csv",
            },
            {
                f"prohibited_phrase_v{VERSION}": "Paper Estrella replacement",
                f"replacement_language_v{VERSION}": "living-lab audit around Paper Estrella",
                f"blocking_artifact_v{VERSION}": "paper4_v374_prohibited_language_register.csv",
            },
            {
                f"prohibited_phrase_v{VERSION}": "full-v55 global optimality",
                f"replacement_language_v{VERSION}": "bounded/gap evidence only",
                f"blocking_artifact_v{VERSION}": "paper4_v375_claim_permission_register.csv",
            },
            {
                f"prohibited_phrase_v{VERSION}": "strict live deployment",
                f"replacement_language_v{VERSION}": "offline/proxy evidence with live gates blocked",
                f"blocking_artifact_v{VERSION}": "paper4_v375_live_gate_data_contract.csv",
            },
            {
                f"prohibited_phrase_v{VERSION}": "contractual IFRS9",
                f"replacement_language_v{VERSION}": "IFRS9-inspired proxy diagnostics",
                f"blocking_artifact_v{VERSION}": "paper4_v375_claim_permission_register.csv",
            },
            {
                f"prohibited_phrase_v{VERSION}": "legal fairness compliance",
                f"replacement_language_v{VERSION}": "fairness proxy diagnostics",
                f"blocking_artifact_v{VERSION}": "paper4_v375_claim_permission_register.csv",
            },
            {
                f"prohibited_phrase_v{VERSION}": "final Paper 4 promotion",
                f"replacement_language_v{VERSION}": "living notebook evidence only",
                f"blocking_artifact_v{VERSION}": "paper4_v375_claim_blockers.csv",
            },
        ]
    )
    prohibited_sentences[f"allowed_v{VERSION}"] = False
    figure_table_plan = pd.DataFrame(
        [
            {
                f"display_id_v{VERSION}": "Table 1",
                f"display_title_v{VERSION}": "Bounded Paper 4 Claim Language",
                f"source_artifact_v{VERSION}": "paper4_v374_claim_language_section_draft.csv",
                f"target_section_v{VERSION}": "Results",
                f"claim_boundary_v{VERSION}": "bounded wording only",
            },
            {
                f"display_id_v{VERSION}": "Table 2",
                f"display_title_v{VERSION}": "Evidence Citation Map",
                f"source_artifact_v{VERSION}": "paper4_v374_evidence_citation_map.csv",
                f"target_section_v{VERSION}": "Results",
                f"claim_boundary_v{VERSION}": "citation support only",
            },
            {
                f"display_id_v{VERSION}": "Table 3",
                f"display_title_v{VERSION}": "Live-Gate Data Contract",
                f"source_artifact_v{VERSION}": "paper4_v375_live_gate_data_contract.csv",
                f"target_section_v{VERSION}": "Methods/Limitations",
                f"claim_boundary_v{VERSION}": "gate requirements only",
            },
            {
                f"display_id_v{VERSION}": "Table 4",
                f"display_title_v{VERSION}": "Claim Permission Register",
                f"source_artifact_v{VERSION}": "paper4_v375_claim_permission_register.csv",
                f"target_section_v{VERSION}": "Limitations",
                f"claim_boundary_v{VERSION}": "permissions and blockers",
            },
            {
                f"display_id_v{VERSION}": "Figure 1",
                f"display_title_v{VERSION}": "Wave Lineage v361-v375",
                f"source_artifact_v{VERSION}": "paper4_living_lab_notebook.md",
                f"target_section_v{VERSION}": "Methods",
                f"claim_boundary_v{VERSION}": "process diagram plan only",
            },
            {
                f"display_id_v{VERSION}": "Appendix A",
                f"display_title_v{VERSION}": "Reproducibility Bundle Manifest",
                f"source_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"target_section_v{VERSION}": "Appendix",
                f"claim_boundary_v{VERSION}": "future artifact",
            },
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "quarto_promotion_not_performed",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 0,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "v376 is a living-notebook patch only",
            },
            {
                f"blocker_id_v{VERSION}": "strict_live_language_blocked_by_v375",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v375_status["live_deployment_gate_met_rows_v375"]),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "live deployment gates remain zero",
            },
            {
                f"blocker_id_v{VERSION}": "global_solver_language_blocked_by_v375",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v375_status["global_solver_gate_met_rows_v375"]),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "global solver gates remain zero",
            },
            {
                f"blocker_id_v{VERSION}": "contractual_legal_language_blocked_by_v375",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    v375_status["contractual_or_legal_gate_met_rows_v375"]
                ),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "contractual/legal gates remain zero",
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
                "claim_id": "v376_publication_integration_patch_created",
                "allowed": True,
                "artifact": "paper4_v376_publication_integration_patch.md",
                "boundary": "future manuscript patch only",
            },
            {
                "claim_id": "v376_section_and_display_plan_created",
                "allowed": True,
                "artifact": "paper4_v376_section_integration_map.csv",
                "boundary": "planning artifact only",
            },
            {
                "claim_id": "v376_quarto_pages_modified_or_promoted",
                "allowed": False,
                "artifact": "paper4_v376_claim_blockers.csv",
                "boundary": "living notebook only",
            },
            {
                "claim_id": "v376_live_legal_or_global_claim_authorized",
                "allowed": False,
                "artifact": "paper4_v376_prohibited_sentence_bank.csv",
                "boundary": "v375 gates remain unmet",
            },
            {
                "claim_id": "v376_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v376_claim_blockers.csv",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v376_section_integration_map.csv", section_map)
    write_csv(TABLE_DIR / "paper4_v376_allowed_sentence_bank.csv", allowed_sentences)
    write_csv(TABLE_DIR / "paper4_v376_prohibited_sentence_bank.csv", prohibited_sentences)
    write_csv(TABLE_DIR / "paper4_v376_figure_table_plan.csv", figure_table_plan)
    write_csv(TABLE_DIR / "paper4_v376_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v376_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v376_publication_integration_patch",
        "schema_version": "2026-05-17.376",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_claim_language_version_v376": PRIOR_CLAIM_LANGUAGE_VERSION,
        "prior_data_contract_version_v376": PRIOR_DATA_CONTRACT_VERSION,
        "prior_v374_draft_section_rows_v376": int(v374_status["draft_section_rows_v374"]),
        "prior_v375_contract_rows_v376": int(v375_status["contract_rows_v375"]),
        "section_integration_rows_v376": int(len(section_map)),
        "allowed_sentence_rows_v376": int(len(allowed_sentences)),
        "prohibited_sentence_rows_v376": int(len(prohibited_sentences)),
        "figure_table_plan_rows_v376": int(len(figure_table_plan)),
        "claim_blocker_rows_v376": int(len(blockers)),
        "claim_matrix_rows_v376": int(len(claim_matrix)),
        "quarto_pages_modified_v376": False,
        "bounded_living_lab_language_allowed_v376": True,
        "offline_proxy_language_allowed_v376": True,
        "strict_live_deployment_language_allowed_v376": False,
        "contractual_or_legal_language_allowed_v376": False,
        "global_optimality_language_allowed_v376": False,
        "working_champion_claim_allowed_v376": False,
        "paper1_promotion_allowed_v376": False,
        "paper4_working_champion_changed_v376": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "patch_artifact_v376": (
            "reports/paper_material/paper4/notes/"
            "paper4_v376_publication_integration_patch.md"
        ),
        "next_artifact_v376": NEXT_ARTIFACT,
        "claim_boundary": (
            "v376 is a publication integration patch; Quarto promotion and stronger "
            "live/legal/global/final claims remain blocked"
        ),
    }
    PATCH.write_text(
        _patch_markdown(status, section_map, allowed_sentences, prohibited_sentences, figure_table_plan),
        encoding="utf-8",
    )
    write_json(STATUS_DIR / "paper4_v376_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v376": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
