#!/usr/bin/env python3
"""Build Paper 4 v378 submission-readiness gap register artifacts."""

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

VERSION = 378
PRIOR_REPRODUCIBILITY_BUNDLE_VERSION = 377
NEXT_VERSION = 379
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_evidence_gap_closure_work_order.csv"


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v378 inventories remaining Paper 4 submission-readiness gaps.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v378_submission_readiness_gap_register.csv"
                ),
                "boundary": "Gap register only; not a submission-ready claim.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v378 routes open gaps to a future evidence-closure work order.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v378_submission_gap_domain_summary.csv"
                ),
                "boundary": "Planning and triage only; evidence must be generated later.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v378 makes Paper 4 submission-ready.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v378_claim_blockers.csv"
                ),
                "boundary": "Open manuscript, evidence, validation and review gaps remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v378 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v378_claim_blockers.csv"
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
                "lane": "Execution Planning",
                "executable_item": (
                    "v378 enumerates submission-readiness gaps and routes the next wave "
                    "to an evidence-closure work order."
                ),
                "status": "submission_readiness_gap_register_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v379 turns the highest-priority open gaps into executable evidence "
                    "closure tasks without promoting Paper 4"
                ),
                "last_wave": "v378",
                "execution_result": "submission_readiness_blockers_prioritized_without_promotion",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v378")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V378_SUBMISSION_READINESS_GAP_REGISTER_START -->"
    end = "<!-- V378_SUBMISSION_READINESS_GAP_REGISTER_END -->"
    block = f"""
{start}

## Wave v378: Submission-Readiness Gap Register

Generated: {status["generated_at_utc"]}

### Objective

v378 converts the v377 reproducibility bundle into a submission-readiness gap
register. The goal is to make remaining manuscript, evidence, validation and
review blockers explicit without promoting Paper 4.

### Results

- Gap register rows:
  `{status["gap_register_rows_v378"]}`.
- Open gap rows:
  `{status["open_gap_rows_v378"]}`.
- Satisfied readiness rows:
  `{status["satisfied_readiness_rows_v378"]}`.
- Domain summary rows:
  `{status["domain_summary_rows_v378"]}`.
- Submission blocker rows:
  `{status["submission_blocker_rows_v378"]}`.
- Highest priority open gaps:
  `{status["highest_priority_open_gap_rows_v378"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v378"]}`.
- Strict live deployment language allowed:
  `{status["strict_live_deployment_language_allowed_v378"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v378"]}`.

### Interpretation

The lab is now packaged enough to audit, but not ready to submit. The remaining
work is concentrated in manuscript integration, external/literature review,
solver/source-governance limits, live/legal/regulatory gates and full validation.

### Claim Impact

- Allowed: submission-readiness gap statements and prioritized next actions.
- Still prohibited: submission-ready, Quarto promotion, live/legal/global
  claims, champion replacement and final promotion.

### Quarto Promotion Decision

Keep v378 in the living notebook. v379 should convert the highest-priority open
gaps into an executable evidence-closure work order.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def _domain_summary(gaps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain, group in gaps.groupby(f"readiness_domain_v{VERSION}", sort=True):
        satisfied = group[f"currently_satisfied_v{VERSION}"].astype(bool)
        blocker = group[f"severity_v{VERSION}"].astype(str).eq("blocker")
        rows.append(
            {
                f"readiness_domain_v{VERSION}": domain,
                f"gap_rows_v{VERSION}": int(len(group)),
                f"open_gap_rows_v{VERSION}": int((~satisfied).sum()),
                f"satisfied_rows_v{VERSION}": int(satisfied.sum()),
                f"blocker_rows_v{VERSION}": int(blocker.sum()),
                f"highest_priority_open_rows_v{VERSION}": int(
                    (
                        group[f"priority_v{VERSION}"].astype(int).eq(1)
                        & ~satisfied
                    ).sum()
                ),
                f"claim_boundary_v{VERSION}": "domain summary only",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v363_status = json.loads((STATUS_DIR / "paper4_v363_status.json").read_text(encoding="utf-8"))
    v373_status = json.loads((STATUS_DIR / "paper4_v373_status.json").read_text(encoding="utf-8"))
    v375_status = json.loads((STATUS_DIR / "paper4_v375_status.json").read_text(encoding="utf-8"))
    v376_status = json.loads((STATUS_DIR / "paper4_v376_status.json").read_text(encoding="utf-8"))
    v377_status = json.loads((STATUS_DIR / "paper4_v377_status.json").read_text(encoding="utf-8"))
    if v377_status["next_artifact_v377"] != "paper4_v378_submission_readiness_gap_register.csv":
        raise RuntimeError("v378 expects v377 to route to the submission-readiness gap register.")

    bundle = read_csv("paper4_v377_reproducibility_bundle_manifest.csv")
    checks = read_csv("paper4_v377_bundle_checks.csv")
    if bundle.empty or checks.empty:
        raise RuntimeError("Missing v377 reproducibility bundle inputs.")

    gaps = pd.DataFrame(
        [
            {
                f"gap_id_v{VERSION}": "full_manuscript_draft_not_created",
                f"readiness_domain_v{VERSION}": "manuscript",
                f"severity_v{VERSION}": "blocker",
                f"priority_v{VERSION}": 1,
                f"current_evidence_artifact_v{VERSION}": "paper4_v376_publication_integration_patch.md",
                f"current_evidence_count_v{VERSION}": int(
                    v376_status["section_integration_rows_v376"]
                ),
                f"required_before_submission_v{VERSION}": True,
                f"currently_satisfied_v{VERSION}": False,
                f"missing_item_v{VERSION}": "complete manuscript text integrated from the patch",
                f"next_action_v{VERSION}": "turn v376 section map into full draft sections",
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "not submission ready",
            },
            {
                f"gap_id_v{VERSION}": "quarto_submission_pages_not_promoted",
                f"readiness_domain_v{VERSION}": "publication_infrastructure",
                f"severity_v{VERSION}": "blocker",
                f"priority_v{VERSION}": 1,
                f"current_evidence_artifact_v{VERSION}": "paper4_v376_status.json",
                f"current_evidence_count_v{VERSION}": 0,
                f"required_before_submission_v{VERSION}": True,
                f"currently_satisfied_v{VERSION}": False,
                f"missing_item_v{VERSION}": "curated Quarto chapter integration and render signoff",
                f"next_action_v{VERSION}": "define exact Quarto integration patch after evidence closure",
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "living notebook only",
            },
            {
                f"gap_id_v{VERSION}": "external_literature_source_log_missing",
                f"readiness_domain_v{VERSION}": "citations",
                f"severity_v{VERSION}": "blocker",
                f"priority_v{VERSION}": 1,
                f"current_evidence_artifact_v{VERSION}": "paper4_v376_section_integration_map.csv",
                f"current_evidence_count_v{VERSION}": 0,
                f"required_before_submission_v{VERSION}": True,
                f"currently_satisfied_v{VERSION}": False,
                f"missing_item_v{VERSION}": "verified bibliography and source log for related work",
                f"next_action_v{VERSION}": "create a verified Paper 4 literature/source log",
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "no fabricated citations",
            },
            {
                f"gap_id_v{VERSION}": "full_v55_global_certificate_missing",
                f"readiness_domain_v{VERSION}": "solver_global",
                f"severity_v{VERSION}": "blocker",
                f"priority_v{VERSION}": 1,
                f"current_evidence_artifact_v{VERSION}": (
                    "paper4_v363_v353_full_dual_bound_or_gap_certificate.csv"
                ),
                f"current_evidence_count_v{VERSION}": int(
                    v363_status["v71_improving_omitted_columns_v363"]
                ),
                f"required_before_submission_v{VERSION}": False,
                f"currently_satisfied_v{VERSION}": False,
                f"missing_item_v{VERSION}": "full-v55 dual-bound certificate if claiming global optimality",
                f"next_action_v{VERSION}": "keep global optimality prohibited or design a separate certificate route",
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "bounded/gap evidence only",
            },
            {
                f"gap_id_v{VERSION}": "source_governance_full_v55_source_exact_missing",
                f"readiness_domain_v{VERSION}": "source_governance",
                f"severity_v{VERSION}": "blocker",
                f"priority_v{VERSION}": 1,
                f"current_evidence_artifact_v{VERSION}": "paper4_v373_sampled_chunk_source_screen.csv",
                f"current_evidence_count_v{VERSION}": int(
                    v373_status["sampled_total_source_exact_rows_v373"]
                ),
                f"required_before_submission_v{VERSION}": False,
                f"currently_satisfied_v{VERSION}": False,
                f"missing_item_v{VERSION}": "source-exact full-v55 evidence beyond sampled zero rows",
                f"next_action_v{VERSION}": "keep source-governance blocker language or design a targeted source audit",
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "diagnostic only",
            },
            {
                f"gap_id_v{VERSION}": "external_live_holdout_missing",
                f"readiness_domain_v{VERSION}": "live_validation",
                f"severity_v{VERSION}": "blocker",
                f"priority_v{VERSION}": 1,
                f"current_evidence_artifact_v{VERSION}": "paper4_v375_live_gate_data_contract.csv",
                f"current_evidence_count_v{VERSION}": int(
                    v375_status["live_deployment_gate_met_rows_v375"]
                ),
                f"required_before_submission_v{VERSION}": False,
                f"currently_satisfied_v{VERSION}": False,
                f"missing_item_v{VERSION}": "external/future holdout panel with live pass rows",
                f"next_action_v{VERSION}": "keep live deployment claims prohibited",
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "offline/proxy only",
            },
            {
                f"gap_id_v{VERSION}": "online_shadow_monitoring_missing",
                f"readiness_domain_v{VERSION}": "live_validation",
                f"severity_v{VERSION}": "blocker",
                f"priority_v{VERSION}": 2,
                f"current_evidence_artifact_v{VERSION}": "paper4_v375_live_gate_data_contract.csv",
                f"current_evidence_count_v{VERSION}": 0,
                f"required_before_submission_v{VERSION}": False,
                f"currently_satisfied_v{VERSION}": False,
                f"missing_item_v{VERSION}": "shadow monitoring log and deployment runbook",
                f"next_action_v{VERSION}": "keep production monitoring readiness prohibited",
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "no production monitoring claim",
            },
            {
                f"gap_id_v{VERSION}": "ifrs9_contractual_coverage_missing",
                f"readiness_domain_v{VERSION}": "regulatory_ifrs9",
                f"severity_v{VERSION}": "blocker",
                f"priority_v{VERSION}": 1,
                f"current_evidence_artifact_v{VERSION}": "paper4_v375_claim_permission_register.csv",
                f"current_evidence_count_v{VERSION}": 76,
                f"required_before_submission_v{VERSION}": False,
                f"currently_satisfied_v{VERSION}": False,
                f"missing_item_v{VERSION}": "complete contractual IFRS9 coverage and approval",
                f"next_action_v{VERSION}": "keep IFRS9 language proxy-inspired and non-contractual",
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "IFRS9 proxy diagnostics only",
            },
            {
                f"gap_id_v{VERSION}": "legal_fairness_review_missing",
                f"readiness_domain_v{VERSION}": "legal_fairness",
                f"severity_v{VERSION}": "blocker",
                f"priority_v{VERSION}": 1,
                f"current_evidence_artifact_v{VERSION}": "paper4_v375_claim_permission_register.csv",
                f"current_evidence_count_v{VERSION}": 0,
                f"required_before_submission_v{VERSION}": False,
                f"currently_satisfied_v{VERSION}": False,
                f"missing_item_v{VERSION}": "approved legal fairness attribute review",
                f"next_action_v{VERSION}": "keep legal fairness compliance claims prohibited",
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "fairness proxy diagnostics only",
            },
            {
                f"gap_id_v{VERSION}": "formal_spo_dla_approval_missing",
                f"readiness_domain_v{VERSION}": "formal_methods",
                f"severity_v{VERSION}": "blocker",
                f"priority_v{VERSION}": 2,
                f"current_evidence_artifact_v{VERSION}": "paper4_v375_live_gate_data_contract.csv",
                f"current_evidence_count_v{VERSION}": 2,
                f"required_before_submission_v{VERSION}": False,
                f"currently_satisfied_v{VERSION}": False,
                f"missing_item_v{VERSION}": "approved formal SPO/DLA claim review",
                f"next_action_v{VERSION}": "limit SPO/DLA language to historical audit only",
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "formal claim blocked",
            },
            {
                f"gap_id_v{VERSION}": "full_regression_suite_not_clean",
                f"readiness_domain_v{VERSION}": "validation",
                f"severity_v{VERSION}": "blocker",
                f"priority_v{VERSION}": 1,
                f"current_evidence_artifact_v{VERSION}": "tests/test_docs/test_paper4_living_lab_guardrails.py",
                f"current_evidence_count_v{VERSION}": int(v377_status["guardrail_manifest_rows_v377"]),
                f"required_before_submission_v{VERSION}": True,
                f"currently_satisfied_v{VERSION}": False,
                f"missing_item_v{VERSION}": "full regression/Quarto guardrail signoff",
                f"next_action_v{VERSION}": "separate known old Quarto registration failure from Paper 4 wave tests",
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "targeted guardrails only",
            },
            {
                f"gap_id_v{VERSION}": "reproducibility_bundle_complete",
                f"readiness_domain_v{VERSION}": "reproducibility",
                f"severity_v{VERSION}": "satisfied",
                f"priority_v{VERSION}": 3,
                f"current_evidence_artifact_v{VERSION}": (
                    "paper4_v377_reproducibility_bundle_manifest.csv"
                ),
                f"current_evidence_count_v{VERSION}": int(
                    v377_status["bundle_manifest_rows_v377"]
                ),
                f"required_before_submission_v{VERSION}": True,
                f"currently_satisfied_v{VERSION}": bool(
                    v377_status["all_required_artifacts_exist_v377"]
                ),
                f"missing_item_v{VERSION}": "none for current living-lab bundle",
                f"next_action_v{VERSION}": "preserve hashes as appendix provenance",
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "appendix packaging support only",
            },
            {
                f"gap_id_v{VERSION}": "guardrail_manifest_available",
                f"readiness_domain_v{VERSION}": "validation",
                f"severity_v{VERSION}": "satisfied",
                f"priority_v{VERSION}": 3,
                f"current_evidence_artifact_v{VERSION}": "paper4_v377_guardrail_manifest.csv",
                f"current_evidence_count_v{VERSION}": int(
                    v377_status["guardrail_manifest_rows_v377"]
                ),
                f"required_before_submission_v{VERSION}": True,
                f"currently_satisfied_v{VERSION}": True,
                f"missing_item_v{VERSION}": "none for wave-level guardrail manifest",
                f"next_action_v{VERSION}": "keep adding guardrails for new waves",
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "wave-level validation only",
            },
            {
                f"gap_id_v{VERSION}": "paper_estrella_protection_active",
                f"readiness_domain_v{VERSION}": "governance",
                f"severity_v{VERSION}": "satisfied",
                f"priority_v{VERSION}": 3,
                f"current_evidence_artifact_v{VERSION}": "paper4_v377_bundle_checks.csv",
                f"current_evidence_count_v{VERSION}": 1,
                f"required_before_submission_v{VERSION}": True,
                f"currently_satisfied_v{VERSION}": not FORBIDDEN_FINAL_PROMOTION.exists(),
                f"missing_item_v{VERSION}": "none; final promotion remains absent",
                f"next_action_v{VERSION}": "continue blocking Paper Estrella replacement",
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "Paper Estrella remains protected",
            },
        ]
    )
    domain_summary = _domain_summary(gaps)
    open_gaps = gaps.loc[~gaps[f"currently_satisfied_v{VERSION}"].astype(bool)].copy()
    open_blocker_count = int(open_gaps[f"severity_v{VERSION}"].astype(str).eq("blocker").sum())
    highest_priority_open = int(open_gaps[f"priority_v{VERSION}"].astype(int).eq(1).sum())

    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "open_submission_readiness_gaps",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(open_gaps)),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "submission-ready claim remains blocked",
            },
            {
                f"blocker_id_v{VERSION}": "highest_priority_open_gaps",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": highest_priority_open,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "v379 must create executable closure work order",
            },
            {
                f"blocker_id_v{VERSION}": "live_legal_global_claims_still_blocked",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 0,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "v375 permissions remain unchanged",
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
                "claim_id": "v378_submission_readiness_gap_register_created",
                "allowed": True,
                "artifact": "paper4_v378_submission_readiness_gap_register.csv",
                "boundary": "gap register only",
            },
            {
                "claim_id": "v378_reproducibility_bundle_reused_for_readiness",
                "allowed": True,
                "artifact": "paper4_v377_reproducibility_bundle_manifest.csv",
                "boundary": "provenance input only",
            },
            {
                "claim_id": "v378_submission_ready_paper",
                "allowed": False,
                "artifact": "paper4_v378_claim_blockers.csv",
                "boundary": "open readiness gaps remain",
            },
            {
                "claim_id": "v378_live_legal_or_global_claim_authorized",
                "allowed": False,
                "artifact": "paper4_v378_claim_blockers.csv",
                "boundary": "v375 gates remain blocked",
            },
            {
                "claim_id": "v378_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v378_claim_blockers.csv",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v378_submission_readiness_gap_register.csv", gaps)
    write_csv(TABLE_DIR / "paper4_v378_submission_gap_domain_summary.csv", domain_summary)
    write_csv(TABLE_DIR / "paper4_v378_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v378_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v378_submission_readiness_gap_register",
        "schema_version": "2026-05-17.378",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_reproducibility_bundle_version_v378": PRIOR_REPRODUCIBILITY_BUNDLE_VERSION,
        "prior_v377_bundle_manifest_rows_v378": int(v377_status["bundle_manifest_rows_v377"]),
        "prior_v377_all_bundle_checks_passed_v378": bool(
            v377_status["all_bundle_checks_passed_v377"]
        ),
        "gap_register_rows_v378": int(len(gaps)),
        "open_gap_rows_v378": int(len(open_gaps)),
        "satisfied_readiness_rows_v378": int(
            gaps[f"currently_satisfied_v{VERSION}"].astype(bool).sum()
        ),
        "domain_summary_rows_v378": int(len(domain_summary)),
        "submission_blocker_rows_v378": open_blocker_count,
        "highest_priority_open_gap_rows_v378": highest_priority_open,
        "claim_blocker_rows_v378": int(len(blockers)),
        "claim_matrix_rows_v378": int(len(claim_matrix)),
        "submission_ready_claim_allowed_v378": False,
        "quarto_promotion_allowed_v378": False,
        "bounded_living_lab_language_allowed_v378": True,
        "reproducibility_appendix_language_allowed_v378": True,
        "strict_live_deployment_language_allowed_v378": False,
        "contractual_or_legal_language_allowed_v378": False,
        "global_optimality_language_allowed_v378": False,
        "working_champion_claim_allowed_v378": False,
        "paper1_promotion_allowed_v378": False,
        "paper4_working_champion_changed_v378": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v378": NEXT_ARTIFACT,
        "claim_boundary": (
            "v378 prioritizes submission-readiness gaps; Paper 4 is not submission-ready "
            "and stronger live/legal/global/final claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v378_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v378": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
