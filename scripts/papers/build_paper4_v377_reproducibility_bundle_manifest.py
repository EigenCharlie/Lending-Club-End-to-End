#!/usr/bin/env python3
"""Build Paper 4 v377 reproducibility bundle manifest artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
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

VERSION = 377
PRIOR_PUBLICATION_PATCH_VERSION = 376
NEXT_VERSION = 378
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_submission_readiness_gap_register.csv"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _artifact_row(
    *,
    bundle_id: str,
    artifact_role: str,
    source_wave: str,
    relative_path: str,
    claim_use: str,
    claim_boundary: str,
    required: bool = True,
) -> dict[str, Any]:
    path = Path(relative_path)
    exists = path.exists()
    return {
        f"bundle_id_v{VERSION}": bundle_id,
        f"artifact_role_v{VERSION}": artifact_role,
        f"source_wave_v{VERSION}": source_wave,
        f"artifact_path_v{VERSION}": relative_path,
        f"required_v{VERSION}": required,
        f"path_exists_v{VERSION}": exists,
        f"byte_size_v{VERSION}": int(path.stat().st_size) if exists else 0,
        f"sha256_v{VERSION}": _hash_file(path),
        f"claim_use_v{VERSION}": claim_use,
        f"claim_boundary_v{VERSION}": claim_boundary,
    }


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v377 inventories citable Paper 4 living-lab artifacts and hashes.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v377_reproducibility_bundle_manifest.csv"
                ),
                "boundary": "Reproducibility manifest only; no claim permissions change.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v377 provides a reproducibility bundle for a future appendix.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v377_bundle_checks.csv"
                ),
                "boundary": "Appendix packaging support only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v377 makes Paper 4 submission-ready or changes claim permissions.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v377_claim_blockers.csv"
                ),
                "boundary": "Submission readiness remains a future gap-register task.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v377 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v377_claim_blockers.csv"
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
                "lane": "Publishability/Scope",
                "executable_item": (
                    "v377 packages citable artifacts, status files and guardrails into a "
                    "reproducibility manifest for the Paper 4 appendix."
                ),
                "status": "reproducibility_bundle_manifest_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v378 lists remaining submission-readiness gaps without promoting "
                    "Paper 4 or changing claim permissions"
                ),
                "last_wave": "v377",
                "execution_result": "citable_artifacts_hashed_and_packaged_without_promotion",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v377")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V377_REPRODUCIBILITY_BUNDLE_MANIFEST_START -->"
    end = "<!-- V377_REPRODUCIBILITY_BUNDLE_MANIFEST_END -->"
    block = f"""
{start}

## Wave v377: Reproducibility Bundle Manifest

Generated: {status["generated_at_utc"]}

### Objective

v377 turns the Paper 4 living-lab evidence into a reproducibility bundle
manifest: citable artifacts, status files, guardrails, hashes and package checks.

### Results

- Bundle manifest rows:
  `{status["bundle_manifest_rows_v377"]}`.
- Status manifest rows:
  `{status["status_manifest_rows_v377"]}`.
- Guardrail manifest rows:
  `{status["guardrail_manifest_rows_v377"]}`.
- Bundle check rows:
  `{status["bundle_check_rows_v377"]}`.
- Required artifacts missing:
  `{status["missing_required_artifact_rows_v377"]}`.
- Required status files missing:
  `{status["missing_required_status_rows_v377"]}`.
- All required artifacts exist:
  `{status["all_required_artifacts_exist_v377"]}`.
- All bundle checks passed:
  `{status["all_bundle_checks_passed_v377"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v377"]}`.

### Interpretation

Paper 4 now has a citable evidence bundle for future appendix work. This does
not make the paper submission-ready; it makes the evidence traceable enough to
audit what can and cannot be claimed.

### Claim Impact

- Allowed: reproducibility and appendix packaging statements.
- Still prohibited: submission readiness, Quarto promotion, live/legal/global
  claims, champion replacement and final promotion.

### Quarto Promotion Decision

Keep v377 in the living notebook. v378 should enumerate submission-readiness
gaps explicitly.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v376_status = json.loads((STATUS_DIR / "paper4_v376_status.json").read_text(encoding="utf-8"))
    if v376_status["next_artifact_v376"] != "paper4_v377_reproducibility_bundle_manifest.csv":
        raise RuntimeError("v377 expects v376 to route to the reproducibility bundle manifest.")

    artifact_specs = [
        ("v361_fourth_order_screen", "table", "v361", "reports/paper_material/paper4/tables/paper4_v361_v353_fourth_order_or_full_dual_bound.csv", "bounded fourth-order no-entry evidence", "bounded solver evidence only"),
        ("v361_status", "status", "v361", "reports/paper_material/paper4/status/paper4_v361_status.json", "status provenance", "status evidence only"),
        ("v363_dual_bound_gap", "table", "v363", "reports/paper_material/paper4/tables/paper4_v363_v353_full_dual_bound_or_gap_certificate.csv", "full-v55 gap disclosure", "not a global proof"),
        ("v363_status", "status", "v363", "reports/paper_material/paper4/status/paper4_v363_status.json", "status provenance", "status evidence only"),
        ("v368_claim_scope_update", "note", "v368", "reports/paper_material/paper4/notes/paper4_v368_publishable_claim_scope_update.md", "bounded publishable claim scope", "scope update only"),
        ("v368_status", "status", "v368", "reports/paper_material/paper4/status/paper4_v368_status.json", "status provenance", "status evidence only"),
        ("v369_proxy_live_separation", "table", "v369", "reports/paper_material/paper4/tables/paper4_v369_proxy_live_gate_separation.csv", "proxy/live separation", "gate separation only"),
        ("v369_gate_requirements", "table", "v369", "reports/paper_material/paper4/tables/paper4_v369_gate_requirement_matrix.csv", "live gate requirements", "requirements only"),
        ("v369_status", "status", "v369", "reports/paper_material/paper4/status/paper4_v369_status.json", "status provenance", "status evidence only"),
        ("v371_source_diagnostic", "table", "v371", "reports/paper_material/paper4/tables/paper4_v371_source_governance_blocker_diagnostic.csv", "grade=A source blocker", "diagnostic only"),
        ("v371_status", "status", "v371", "reports/paper_material/paper4/status/paper4_v371_status.json", "status provenance", "status evidence only"),
        ("v372_grade_a_prefilter", "table", "v372", "reports/paper_material/paper4/tables/paper4_v372_grade_a_source_relief_prefilter.csv", "grade-A relief prefilter", "diagnostic only"),
        ("v372_status", "status", "v372", "reports/paper_material/paper4/status/paper4_v372_status.json", "status provenance", "status evidence only"),
        ("v373_chunk_stop_rule", "table", "v373", "reports/paper_material/paper4/tables/paper4_v373_full_v55_chunk_002_or_stop_rule.csv", "blind chunking stop rule", "execution decision only"),
        ("v373_sampled_chunks", "table", "v373", "reports/paper_material/paper4/tables/paper4_v373_sampled_chunk_source_screen.csv", "sampled source screens", "sampled diagnostic only"),
        ("v373_status", "status", "v373", "reports/paper_material/paper4/status/paper4_v373_status.json", "status provenance", "status evidence only"),
        ("v374_claim_draft_note", "note", "v374", "reports/paper_material/paper4/notes/paper4_v374_paper4_claim_language_section_draft.md", "paper language draft", "draft only"),
        ("v374_claim_draft_table", "table", "v374", "reports/paper_material/paper4/tables/paper4_v374_claim_language_section_draft.csv", "paper language rows", "draft only"),
        ("v374_citation_map", "table", "v374", "reports/paper_material/paper4/tables/paper4_v374_evidence_citation_map.csv", "citation evidence map", "citation support only"),
        ("v374_status", "status", "v374", "reports/paper_material/paper4/status/paper4_v374_status.json", "status provenance", "status evidence only"),
        ("v375_data_contract", "table", "v375", "reports/paper_material/paper4/tables/paper4_v375_live_gate_data_contract.csv", "live-gate data contract", "requirements only"),
        ("v375_claim_permissions", "table", "v375", "reports/paper_material/paper4/tables/paper4_v375_claim_permission_register.csv", "claim permissions", "permission register only"),
        ("v375_status", "status", "v375", "reports/paper_material/paper4/status/paper4_v375_status.json", "status provenance", "status evidence only"),
        ("v376_publication_patch", "note", "v376", "reports/paper_material/paper4/notes/paper4_v376_publication_integration_patch.md", "publication integration patch", "patch only"),
        ("v376_section_map", "table", "v376", "reports/paper_material/paper4/tables/paper4_v376_section_integration_map.csv", "section integration map", "planning artifact only"),
        ("v376_allowed_sentences", "table", "v376", "reports/paper_material/paper4/tables/paper4_v376_allowed_sentence_bank.csv", "allowed sentence bank", "bounded wording only"),
        ("v376_prohibited_sentences", "table", "v376", "reports/paper_material/paper4/tables/paper4_v376_prohibited_sentence_bank.csv", "prohibited sentence bank", "prohibition register only"),
        ("v376_status", "status", "v376", "reports/paper_material/paper4/status/paper4_v376_status.json", "status provenance", "status evidence only"),
        ("living_lab_notebook", "note", "v1-v377", "reports/paper_material/paper4/notes/paper4_living_lab_notebook.md", "wave history", "living notebook only"),
        ("living_lab_guardrails", "test", "v1-v377", "tests/test_docs/test_paper4_living_lab_guardrails.py", "guardrail tests", "test evidence only"),
    ]
    manifest = pd.DataFrame(
        [
            _artifact_row(
                bundle_id=bundle_id,
                artifact_role=role,
                source_wave=source_wave,
                relative_path=path,
                claim_use=claim_use,
                claim_boundary=claim_boundary,
            )
            for bundle_id, role, source_wave, path, claim_use, claim_boundary in artifact_specs
        ]
    )
    status_manifest = manifest.loc[manifest[f"artifact_role_v{VERSION}"].eq("status")].copy()
    guardrails = pd.DataFrame(
        [
            {
                f"guardrail_id_v{VERSION}": "v371_source_governance",
                f"test_name_v{VERSION}": (
                    "test_paper4_v371_source_governance_blocker_diagnostic_identifies_grade_a"
                ),
                f"source_wave_v{VERSION}": "v371",
                f"claim_boundary_v{VERSION}": "diagnostic only",
            },
            {
                f"guardrail_id_v{VERSION}": "v372_grade_a_relief",
                f"test_name_v{VERSION}": (
                    "test_paper4_v372_grade_a_source_relief_prefilter_finds_no_return_candidate"
                ),
                f"source_wave_v{VERSION}": "v372",
                f"claim_boundary_v{VERSION}": "prefilter only",
            },
            {
                f"guardrail_id_v{VERSION}": "v373_chunk_stop_rule",
                f"test_name_v{VERSION}": "test_paper4_v373_chunk_002_or_stop_rule_stops_blind_chunking",
                f"source_wave_v{VERSION}": "v373",
                f"claim_boundary_v{VERSION}": "stop rule only",
            },
            {
                f"guardrail_id_v{VERSION}": "v374_claim_language",
                f"test_name_v{VERSION}": (
                    "test_paper4_v374_claim_language_draft_blocks_prohibited_phrases"
                ),
                f"source_wave_v{VERSION}": "v374",
                f"claim_boundary_v{VERSION}": "draft language only",
            },
            {
                f"guardrail_id_v{VERSION}": "v375_live_gate_contract",
                f"test_name_v{VERSION}": (
                    "test_paper4_v375_live_gate_data_contract_blocks_live_legal_global_claims"
                ),
                f"source_wave_v{VERSION}": "v375",
                f"claim_boundary_v{VERSION}": "data contract only",
            },
            {
                f"guardrail_id_v{VERSION}": "v376_publication_patch",
                f"test_name_v{VERSION}": (
                    "test_paper4_v376_publication_integration_patch_stays_living_notebook_only"
                ),
                f"source_wave_v{VERSION}": "v376",
                f"claim_boundary_v{VERSION}": "publication patch only",
            },
            {
                f"guardrail_id_v{VERSION}": "v377_reproducibility_bundle",
                f"test_name_v{VERSION}": (
                    "test_paper4_v377_reproducibility_bundle_manifest_is_complete"
                ),
                f"source_wave_v{VERSION}": "v377",
                f"claim_boundary_v{VERSION}": "manifest completeness only",
            },
        ]
    )
    required_manifest = manifest.loc[manifest[f"required_v{VERSION}"].astype(bool)]
    missing_required = int((~required_manifest[f"path_exists_v{VERSION}"].astype(bool)).sum())
    missing_status = int((~status_manifest[f"path_exists_v{VERSION}"].astype(bool)).sum())
    bundle_checks = pd.DataFrame(
        [
            {
                f"check_id_v{VERSION}": "required_artifacts_exist",
                f"passed_v{VERSION}": missing_required == 0,
                f"evidence_count_v{VERSION}": int(len(required_manifest)),
                f"claim_boundary_v{VERSION}": "artifact existence only",
            },
            {
                f"check_id_v{VERSION}": "status_files_exist",
                f"passed_v{VERSION}": missing_status == 0,
                f"evidence_count_v{VERSION}": int(len(status_manifest)),
                f"claim_boundary_v{VERSION}": "status provenance only",
            },
            {
                f"check_id_v{VERSION}": "required_hashes_created",
                f"passed_v{VERSION}": required_manifest[f"sha256_v{VERSION}"].astype(str).str.len().eq(64).all(),
                f"evidence_count_v{VERSION}": int(len(required_manifest)),
                f"claim_boundary_v{VERSION}": "hash manifest only",
            },
            {
                f"check_id_v{VERSION}": "final_promotion_absent",
                f"passed_v{VERSION}": not FORBIDDEN_FINAL_PROMOTION.exists(),
                f"evidence_count_v{VERSION}": 1,
                f"claim_boundary_v{VERSION}": "final promotion remains forbidden",
            },
            {
                f"check_id_v{VERSION}": "publication_patch_is_living_notebook_only",
                f"passed_v{VERSION}": bool(v376_status["quarto_pages_modified_v376"]) is False,
                f"evidence_count_v{VERSION}": int(v376_status["section_integration_rows_v376"]),
                f"claim_boundary_v{VERSION}": "no Quarto promotion",
            },
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "manifest_is_not_submission_package",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(manifest)),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "submission readiness remains future work",
            },
            {
                f"blocker_id_v{VERSION}": "live_legal_global_claims_still_blocked",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 0,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "v375 permissions remain unchanged",
            },
            {
                f"blocker_id_v{VERSION}": "quarto_pages_not_promoted",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 0,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "living notebook only",
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
                "claim_id": "v377_reproducibility_manifest_created",
                "allowed": True,
                "artifact": "paper4_v377_reproducibility_bundle_manifest.csv",
                "boundary": "manifest only",
            },
            {
                "claim_id": "v377_status_and_guardrail_manifest_created",
                "allowed": True,
                "artifact": "paper4_v377_guardrail_manifest.csv",
                "boundary": "test provenance only",
            },
            {
                "claim_id": "v377_submission_ready_package",
                "allowed": False,
                "artifact": "paper4_v377_claim_blockers.csv",
                "boundary": "submission readiness not claimed",
            },
            {
                "claim_id": "v377_live_legal_or_global_claim_authorized",
                "allowed": False,
                "artifact": "paper4_v377_claim_blockers.csv",
                "boundary": "v375 gates remain blocked",
            },
            {
                "claim_id": "v377_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v377_claim_blockers.csv",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v377_reproducibility_bundle_manifest.csv", manifest)
    write_csv(TABLE_DIR / "paper4_v377_status_manifest.csv", status_manifest)
    write_csv(TABLE_DIR / "paper4_v377_guardrail_manifest.csv", guardrails)
    write_csv(TABLE_DIR / "paper4_v377_bundle_checks.csv", bundle_checks)
    write_csv(TABLE_DIR / "paper4_v377_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v377_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v377_reproducibility_bundle_manifest",
        "schema_version": "2026-05-17.377",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_publication_patch_version_v377": PRIOR_PUBLICATION_PATCH_VERSION,
        "prior_v376_section_integration_rows_v377": int(
            v376_status["section_integration_rows_v376"]
        ),
        "bundle_manifest_rows_v377": int(len(manifest)),
        "status_manifest_rows_v377": int(len(status_manifest)),
        "guardrail_manifest_rows_v377": int(len(guardrails)),
        "bundle_check_rows_v377": int(len(bundle_checks)),
        "claim_blocker_rows_v377": int(len(blockers)),
        "claim_matrix_rows_v377": int(len(claim_matrix)),
        "missing_required_artifact_rows_v377": missing_required,
        "missing_required_status_rows_v377": missing_status,
        "all_required_artifacts_exist_v377": missing_required == 0,
        "all_required_status_files_exist_v377": missing_status == 0,
        "all_bundle_checks_passed_v377": bool(bundle_checks[f"passed_v{VERSION}"].astype(bool).all()),
        "quarto_pages_modified_v377": False,
        "bounded_living_lab_language_allowed_v377": True,
        "reproducibility_appendix_language_allowed_v377": True,
        "submission_ready_claim_allowed_v377": False,
        "strict_live_deployment_language_allowed_v377": False,
        "contractual_or_legal_language_allowed_v377": False,
        "global_optimality_language_allowed_v377": False,
        "working_champion_claim_allowed_v377": False,
        "paper1_promotion_allowed_v377": False,
        "paper4_working_champion_changed_v377": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v377": NEXT_ARTIFACT,
        "claim_boundary": (
            "v377 packages reproducibility evidence; submission readiness and stronger "
            "live/legal/global/final claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v377_status.json", status)
    _update_notebook(status)
    final_manifest = pd.DataFrame(
        [
            _artifact_row(
                bundle_id=bundle_id,
                artifact_role=role,
                source_wave=source_wave,
                relative_path=path,
                claim_use=claim_use,
                claim_boundary=claim_boundary,
            )
            for bundle_id, role, source_wave, path, claim_use, claim_boundary in artifact_specs
        ]
    )
    final_status_manifest = final_manifest.loc[
        final_manifest[f"artifact_role_v{VERSION}"].eq("status")
    ].copy()
    write_csv(TABLE_DIR / "paper4_v377_reproducibility_bundle_manifest.csv", final_manifest)
    write_csv(TABLE_DIR / "paper4_v377_status_manifest.csv", final_status_manifest)
    print(json.dumps({"v377": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
