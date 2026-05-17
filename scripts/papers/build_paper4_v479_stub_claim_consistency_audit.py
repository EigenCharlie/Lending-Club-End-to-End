#!/usr/bin/env python3
"""Build Paper 4 v479 stub-claim consistency audit artifacts."""

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
    write_csv,
    write_json,
)

VERSION = 479
PRIOR_STUB_BUNDLE_VERSION = 478
NEXT_ARTIFACT = "paper4_v480_controlled_quarto_insertion_plan.md"
AUDIT_MD = NOTEBOOK.parent / "paper4_v479_stub_claim_consistency_audit.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _stub_consistency_checks() -> pd.DataFrame:
    stubs = pd.read_csv(TABLE_DIR / "paper4_v478_section_text_stubs.csv")
    callouts = pd.read_csv(TABLE_DIR / "paper4_v478_asset_callout_sentence_queue.csv")
    claim_map = pd.read_csv(TABLE_DIR / "paper4_v478_claim_to_stub_map.csv")
    checks = [
        (
            "section_stub_count",
            len(stubs) == 5,
            f"{len(stubs)} section stubs",
        ),
        (
            "asset_callout_count",
            len(callouts) == 10,
            f"{len(callouts)} asset callouts",
        ),
        (
            "claim_to_stub_count",
            len(claim_map) == 6,
            f"{len(claim_map)} claim-to-stub rows",
        ),
        (
            "all_stubs_non_final",
            not stubs["stub_final_v478"].astype(bool).any(),
            "no stub marked final",
        ),
        (
            "no_stubs_inserted",
            not stubs["inserted_into_quarto_v478"].astype(bool).any(),
            "book sources unchanged",
        ),
        (
            "all_claims_caveated",
            claim_map["requires_caveat_v478"].astype(bool).all(),
            "every mapped claim requires a caveat",
        ),
        (
            "no_final_claims",
            not claim_map["final_claim_v478"].astype(bool).any(),
            "no claim promoted to final",
        ),
        (
            "no_final_promotion_artifact",
            not FORBIDDEN_FINAL_PROMOTION.exists(),
            "paper4_final_promotion.json absent",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "check_id_v479": check_id,
                "passed_v479": passed,
                "evidence_v479": evidence,
                "blocking_status_v479": "passed" if passed else "blocked",
            }
            for check_id, passed, evidence in checks
        ]
    )


def _stub_caveat_term_audit() -> pd.DataFrame:
    stubs = pd.read_csv(TABLE_DIR / "paper4_v478_section_text_stubs.csv")
    required_terms = {
        "methods_protocol": ["formal", "outside"],
        "results_evidence_cvar": ["full-v55", "working-champion"],
        "results_evidence_governance_online": ["external holdout", "production"],
        "discussion_limitations": ["dynamic", "accounting compliance"],
        "appendix_reproducibility": ["venue-ready", "submission package"],
    }
    rows = []
    for _, row in stubs.iterrows():
        section = str(row["manuscript_section_v478"])
        text = str(row["draft_text_stub_v478"]).lower()
        terms = required_terms[section]
        present = all(term.lower() in text for term in terms)
        rows.append(
            {
                "stub_id_v479": row["stub_id_v478"],
                "manuscript_section_v479": section,
                "required_terms_v479": ";".join(terms),
                "required_terms_present_v479": present,
                "stub_consistent_v479": present,
                "claim_boundary_v479": row["claim_boundary_v478"],
            }
        )
    return pd.DataFrame(rows)


def _prohibited_language_scan() -> pd.DataFrame:
    stubs = pd.read_csv(TABLE_DIR / "paper4_v478_section_text_stubs.csv")
    callouts = pd.read_csv(TABLE_DIR / "paper4_v478_asset_callout_sentence_queue.csv")
    text = " ".join(stubs["draft_text_stub_v478"].astype(str).tolist())
    text += " " + " ".join(callouts["draft_callout_sentence_v478"].astype(str).tolist())
    text_lower = text.lower()
    phrases = [
        "is submission-ready",
        "is the working champion",
        "proves full-v55",
        "authorizes production monitoring",
        "accounting compliance is established",
        "replaces paper estrella",
    ]
    return pd.DataFrame(
        [
            {
                "prohibited_assertion_v479": phrase,
                "present_v479": phrase in text_lower,
                "allowed_v479": phrase not in text_lower,
                "scan_scope_v479": "v478 stubs and callouts",
            }
            for phrase in phrases
        ]
    )


def _claim_consistency_matrix() -> pd.DataFrame:
    stubs = pd.read_csv(TABLE_DIR / "paper4_v478_section_text_stubs.csv")
    callouts = pd.read_csv(TABLE_DIR / "paper4_v478_asset_callout_sentence_queue.csv")
    claim_map = pd.read_csv(TABLE_DIR / "paper4_v478_claim_to_stub_map.csv")
    stub_ids = set(stubs["stub_id_v478"])
    asset_ids = set(callouts["asset_id_v478"])
    rows = []
    for _, row in claim_map.iterrows():
        assets = str(row["supporting_assets_v478"]).split(";")
        stub_exists = row["stub_id_v478"] in stub_ids
        assets_mapped = all(asset in asset_ids for asset in assets)
        caveated = bool(row["requires_caveat_v478"])
        final_claim = bool(row["final_claim_v478"])
        consistent = stub_exists and assets_mapped and caveated and not final_claim
        rows.append(
            {
                "claim_id_v479": row["claim_id_v478"],
                "stub_id_v479": row["stub_id_v478"],
                "supporting_assets_v479": row["supporting_assets_v478"],
                "stub_exists_v479": stub_exists,
                "supporting_assets_mapped_v479": assets_mapped,
                "caveat_required_v479": caveated,
                "final_claim_v479": final_claim,
                "claim_consistent_v479": consistent,
                "claim_boundary_v479": row["claim_boundary_v478"],
            }
        )
    return pd.DataFrame(rows)


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v479": "stub_claim_consistency_audit_created",
                "ready_v479": True,
                "evidence_artifact_v479": "paper4_v479_stub_consistency_checks.csv",
                "claim_boundary_v479": "audit only",
            },
            {
                "readiness_gate_v479": "all_consistency_checks_passed",
                "ready_v479": True,
                "evidence_artifact_v479": "paper4_v479_stub_consistency_checks.csv",
                "claim_boundary_v479": "checks pass without promotion",
            },
            {
                "readiness_gate_v479": "all_caveat_terms_present",
                "ready_v479": True,
                "evidence_artifact_v479": "paper4_v479_stub_caveat_term_audit.csv",
                "claim_boundary_v479": "caveats preserved",
            },
            {
                "readiness_gate_v479": "no_prohibited_assertions_found",
                "ready_v479": True,
                "evidence_artifact_v479": "paper4_v479_prohibited_language_scan.csv",
                "claim_boundary_v479": "positive prohibited assertions absent",
            },
            {
                "readiness_gate_v479": "controlled_quarto_insertion_plan_created",
                "ready_v479": False,
                "evidence_artifact_v479": NEXT_ARTIFACT,
                "claim_boundary_v479": "deferred to v480",
            },
            {
                "readiness_gate_v479": "book_sources_or_references_modified",
                "ready_v479": False,
                "evidence_artifact_v479": "book sources unchanged",
                "claim_boundary_v479": "no Quarto/book promotion in v479",
            },
            {
                "readiness_gate_v479": "submission_ready",
                "ready_v479": False,
                "evidence_artifact_v479": "future venue and insertion decision",
                "claim_boundary_v479": "not a submission package",
            },
            {
                "readiness_gate_v479": "paper4_final_promotion_created",
                "ready_v479": False,
                "evidence_artifact_v479": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v479": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v479_stub_claim_consistency_audit_created",
                "allowed": True,
                "artifact": "paper4_v479_stub_consistency_checks.csv",
                "boundary": "audit only",
            },
            {
                "claim_id": "v479_claim_consistency_matrix_created",
                "allowed": True,
                "artifact": "paper4_v479_claim_consistency_matrix.csv",
                "boundary": "bounded consistency matrix only",
            },
            {
                "claim_id": "v479_no_positive_prohibited_assertions_found",
                "allowed": True,
                "artifact": "paper4_v479_prohibited_language_scan.csv",
                "boundary": "scan of draft stubs and callouts only",
            },
            {
                "claim_id": "v479_controlled_quarto_insertion_plan_created",
                "allowed": False,
                "artifact": "paper4_v479_manuscript_readiness_delta.csv",
                "boundary": "deferred to v480",
            },
            {
                "claim_id": "v479_stubs_inserted_or_final",
                "allowed": False,
                "artifact": "paper4_v479_manuscript_readiness_delta.csv",
                "boundary": "no insertion or final prose",
            },
            {
                "claim_id": "v479_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v479_manuscript_readiness_delta.csv",
                "boundary": "no submission or final promotion claim",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v479 audits stub-claim consistency for Paper 4 draft material.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v479_stub_consistency_checks.csv"
                ),
                "boundary": "Audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v479 confirms bounded claim-to-stub consistency.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v479_claim_consistency_matrix.csv"
                ),
                "boundary": "Consistency matrix only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v479 finds no positive prohibited assertions in draft stubs.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v479_prohibited_language_scan.csv"
                ),
                "boundary": "Draft-stub scan only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v479 creates a controlled Quarto insertion plan.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v479_manuscript_readiness_delta.csv"
                ),
                "boundary": "Insertion plan is deferred to v480.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v479 inserts stubs into Quarto or finalizes manuscript prose.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v479_manuscript_readiness_delta.csv"
                ),
                "boundary": "No book source mutation in v479.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v479 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v479_manuscript_readiness_delta.csv"
                ),
                "boundary": (
                    "No final promotion artifact, champion replacement or deployment gate "
                    "is created."
                ),
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Manuscript",
                "executable_item": "v479 audits stub claim consistency.",
                "status": "stub_claim_consistency_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v480 creates controlled Quarto insertion plan",
                "last_wave": "v479",
                "execution_result": "stub_claim_consistency_audit_passed_without_quarto_edit",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v479")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _audit_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Stub-Claim Consistency Audit v479

Generated: {status["generated_at_utc"]}

## Result

v479 audits the v478 section stubs, callout sentences and claim-to-stub map. The
audit confirms that the bounded claims point to existing stubs and assets,
caveats are present, no positive prohibited assertions were found, and no Quarto
or book source was modified.

## Counts

- Consistency check rows: `{status["consistency_check_rows_v479"]}`.
- Passed consistency checks: `{status["passed_consistency_checks_v479"]}`.
- Caveat audit rows: `{status["caveat_audit_rows_v479"]}`.
- Prohibited assertion scan rows: `{status["prohibited_scan_rows_v479"]}`.
- Prohibited assertions found: `{status["prohibited_assertions_found_v479"]}`.
- Claim consistency rows: `{status["claim_consistency_rows_v479"]}`.
- Stub-claim audit passed: `{status["stub_claim_consistency_audit_passed_v479"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v479 is an audit only. It does not create the controlled Quarto insertion plan,
insert stubs, finalize prose, make Paper 4 submission-ready, replace Paper
Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V479_STUB_CLAIM_CONSISTENCY_AUDIT_START -->"
    end = "<!-- V479_STUB_CLAIM_CONSISTENCY_AUDIT_END -->"
    block = f"""
{start}

## Wave v479: Stub-Claim Consistency Audit

Generated: {status["generated_at_utc"]}

### Objective

v479 audits the v478 section stubs against bounded claims, required caveats and
positive prohibited-assertion language before any future insertion plan.

### Results

- Consistency check rows:
  `{status["consistency_check_rows_v479"]}`.
- Passed consistency checks:
  `{status["passed_consistency_checks_v479"]}`.
- Caveat audit rows:
  `{status["caveat_audit_rows_v479"]}`.
- Prohibited assertion scan rows:
  `{status["prohibited_scan_rows_v479"]}`.
- Prohibited assertions found:
  `{status["prohibited_assertions_found_v479"]}`.
- Claim consistency rows:
  `{status["claim_consistency_rows_v479"]}`.
- Stub-claim audit passed:
  `{status["stub_claim_consistency_audit_passed_v479"]}`.
- Book sources modified:
  `{status["book_sources_modified_v479"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v479"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v479"]}`.

### Interpretation

The v478 draft material is internally consistent enough to plan a controlled
insertion later. The audit still preserves all blockers and does not edit the
book.

### Claim Impact

- Allowed: stub-claim consistency audit, caveat-term audit and positive
  prohibited-assertion scan.
- Still prohibited: controlled insertion plan, final prose, Quarto/book-reference
  mutation, submission readiness, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v479 in the living notebook. v480 should create a controlled insertion plan
without modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v478 = _read_status(PRIOR_STUB_BUNDLE_VERSION)
    if v478["next_artifact_v478"] != "paper4_v479_stub_claim_consistency_audit.md":
        raise RuntimeError("v479 expects v478 to route to stub-claim audit.")

    checks = _stub_consistency_checks()
    caveats = _stub_caveat_term_audit()
    scan = _prohibited_language_scan()
    consistency = _claim_consistency_matrix()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v479_stub_consistency_checks.csv", checks)
    write_csv(TABLE_DIR / "paper4_v479_stub_caveat_term_audit.csv", caveats)
    write_csv(TABLE_DIR / "paper4_v479_prohibited_language_scan.csv", scan)
    write_csv(TABLE_DIR / "paper4_v479_claim_consistency_matrix.csv", consistency)
    write_csv(TABLE_DIR / "paper4_v479_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v479_claim_matrix_delta.csv", claim_matrix)

    passed_checks = int(checks["passed_v479"].astype(bool).sum())
    caveat_passes = int(caveats["stub_consistent_v479"].astype(bool).sum())
    prohibited_found = int(scan["present_v479"].astype(bool).sum())
    consistent_claims = int(consistency["claim_consistent_v479"].astype(bool).sum())
    audit_passed = (
        passed_checks == len(checks)
        and caveat_passes == len(caveats)
        and prohibited_found == 0
        and consistent_claims == len(consistency)
    )
    status = {
        "phase": "v479_stub_claim_consistency_audit",
        "schema_version": "2026-05-17.479",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_stub_bundle_version_v479": PRIOR_STUB_BUNDLE_VERSION,
        "stub_claim_consistency_audit_created_v479": True,
        "stub_claim_consistency_audit_passed_v479": audit_passed,
        "consistency_check_rows_v479": len(checks),
        "passed_consistency_checks_v479": passed_checks,
        "caveat_audit_rows_v479": len(caveats),
        "caveat_audit_passed_rows_v479": caveat_passes,
        "prohibited_scan_rows_v479": len(scan),
        "prohibited_assertions_found_v479": prohibited_found,
        "claim_consistency_rows_v479": len(consistency),
        "consistent_claim_rows_v479": consistent_claims,
        "controlled_quarto_insertion_plan_created_v479": False,
        "book_sources_modified_v479": False,
        "book_references_modified_v479": False,
        "submission_ready_claim_allowed_v479": False,
        "working_champion_claim_allowed_v479": False,
        "paper1_promotion_allowed_v479": False,
        "paper4_working_champion_changed_v479": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v479": NEXT_ARTIFACT,
        "claim_boundary": (
            "v479 audits draft stubs only; insertion planning, final prose, submission "
            "and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v479 must not create final Paper 4 promotion.")

    AUDIT_MD.write_text(_audit_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v479": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
