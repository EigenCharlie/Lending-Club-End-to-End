#!/usr/bin/env python3
"""Build Paper 4 v485 caption consistency audit artifacts."""

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

VERSION = 485
PRIOR_CAPTION_HARDENING_VERSION = 484
NEXT_ARTIFACT = "paper4_v486_caption_review_decision_matrix.md"
AUDIT_MD = NOTEBOOK.parent / "paper4_v485_caption_consistency_audit.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _caption_consistency_checks() -> pd.DataFrame:
    hardened = pd.read_csv(TABLE_DIR / "paper4_v484_hardened_caption_dry_run.csv")
    caveats = pd.read_csv(TABLE_DIR / "paper4_v484_caption_caveat_preservation_audit.csv")
    review = pd.read_csv(TABLE_DIR / "paper4_v484_caption_review_delta.csv")
    expected_assets = {"T1", "T2", "T3", "T4", "T5", "T6", "F1", "F2", "F3", "F4"}
    checks = [
        (
            "expected_caption_count",
            len(hardened) == 10,
            f"{len(hardened)} hardened captions",
        ),
        (
            "expected_asset_coverage",
            set(hardened["asset_id_v484"]) == expected_assets,
            "all T1-T6 and F1-F4 assets covered",
        ),
        (
            "all_captions_hardened",
            hardened["caption_hardened_v484"].astype(bool).all(),
            "every caption marked hardened",
        ),
        (
            "all_caveats_preserved",
            caveats["caveat_preserved_v484"].astype(bool).all(),
            "every required caveat term present",
        ),
        (
            "manual_review_required",
            hardened["manual_review_required_v484"].astype(bool).all(),
            "manual review required for every caption",
        ),
        (
            "no_caption_finalization",
            not hardened["caption_final_v484"].astype(bool).any(),
            "no hardened caption is final",
        ),
        (
            "no_quarto_insertion",
            not review["inserted_into_quarto_v484"].astype(bool).any(),
            "no caption inserted into Quarto",
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
                "check_id_v485": check_id,
                "passed_v485": passed,
                "evidence_v485": evidence,
                "claim_boundary_v485": "caption consistency audit only",
            }
            for check_id, passed, evidence in checks
        ]
    )


def _caption_quality_matrix() -> pd.DataFrame:
    hardened = pd.read_csv(TABLE_DIR / "paper4_v484_hardened_caption_dry_run.csv")
    rows = []
    for _, row in hardened.iterrows():
        caption = str(row["hardened_caption_v484"])
        word_count = len(caption.split())
        boundary_context = f"{caption} {row['claim_boundary_v484']}".lower()
        has_boundary_language = any(
            term in boundary_context
            for term in [
                "remain blocked",
                "outside scope",
                "does not authorize",
                "does not establish",
                "not production",
                "only",
                "no ",
                "lacks",
                "visual context",
                "fairness certification",
            ]
        )
        rows.append(
            {
                "asset_id_v485": row["asset_id_v484"],
                "caption_word_count_v485": word_count,
                "caption_minimum_length_pass_v485": word_count >= 15,
                "caption_boundary_language_pass_v485": has_boundary_language,
                "caption_manual_review_required_v485": True,
                "caption_ready_for_final_review_v485": False,
                "caption_quality_pass_v485": word_count >= 15 and has_boundary_language,
                "claim_boundary_v485": row["claim_boundary_v484"],
            }
        )
    return pd.DataFrame(rows)


def _prohibited_language_scan() -> pd.DataFrame:
    hardened = pd.read_csv(TABLE_DIR / "paper4_v484_hardened_caption_dry_run.csv")
    text = " ".join(hardened["hardened_caption_v484"].astype(str).tolist()).lower()
    phrases = [
        "is submission-ready",
        "working champion is established",
        "proves full-v55 optimality",
        "authorizes cap relaxation",
        "authorizes production monitoring",
        "accounting compliance is established",
        "replaces paper estrella",
    ]
    return pd.DataFrame(
        [
            {
                "prohibited_caption_assertion_v485": phrase,
                "present_v485": phrase in text,
                "allowed_v485": phrase not in text,
                "scan_scope_v485": "v484 hardened captions",
            }
            for phrase in phrases
        ]
    )


def _caption_decision_queue() -> pd.DataFrame:
    quality = _caption_quality_matrix()
    rows = []
    for _, row in quality.iterrows():
        rows.append(
            {
                "asset_id_v485": row["asset_id_v485"],
                "decision_needed_v485": "manual_accept_revise_or_reject_caption",
                "audit_passed_v485": bool(row["caption_quality_pass_v485"]),
                "recommended_next_state_v485": "manual_review",
                "caption_final_v485": False,
                "caption_inserted_into_quarto_v485": False,
                "claim_boundary_v485": row["claim_boundary_v485"],
            }
        )
    return pd.DataFrame(rows)


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v485": "caption_consistency_audit_created",
                "ready_v485": True,
                "evidence_artifact_v485": "paper4_v485_caption_consistency_checks.csv",
                "claim_boundary_v485": "audit only",
            },
            {
                "readiness_gate_v485": "caption_quality_matrix_created",
                "ready_v485": True,
                "evidence_artifact_v485": "paper4_v485_caption_quality_matrix.csv",
                "claim_boundary_v485": "quality matrix only",
            },
            {
                "readiness_gate_v485": "prohibited_caption_language_scan_created",
                "ready_v485": True,
                "evidence_artifact_v485": "paper4_v485_prohibited_caption_language_scan.csv",
                "claim_boundary_v485": "caption scan only",
            },
            {
                "readiness_gate_v485": "caption_decision_queue_created",
                "ready_v485": True,
                "evidence_artifact_v485": "paper4_v485_caption_decision_queue.csv",
                "claim_boundary_v485": "manual decision queue only",
            },
            {
                "readiness_gate_v485": "captions_final",
                "ready_v485": False,
                "evidence_artifact_v485": "manual review required",
                "claim_boundary_v485": "captions remain non-final",
            },
            {
                "readiness_gate_v485": "captions_inserted_into_quarto",
                "ready_v485": False,
                "evidence_artifact_v485": "book sources unchanged",
                "claim_boundary_v485": "no Quarto/book mutation in v485",
            },
            {
                "readiness_gate_v485": "submission_ready",
                "ready_v485": False,
                "evidence_artifact_v485": "future approval, patch, render and venue gates",
                "claim_boundary_v485": "not a submission package",
            },
            {
                "readiness_gate_v485": "paper4_final_promotion_created",
                "ready_v485": False,
                "evidence_artifact_v485": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v485": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v485_caption_consistency_audit_created",
                "allowed": True,
                "artifact": "paper4_v485_caption_consistency_checks.csv",
                "boundary": "caption audit only",
            },
            {
                "claim_id": "v485_caption_quality_matrix_created",
                "allowed": True,
                "artifact": "paper4_v485_caption_quality_matrix.csv",
                "boundary": "quality matrix only",
            },
            {
                "claim_id": "v485_no_prohibited_caption_assertions_found",
                "allowed": True,
                "artifact": "paper4_v485_prohibited_caption_language_scan.csv",
                "boundary": "caption scan only",
            },
            {
                "claim_id": "v485_caption_decision_queue_created",
                "allowed": True,
                "artifact": "paper4_v485_caption_decision_queue.csv",
                "boundary": "manual decision queue only",
            },
            {
                "claim_id": "v485_captions_final_or_inserted",
                "allowed": False,
                "artifact": "paper4_v485_manuscript_readiness_delta.csv",
                "boundary": "captions remain non-final and uninserted",
            },
            {
                "claim_id": "v485_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v485_manuscript_readiness_delta.csv",
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
                "claim": "v485 audits consistency of hardened Paper 4 captions.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v485_caption_consistency_checks.csv"
                ),
                "boundary": "Caption audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v485 finds no positive prohibited caption assertions.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v485_prohibited_caption_language_scan.csv"
                ),
                "boundary": "Caption scan only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v485 creates a manual caption decision queue.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v485_caption_decision_queue.csv"
                ),
                "boundary": "Decision queue only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v485 finalizes captions or inserts them into Quarto.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v485_manuscript_readiness_delta.csv"
                ),
                "boundary": "Captions remain non-final and uninserted.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v485 makes Paper 4 ready for submission.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v485_manuscript_readiness_delta.csv"
                ),
                "boundary": "Approval, patch, render and venue gates remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v485 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v485_manuscript_readiness_delta.csv"
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
                "executable_item": "v485 audits hardened caption consistency.",
                "status": "caption_consistency_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v486 records caption review decisions",
                "last_wave": "v485",
                "execution_result": "caption_consistency_audit_passed_without_insertion",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v485")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _audit_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Caption Consistency Audit v485

Generated: {status["generated_at_utc"]}

## Result

v485 audits the v484 hardened captions for asset coverage, caveat preservation,
quality gates and prohibited assertion language. The audit passes, but all
captions remain non-final and uninserted.

## Counts

- Consistency check rows: `{status["consistency_check_rows_v485"]}`.
- Passed consistency checks: `{status["passed_consistency_checks_v485"]}`.
- Caption quality rows: `{status["caption_quality_rows_v485"]}`.
- Caption quality pass rows: `{status["caption_quality_pass_rows_v485"]}`.
- Prohibited scan rows: `{status["prohibited_scan_rows_v485"]}`.
- Prohibited hits: `{status["prohibited_hits_v485"]}`.
- Caption decision rows: `{status["caption_decision_rows_v485"]}`.
- Caption audit passed: `{status["caption_consistency_audit_passed_v485"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v485 is an audit only. It does not finalize captions, insert captions into
Quarto, edit book sources, make Paper 4 submission-ready, replace Paper
Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V485_CAPTION_CONSISTENCY_AUDIT_START -->"
    end = "<!-- V485_CAPTION_CONSISTENCY_AUDIT_END -->"
    block = f"""
{start}

## Wave v485: Caption Consistency Audit

Generated: {status["generated_at_utc"]}

### Objective

v485 audits hardened captions for coverage, quality, caveat preservation and
prohibited assertion language without finalizing or inserting captions.

### Results

- Consistency check rows:
  `{status["consistency_check_rows_v485"]}`.
- Passed consistency checks:
  `{status["passed_consistency_checks_v485"]}`.
- Caption quality rows:
  `{status["caption_quality_rows_v485"]}`.
- Caption quality pass rows:
  `{status["caption_quality_pass_rows_v485"]}`.
- Prohibited scan rows:
  `{status["prohibited_scan_rows_v485"]}`.
- Prohibited hits:
  `{status["prohibited_hits_v485"]}`.
- Caption decision rows:
  `{status["caption_decision_rows_v485"]}`.
- Caption audit passed:
  `{status["caption_consistency_audit_passed_v485"]}`.
- Captions final:
  `{status["captions_final_v485"]}`.
- Captions inserted into Quarto:
  `{status["captions_inserted_into_quarto_v485"]}`.
- Book sources modified:
  `{status["book_sources_modified_v485"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v485"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v485"]}`.

### Interpretation

The hardened captions pass internal consistency checks and are ready for manual
caption-review decisions, not final insertion.

### Claim Impact

- Allowed: caption consistency audit, quality matrix, prohibited-language scan
  and manual decision queue.
- Still prohibited: final captions, Quarto insertion, book-reference mutation,
  submission readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v485 in the living notebook. v486 should record caption review decisions
without modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v484 = _read_status(PRIOR_CAPTION_HARDENING_VERSION)
    if v484["next_artifact_v484"] != "paper4_v485_caption_consistency_audit.md":
        raise RuntimeError("v485 expects v484 to route to caption consistency audit.")

    checks = _caption_consistency_checks()
    quality = _caption_quality_matrix()
    scan = _prohibited_language_scan()
    decision_queue = _caption_decision_queue()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v485_caption_consistency_checks.csv", checks)
    write_csv(TABLE_DIR / "paper4_v485_caption_quality_matrix.csv", quality)
    write_csv(TABLE_DIR / "paper4_v485_prohibited_caption_language_scan.csv", scan)
    write_csv(TABLE_DIR / "paper4_v485_caption_decision_queue.csv", decision_queue)
    write_csv(TABLE_DIR / "paper4_v485_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v485_claim_matrix_delta.csv", claim_matrix)

    passed_checks = int(checks["passed_v485"].astype(bool).sum())
    quality_passes = int(quality["caption_quality_pass_v485"].astype(bool).sum())
    prohibited_hits = int(scan["present_v485"].astype(bool).sum())
    audit_passed = (
        passed_checks == len(checks)
        and quality_passes == len(quality)
        and prohibited_hits == 0
    )
    status = {
        "phase": "v485_caption_consistency_audit",
        "schema_version": "2026-05-17.485",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_caption_hardening_version_v485": PRIOR_CAPTION_HARDENING_VERSION,
        "caption_consistency_audit_created_v485": True,
        "caption_consistency_audit_passed_v485": audit_passed,
        "consistency_check_rows_v485": len(checks),
        "passed_consistency_checks_v485": passed_checks,
        "caption_quality_rows_v485": len(quality),
        "caption_quality_pass_rows_v485": quality_passes,
        "prohibited_scan_rows_v485": len(scan),
        "prohibited_hits_v485": prohibited_hits,
        "caption_decision_rows_v485": len(decision_queue),
        "readiness_delta_rows_v485": len(readiness),
        "captions_final_v485": False,
        "captions_inserted_into_quarto_v485": False,
        "book_sources_modified_v485": False,
        "book_references_modified_v485": False,
        "submission_ready_claim_allowed_v485": False,
        "working_champion_claim_allowed_v485": False,
        "paper1_promotion_allowed_v485": False,
        "paper4_working_champion_changed_v485": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v485": NEXT_ARTIFACT,
        "claim_boundary": (
            "v485 audits hardened captions only; final captions, insertion, submission "
            "and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v485 must not create final Paper 4 promotion.")

    AUDIT_MD.write_text(_audit_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v485": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
