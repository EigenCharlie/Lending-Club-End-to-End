#!/usr/bin/env python3
"""Build Paper 4 v478 section text stub bundle artifacts."""

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

VERSION = 478
PRIOR_VISUAL_DELTA_VERSION = 477
NEXT_ARTIFACT = "paper4_v479_stub_claim_consistency_audit.md"
STUB_MD = NOTEBOOK.parent / "paper4_v478_section_text_stub_bundle.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _word_count(text: str) -> int:
    return len(text.split())


def _section_text_stubs() -> pd.DataFrame:
    sections = pd.read_csv(TABLE_DIR / "paper4_v477_visual_section_delta.csv")
    section_to_stub = {
        "methods_protocol": (
            "The methods section should first establish the claim-boundary protocol "
            "with the SPO-DLA boundary matrix and regret-auditability visual context. "
            "The text may describe bounded historical audit evidence, while stating "
            "that formal SPO+/DLA and CRC guarantees remain outside the present claim."
        ),
        "results_evidence_cvar": (
            "The results section should report the local return/CVaR frontier as a "
            "bounded v338-v347-v353 chain. The table carries the current evidence and "
            "the figure provides context; the paragraph must avoid full-v55 proof, "
            "global optimality, working-champion and final-promotion language."
        ),
        "results_evidence_governance_online": (
            "The source-governance and online-monitoring paragraph should present "
            "grade-A source pressure and internal online proxy gates as diagnostics. "
            "The text should preserve that no cap relaxation, external holdout, live "
            "monitoring approval or production control is established."
        ),
        "discussion_limitations": (
            "The limitations section should make the dynamic replay and IFRS9 gaps "
            "visible as remaining blockers. The text may cite the current gap tables, "
            "but it must keep dynamic deployment, contractual cashflow and accounting "
            "compliance claims blocked."
        ),
        "appendix_reproducibility": (
            "The appendix text should index the visual-selection, caption, blocker "
            "caveat and stub artifacts as reproducibility provenance. The appendix "
            "does not become a venue-ready supplement or final submission package."
        ),
    }
    rows = []
    for _, row in sections.iterrows():
        section = str(row["manuscript_section_v477"])
        stub_text = section_to_stub[section]
        rows.append(
            {
                "stub_id_v478": f"stub_{section}",
                "manuscript_section_v478": section,
                "source_asset_bundle_v478": row["asset_bundle_v477"],
                "draft_text_stub_v478": stub_text,
                "word_count_v478": _word_count(stub_text),
                "main_text_claim_allowed_v478": row["main_text_claim_allowed_v477"],
                "appendix_only_v478": row["appendix_only_v477"],
                "required_caveat_v478": row["required_caveat_v477"],
                "stub_final_v478": False,
                "inserted_into_quarto_v478": False,
                "claim_boundary_v478": row["claim_boundary_v477"],
            }
        )
    return pd.DataFrame(rows)


def _asset_callout_queue() -> pd.DataFrame:
    asset_map = pd.read_csv(TABLE_DIR / "paper4_v477_visual_asset_manuscript_map.csv")
    callouts = {
        "T5": "Call out T5 as the boundary matrix that limits SPO-DLA language.",
        "F4": "Call out F4 as historical regret-auditability context only.",
        "T1": "Call out T1 as the current local return/CVaR evidence table.",
        "F1": "Call out F1 as legacy frontier context paired with T1.",
        "T2": "Call out T2 as source-governance blocker evidence.",
        "F2": "Call out F2 as source-governance visual context only.",
        "T4": "Call out T4 as internal online monitoring proxy evidence.",
        "F3": "Call out F3 as internal online method-search context only.",
        "T3": "Call out T3 as the v353 dynamic replay gap table.",
        "T6": "Call out T6 as the contractual IFRS9 requirement-gap audit.",
    }
    rows = []
    for _, row in asset_map.sort_values("insertion_order_v477").iterrows():
        asset_id = str(row["asset_id_v477"])
        rows.append(
            {
                "asset_id_v478": asset_id,
                "asset_type_v478": row["asset_type_v477"],
                "manuscript_section_v478": row["manuscript_section_v477"],
                "callout_order_v478": int(row["insertion_order_v477"]),
                "draft_callout_sentence_v478": callouts[asset_id],
                "callout_sentence_final_v478": False,
                "inserted_into_quarto_v478": False,
                "must_preserve_caveat_v478": True,
                "claim_boundary_v478": row["claim_boundary_v477"],
            }
        )
    return pd.DataFrame(rows)


def _claim_to_stub_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id_v478": "v467_v353_local_return_cvar_frontier",
                "stub_id_v478": "stub_results_evidence_cvar",
                "supporting_assets_v478": "T1;F1",
                "allowed_in_stub_v478": True,
                "requires_caveat_v478": True,
                "final_claim_v478": False,
                "claim_boundary_v478": "local frontier only",
            },
            {
                "claim_id_v478": "v468_grade_a_primary_blocker_documented",
                "stub_id_v478": "stub_results_evidence_governance_online",
                "supporting_assets_v478": "T2;F2",
                "allowed_in_stub_v478": True,
                "requires_caveat_v478": True,
                "final_claim_v478": False,
                "claim_boundary_v478": "source diagnostic only",
            },
            {
                "claim_id_v478": "v470_online_monitoring_proxy_created",
                "stub_id_v478": "stub_results_evidence_governance_online",
                "supporting_assets_v478": "T4;F3",
                "allowed_in_stub_v478": True,
                "requires_caveat_v478": True,
                "final_claim_v478": False,
                "claim_boundary_v478": "internal proxy only",
            },
            {
                "claim_id_v478": "v471_bounded_historical_spo_dla_language",
                "stub_id_v478": "stub_methods_protocol",
                "supporting_assets_v478": "T5;F4",
                "allowed_in_stub_v478": True,
                "requires_caveat_v478": True,
                "final_claim_v478": False,
                "claim_boundary_v478": "bounded historical audit only",
            },
            {
                "claim_id_v478": "v469_v353_dynamic_gap_documented",
                "stub_id_v478": "stub_discussion_limitations",
                "supporting_assets_v478": "T3",
                "allowed_in_stub_v478": True,
                "requires_caveat_v478": True,
                "final_claim_v478": False,
                "claim_boundary_v478": "gap documentation only",
            },
            {
                "claim_id_v478": "v472_contractual_requirement_gap_documented",
                "stub_id_v478": "stub_discussion_limitations",
                "supporting_assets_v478": "T6",
                "allowed_in_stub_v478": True,
                "requires_caveat_v478": True,
                "final_claim_v478": False,
                "claim_boundary_v478": "requirement audit only",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v478": "section_text_stubs_created",
                "ready_v478": True,
                "evidence_artifact_v478": "paper4_v478_section_text_stubs.csv",
                "claim_boundary_v478": "draft text stubs only",
            },
            {
                "readiness_gate_v478": "asset_callout_sentences_created",
                "ready_v478": True,
                "evidence_artifact_v478": "paper4_v478_asset_callout_sentence_queue.csv",
                "claim_boundary_v478": "draft callouts only",
            },
            {
                "readiness_gate_v478": "claim_to_stub_map_created",
                "ready_v478": True,
                "evidence_artifact_v478": "paper4_v478_claim_to_stub_map.csv",
                "claim_boundary_v478": "bounded claim map only",
            },
            {
                "readiness_gate_v478": "stub_caveats_preserved",
                "ready_v478": True,
                "evidence_artifact_v478": "paper4_v478_section_text_stubs.csv",
                "claim_boundary_v478": "caveats remain mandatory",
            },
            {
                "readiness_gate_v478": "stubs_final",
                "ready_v478": False,
                "evidence_artifact_v478": "future manuscript editing",
                "claim_boundary_v478": "stubs are not final prose",
            },
            {
                "readiness_gate_v478": "stubs_inserted_into_quarto",
                "ready_v478": False,
                "evidence_artifact_v478": "book sources unchanged",
                "claim_boundary_v478": "no Quarto/book promotion in v478",
            },
            {
                "readiness_gate_v478": "submission_ready",
                "ready_v478": False,
                "evidence_artifact_v478": "future venue and manuscript edit",
                "claim_boundary_v478": "not a submission package",
            },
            {
                "readiness_gate_v478": "paper4_final_promotion_created",
                "ready_v478": False,
                "evidence_artifact_v478": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v478": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v478_section_text_stubs_created",
                "allowed": True,
                "artifact": "paper4_v478_section_text_stubs.csv",
                "boundary": "draft section stubs only",
            },
            {
                "claim_id": "v478_asset_callout_sentences_created",
                "allowed": True,
                "artifact": "paper4_v478_asset_callout_sentence_queue.csv",
                "boundary": "draft callouts only",
            },
            {
                "claim_id": "v478_claim_to_stub_map_created",
                "allowed": True,
                "artifact": "paper4_v478_claim_to_stub_map.csv",
                "boundary": "bounded claim map only",
            },
            {
                "claim_id": "v478_stubs_inserted_or_final",
                "allowed": False,
                "artifact": "paper4_v478_manuscript_readiness_delta.csv",
                "boundary": "no insertion or final prose",
            },
            {
                "claim_id": "v478_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v478_manuscript_readiness_delta.csv",
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
                "claim": "v478 drafts section text stubs from the visual manuscript delta.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v478_section_text_stubs.csv"
                ),
                "boundary": "Draft text stubs only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v478 drafts asset callout sentences for future editing.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v478_asset_callout_sentence_queue.csv"
                ),
                "boundary": "Draft callouts only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v478 maps bounded claims to draft section stubs.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v478_claim_to_stub_map.csv"
                ),
                "boundary": "Bounded claim-to-stub map only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v478 inserts stubs into Quarto or finalizes manuscript prose.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v478_manuscript_readiness_delta.csv"
                ),
                "boundary": "No book source mutation in v478.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v478 makes Paper 4 submission-ready.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v478_manuscript_readiness_delta.csv"
                ),
                "boundary": "Venue, prose finalization and insertion gates remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v478 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v478_manuscript_readiness_delta.csv"
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
                "executable_item": "v478 drafts section text stubs.",
                "status": "section_text_stub_bundle_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v479 audits stub claim consistency",
                "last_wave": "v478",
                "execution_result": "section_text_stubs_created_without_quarto_edit",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v478")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _stub_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Section Text Stub Bundle v478

Generated: {status["generated_at_utc"]}

## Result

v478 drafts section-level text stubs and asset callout sentences from the v477
visual manuscript delta. It maps bounded claims to draft stubs and preserves the
required caveats. It does not insert text into Quarto, finalize manuscript prose,
make Paper 4 submission-ready, or promote Paper 4.

## Counts

- Section text stub rows: `{status["section_text_stub_rows_v478"]}`.
- Asset callout rows: `{status["asset_callout_rows_v478"]}`.
- Claim-to-stub rows: `{status["claim_to_stub_rows_v478"]}`.
- Stubs final: `{status["stubs_final_v478"]}`.
- Stubs inserted into Quarto: `{status["stubs_inserted_into_quarto_v478"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v478 is a draft text-stub bundle only. Quarto insertion, final prose, venue
formatting, external validation, submission readiness, Paper Estrella
replacement and final Paper 4 promotion remain blocked.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V478_SECTION_TEXT_STUB_BUNDLE_START -->"
    end = "<!-- V478_SECTION_TEXT_STUB_BUNDLE_END -->"
    block = f"""
{start}

## Wave v478: Section Text Stub Bundle

Generated: {status["generated_at_utc"]}

### Objective

v478 drafts section text stubs and asset callout sentences from the v477 visual
manuscript delta without editing Quarto.

### Results

- Section text stub rows:
  `{status["section_text_stub_rows_v478"]}`.
- Asset callout rows:
  `{status["asset_callout_rows_v478"]}`.
- Claim-to-stub rows:
  `{status["claim_to_stub_rows_v478"]}`.
- Stubs final:
  `{status["stubs_final_v478"]}`.
- Stubs inserted into Quarto:
  `{status["stubs_inserted_into_quarto_v478"]}`.
- Book sources modified:
  `{status["book_sources_modified_v478"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v478"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v478"]}`.

### Interpretation

The manuscript now has bounded draft text stubs tied to the selected visual
package. These stubs are useful drafting material, but they are not final prose
and have not been inserted into the book.

### Claim Impact

- Allowed: section text stubs, asset callout sentences and bounded
  claim-to-stub mapping.
- Still prohibited: final prose, Quarto/book-reference mutation, submission
  readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v478 in the living notebook. v479 should audit stub claim consistency
before any future insertion plan.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v477 = _read_status(PRIOR_VISUAL_DELTA_VERSION)
    if v477["next_artifact_v477"] != "paper4_v478_section_text_stub_bundle.md":
        raise RuntimeError("v478 expects v477 to route to section text stub bundle.")

    section_stubs = _section_text_stubs()
    callouts = _asset_callout_queue()
    claim_to_stub = _claim_to_stub_map()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v478_section_text_stubs.csv", section_stubs)
    write_csv(TABLE_DIR / "paper4_v478_asset_callout_sentence_queue.csv", callouts)
    write_csv(TABLE_DIR / "paper4_v478_claim_to_stub_map.csv", claim_to_stub)
    write_csv(TABLE_DIR / "paper4_v478_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v478_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v478_section_text_stub_bundle",
        "schema_version": "2026-05-17.478",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_visual_delta_version_v478": PRIOR_VISUAL_DELTA_VERSION,
        "section_text_stub_bundle_created_v478": True,
        "section_text_stub_rows_v478": len(section_stubs),
        "asset_callout_rows_v478": len(callouts),
        "claim_to_stub_rows_v478": len(claim_to_stub),
        "readiness_delta_rows_v478": len(readiness),
        "stubs_final_v478": False,
        "stubs_inserted_into_quarto_v478": False,
        "book_sources_modified_v478": False,
        "book_references_modified_v478": False,
        "submission_ready_claim_allowed_v478": False,
        "working_champion_claim_allowed_v478": False,
        "paper1_promotion_allowed_v478": False,
        "paper4_working_champion_changed_v478": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v478": NEXT_ARTIFACT,
        "claim_boundary": (
            "v478 creates draft text stubs only; final prose, insertion, submission "
            "and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v478 must not create final Paper 4 promotion.")

    STUB_MD.write_text(_stub_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v478": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
