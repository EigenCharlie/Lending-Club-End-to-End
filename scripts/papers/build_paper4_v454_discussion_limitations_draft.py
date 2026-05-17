#!/usr/bin/env python3
"""Build Paper 4 v454 discussion/limitations draft artifacts."""

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

VERSION = 454
PRIOR_METHODS_RESULTS_DRAFT_VERSION = 453
NEXT_ARTIFACT = "paper4_v455_abstract_conclusion_draft.md"
DRAFT_MD = NOTEBOOK.parent / "paper4_v454_discussion_limitations_draft.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _limitation_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "limitation_id_v454": "internal_living_lab_not_external_validation",
                "must_state_v454": True,
                "evidence_artifact_v454": "paper4_v451_remaining_blockers.csv",
                "discussion_role_v454": "separate reproducibility from external generalization",
                "claim_boundary_v454": "no external dataset validation was run",
            },
            {
                "limitation_id_v454": "proxy_fairness_not_legal_certification",
                "must_state_v454": True,
                "evidence_artifact_v454": "paper4_current_claim_boundaries.csv",
                "discussion_role_v454": "avoid legal/global fair-lending claim",
                "claim_boundary_v454": "proxy/intersectional audit only",
            },
            {
                "limitation_id_v454": "bounded_readiness_not_final_submission",
                "must_state_v454": True,
                "evidence_artifact_v454": "paper4_v451_publishable_language_bank.csv",
                "discussion_role_v454": "prevent release-ready overclaim",
                "claim_boundary_v454": "not final, submitted, or accepted",
            },
            {
                "limitation_id_v454": "archive_governance_not_full_history_render",
                "must_state_v454": True,
                "evidence_artifact_v454": "paper4_v449_paper4_surface_in_full_book.csv",
                "discussion_role_v454": "explain compact official chapter surface",
                "claim_boundary_v454": "historical archive remains intentionally not rendered",
            },
            {
                "limitation_id_v454": "no_champion_replacement",
                "must_state_v454": True,
                "evidence_artifact_v454": "paper4_final_promotion_gate_not_created",
                "discussion_role_v454": "protect Paper Estrella champion/promotion boundary",
                "claim_boundary_v454": "no deployment or champion update",
            },
        ]
    )


def _discussion_claim_trace() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sentence_id_v454": "discussion_value",
                "allowed_v454": True,
                "sentence_v454": (
                    "The contribution of the current Paper 4 package is an "
                    "auditable execution protocol that turns experimental waves "
                    "into reproducible validation gates."
                ),
                "source_artifact_v454": "paper4_v451_release_readiness_gate_summary.csv",
                "claim_boundary_v454": "protocol/readiness contribution",
            },
            {
                "sentence_id_v454": "discussion_reproducibility",
                "allowed_v454": True,
                "sentence_v454": (
                    "Its strongest current evidence is internal reproducibility: "
                    "full pytest, Ruff, Paper 4 render, full-book render and "
                    "no-promotion gates are clean."
                ),
                "source_artifact_v454": "paper4_v451_release_readiness_gate_summary.csv",
                "claim_boundary_v454": "internal validation gates",
            },
            {
                "sentence_id_v454": "limitation_external",
                "allowed_v454": True,
                "sentence_v454": (
                    "The package does not yet establish external generalization, "
                    "because no external dataset validation has been run."
                ),
                "source_artifact_v454": "paper4_v454_limitation_register.csv",
                "claim_boundary_v454": "explicit limitation",
            },
            {
                "sentence_id_v454": "limitation_fairness",
                "allowed_v454": True,
                "sentence_v454": (
                    "Fairness language remains proxy/intersectional and should not "
                    "be framed as legal certification."
                ),
                "source_artifact_v454": "paper4_current_claim_boundaries.csv",
                "claim_boundary_v454": "proxy fairness only",
            },
            {
                "sentence_id_v454": "prohibited_final",
                "allowed_v454": False,
                "sentence_v454": (
                    "The discussion can state that Paper 4 is final and externally validated."
                ),
                "source_artifact_v454": "paper4_v454_remaining_blockers.csv",
                "claim_boundary_v454": "final/external claims remain blocked",
            },
        ]
    )


def _remaining_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v454": "abstract_conclusion_not_drafted",
                "blocking_v454": True,
                "evidence_count_v454": 1,
                "required_next_artifact_v454": NEXT_ARTIFACT,
                "claim_boundary_v454": "front/back matter remains pending",
            },
            {
                "blocker_id_v454": "complete_manuscript_not_assembled",
                "blocking_v454": True,
                "evidence_count_v454": 1,
                "required_next_artifact_v454": "paper4_v456_manuscript_assembly_packet.md",
                "claim_boundary_v454": "draft sections are not assembled manuscript",
            },
            {
                "blocker_id_v454": "external_dataset_validation_not_run",
                "blocking_v454": True,
                "evidence_count_v454": 0,
                "required_next_artifact_v454": "future_external_validation_protocol",
                "claim_boundary_v454": "do not claim external generalization",
            },
            {
                "blocker_id_v454": "paper4_final_promotion_forbidden",
                "blocking_v454": True,
                "evidence_count_v454": 1,
                "required_next_artifact_v454": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v454": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v454_discussion_limitations_draft_created",
                "allowed": True,
                "artifact": "paper4_v454_discussion_limitations_draft.md",
                "boundary": "discussion/limitations prose only",
            },
            {
                "claim_id": "v454_limitation_register_created",
                "allowed": True,
                "artifact": "paper4_v454_limitation_register.csv",
                "boundary": "required caveat register",
            },
            {
                "claim_id": "v454_abstract_conclusion_complete",
                "allowed": False,
                "artifact": "paper4_v454_remaining_blockers.csv",
                "boundary": "deferred to v455",
            },
            {
                "claim_id": "v454_complete_manuscript_or_submission",
                "allowed": False,
                "artifact": "paper4_v454_remaining_blockers.csv",
                "boundary": "draft is still partial",
            },
            {
                "claim_id": "v454_external_validation_complete",
                "allowed": False,
                "artifact": "paper4_v454_remaining_blockers.csv",
                "boundary": "no external validation run",
            },
            {
                "claim_id": "v454_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v454 drafts Paper 4 discussion and limitations prose.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v454_discussion_limitations_draft.md"
                ),
                "boundary": "Partial manuscript draft only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v454 registers required limitations for Paper 4 manuscript extraction.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v454_limitation_register.csv"
                ),
                "boundary": "Limitations register only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v454 completes abstract, conclusion or manuscript assembly.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v454_remaining_blockers.csv"
                ),
                "boundary": "Those artifacts remain pending.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v454 makes Paper 4 final, submitted, or externally validated.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v454_remaining_blockers.csv"
                ),
                "boundary": "Discussion draft does not change validation scope.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v454 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v454_remaining_blockers.csv"
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
                "executable_item": "v454 drafts discussion and limitations prose.",
                "status": "discussion_limitations_draft_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v455 drafts abstract/conclusion without final promotion",
                "last_wave": "v454",
                "execution_result": "discussion_limitations_draft_created_without_finalization",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v454")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _draft_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Discussion/Limitations Draft v454

Generated: {status["generated_at_utc"]}

## Discussion Draft

The strongest current result of the Paper 4 living lab is not a single final
policy claim, but a reproducible execution protocol. The validation stack now
connects executable waves, generated artifacts, guardrail tests, repository
lint, and Quarto rendering into one bounded evidence package. This makes the lab
useful as a future manuscript source because every claim can be traced to an
artifact and every prohibited claim is explicitly represented.

The readiness result is therefore intentionally narrow. Paper 4 can say that its
current evidence package is internally reproducible and manuscript-extractable:
post-render pytest is clean, repository Ruff is clean, the registered Paper 4
chapter renders, the full official Quarto book renders, and the final promotion
artifact is absent.

## Limitations Draft

This does not establish external generalization, because no external dataset
validation has been run. It also does not establish legal fair-lending
certification; fairness language must remain proxy/intersectional. Finally,
bounded readiness is not final paper promotion. Paper 4 does not replace Paper
Estrella, update the official champion, create a deployment gate, or claim
submission readiness.

## Required Caveat

v454 is a discussion/limitations draft. Abstract, conclusion, full manuscript
assembly, external validation, and final promotion remain blocked.

## Next Executable Wave

Build `{status["next_artifact_v454"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V454_DISCUSSION_LIMITATIONS_DRAFT_START -->"
    end = "<!-- V454_DISCUSSION_LIMITATIONS_DRAFT_END -->"
    block = f"""
{start}

## Wave v454: Discussion/Limitations Draft

Generated: {status["generated_at_utc"]}

### Objective

v454 drafts discussion/limitations prose and registers the required caveats that
prevent bounded readiness from becoming a final-paper claim.

### Results

- Limitation rows:
  `{status["limitation_count_v454"]}`.
- Discussion trace sentences:
  `{status["discussion_sentence_count_v454"]}`.
- Allowed discussion sentences:
  `{status["allowed_discussion_sentence_count_v454"]}`.
- Discussion/limitations draft created:
  `{status["discussion_limitations_draft_created_v454"]}`.
- Abstract/conclusion complete:
  `{status["abstract_conclusion_complete_v454"]}`.
- Complete manuscript:
  `{status["complete_manuscript_v454"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v454"]}`.

### Interpretation

Discussion/limitations prose now exists and explicitly separates internal
readiness from external validation, legal certification, submission readiness and
final promotion.

### Claim Impact

- Allowed: discussion/limitations draft and limitation register.
- Still prohibited: abstract/conclusion complete, full manuscript, external
  validation, champion replacement and final-promotion claims.

### Quarto Promotion Decision

Keep v454 in the living notebook. v455 should draft abstract/conclusion without
final promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v453 = _read_status(453)
    if v453["next_artifact_v453"] != "paper4_v454_discussion_limitations_draft.md":
        raise RuntimeError("v454 expects v453 to route to discussion/limitations draft.")
    if v453["methods_results_draft_created_v453"] is not True:
        raise RuntimeError("v454 expects v453 Methods/Results draft to exist.")

    limitations = _limitation_register()
    trace = _discussion_claim_trace()
    blockers = _remaining_blockers()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v454_limitation_register.csv", limitations)
    write_csv(TABLE_DIR / "paper4_v454_discussion_claim_trace.csv", trace)
    write_csv(TABLE_DIR / "paper4_v454_remaining_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v454_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v454_discussion_limitations_draft",
        "schema_version": "2026-05-17.454",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_methods_results_draft_version_v454": PRIOR_METHODS_RESULTS_DRAFT_VERSION,
        "limitation_count_v454": len(limitations),
        "discussion_sentence_count_v454": len(trace),
        "allowed_discussion_sentence_count_v454": int(trace["allowed_v454"].astype(bool).sum()),
        "prohibited_discussion_sentence_count_v454": int(
            (~trace["allowed_v454"].astype(bool)).sum()
        ),
        "discussion_limitations_draft_created_v454": True,
        "limitation_register_created_v454": True,
        "abstract_conclusion_complete_v454": False,
        "complete_manuscript_v454": False,
        "external_validation_complete_v454": False,
        "working_champion_claim_allowed_v454": False,
        "paper1_promotion_allowed_v454": False,
        "paper4_working_champion_changed_v454": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v454": NEXT_ARTIFACT,
        "claim_boundary": (
            "v454 is discussion/limitations draft only; abstract, conclusion, "
            "external validation and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v454 must not create final Paper 4 promotion.")

    DRAFT_MD.write_text(_draft_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v454": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
