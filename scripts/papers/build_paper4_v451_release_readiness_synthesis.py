#!/usr/bin/env python3
"""Build Paper 4 v451 bounded release-readiness synthesis artifacts."""

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

VERSION = 451
PRIOR_POST_RENDER_PYTEST_VERSION = 450
NEXT_ARTIFACT = "paper4_v452_manuscript_extraction_scaffold.md"
SYNTHESIS_MD = NOTEBOOK.parent / "paper4_v451_release_readiness_synthesis.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _gate_summary(
    *,
    v448: dict[str, Any],
    v449: dict[str, Any],
    v450: dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gate_id_v451": "post_render_full_repository_pytest",
                "source_version_v451": 450,
                "passed_v451": bool(v450["post_render_full_repository_pytest_clean_v450"]),
                "evidence_count_v451": int(v450["pytest_collected_items_v450"]),
                "evidence_artifact_v451": "paper4_v450_pytest_probe_summary.csv",
                "bounded_readiness_role_v451": "regression gate",
                "claim_boundary_v451": "full pytest clean after v448-v449 guardrails",
            },
            {
                "gate_id_v451": "repository_ruff_clean",
                "source_version_v451": 450,
                "passed_v451": bool(v450["repository_ruff_clean_v450"]),
                "evidence_count_v451": int(v450["repo_ruff_total_v450"]),
                "evidence_artifact_v451": "paper4_v450_validation_gate_summary.csv",
                "bounded_readiness_role_v451": "lint gate",
                "claim_boundary_v451": "repository Ruff emits zero diagnostics",
            },
            {
                "gate_id_v451": "paper4_official_quarto_render",
                "source_version_v451": 448,
                "passed_v451": bool(v448["paper4_official_quarto_render_clean_v448"]),
                "evidence_count_v451": int(v448["rendered_page_count_v448"]),
                "evidence_artifact_v451": "paper4_v448_quarto_render_probe_summary.csv",
                "bounded_readiness_role_v451": "chapter render gate",
                "claim_boundary_v451": "official registered Paper 4 chapter only",
            },
            {
                "gate_id_v451": "full_book_quarto_render",
                "source_version_v451": 449,
                "passed_v451": bool(v449["full_book_render_clean_v449"]),
                "evidence_count_v451": int(v449["rendered_page_count_v449"]),
                "evidence_artifact_v451": "paper4_v449_full_book_render_probe_summary.csv",
                "bounded_readiness_role_v451": "book render gate",
                "claim_boundary_v451": "full official Quarto book render",
            },
            {
                "gate_id_v451": "paper4_archive_policy_preserved",
                "source_version_v451": 449,
                "passed_v451": bool(v449["paper4_archive_policy_preserved_v449"]),
                "evidence_count_v451": int(v449["paper4_unregistered_nonarchived_page_count_v449"]),
                "evidence_artifact_v451": "paper4_v449_paper4_surface_in_full_book.csv",
                "bounded_readiness_role_v451": "surface governance gate",
                "claim_boundary_v451": "historical archive stays out of official render",
            },
            {
                "gate_id_v451": "paper4_final_promotion_absent",
                "source_version_v451": 451,
                "passed_v451": not FORBIDDEN_FINAL_PROMOTION.exists(),
                "evidence_count_v451": 1,
                "evidence_artifact_v451": "paper4_final_promotion_gate_not_created",
                "bounded_readiness_role_v451": "negative promotion gate",
                "claim_boundary_v451": "final promotion remains forbidden and absent",
            },
        ]
    )


def _language_bank() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "language_id_v451": "bounded_readiness_allowed",
                "allowed_v451": True,
                "sentence_v451": (
                    "The living-lab evidence package is bounded-release ready: "
                    "full pytest, repository Ruff, Paper 4 Quarto, and full-book "
                    "Quarto gates are clean as of v450."
                ),
                "use_context_v451": "readiness memo or manuscript extraction preface",
                "claim_boundary_v451": "validation gates only, not final paper acceptance",
            },
            {
                "language_id_v451": "paper4_inside_full_book_allowed",
                "allowed_v451": True,
                "sentence_v451": (
                    "The compact registered Paper 4 chapter renders both alone "
                    "and inside the full official Quarto book."
                ),
                "use_context_v451": "publication surface readiness",
                "claim_boundary_v451": "registered Paper 4 pages only",
            },
            {
                "language_id_v451": "release_ready_prohibited",
                "allowed_v451": False,
                "sentence_v451": (
                    "Paper 4 is final, submitted, externally validated, or promoted "
                    "as a replacement for Paper Estrella."
                ),
                "use_context_v451": "do not use",
                "claim_boundary_v451": "requires manuscript extraction and explicit promotion gate",
            },
            {
                "language_id_v451": "champion_replacement_prohibited",
                "allowed_v451": False,
                "sentence_v451": "Paper 4 changes the official economic champion.",
                "use_context_v451": "do not use",
                "claim_boundary_v451": "no champion or deployment artifact is created",
            },
        ]
    )


def _remaining_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v451": "manuscript_extraction_not_written",
                "blocking_v451": True,
                "evidence_count_v451": 1,
                "required_next_artifact_v451": NEXT_ARTIFACT,
                "claim_boundary_v451": "validated lab evidence still needs manuscript extraction",
            },
            {
                "blocker_id_v451": "external_dataset_validation_not_run",
                "blocking_v451": True,
                "evidence_count_v451": 0,
                "required_next_artifact_v451": "future_external_validation_protocol",
                "claim_boundary_v451": "do not claim external generalization",
            },
            {
                "blocker_id_v451": "paper4_final_promotion_forbidden",
                "blocking_v451": True,
                "evidence_count_v451": 1,
                "required_next_artifact_v451": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v451": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix(all_gates_clean: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v451_bounded_release_readiness_synthesis_written",
                "allowed": True,
                "artifact": "paper4_v451_release_readiness_synthesis.md",
                "boundary": "synthesis artifact only",
            },
            {
                "claim_id": "v451_validation_gates_clean",
                "allowed": all_gates_clean,
                "artifact": "paper4_v451_release_readiness_gate_summary.csv",
                "boundary": "pytest, Ruff and Quarto gates only",
            },
            {
                "claim_id": "v451_manuscript_extraction_complete",
                "allowed": False,
                "artifact": "paper4_v451_remaining_blockers.csv",
                "boundary": "manuscript extraction deferred to v452",
            },
            {
                "claim_id": "v451_external_validation_complete",
                "allowed": False,
                "artifact": "paper4_v451_remaining_blockers.csv",
                "boundary": "no external dataset validation is run",
            },
            {
                "claim_id": "v451_release_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v451_remaining_blockers.csv",
                "boundary": "bounded readiness is not final promotion",
            },
            {
                "claim_id": "v451_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(all_gates_clean: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v451 synthesizes bounded release-readiness gates for Paper 4.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v451_release_readiness_synthesis.md"
                ),
                "boundary": "Synthesis only; not final release or manuscript submission.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v451 validation gates are clean for bounded release readiness.",
                "allowed": all_gates_clean,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v451_release_readiness_gate_summary.csv"
                ),
                "boundary": "Full pytest, Ruff, Quarto render and no-promotion gates only.",
                "prohibited_claim_flag": not all_gates_clean,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v451 completes manuscript extraction or external validation.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v451_remaining_blockers.csv"
                ),
                "boundary": "Both remain future work.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v451 makes Paper 4 final, submitted, or release-promoted.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v451_remaining_blockers.csv"
                ),
                "boundary": "Bounded readiness is not final promotion.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v451 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v451_remaining_blockers.csv"
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
                "lane": "Publication",
                "executable_item": "v451 synthesizes bounded release-readiness gates.",
                "status": "bounded_release_readiness_synthesis_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v452 extracts a manuscript scaffold without final promotion",
                "last_wave": "v451",
                "execution_result": "clean_validation_gates_synthesized_without_promotion",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v451")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _synthesis_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Bounded Release-Readiness Synthesis v451

Generated: {status["generated_at_utc"]}

v451 consolidates the clean validation gates produced by v447-v450 into a
bounded readiness statement for the Paper 4 living lab.

## Clean Gates

- Full repository pytest after full-book render:
  `{status["post_render_full_repository_pytest_clean_v451"]}`.
- Repository Ruff:
  `{status["repository_ruff_clean_v451"]}`.
- Official Paper 4 Quarto render:
  `{status["paper4_official_quarto_render_clean_v451"]}`.
- Full official Quarto book render:
  `{status["full_book_render_clean_v451"]}`.
- Paper 4 archive policy preserved:
  `{status["paper4_archive_policy_preserved_v451"]}`.
- Final promotion absent:
  `{status["paper4_final_promotion_absent_v451"]}`.

## Allowed Language

The living-lab evidence package is bounded-release ready: full pytest,
repository Ruff, Paper 4 Quarto, and full-book Quarto gates are clean as of
v450.

## Prohibited Language

Do not claim that Paper 4 is final, submitted, externally validated, promoted,
or replacing Paper Estrella.

## Next Executable Wave

Build `{status["next_artifact_v451"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V451_RELEASE_READINESS_SYNTHESIS_START -->"
    end = "<!-- V451_RELEASE_READINESS_SYNTHESIS_END -->"
    block = f"""
{start}

## Wave v451: Bounded Release-Readiness Synthesis

Generated: {status["generated_at_utc"]}

### Objective

v451 consolidates v447-v450 clean validation gates into bounded release-readiness
language without creating a final Paper 4 promotion.

### Results

- Clean validation gates:
  `{status["clean_validation_gate_count_v451"]}` /
  `{status["validation_gate_count_v451"]}`.
- Bounded release-readiness language allowed:
  `{status["bounded_release_readiness_language_allowed_v451"]}`.
- Manuscript extraction complete:
  `{status["manuscript_extraction_complete_v451"]}`.
- External validation complete:
  `{status["external_validation_complete_v451"]}`.
- Final promotion absent:
  `{status["paper4_final_promotion_absent_v451"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v451"]}`.

### Interpretation

Paper 4 now has a clean bounded-readiness validation stack: full pytest, global
Ruff, official Paper 4 Quarto render, full-book render, archive governance and
no-promotion gate. This supports extraction into manuscript language, but still
does not make the paper final or externally validated.

### Claim Impact

- Allowed: bounded release-readiness synthesis and clean validation-gate
  language.
- Still prohibited: manuscript-complete, external-validation, champion
  replacement and final-promotion claims.

### Quarto Promotion Decision

Keep v451 in the living notebook. v452 should extract a manuscript scaffold
without final promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v448 = _read_status(448)
    v449 = _read_status(449)
    v450 = _read_status(450)
    if v450["next_artifact_v450"] != "paper4_v451_release_readiness_synthesis.md":
        raise RuntimeError("v451 expects v450 to route to release readiness synthesis.")

    gates = _gate_summary(v448=v448, v449=v449, v450=v450)
    clean_gate_count = int(gates["passed_v451"].astype(bool).sum())
    all_gates_clean = clean_gate_count == len(gates)
    language = _language_bank()
    blockers = _remaining_blockers()
    claim_matrix = _claim_matrix(all_gates_clean)
    _update_claim_boundaries(all_gates_clean)
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v451_release_readiness_gate_summary.csv", gates)
    write_csv(TABLE_DIR / "paper4_v451_publishable_language_bank.csv", language)
    write_csv(TABLE_DIR / "paper4_v451_remaining_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v451_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v451_release_readiness_synthesis",
        "schema_version": "2026-05-17.451",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_post_render_pytest_version_v451": PRIOR_POST_RENDER_PYTEST_VERSION,
        "validation_gate_count_v451": len(gates),
        "clean_validation_gate_count_v451": clean_gate_count,
        "all_validation_gates_clean_v451": all_gates_clean,
        "post_render_full_repository_pytest_clean_v451": bool(
            v450["post_render_full_repository_pytest_clean_v450"]
        ),
        "repository_ruff_clean_v451": bool(v450["repository_ruff_clean_v450"]),
        "paper4_official_quarto_render_clean_v451": bool(
            v448["paper4_official_quarto_render_clean_v448"]
        ),
        "full_book_render_clean_v451": bool(v449["full_book_render_clean_v449"]),
        "paper4_archive_policy_preserved_v451": bool(
            v449["paper4_archive_policy_preserved_v449"]
        ),
        "paper4_final_promotion_absent_v451": not FORBIDDEN_FINAL_PROMOTION.exists(),
        "bounded_release_readiness_language_allowed_v451": all_gates_clean,
        "release_readiness_synthesis_written_v451": True,
        "manuscript_extraction_complete_v451": False,
        "external_validation_complete_v451": False,
        "working_champion_claim_allowed_v451": False,
        "paper1_promotion_allowed_v451": False,
        "paper4_working_champion_changed_v451": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v451": NEXT_ARTIFACT,
        "claim_boundary": (
            "v451 is bounded release-readiness synthesis only; manuscript "
            "extraction, external validation and final promotion remain blocked"
        ),
    }
    if not all_gates_clean:
        raise RuntimeError("v451 expected all validation gates to be clean.")
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v451 must not create final Paper 4 promotion.")

    SYNTHESIS_MD.write_text(_synthesis_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v451": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
