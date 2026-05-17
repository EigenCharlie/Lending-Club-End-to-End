#!/usr/bin/env python3
"""Build Paper 4 v368 publishable claim-scope update artifacts."""

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

VERSION = 368
BASE_VERSION = 353
PRIOR_ROUTE_VERSION = 367
PRIOR_CHUNK_VERSION = 366
PRIOR_PLAN_VERSION = 365
PRIOR_GAP_VERSION = 363
PRIOR_BOUND_VERSION = 361
NEXT_VERSION = 369
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_proxy_live_gate_separation.csv"
SCOPE_MEMO = NOTEBOOK.parent / "paper4_v368_publishable_claim_scope_update.md"
RECOMMENDED_FRAMING = "bounded_living_lab_claim_scope"
STRONGEST_TRUE_CLAIM = (
    "Paper 4 can be framed as a reproducible living-lab and governance protocol "
    "that audits bounded candidate-improvement evidence around the protected Paper "
    "Estrella economic champion; v361-v367 support bounded no-entry/source-blocker "
    "claims, but not a new working champion, final promotion or full-v55 global "
    "optimality certificate."
)


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v368 defines the strongest publishable Paper 4 claim after v361-v367.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v368_publishable_claim_scope_update.md"
                ),
                "boundary": "Bounded living-lab and governance framing only; no champion promotion.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v368 allows bounded no-entry and source-blocker result statements.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v368_evidence_trace.csv"
                ),
                "boundary": "Statements must cite v361 bounded search and v366 chunk evidence.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v368 proves a full-v55 global optimality certificate.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v368_claim_blockers.csv"
                ),
                "boundary": "v363 remains a gap certificate; v365-v366 did not terminate full pricing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v368 authorizes a Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v368_claim_blockers.csv"
                ),
                "boundary": "No full solver certificate, proxy-live bridge or deployment gate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v368 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v368_claim_blockers.csv"
                ),
                "boundary": "No final promotion artifact, final champion or paper replacement is created.",
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
                    "v368 records the strongest true publishable claim after v361-v367 "
                    "and routes the next wave to separate proxy evidence from live gates."
                ),
                "status": "publishable_bounded_claim_scope_update_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v369 cleanly separates offline proxy evidence, live readiness, "
                    "deployment gates and final-promotion blockers"
                ),
                "last_wave": "v368",
                "execution_result": "strongest_true_claim_after_v361_v367_recorded",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v368")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _scope_memo(status: dict[str, Any], evidence_trace: pd.DataFrame) -> str:
    evidence_lines = "\n".join(
        (
            f"- v{int(row['wave_v368'])}: {row['evidence_statement_v368']}. "
            f"Artifact: `{row['artifact_v368']}`."
        )
        for _, row in evidence_trace.iterrows()
    )
    return f"""# Paper 4 Publishable Claim Scope Update v368

Generated: {status["generated_at_utc"]}

## Recommended Framing

`{status["recommended_framing_v368"]}`

## Strongest True Claim

{status["strongest_true_claim_v368"]}

## Evidence That Supports This Claim

{evidence_lines}

## Allowed Publishable Language

- Paper 4 can present a reproducible living-lab protocol for auditing candidate
  policy improvements around the protected Paper Estrella economic champion.
- Paper 4 can report that the bounded v361 fourth-order source-tight search
  screened {status["v361_ordered_fourth_order_rows_v368"]} ordered rows,
  found {status["v361_source_exact_fourth_order_rows_v368"]} source-exact rows,
  and found {status["v361_cvar_feasible_entering_rows_v368"]} CVaR-feasible
  entering rows.
- Paper 4 can report that the v366 chunk-0001 full-v55 one-swap prototype
  screened {status["v366_ordered_one_swap_rows_v368"]} ordered one-swaps and
  found {status["v366_source_exact_rows_v368"]} source-exact rows.
- Paper 4 can report the open global-proof gap: v71 still had
  {status["v71_improving_omitted_columns_v368"]} improving omitted columns,
  v365 left {status["planned_chunk_count_v368"]} planned chunks, and v366 left
  {status["remaining_unpriced_chunks_v368"]} chunks unpriced.

## Prohibited Language

- Do not claim a Paper 4 working champion.
- Do not claim Paper Estrella replacement.
- Do not claim full-v55 reduced-cost termination, full-universe integer
  optimality or deployment readiness.
- Do not describe IFRS9, fairness, live monitoring or governance outputs as
  contractual/legal production controls.

## Next Executable Wave

Build `{status["next_artifact_v368"]}` so the paper can clearly distinguish
offline proxy evidence from live deployment and final-promotion gates.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V368_PUBLISHABLE_CLAIM_SCOPE_UPDATE_START -->"
    end = "<!-- V368_PUBLISHABLE_CLAIM_SCOPE_UPDATE_END -->"
    block = f"""
{start}

## Wave v368: Publishable Claim Scope Update

Generated: {status["generated_at_utc"]}

### Objective

v367 selected a bounded claim-scope update after the v366 chunk probe found no
source-exact rows. v368 writes the strongest Paper 4 claim that remains true
after v361-v367 without pretending that a final solver, champion or deployment
gate exists.

### Results

- Recommended framing:
  `{status["recommended_framing_v368"]}`.
- Strongest true claim:
  {status["strongest_true_claim_v368"]}
- V361 ordered fourth-order rows:
  `{status["v361_ordered_fourth_order_rows_v368"]}`.
- V361 source-exact rows:
  `{status["v361_source_exact_fourth_order_rows_v368"]}`.
- V361 CVaR-feasible entering rows:
  `{status["v361_cvar_feasible_entering_rows_v368"]}`.
- v71 improving omitted columns:
  `{status["v71_improving_omitted_columns_v368"]}`.
- V366 ordered one-swap rows:
  `{status["v366_ordered_one_swap_rows_v368"]}`.
- V366 source-exact rows:
  `{status["v366_source_exact_rows_v368"]}`.
- Allowed publishable claims:
  `{status["allowed_publishable_claim_rows_v368"]}`.
- Prohibited claim rows:
  `{status["prohibited_claim_rows_v368"]}`.
- Next artifact:
  `{status["next_artifact_v368"]}`.

### Interpretation

v368 is the paper-facing consolidation point. The useful result is not a new
champion; it is a disciplined claim boundary: Paper 4 can be written as a
governed living-lab method with bounded evidence and explicit global-proof
blockers.

### Claim Impact

- Allowed: publishable bounded living-lab framing, bounded v361 no-entry
  evidence, v366 source-blocker evidence, and transparent global-proof gap.
- Still prohibited: full-v55 termination, valid global integer optimality,
  working champion, Paper Estrella replacement, live deployment and final
  promotion.

### Quarto Promotion Decision

Keep v368 in the living notebook. v369 should separate offline proxy evidence
from live readiness and deployment/final-promotion gates.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v361_status = json.loads((STATUS_DIR / "paper4_v361_status.json").read_text(encoding="utf-8"))
    v363_status = json.loads((STATUS_DIR / "paper4_v363_status.json").read_text(encoding="utf-8"))
    v365_status = json.loads((STATUS_DIR / "paper4_v365_status.json").read_text(encoding="utf-8"))
    v366_status = json.loads((STATUS_DIR / "paper4_v366_status.json").read_text(encoding="utf-8"))
    v367_status = json.loads((STATUS_DIR / "paper4_v367_status.json").read_text(encoding="utf-8"))
    if v367_status["recommended_route_v367"] != "bounded_claim_scope_update":
        raise RuntimeError("v368 requires v367 to select bounded claim-scope update.")
    if bool(v367_status["valid_full_v55_dual_bound_certificate_v367"]):
        raise RuntimeError("v368 expects the full-v55 certificate to remain blocked.")

    strongest_claim = STRONGEST_TRUE_CLAIM
    scope_update = pd.DataFrame(
        [
            {
                f"scope_update_id_v{VERSION}": "v368_publishable_claim_scope_update",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"prior_route_version_v{VERSION}": PRIOR_ROUTE_VERSION,
                f"prior_chunk_version_v{VERSION}": PRIOR_CHUNK_VERSION,
                f"prior_plan_version_v{VERSION}": PRIOR_PLAN_VERSION,
                f"prior_gap_version_v{VERSION}": PRIOR_GAP_VERSION,
                f"prior_bound_version_v{VERSION}": PRIOR_BOUND_VERSION,
                f"recommended_framing_v{VERSION}": RECOMMENDED_FRAMING,
                f"strongest_true_claim_v{VERSION}": strongest_claim,
                f"v361_ordered_fourth_order_rows_v{VERSION}": int(
                    v361_status["ordered_fourth_order_rows_screened_v361"]
                ),
                f"v361_source_exact_fourth_order_rows_v{VERSION}": int(
                    v361_status["source_exact_fourth_order_rows_v361"]
                ),
                f"v361_cvar_feasible_entering_rows_v{VERSION}": int(
                    v361_status["cvar_feasible_entering_rows_v361"]
                ),
                f"v363_bounded_candidate_pool_share_v{VERSION}": float(
                    v363_status["bounded_candidate_pool_share_v363"]
                ),
                f"v71_improving_omitted_columns_v{VERSION}": int(
                    v363_status["v71_improving_omitted_columns_v363"]
                ),
                f"planned_chunk_count_v{VERSION}": int(
                    v365_status["planned_chunk_count_v365"]
                ),
                f"remaining_unpriced_chunks_v{VERSION}": int(
                    v367_status["remaining_unpriced_chunks_v367"]
                ),
                f"v366_ordered_one_swap_rows_v{VERSION}": int(
                    v366_status["ordered_one_swap_rows_v366"]
                ),
                f"v366_source_exact_rows_v{VERSION}": int(v366_status["source_exact_rows_v366"]),
                f"v366_cvar_feasible_entering_rows_v{VERSION}": int(
                    v366_status["cvar_feasible_entering_rows_v366"]
                ),
                f"valid_full_v55_dual_bound_certificate_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
            }
        ]
    )
    evidence_trace = pd.DataFrame(
        [
            {
                f"evidence_id_v{VERSION}": "bounded_fourth_order_no_entry",
                f"wave_v{VERSION}": 361,
                f"artifact_v{VERSION}": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v361_v353_fourth_order_or_full_dual_bound.csv"
                ),
                f"evidence_statement_v{VERSION}": (
                    "bounded fourth-order source-tight loop found zero CVaR-feasible "
                    "entering rows"
                ),
                f"publishable_v{VERSION}": True,
                f"claim_boundary_v{VERSION}": "bounded v361 search scope only",
            },
            {
                f"evidence_id_v{VERSION}": "full_dual_bound_gap",
                f"wave_v{VERSION}": 363,
                f"artifact_v{VERSION}": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v363_v353_full_dual_bound_or_gap_certificate.csv"
                ),
                f"evidence_statement_v{VERSION}": (
                    "full-v55 dual-bound certificate remains unavailable with "
                    "v71 improving omitted columns"
                ),
                f"publishable_v{VERSION}": True,
                f"claim_boundary_v{VERSION}": "gap disclosure only",
            },
            {
                f"evidence_id_v{VERSION}": "full_v55_chunk_plan",
                f"wave_v{VERSION}": 365,
                f"artifact_v{VERSION}": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v365_v353_full_v55_pricing_chunk_plan.csv"
                ),
                f"evidence_statement_v{VERSION}": (
                    "full omitted pricing was decomposed into 28 reproducible chunks"
                ),
                f"publishable_v{VERSION}": True,
                f"claim_boundary_v{VERSION}": "resource plan only",
            },
            {
                f"evidence_id_v{VERSION}": "chunk_0001_source_blocker",
                f"wave_v{VERSION}": 366,
                f"artifact_v{VERSION}": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v366_v353_full_v55_pricing_chunk_prototype.csv"
                ),
                f"evidence_statement_v{VERSION}": (
                    "chunk 1 screened 1.71M one-swaps and found zero source-exact rows"
                ),
                f"publishable_v{VERSION}": True,
                f"claim_boundary_v{VERSION}": "single chunk only",
            },
            {
                f"evidence_id_v{VERSION}": "route_scope_decision",
                f"wave_v{VERSION}": 367,
                f"artifact_v{VERSION}": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v367_route_decision_after_chunk_probe.csv"
                ),
                f"evidence_statement_v{VERSION}": (
                    "route decision selects bounded claim-scope update before more "
                    "solver expansion"
                ),
                f"publishable_v{VERSION}": True,
                f"claim_boundary_v{VERSION}": "recommendation only",
            },
        ]
    )
    publishable_claims = pd.DataFrame(
        [
            {
                f"claim_id_v{VERSION}": "bounded_living_lab_protocol",
                f"allowed_v{VERSION}": True,
                f"publishable_language_v{VERSION}": (
                    "Paper 4 contributes a reproducible living-lab protocol for "
                    "auditing candidate-improvement evidence around the protected "
                    "Paper Estrella champion."
                ),
                f"required_citation_v{VERSION}": "v361-v368 evidence trace",
                f"boundary_v{VERSION}": "method/governance claim only",
            },
            {
                f"claim_id_v{VERSION}": "bounded_fourth_order_no_entry",
                f"allowed_v{VERSION}": True,
                f"publishable_language_v{VERSION}": (
                    "Within the bounded v361 fourth-order source-tight scope, no "
                    "CVaR-feasible entering row was found."
                ),
                f"required_citation_v{VERSION}": "paper4_v361_v353_fourth_order_or_full_dual_bound.csv",
                f"boundary_v{VERSION}": "bounded search scope only",
            },
            {
                f"claim_id_v{VERSION}": "chunk_source_governance_blocker",
                f"allowed_v{VERSION}": True,
                f"publishable_language_v{VERSION}": (
                    "The v366 chunk prototype shows source governance can bind before "
                    "any source-exact full-v55 one-swap row survives."
                ),
                f"required_citation_v{VERSION}": "paper4_v366_v353_full_v55_pricing_chunk_prototype.csv",
                f"boundary_v{VERSION}": "chunk 1 only",
            },
            {
                f"claim_id_v{VERSION}": "global_proof_gap_transparency",
                f"allowed_v{VERSION}": True,
                f"publishable_language_v{VERSION}": (
                    "The paper explicitly reports that full-v55 global proof remains "
                    "open after v363-v367."
                ),
                f"required_citation_v{VERSION}": "paper4_v363_status.json; paper4_v367_status.json",
                f"boundary_v{VERSION}": "negative/gap statement only",
            },
        ]
    )
    prohibited_claims = pd.DataFrame(
        [
            {
                f"claim_id_v{VERSION}": "full_v55_global_optimality",
                f"allowed_v{VERSION}": False,
                f"reason_v{VERSION}": (
                    "v363 is a gap certificate and v366 priced only one chunk."
                ),
                f"blocking_evidence_count_v{VERSION}": int(
                    v367_status["remaining_unpriced_chunks_v367"]
                ),
            },
            {
                f"claim_id_v{VERSION}": "paper4_working_champion",
                f"allowed_v{VERSION}": False,
                f"reason_v{VERSION}": (
                    "no valid full solver certificate, proxy/live bridge or promotion gate"
                ),
                f"blocking_evidence_count_v{VERSION}": 1,
            },
            {
                f"claim_id_v{VERSION}": "paper_estrella_replacement",
                f"allowed_v{VERSION}": False,
                f"reason_v{VERSION}": "Paper Estrella champion remains protected",
                f"blocking_evidence_count_v{VERSION}": 1,
            },
            {
                f"claim_id_v{VERSION}": "live_deployment_readiness",
                f"allowed_v{VERSION}": False,
                f"reason_v{VERSION}": "offline proxy evidence has not been separated from live gates",
                f"blocking_evidence_count_v{VERSION}": 1,
            },
        ]
    )
    claim_blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "full_v55_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v367_status["remaining_unpriced_chunks_v367"]),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "global proof remains open",
            },
            {
                f"blocker_id_v{VERSION}": "source_exact_chunk_rows_zero",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v366_status["source_exact_rows_v366"]),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "source governance blocker must be reported",
            },
            {
                f"blocker_id_v{VERSION}": "proxy_live_gate_not_separated",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "next wave must separate offline and live claims",
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
                "claim_id": "v368_publishable_scope_update_created",
                "allowed": True,
                "artifact": "paper4_v368_publishable_claim_scope_update.md",
                "boundary": "scope update only",
            },
            {
                "claim_id": "v368_bounded_living_lab_claim_allowed",
                "allowed": True,
                "artifact": "paper4_v368_publishable_claims.csv",
                "boundary": "method/governance framing",
            },
            {
                "claim_id": "v368_valid_full_v55_dual_bound_certificate",
                "allowed": False,
                "artifact": "paper4_v368_claim_blockers.csv",
                "boundary": "global proof remains blocked",
            },
            {
                "claim_id": "v368_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v368_claim_blockers.csv",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v368_publishable_claim_scope_update.csv", scope_update)
    write_csv(TABLE_DIR / "paper4_v368_evidence_trace.csv", evidence_trace)
    write_csv(TABLE_DIR / "paper4_v368_publishable_claims.csv", publishable_claims)
    write_csv(TABLE_DIR / "paper4_v368_prohibited_claims.csv", prohibited_claims)
    write_csv(TABLE_DIR / "paper4_v368_claim_blockers.csv", claim_blockers)
    write_csv(TABLE_DIR / "paper4_v368_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = scope_update.iloc[0]
    status = {
        "phase": "v368_publishable_claim_scope_update",
        "schema_version": "2026-05-17.368",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v368": BASE_VERSION,
        "prior_route_version_v368": PRIOR_ROUTE_VERSION,
        "prior_chunk_version_v368": PRIOR_CHUNK_VERSION,
        "prior_plan_version_v368": PRIOR_PLAN_VERSION,
        "prior_gap_version_v368": PRIOR_GAP_VERSION,
        "prior_bound_version_v368": PRIOR_BOUND_VERSION,
        "recommended_framing_v368": str(row[f"recommended_framing_v{VERSION}"]),
        "strongest_true_claim_v368": strongest_claim,
        "v361_ordered_fourth_order_rows_v368": int(
            row[f"v361_ordered_fourth_order_rows_v{VERSION}"]
        ),
        "v361_source_exact_fourth_order_rows_v368": int(
            row[f"v361_source_exact_fourth_order_rows_v{VERSION}"]
        ),
        "v361_cvar_feasible_entering_rows_v368": int(
            row[f"v361_cvar_feasible_entering_rows_v{VERSION}"]
        ),
        "v363_bounded_candidate_pool_share_v368": float(
            row[f"v363_bounded_candidate_pool_share_v{VERSION}"]
        ),
        "v71_improving_omitted_columns_v368": int(
            row[f"v71_improving_omitted_columns_v{VERSION}"]
        ),
        "planned_chunk_count_v368": int(row[f"planned_chunk_count_v{VERSION}"]),
        "remaining_unpriced_chunks_v368": int(row[f"remaining_unpriced_chunks_v{VERSION}"]),
        "v366_ordered_one_swap_rows_v368": int(row[f"v366_ordered_one_swap_rows_v{VERSION}"]),
        "v366_source_exact_rows_v368": int(row[f"v366_source_exact_rows_v{VERSION}"]),
        "v366_cvar_feasible_entering_rows_v368": int(
            row[f"v366_cvar_feasible_entering_rows_v{VERSION}"]
        ),
        "allowed_publishable_claim_rows_v368": int(len(publishable_claims)),
        "prohibited_claim_rows_v368": int(len(prohibited_claims)),
        "evidence_trace_rows_v368": int(len(evidence_trace)),
        "claim_blocker_rows_v368": int(len(claim_blockers)),
        "claim_matrix_rows_v368": int(len(claim_matrix)),
        "valid_full_v55_dual_bound_certificate_v368": False,
        "full_universe_integer_optimality_claim_allowed_v368": False,
        "working_champion_claim_allowed_v368": False,
        "paper1_promotion_allowed_v368": False,
        "paper4_working_champion_changed_v368": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "scope_memo_artifact_v368": (
            "reports/paper_material/paper4/notes/"
            "paper4_v368_publishable_claim_scope_update.md"
        ),
        "next_artifact_v368": NEXT_ARTIFACT,
        "claim_boundary": (
            "v368 permits bounded living-lab publishable framing but blocks full solver, "
            "champion, deployment and promotion claims"
        ),
    }
    write_json(STATUS_DIR / "paper4_v368_status.json", status)
    SCOPE_MEMO.write_text(_scope_memo(status, evidence_trace), encoding="utf-8")
    _update_notebook(status)
    print(json.dumps({"v368": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
