#!/usr/bin/env python3
"""Build Paper 4 v367 route decision after the v366 chunk probe."""

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

VERSION = 367
BASE_VERSION = 353
PRIOR_CHUNK_VERSION = 366
PRIOR_PLAN_VERSION = 365
PRIOR_GAP_VERSION = 363
NEXT_VERSION = 368
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_publishable_claim_scope_update.md"
RECOMMENDED_ROUTE = "bounded_claim_scope_update"


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v367 records a Paper 4 route decision after the v366 chunk probe.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v367_route_decision_after_chunk_probe.csv"
                ),
                "boundary": "Route decision only; it narrows future execution and claims.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v367 recommends bounded publishable scope over immediate full-v55 proof.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v367_route_option_scorecard.csv"
                ),
                "boundary": "Recommendation only; no final paper promotion or champion.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v367 proves full-v55 reduced-cost termination.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v367_claim_blockers.csv"
                ),
                "boundary": "v366 priced only chunk 1 and found zero source-exact rows.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v367 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v367_claim_blockers.csv"
                ),
                "boundary": "No final promotion, working champion or deployment gate is created.",
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
                    "v367 converts the v366 chunk probe into a route decision: "
                    "update publishable bounded scope before more solver expansion."
                ),
                "status": "route_decision_recommends_bounded_claim_scope_update",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v368 states the strongest true Paper 4 claim after v361-v367 "
                    "without global optimality or promotion"
                ),
                "last_wave": "v367",
                "execution_result": "bounded_claim_scope_route_selected_after_chunk_source_blocker",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v367")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V367_ROUTE_DECISION_AFTER_CHUNK_PROBE_START -->"
    end = "<!-- V367_ROUTE_DECISION_AFTER_CHUNK_PROBE_END -->"
    block = f"""
{start}

## Wave v367: Route Decision After Chunk Probe

Generated: {status["generated_at_utc"]}

### Objective

v366 executed the first full-v55 pricing chunk prototype and found zero
source-exact one-swap rows. v367 decides whether the lab should keep spending
the next wave on full-v55 chunks, switch to bounded fifth-order search, or
narrow the publishable claim.

### Results

- Recommended route:
  `{status["recommended_route_v367"]}`.
- V366 ordered one-swap rows:
  `{status["v366_ordered_one_swap_rows_v367"]}`.
- V366 source-exact rows:
  `{status["v366_source_exact_rows_v367"]}`.
- V366 CVaR-feasible entering rows:
  `{status["v366_cvar_feasible_entering_rows_v367"]}`.
- Remaining v365 chunks:
  `{status["remaining_unpriced_chunks_v367"]}`.
- Continue full-v55 chunks immediately:
  `{status["continue_full_v55_chunks_immediately_v367"]}`.
- Switch immediately to fifth-order bounded search:
  `{status["switch_to_fifth_order_bounded_search_v367"]}`.
- Next artifact:
  `{status["next_artifact_v367"]}`.

### Interpretation

v367 selects a conservative but useful route: update the publishable claim scope
before spending more compute. The first full-v55 chunk was not merely CVaR-
blocked; it was source-governance blocked before any source-exact row survived.
That makes immediate full-v55 chunk continuation less informative than a clear
paper-scope update.

### Claim Impact

- Allowed: route decision after measured chunk evidence.
- Still prohibited: full-v55 termination, valid global integer optimality,
  working champion, Paper Estrella replacement and final promotion.

### Quarto Promotion Decision

Keep v367 in the living notebook. v368 should write the strongest true
publishable claim after v361-v367.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v363_status = json.loads((STATUS_DIR / "paper4_v363_status.json").read_text(encoding="utf-8"))
    v365_status = json.loads((STATUS_DIR / "paper4_v365_status.json").read_text(encoding="utf-8"))
    v366_status = json.loads((STATUS_DIR / "paper4_v366_status.json").read_text(encoding="utf-8"))
    v366_stage = read_csv("paper4_v366_chunk_stage_summary.csv")
    if v366_stage.empty:
        raise RuntimeError("Missing v367 route-decision inputs.")
    if not bool(v366_status["chunk_prototype_executed_v366"]):
        raise RuntimeError("v367 expects v366 chunk prototype evidence.")

    remaining_chunks = int(v365_status["planned_chunk_count_v365"]) - int(v366_status["chunk_id_v366"])
    continue_chunks = False
    switch_fifth_order = False
    option_scorecard = pd.DataFrame(
        [
            {
                f"route_option_v{VERSION}": "continue_full_v55_chunking",
                f"recommended_v{VERSION}": continue_chunks,
                f"evidence_for_v{VERSION}": "v365 plan is executable and resumable",
                f"evidence_against_v{VERSION}": (
                    "v366 chunk 1 produced zero source-exact rows after 1.71M one-swaps"
                ),
                f"next_artifact_if_chosen_v{VERSION}": "paper4_v368_full_v55_chunk_002_probe.csv",
                f"claim_boundary_v{VERSION}": "would remain prototype evidence only",
            },
            {
                f"route_option_v{VERSION}": "switch_to_fifth_order_bounded_search",
                f"recommended_v{VERSION}": switch_fifth_order,
                f"evidence_for_v{VERSION}": "bounded fourth-order no-entry is already documented",
                f"evidence_against_v{VERSION}": (
                    "fifth-order expansion is likely expensive and still not full-v55 termination"
                ),
                f"next_artifact_if_chosen_v{VERSION}": "paper4_v368_fifth_order_resource_gate.csv",
                f"claim_boundary_v{VERSION}": "bounded-depth evidence only",
            },
            {
                f"route_option_v{VERSION}": RECOMMENDED_ROUTE,
                f"recommended_v{VERSION}": True,
                f"evidence_for_v{VERSION}": (
                    "v361-v366 give strong bounded no-entry/source-blocker evidence but no global proof"
                ),
                f"evidence_against_v{VERSION}": "does not produce a new solver certificate",
                f"next_artifact_if_chosen_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "publishable claim scope only; no promotion",
            },
            {
                f"route_option_v{VERSION}": "proxy_live_gate_separation",
                f"recommended_v{VERSION}": False,
                f"evidence_for_v{VERSION}": "proxy/live blockers remain active",
                f"evidence_against_v{VERSION}": "should follow the publishable scope update",
                f"next_artifact_if_chosen_v{VERSION}": "paper4_v369_proxy_live_gate_separation.csv",
                f"claim_boundary_v{VERSION}": "governance separation only",
            },
        ]
    )
    decision = pd.DataFrame(
        [
            {
                f"decision_id_v{VERSION}": "v367_route_decision_after_chunk_probe",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"prior_chunk_version_v{VERSION}": PRIOR_CHUNK_VERSION,
                f"prior_plan_version_v{VERSION}": PRIOR_PLAN_VERSION,
                f"prior_gap_version_v{VERSION}": PRIOR_GAP_VERSION,
                f"recommended_route_v{VERSION}": RECOMMENDED_ROUTE,
                f"v366_ordered_one_swap_rows_v{VERSION}": int(
                    v366_status["ordered_one_swap_rows_v366"]
                ),
                f"v366_return_improving_rows_v{VERSION}": int(
                    v366_status["return_improving_rows_v366"]
                ),
                f"v366_budget_return_feasible_rows_v{VERSION}": int(
                    v366_status["budget_return_feasible_rows_v366"]
                ),
                f"v366_source_exact_rows_v{VERSION}": int(v366_status["source_exact_rows_v366"]),
                f"v366_cvar_feasible_entering_rows_v{VERSION}": int(
                    v366_status["cvar_feasible_entering_rows_v366"]
                ),
                f"remaining_unpriced_chunks_v{VERSION}": remaining_chunks,
                f"v71_improving_omitted_columns_v{VERSION}": int(
                    v363_status["v71_improving_omitted_columns_v363"]
                ),
                f"continue_full_v55_chunks_immediately_v{VERSION}": continue_chunks,
                f"switch_to_fifth_order_bounded_search_v{VERSION}": switch_fifth_order,
                f"update_publishable_claim_scope_v{VERSION}": True,
                f"valid_full_v55_dual_bound_certificate_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "route decision only; no full-v55 termination or champion claim"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "full_v55_termination_still_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": remaining_chunks,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "remaining chunks are unpriced",
            },
            {
                f"blocker_id_v{VERSION}": "chunk_0001_source_exact_rows_zero",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v366_status["source_exact_rows_v366"]),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "source governance blocked the first chunk",
            },
            {
                f"blocker_id_v{VERSION}": "paper4_final_promotion_forbidden",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "paper4_final_promotion_gate_not_created",
                f"claim_boundary_v{VERSION}": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v367_route_decision_created",
                "allowed": True,
                "artifact": "paper4_v367_route_decision_after_chunk_probe.csv",
                "boundary": "route decision only",
            },
            {
                "claim_id": "v367_bounded_claim_scope_route_recommended",
                "allowed": True,
                "artifact": "paper4_v367_route_option_scorecard.csv",
                "boundary": "recommendation only",
            },
            {
                "claim_id": "v367_valid_full_v55_dual_bound_certificate",
                "allowed": False,
                "artifact": "paper4_v367_claim_blockers.csv",
                "boundary": "termination proof missing",
            },
            {
                "claim_id": "v367_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v367_claim_blockers.csv",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v367_route_decision_after_chunk_probe.csv", decision)
    write_csv(TABLE_DIR / "paper4_v367_route_option_scorecard.csv", option_scorecard)
    write_csv(TABLE_DIR / "paper4_v367_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v367_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = decision.iloc[0]
    status = {
        "phase": "v367_route_decision_after_chunk_probe",
        "schema_version": "2026-05-17.367",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v367": BASE_VERSION,
        "prior_chunk_version_v367": PRIOR_CHUNK_VERSION,
        "prior_plan_version_v367": PRIOR_PLAN_VERSION,
        "prior_gap_version_v367": PRIOR_GAP_VERSION,
        "recommended_route_v367": str(row[f"recommended_route_v{VERSION}"]),
        "v366_ordered_one_swap_rows_v367": int(row[f"v366_ordered_one_swap_rows_v{VERSION}"]),
        "v366_return_improving_rows_v367": int(row[f"v366_return_improving_rows_v{VERSION}"]),
        "v366_budget_return_feasible_rows_v367": int(
            row[f"v366_budget_return_feasible_rows_v{VERSION}"]
        ),
        "v366_source_exact_rows_v367": int(row[f"v366_source_exact_rows_v{VERSION}"]),
        "v366_cvar_feasible_entering_rows_v367": int(
            row[f"v366_cvar_feasible_entering_rows_v{VERSION}"]
        ),
        "remaining_unpriced_chunks_v367": int(row[f"remaining_unpriced_chunks_v{VERSION}"]),
        "v71_improving_omitted_columns_v367": int(
            row[f"v71_improving_omitted_columns_v{VERSION}"]
        ),
        "continue_full_v55_chunks_immediately_v367": bool(
            row[f"continue_full_v55_chunks_immediately_v{VERSION}"]
        ),
        "switch_to_fifth_order_bounded_search_v367": bool(
            row[f"switch_to_fifth_order_bounded_search_v{VERSION}"]
        ),
        "update_publishable_claim_scope_v367": bool(
            row[f"update_publishable_claim_scope_v{VERSION}"]
        ),
        "valid_full_v55_dual_bound_certificate_v367": False,
        "full_universe_integer_optimality_claim_allowed_v367": False,
        "working_champion_claim_allowed_v367": False,
        "paper1_promotion_allowed_v367": False,
        "paper4_working_champion_changed_v367": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "route_option_rows_v367": int(len(option_scorecard)),
        "claim_blocker_rows_v367": int(len(blockers)),
        "claim_matrix_rows_v367": int(len(claim_matrix)),
        "next_artifact_v367": NEXT_ARTIFACT,
        "claim_boundary": (
            "v367 recommends bounded publishable scope update; full dual-bound, "
            "champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v367_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v367": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
