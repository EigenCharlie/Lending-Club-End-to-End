#!/usr/bin/env python3
"""Build Paper 4 v362 v353 fourth-order disposition / bound memo."""

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

VERSION = 362
BASE_VERSION = 353
PRIOR_BRANCH_VERSION = 361
READINESS_VERSION = 356
DISPOSITION_VERSION = 360
NEXT_VERSION = 363
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_v353_full_dual_bound_or_gap_certificate.csv"


def _maybe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _update_claim_boundaries(*, no_entry: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v362 records the v361 fourth-order candidate disposition memo.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v362_v353_apply_fourth_order_candidate_or_bound_memo.csv"
                ),
                "boundary": (
                    "Disposition memo only; no candidate is applied when v361 has no entering row."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v362 allows no-apply disposition after v361 no-entry evidence.",
                "allowed": no_entry,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v362_candidate_disposition.csv"
                ),
                "boundary": "Allowed only for the bounded v361 fourth-order scope.",
                "prohibited_claim_flag": not no_entry,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v362 proves a valid full-universe branch-price bound.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v362_claim_blockers.csv"
                ),
                "boundary": "v361 was bounded fourth-order only; full-v55 termination remains missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v362 authorizes a Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v362_claim_blockers.csv"
                ),
                "boundary": "Proxy, global, dynamic, online and deployment gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v362 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v362_claim_blockers.csv"
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


def _update_backlog(*, no_entry: bool) -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Source Governance/Global",
                "executable_item": (
                    "v362 converts the v361 bounded fourth-order no-entry result "
                    "into a candidate-disposition and partial-bound memo."
                ),
                "status": (
                    "fourth_order_no_entry_disposition_memo_created"
                    if no_entry
                    else "fourth_order_entering_candidate_requires_apply_reprice"
                ),
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "produce a valid full-v55 dual-bound/gap certificate or explicitly "
                    "register why the remaining global proof remains infeasible"
                ),
                "last_wave": "v362",
                "execution_result": (
                    "no_apply_after_fourth_order_no_entry_partial_bound_blocker_recorded"
                    if no_entry
                    else "bounded_fourth_order_entering_candidate_ready_for_apply"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v362")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V362_V353_FOURTH_ORDER_CANDIDATE_DISPOSITION_MEMO_START -->"
    end = "<!-- V362_V353_FOURTH_ORDER_CANDIDATE_DISPOSITION_MEMO_END -->"
    block = f"""
{start}

## Wave v362: v353 Fourth-Order Disposition Memo

Generated: {status["generated_at_utc"]}

### Objective

v361 expanded the post-v353 branch-price evidence to a bounded fourth-order
loop and found no CVaR-feasible entering row. v362 records the disposition: no
fourth-order candidate is applied, and the result becomes a deeper partial
no-entry blocker rather than a global certificate.

### Results

- Prior branch version: `{status["prior_branch_version_v362"]}`.
- V361 three-swap seed rows: `{status["v361_three_swap_seed_rows_v362"]}`.
- V361 ordered rows screened:
  `{status["v361_ordered_fourth_order_rows_screened_v362"]}`.
- V361 source-exact rows:
  `{status["v361_source_exact_fourth_order_rows_v362"]}`.
- V361 CVaR-feasible entering rows:
  `{status["v361_cvar_feasible_entering_rows_v362"]}`.
- No-apply disposition allowed:
  `{status["no_apply_disposition_allowed_v362"]}`.
- Best source-exact return delta:
  `{status["best_source_exact_return_delta_v362"]}`.
- Best source-exact CVaR gap versus v353 cap:
  `{status["best_source_exact_cvar_gap_vs_cap_v362"]}`.
- Valid branch-price bound:
  `{status["valid_branch_price_bound_v362"]}`.

### Interpretation

v362 turns the 371.10M-row fourth-order screen into a clean paper artifact. The
bounded fourth-order path does not justify applying a new candidate, and it
still does not close the global proof obligation. The important negative result
is explicit: even the best tail-risk row in the v361 source-exact frontier
remains above the v353 CVaR cap.

### Claim Impact

- Allowed: no-apply disposition after v361 no-entry evidence.
- Allowed: bounded fourth-order no-entry blocker for the v353 candidate.
- Still prohibited: full-v55 branch-price termination, valid global integer
  optimality, contractual IFRS9, live deployability, Paper Estrella replacement,
  final Paper 4 promotion and working champion claims.

### Quarto Promotion Decision

Keep v362 in the living notebook. The next wave should target a full-v55
dual-bound/gap certificate or document the remaining proof infeasibility,
without promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v353_status = json.loads((STATUS_DIR / "paper4_v353_status.json").read_text(encoding="utf-8"))
    v361_status = json.loads((STATUS_DIR / "paper4_v361_status.json").read_text(encoding="utf-8"))
    v361_protocol = read_csv("paper4_v361_v353_fourth_order_or_full_dual_bound.csv")
    v361_stage = read_csv("paper4_v361_fourth_order_branch_price_stage_summary.csv")
    v361_candidates = read_csv("paper4_v361_branch_price_candidate_screen.csv")
    v361_entering = read_csv("paper4_v361_entering_candidate_summary.csv")
    if any(df.empty for df in [v361_protocol, v361_stage, v361_candidates]):
        raise RuntimeError("Missing v362 disposition memo inputs.")
    if not bool(v361_status["branch_price_loop_executed_v361"]):
        raise RuntimeError("v362 expects v361 branch-price loop execution evidence.")

    entering_rows = int(v361_status["cvar_feasible_entering_rows_v361"])
    no_entry = entering_rows == 0 and v361_entering.empty
    best_source_exact_cvar = _maybe_float(v361_status["best_source_exact_cvar90_v361"])
    v353_cvar_cap = float(v353_status["scenario_loss_cvar90_v353"])
    cvar_gap = None if best_source_exact_cvar is None else best_source_exact_cvar - v353_cvar_cap
    best_source_exact_return = _maybe_float(v361_status["best_source_exact_return_delta_v361"])
    stage_map = dict(zip(v361_stage["stage_v361"], v361_stage["row_count_v361"], strict=False))

    disposition = pd.DataFrame(
        [
            {
                f"disposition_id_v{VERSION}": "v361_no_entry_no_apply",
                f"prior_branch_version_v{VERSION}": PRIOR_BRANCH_VERSION,
                f"base_version_v{VERSION}": BASE_VERSION,
                f"entering_candidate_rows_v{VERSION}": entering_rows,
                f"no_apply_disposition_allowed_v{VERSION}": no_entry,
                f"candidate_applied_v{VERSION}": False,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "no fourth-order branch-price candidate applied because v361 has zero "
                    "bounded fourth-order entering rows"
                    if no_entry
                    else "bounded fourth-order entering candidate requires apply/reprice before any claim expansion"
                ),
            }
        ]
    )
    scope = pd.DataFrame(
        [
            {
                f"scope_id_v{VERSION}": "v361_fourth_order_scope_covered",
                f"covered_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    v361_status["ordered_fourth_order_rows_screened_v361"]
                ),
                f"evidence_artifact_v{VERSION}": (
                    "paper4_v361_fourth_order_branch_price_stage_summary.csv"
                ),
                f"claim_boundary_v{VERSION}": "bounded v359-seeded fourth-order screen only",
            },
            {
                f"scope_id_v{VERSION}": "v361_source_exact_frontier_covered",
                f"covered_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    v361_status["source_exact_fourth_order_rows_v361"]
                ),
                f"evidence_artifact_v{VERSION}": "paper4_v361_branch_price_candidate_screen.csv",
                f"claim_boundary_v{VERSION}": "4,631 source-exact rows, zero entering rows",
            },
            {
                f"scope_id_v{VERSION}": "full_v55_dual_bound_covered",
                f"covered_v{VERSION}": False,
                f"evidence_count_v{VERSION}": 0,
                f"evidence_artifact_v{VERSION}": "missing",
                f"claim_boundary_v{VERSION}": "no full-v55 branch-price termination certificate exists",
            },
            {
                f"scope_id_v{VERSION}": "full_integer_or_fifth_order_scope_covered",
                f"covered_v{VERSION}": False,
                f"evidence_count_v{VERSION}": 0,
                f"evidence_artifact_v{VERSION}": "missing",
                f"claim_boundary_v{VERSION}": (
                    "fourth-order no-entry is still not a full integer optimality proof"
                ),
            },
            {
                f"scope_id_v{VERSION}": "proxy_dynamic_online_gates_covered",
                f"covered_v{VERSION}": False,
                f"evidence_count_v{VERSION}": int(v353_status["missing_proxy_rows_v353"]),
                f"evidence_artifact_v{VERSION}": "paper4_v355_proxy_repair_tier_summary.csv",
                f"claim_boundary_v{VERSION}": "proxy gap and live gates remain open",
            },
        ]
    )
    memo = pd.DataFrame(
        [
            {
                f"memo_id_v{VERSION}": "v353_post_v361_no_entry_partial_bound_memo",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"prior_branch_version_v{VERSION}": PRIOR_BRANCH_VERSION,
                f"readiness_version_v{VERSION}": READINESS_VERSION,
                f"disposition_version_v{VERSION}": DISPOSITION_VERSION,
                f"v361_three_swap_seed_rows_v{VERSION}": int(
                    v361_status["three_swap_seed_rows_v361"]
                ),
                f"v361_positive_source_tight_candidate_rows_v{VERSION}": int(
                    v361_status["positive_source_tight_candidate_rows_v361"]
                ),
                f"v361_ordered_fourth_order_rows_screened_v{VERSION}": int(
                    v361_status["ordered_fourth_order_rows_screened_v361"]
                ),
                f"v361_return_improving_rows_v{VERSION}": int(
                    v361_status["return_improving_rows_v361"]
                ),
                f"v361_budget_return_feasible_rows_v{VERSION}": int(
                    v361_status["budget_return_feasible_rows_v361"]
                ),
                f"v361_source_exact_fourth_order_rows_v{VERSION}": int(
                    v361_status["source_exact_fourth_order_rows_v361"]
                ),
                f"v361_unique_source_exact_action_signatures_v{VERSION}": int(
                    v361_status["unique_source_exact_action_signatures_v361"]
                ),
                f"v361_cvar_feasible_entering_rows_v{VERSION}": entering_rows,
                f"no_apply_disposition_allowed_v{VERSION}": no_entry,
                f"best_source_exact_return_delta_v{VERSION}": best_source_exact_return,
                f"best_source_exact_cvar90_v{VERSION}": best_source_exact_cvar,
                f"v353_cvar_cap_v{VERSION}": v353_cvar_cap,
                f"best_source_exact_cvar_gap_vs_cap_v{VERSION}": cvar_gap,
                f"fourth_order_scope_covered_v{VERSION}": True,
                f"full_v55_dual_bound_covered_v{VERSION}": False,
                f"valid_branch_price_bound_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "fourth-order candidate disposition memo only; no full-v55 dual-bound certificate"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "no_apply_fourth_order_candidate_available",
                f"blocking_v{VERSION}": no_entry,
                f"evidence_count_v{VERSION}": entering_rows,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "zero bounded v361 entering rows",
            },
            {
                f"blocker_id_v{VERSION}": "best_source_exact_cvar_above_cap",
                f"blocking_v{VERSION}": cvar_gap is not None and cvar_gap > 0,
                f"evidence_count_v{VERSION}": int(round(cvar_gap or 0)),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "best fourth-order source-exact row still violates v353 CVaR cap",
            },
            {
                f"blocker_id_v{VERSION}": "full_v55_dual_bound_or_gap_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "bounded fourth-order no-entry is not full branch-price termination",
            },
            {
                f"blocker_id_v{VERSION}": "proxy_gap_persists",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v353_status["missing_proxy_rows_v353"]),
                f"required_next_artifact_v{VERSION}": "future_proxy_or_ifrs9_gate",
                f"claim_boundary_v{VERSION}": "v353 candidate still contains missing proxy rows",
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
                "claim_id": "v362_fourth_order_disposition_memo_created",
                "allowed": True,
                "artifact": "paper4_v362_v353_apply_fourth_order_candidate_or_bound_memo.csv",
                "boundary": "memo only",
            },
            {
                "claim_id": "v362_no_apply_after_v361_no_entry",
                "allowed": no_entry,
                "artifact": "paper4_v362_candidate_disposition.csv",
                "boundary": "bounded v361 scope only",
            },
            {
                "claim_id": "v362_valid_full_universe_branch_price_bound",
                "allowed": False,
                "artifact": "paper4_v362_claim_blockers.csv",
                "boundary": "full-v55 dual-bound/gap certificate missing",
            },
            {
                "claim_id": "v362_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v362_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(
        TABLE_DIR / "paper4_v362_v353_apply_fourth_order_candidate_or_bound_memo.csv",
        memo,
    )
    write_csv(TABLE_DIR / "paper4_v362_candidate_disposition.csv", disposition)
    write_csv(TABLE_DIR / "paper4_v362_no_entry_scope_register.csv", scope)
    write_csv(TABLE_DIR / "paper4_v362_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v362_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries(no_entry=no_entry)
    _update_backlog(no_entry=no_entry)

    row = memo.iloc[0]
    status = {
        "phase": "v362_v353_apply_fourth_order_candidate_or_bound_memo",
        "schema_version": "2026-05-17.362",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v362": BASE_VERSION,
        "prior_branch_version_v362": PRIOR_BRANCH_VERSION,
        "readiness_version_v362": READINESS_VERSION,
        "disposition_version_v362": DISPOSITION_VERSION,
        "v361_three_swap_seed_rows_v362": int(row[f"v361_three_swap_seed_rows_v{VERSION}"]),
        "v361_ordered_fourth_order_rows_screened_v362": int(
            row[f"v361_ordered_fourth_order_rows_screened_v{VERSION}"]
        ),
        "v361_source_exact_fourth_order_rows_v362": int(
            row[f"v361_source_exact_fourth_order_rows_v{VERSION}"]
        ),
        "v361_cvar_feasible_entering_rows_v362": int(
            row[f"v361_cvar_feasible_entering_rows_v{VERSION}"]
        ),
        "no_apply_disposition_allowed_v362": bool(row[f"no_apply_disposition_allowed_v{VERSION}"]),
        "candidate_applied_v362": False,
        "best_source_exact_return_delta_v362": _maybe_float(
            row[f"best_source_exact_return_delta_v{VERSION}"]
        ),
        "best_source_exact_cvar90_v362": _maybe_float(row[f"best_source_exact_cvar90_v{VERSION}"]),
        "v353_cvar_cap_v362": float(row[f"v353_cvar_cap_v{VERSION}"]),
        "best_source_exact_cvar_gap_vs_cap_v362": _maybe_float(
            row[f"best_source_exact_cvar_gap_vs_cap_v{VERSION}"]
        ),
        "fourth_order_scope_covered_v362": True,
        "full_v55_dual_bound_covered_v362": False,
        "valid_branch_price_bound_v362": False,
        "full_universe_integer_optimality_claim_allowed_v362": False,
        "working_champion_claim_allowed_v362": False,
        "paper1_promotion_allowed_v362": False,
        "paper4_working_champion_changed_v362": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v362": int(len(blockers)),
        "claim_matrix_rows_v362": int(len(claim_matrix)),
        "scope_register_rows_v362": int(len(scope)),
        "stage_map_snapshot_v362": {str(k): int(v) for k, v in stage_map.items()},
        "next_artifact_v362": NEXT_ARTIFACT,
        "claim_boundary": (
            "v362 records no-apply disposition after bounded v361 fourth-order no-entry; "
            "full dual-bound, proxy, live, champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v362_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v362": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
