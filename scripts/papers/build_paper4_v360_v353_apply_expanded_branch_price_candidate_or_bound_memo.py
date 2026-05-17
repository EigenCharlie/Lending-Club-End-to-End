#!/usr/bin/env python3
"""Build Paper 4 v360 v353 expanded branch-price disposition / bound memo."""

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

VERSION = 360
BASE_VERSION = 353
PRIOR_BRANCH_VERSION = 359
READINESS_VERSION = 356
DISPOSITION_VERSION = 358
NEXT_VERSION = 361
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_v353_fourth_order_or_full_dual_bound.csv"


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
                "claim": "v360 records the v359 post-v353 expanded candidate disposition memo.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v360_v353_apply_expanded_branch_price_candidate_or_bound_memo.csv"
                ),
                "boundary": "Disposition memo only; no candidate is applied when v359 has no entering row.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v360 allows no-apply disposition after v359 no-entry evidence.",
                "allowed": no_entry,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v360_candidate_disposition.csv"
                ),
                "boundary": "Allowed only for the bounded v359 third-order scope.",
                "prohibited_claim_flag": not no_entry,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v360 proves a valid full-universe branch-price bound.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v360_claim_blockers.csv"
                ),
                "boundary": "v359 was bounded third-order only; full-v55 termination remains missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v360 authorizes a Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v360_claim_blockers.csv"
                ),
                "boundary": "Proxy, global, dynamic, online and deployment gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v360 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v360_claim_blockers.csv"
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
                    "v360 converts the v359 bounded post-v353 third-order no-entry "
                    "result into a candidate-disposition and partial-bound memo."
                ),
                "status": (
                    "expanded_no_entry_disposition_memo_created"
                    if no_entry
                    else "expanded_entering_candidate_requires_apply_reprice"
                ),
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "expand branch-price depth again or produce a valid full-v55 "
                    "dual-bound certificate without promotion"
                ),
                "last_wave": "v360",
                "execution_result": (
                    "no_apply_after_third_order_no_entry_partial_bound_blocker_recorded"
                    if no_entry
                    else "bounded_expanded_entering_candidate_ready_for_apply"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v360")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V360_V353_EXPANDED_CANDIDATE_DISPOSITION_MEMO_START -->"
    end = "<!-- V360_V353_EXPANDED_CANDIDATE_DISPOSITION_MEMO_END -->"
    block = f"""
{start}

## Wave v360: v353 Expanded Branch-Price Disposition Memo

Generated: {status["generated_at_utc"]}

### Objective

v359 expanded the post-v353 branch-price evidence to a bounded third-order loop
and still found no CVaR-feasible entering row. v360 records the candidate
disposition: no expanded branch-price candidate is applied, and the result
becomes a stronger partial no-entry blocker rather than a global certificate.

### Results

- Prior branch version: `{status["prior_branch_version_v360"]}`.
- V359 two-swap seed rows: `{status["v359_two_swap_seed_rows_v360"]}`.
- V359 ordered rows screened:
  `{status["v359_ordered_third_order_rows_screened_v360"]}`.
- V359 source-exact rows:
  `{status["v359_source_exact_third_order_rows_v360"]}`.
- V359 CVaR-feasible entering rows:
  `{status["v359_cvar_feasible_entering_rows_v360"]}`.
- No-apply disposition allowed:
  `{status["no_apply_disposition_allowed_v360"]}`.
- Best source-exact return delta:
  `{status["best_source_exact_return_delta_v360"]}`.
- Best source-exact CVaR gap versus v353 cap:
  `{status["best_source_exact_cvar_gap_vs_cap_v360"]}`.
- Valid branch-price bound:
  `{status["valid_branch_price_bound_v360"]}`.

### Interpretation

v360 turns the v359 no-entry evidence into a clean paper artifact. The bounded
third-order branch-price path does not justify applying a new candidate, and it
still does not close the global proof obligation. The useful frontier is now
sharper: even after 62.97M ordered third-order rows and 843 source-exact rows,
the best tail-risk row remains above the v353 CVaR cap.

### Claim Impact

- Allowed: no-apply disposition after v359 no-entry evidence.
- Allowed: bounded third-order no-entry blocker for the v353 candidate.
- Still prohibited: full-v55 branch-price termination, valid global integer
  optimality, contractual IFRS9, live deployability, Paper Estrella replacement,
  final Paper 4 promotion and working champion claims.

### Quarto Promotion Decision

Keep v360 in the living notebook. The next wave should either expand branch-
price depth again or produce a valid full-v55 dual-bound certificate, without
promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v353_status = json.loads((STATUS_DIR / "paper4_v353_status.json").read_text(encoding="utf-8"))
    v359_status = json.loads((STATUS_DIR / "paper4_v359_status.json").read_text(encoding="utf-8"))
    v359_protocol = read_csv("paper4_v359_v353_expand_branch_price_or_full_dual_bound.csv")
    v359_stage = read_csv("paper4_v359_third_order_branch_price_stage_summary.csv")
    v359_candidates = read_csv("paper4_v359_branch_price_candidate_screen.csv")
    v359_entering = read_csv("paper4_v359_entering_candidate_summary.csv")
    if any(df.empty for df in [v359_protocol, v359_stage, v359_candidates]):
        raise RuntimeError("Missing v360 disposition memo inputs.")
    if not bool(v359_status["branch_price_loop_executed_v359"]):
        raise RuntimeError("v360 expects v359 branch-price loop execution evidence.")

    entering_rows = int(v359_status["cvar_feasible_entering_rows_v359"])
    no_entry = entering_rows == 0 and v359_entering.empty
    best_source_exact_cvar = _maybe_float(v359_status["best_source_exact_cvar90_v359"])
    v353_cvar_cap = float(v353_status["scenario_loss_cvar90_v353"])
    cvar_gap = None if best_source_exact_cvar is None else best_source_exact_cvar - v353_cvar_cap
    best_source_exact_return = _maybe_float(v359_status["best_source_exact_return_delta_v359"])
    stage_map = dict(zip(v359_stage["stage_v359"], v359_stage["row_count_v359"], strict=False))

    disposition = pd.DataFrame(
        [
            {
                f"disposition_id_v{VERSION}": "v359_no_entry_no_apply",
                f"prior_branch_version_v{VERSION}": PRIOR_BRANCH_VERSION,
                f"base_version_v{VERSION}": BASE_VERSION,
                f"entering_candidate_rows_v{VERSION}": entering_rows,
                f"no_apply_disposition_allowed_v{VERSION}": no_entry,
                f"candidate_applied_v{VERSION}": False,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "no expanded branch-price candidate applied because v359 has zero "
                    "bounded third-order entering rows"
                    if no_entry
                    else "bounded entering candidate requires apply/reprice before any claim expansion"
                ),
            }
        ]
    )
    scope = pd.DataFrame(
        [
            {
                f"scope_id_v{VERSION}": "v359_third_order_scope_covered",
                f"covered_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    v359_status["ordered_third_order_rows_screened_v359"]
                ),
                f"evidence_artifact_v{VERSION}": (
                    "paper4_v359_third_order_branch_price_stage_summary.csv"
                ),
                f"claim_boundary_v{VERSION}": "bounded v357-seeded third-order screen only",
            },
            {
                f"scope_id_v{VERSION}": "v359_source_exact_frontier_covered",
                f"covered_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    v359_status["source_exact_third_order_rows_v359"]
                ),
                f"evidence_artifact_v{VERSION}": "paper4_v359_branch_price_candidate_screen.csv",
                f"claim_boundary_v{VERSION}": "843 source-exact rows, zero entering rows",
            },
            {
                f"scope_id_v{VERSION}": "fourth_order_expansion_covered",
                f"covered_v{VERSION}": False,
                f"evidence_count_v{VERSION}": 0,
                f"evidence_artifact_v{VERSION}": "missing",
                f"claim_boundary_v{VERSION}": "fourth-order or deeper branch-price expansion remains open",
            },
            {
                f"scope_id_v{VERSION}": "full_v55_dual_bound_covered",
                f"covered_v{VERSION}": False,
                f"evidence_count_v{VERSION}": 0,
                f"evidence_artifact_v{VERSION}": "missing",
                f"claim_boundary_v{VERSION}": "no full-v55 branch-price termination certificate exists",
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
                f"memo_id_v{VERSION}": "v353_post_v359_no_entry_partial_bound_memo",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"prior_branch_version_v{VERSION}": PRIOR_BRANCH_VERSION,
                f"readiness_version_v{VERSION}": READINESS_VERSION,
                f"disposition_version_v{VERSION}": DISPOSITION_VERSION,
                f"v359_two_swap_seed_rows_v{VERSION}": int(v359_status["two_swap_seed_rows_v359"]),
                f"v359_positive_source_tight_candidate_rows_v{VERSION}": int(
                    v359_status["positive_source_tight_candidate_rows_v359"]
                ),
                f"v359_ordered_third_order_rows_screened_v{VERSION}": int(
                    v359_status["ordered_third_order_rows_screened_v359"]
                ),
                f"v359_return_improving_rows_v{VERSION}": int(
                    v359_status["return_improving_rows_v359"]
                ),
                f"v359_budget_return_feasible_rows_v{VERSION}": int(
                    v359_status["budget_return_feasible_rows_v359"]
                ),
                f"v359_source_exact_third_order_rows_v{VERSION}": int(
                    v359_status["source_exact_third_order_rows_v359"]
                ),
                f"v359_unique_source_exact_action_signatures_v{VERSION}": int(
                    v359_status["unique_source_exact_action_signatures_v359"]
                ),
                f"v359_cvar_feasible_entering_rows_v{VERSION}": entering_rows,
                f"no_apply_disposition_allowed_v{VERSION}": no_entry,
                f"best_source_exact_return_delta_v{VERSION}": best_source_exact_return,
                f"best_source_exact_cvar90_v{VERSION}": best_source_exact_cvar,
                f"v353_cvar_cap_v{VERSION}": v353_cvar_cap,
                f"best_source_exact_cvar_gap_vs_cap_v{VERSION}": cvar_gap,
                f"third_order_scope_covered_v{VERSION}": True,
                f"fourth_order_scope_covered_v{VERSION}": False,
                f"valid_branch_price_bound_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "expanded candidate disposition memo only; no full-v55 dual-bound certificate"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "no_apply_expanded_candidate_available",
                f"blocking_v{VERSION}": no_entry,
                f"evidence_count_v{VERSION}": entering_rows,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "zero bounded v359 entering rows",
            },
            {
                f"blocker_id_v{VERSION}": "best_source_exact_cvar_above_cap",
                f"blocking_v{VERSION}": cvar_gap is not None and cvar_gap > 0,
                f"evidence_count_v{VERSION}": int(round(cvar_gap or 0)),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "best third-order source-exact row still violates v353 CVaR cap",
            },
            {
                f"blocker_id_v{VERSION}": "fourth_order_or_full_dual_bound_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "third-order no-entry is not full branch-price termination",
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
                "claim_id": "v360_expanded_candidate_disposition_memo_created",
                "allowed": True,
                "artifact": "paper4_v360_v353_apply_expanded_branch_price_candidate_or_bound_memo.csv",
                "boundary": "memo only",
            },
            {
                "claim_id": "v360_no_apply_after_v359_no_entry",
                "allowed": no_entry,
                "artifact": "paper4_v360_candidate_disposition.csv",
                "boundary": "bounded v359 scope only",
            },
            {
                "claim_id": "v360_valid_full_universe_branch_price_bound",
                "allowed": False,
                "artifact": "paper4_v360_claim_blockers.csv",
                "boundary": "fourth-order/full-v55 dual-bound missing",
            },
            {
                "claim_id": "v360_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v360_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(
        TABLE_DIR / "paper4_v360_v353_apply_expanded_branch_price_candidate_or_bound_memo.csv",
        memo,
    )
    write_csv(TABLE_DIR / "paper4_v360_candidate_disposition.csv", disposition)
    write_csv(TABLE_DIR / "paper4_v360_no_entry_scope_register.csv", scope)
    write_csv(TABLE_DIR / "paper4_v360_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v360_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries(no_entry=no_entry)
    _update_backlog(no_entry=no_entry)

    row = memo.iloc[0]
    status = {
        "phase": "v360_v353_apply_expanded_branch_price_candidate_or_bound_memo",
        "schema_version": "2026-05-17.360",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v360": BASE_VERSION,
        "prior_branch_version_v360": PRIOR_BRANCH_VERSION,
        "readiness_version_v360": READINESS_VERSION,
        "disposition_version_v360": DISPOSITION_VERSION,
        "v359_two_swap_seed_rows_v360": int(row[f"v359_two_swap_seed_rows_v{VERSION}"]),
        "v359_ordered_third_order_rows_screened_v360": int(
            row[f"v359_ordered_third_order_rows_screened_v{VERSION}"]
        ),
        "v359_source_exact_third_order_rows_v360": int(
            row[f"v359_source_exact_third_order_rows_v{VERSION}"]
        ),
        "v359_cvar_feasible_entering_rows_v360": int(
            row[f"v359_cvar_feasible_entering_rows_v{VERSION}"]
        ),
        "no_apply_disposition_allowed_v360": bool(row[f"no_apply_disposition_allowed_v{VERSION}"]),
        "candidate_applied_v360": False,
        "best_source_exact_return_delta_v360": _maybe_float(
            row[f"best_source_exact_return_delta_v{VERSION}"]
        ),
        "best_source_exact_cvar90_v360": _maybe_float(row[f"best_source_exact_cvar90_v{VERSION}"]),
        "v353_cvar_cap_v360": float(row[f"v353_cvar_cap_v{VERSION}"]),
        "best_source_exact_cvar_gap_vs_cap_v360": _maybe_float(
            row[f"best_source_exact_cvar_gap_vs_cap_v{VERSION}"]
        ),
        "third_order_scope_covered_v360": True,
        "fourth_order_scope_covered_v360": False,
        "valid_branch_price_bound_v360": False,
        "full_universe_integer_optimality_claim_allowed_v360": False,
        "working_champion_claim_allowed_v360": False,
        "paper1_promotion_allowed_v360": False,
        "paper4_working_champion_changed_v360": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v360": int(len(blockers)),
        "claim_matrix_rows_v360": int(len(claim_matrix)),
        "scope_register_rows_v360": int(len(scope)),
        "stage_map_snapshot_v360": {str(k): int(v) for k, v in stage_map.items()},
        "next_artifact_v360": NEXT_ARTIFACT,
        "claim_boundary": (
            "v360 records no-apply disposition after bounded v359 no-entry; full dual-bound, "
            "proxy, live, champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v360_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v360": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
