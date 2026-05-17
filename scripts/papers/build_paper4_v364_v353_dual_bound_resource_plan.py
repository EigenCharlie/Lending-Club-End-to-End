#!/usr/bin/env python3
"""Build Paper 4 v364 v353 dual-bound resource plan and goal prompt."""

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

VERSION = 364
BASE_VERSION = 353
PRIOR_GAP_VERSION = 363
NEXT_VERSION = 365
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_v353_full_v55_pricing_chunk_plan.csv"
GOAL_PROMPT = NOTEBOOK.parent / "paper4_v364_goal_prompt.md"


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v364 records executable Paper 4 dual-bound next steps.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v364_v353_dual_bound_resource_plan.csv"
                ),
                "boundary": "Planning and execution queue only; no global optimality claim.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v364 provides a reusable goal prompt for continued Paper 4 iteration.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/notes/paper4_v364_goal_prompt.md",
                "boundary": "Prompt preserves no-promotion, no-working-champion and bounded-claim rules.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v364 proves a valid full-v55 dual-bound termination.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v364_claim_blockers.csv"
                ),
                "boundary": "v364 is a resource plan after v363 gap certificate, not a solver proof.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v364 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v364_claim_blockers.csv"
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
                "lane": "Source Governance/Global",
                "executable_item": (
                    "v364 materializes the post-v363 executable pending register and "
                    "goal prompt for the full-v55 dual-bound route."
                ),
                "status": "dual_bound_resource_plan_and_goal_prompt_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v365 estimates chunk sizes and starts a reproducible full-v55 pricing plan "
                    "without promotion"
                ),
                "last_wave": "v364",
                "execution_result": "pending_register_goal_prompt_and_resource_plan_recorded",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v364")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _goal_prompt(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Living-Lab Goal Prompt v364

Goal: continue Paper 4 living-lab execution from v364 without time limit, advancing the executable backlog while preserving all claim boundaries.

Current hard facts:
- Latest committed wave: v364 resource plan, after v363 negative full-dual-bound gap certificate.
- v361 screened 371100576 bounded fourth-order rows, with 4631 source-exact rows and 0 CVaR-feasible entering rows.
- v362 recorded no-apply disposition; best source-exact CVaR gap vs v353 cap is {status["best_source_exact_cvar_gap_vs_cap_v364"]}.
- v363 showed the bounded source-tight pool covers {status["bounded_candidate_pool_share_v364"]} of the full omitted universe, and v71 restricted-master pricing still had {status["v71_improving_omitted_columns_v364"]} improving omitted columns.
- Valid full-v55 dual-bound certificate remains false.

Non-negotiable constraints:
- Before every run, commit, or push, verify reports/paper_material/paper4/status/paper4_final_promotion.json does not exist.
- Do not create Paper 4 final promotion, working champion, Paper Estrella replacement, live deployment, contractual IFRS9, fairness legal, or full-universe optimality claims.
- Keep new evidence in the living notebook and claim-boundary tables unless a later explicit promotion gate is built and approved.
- Use --no-verify for commits and pushes in this branch.

Immediate executable order:
1. v365: build paper4_v365_v353_full_v55_pricing_chunk_plan.csv. Estimate chunk sizes, memory, scenario-matrix requirements, exact input artifacts, and a resumable pricing-loop layout over the full omitted universe.
2. v366: run a small deterministic chunk prototype over full omitted candidates, recording runtime, candidate coverage, reduced-cost/proxy limitations, and why it is or is not scalable.
3. v367: decide whether to continue full-v55 pricing, switch to fifth-order bounded search, or narrow the publishable claim to bounded fourth-order no-entry plus governance.
4. v368: update the Paper 4 publishability focus memo with the strongest claim that remains true after v363/v364.
5. Keep guardrails for every wave: status JSON, CSV artifacts, living notebook block, claim boundaries, backlog row, and pytest assertions.

Definition of done for the next iteration:
- At least one new wave is executed and committed.
- All new claims are bounded and tested.
- The final-promotion artifact remains absent.
- The final answer reports both useful evidence and remaining blockers.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V364_V353_DUAL_BOUND_RESOURCE_PLAN_START -->"
    end = "<!-- V364_V353_DUAL_BOUND_RESOURCE_PLAN_END -->"
    block = f"""
{start}

## Wave v364: v353 Dual-Bound Resource Plan

Generated: {status["generated_at_utc"]}

### Objective

v363 established a negative full-dual-bound gap certificate. v364 turns that
gap into an executable queue and a reusable goal prompt so future iterations can
continue from the real frontier rather than restarting the analysis.

### Results

- Pending register rows: `{status["pending_register_rows_v364"]}`.
- Recommended next wave:
  `{status["recommended_next_wave_v364"]}`.
- Goal prompt artifact:
  `{status["goal_prompt_artifact_v364"]}`.
- Bounded candidate-pool share from v363:
  `{status["bounded_candidate_pool_share_v364"]}`.
- v71 improving omitted columns from v363:
  `{status["v71_improving_omitted_columns_v364"]}`.
- Valid full-v55 dual-bound certificate:
  `{status["valid_full_v55_dual_bound_certificate_v364"]}`.

### Interpretation

v364 is a practical bridge from evidence to execution. The next useful move is
not another promotion memo; it is a resource plan for full-v55 pricing chunks,
or an explicit paper-scope decision if that route remains too expensive.

### Claim Impact

- Allowed: executable pending register and goal prompt for continued work.
- Still prohibited: full-v55 branch-price termination, valid global integer
  optimality, working champion, Paper Estrella replacement and final promotion.

### Quarto Promotion Decision

Keep v364 in the living notebook. Use the goal prompt artifact to continue
iterating from v365.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v363_status = json.loads((STATUS_DIR / "paper4_v363_status.json").read_text(encoding="utf-8"))
    v363_certificate = read_csv("paper4_v363_v353_full_dual_bound_or_gap_certificate.csv")
    if v363_certificate.empty:
        raise RuntimeError("Missing v364 resource-plan input.")
    if bool(v363_status["valid_full_v55_dual_bound_certificate_v363"]):
        raise RuntimeError("v364 expects v363 to be a negative/gap certificate.")

    pending = pd.DataFrame(
        [
            {
                f"pending_id_v{VERSION}": "v365_full_v55_pricing_chunk_plan",
                f"priority_v{VERSION}": 1,
                f"lane_v{VERSION}": "Source Governance/Global",
                f"executable_now_v{VERSION}": True,
                f"expected_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"success_condition_v{VERSION}": (
                    "chunk sizes, memory limits, resumable state and full omitted coverage plan exist"
                ),
                f"risk_v{VERSION}": "medium_runtime_high_claim_value",
                f"claim_boundary_v{VERSION}": "planning only; no dual-bound claim",
            },
            {
                f"pending_id_v{VERSION}": "v366_full_v55_pricing_chunk_prototype",
                f"priority_v{VERSION}": 2,
                f"lane_v{VERSION}": "Source Governance/Global",
                f"executable_now_v{VERSION}": True,
                f"expected_artifact_v{VERSION}": "paper4_v366_v353_full_v55_pricing_chunk_prototype.csv",
                f"success_condition_v{VERSION}": (
                    "one deterministic full omitted chunk is priced with runtime and limitation evidence"
                ),
                f"risk_v{VERSION}": "bounded_compute_probe",
                f"claim_boundary_v{VERSION}": "prototype only; no termination claim",
            },
            {
                f"pending_id_v{VERSION}": "v367_route_decision_after_chunk_probe",
                f"priority_v{VERSION}": 3,
                f"lane_v{VERSION}": "Publishability/Scope",
                f"executable_now_v{VERSION}": True,
                f"expected_artifact_v{VERSION}": "paper4_v367_route_decision_after_chunk_probe.csv",
                f"success_condition_v{VERSION}": (
                    "choose full-v55 pricing, bounded fifth-order search, or bounded-claim paper scope"
                ),
                f"risk_v{VERSION}": "editorial_decision",
                f"claim_boundary_v{VERSION}": "route decision only",
            },
            {
                f"pending_id_v{VERSION}": "v368_publishable_claim_scope_update",
                f"priority_v{VERSION}": 4,
                f"lane_v{VERSION}": "Academic synthesis",
                f"executable_now_v{VERSION}": True,
                f"expected_artifact_v{VERSION}": "paper4_v368_publishable_claim_scope_update.md",
                f"success_condition_v{VERSION}": (
                    "the paper claim is narrowed to evidence that remains true after v363"
                ),
                f"risk_v{VERSION}": "low_compute_high_narrative_value",
                f"claim_boundary_v{VERSION}": "no global optimality unless later certified",
            },
            {
                f"pending_id_v{VERSION}": "v369_proxy_live_gate_separation",
                f"priority_v{VERSION}": 5,
                f"lane_v{VERSION}": "Proxy/Dynamic/Online",
                f"executable_now_v{VERSION}": True,
                f"expected_artifact_v{VERSION}": "paper4_v369_proxy_live_gate_separation.csv",
                f"success_condition_v{VERSION}": (
                    "proxy, dynamic, online and live-deployment blockers are separated from solver blockers"
                ),
                f"risk_v{VERSION}": "governance_cleanup",
                f"claim_boundary_v{VERSION}": "no live or contractual claims",
            },
        ]
    )
    plan = pd.DataFrame(
        [
            {
                f"plan_id_v{VERSION}": "v353_dual_bound_resource_plan",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"prior_gap_version_v{VERSION}": PRIOR_GAP_VERSION,
                f"pending_register_rows_v{VERSION}": int(len(pending)),
                f"recommended_next_wave_v{VERSION}": "v365_full_v55_pricing_chunk_plan",
                f"goal_prompt_artifact_v{VERSION}": (
                    "reports/paper_material/paper4/notes/paper4_v364_goal_prompt.md"
                ),
                f"bounded_candidate_pool_share_v{VERSION}": float(
                    v363_status["bounded_candidate_pool_share_v363"]
                ),
                f"v71_improving_omitted_columns_v{VERSION}": int(
                    v363_status["v71_improving_omitted_columns_v363"]
                ),
                f"best_source_exact_cvar_gap_vs_cap_v{VERSION}": float(
                    v363_status["best_source_exact_cvar_gap_vs_cap_v363"]
                ),
                f"valid_full_v55_dual_bound_certificate_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "resource plan and goal prompt only; no full-v55 termination claim"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "resource_plan_not_solver_certificate",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "v364 does not solve or price all full-v55 columns",
            },
            {
                f"blocker_id_v{VERSION}": "v363_gap_certificate_still_active",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    v363_status["v71_improving_omitted_columns_v363"]
                ),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "v71 improving omitted columns remain a blocker",
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
                "claim_id": "v364_dual_bound_resource_plan_created",
                "allowed": True,
                "artifact": "paper4_v364_v353_dual_bound_resource_plan.csv",
                "boundary": "planning only",
            },
            {
                "claim_id": "v364_goal_prompt_created",
                "allowed": True,
                "artifact": "paper4_v364_goal_prompt.md",
                "boundary": "continued execution prompt only",
            },
            {
                "claim_id": "v364_valid_full_v55_dual_bound_certificate",
                "allowed": False,
                "artifact": "paper4_v364_claim_blockers.csv",
                "boundary": "solver proof still missing",
            },
            {
                "claim_id": "v364_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v364_claim_blockers.csv",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v364_v353_dual_bound_resource_plan.csv", plan)
    write_csv(TABLE_DIR / "paper4_v364_executable_pending_register.csv", pending)
    write_csv(TABLE_DIR / "paper4_v364_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v364_claim_matrix_delta.csv", claim_matrix)
    row = plan.iloc[0]
    status = {
        "phase": "v364_v353_dual_bound_resource_plan",
        "schema_version": "2026-05-17.364",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v364": BASE_VERSION,
        "prior_gap_version_v364": PRIOR_GAP_VERSION,
        "pending_register_rows_v364": int(row[f"pending_register_rows_v{VERSION}"]),
        "recommended_next_wave_v364": str(row[f"recommended_next_wave_v{VERSION}"]),
        "goal_prompt_artifact_v364": str(row[f"goal_prompt_artifact_v{VERSION}"]),
        "bounded_candidate_pool_share_v364": float(
            row[f"bounded_candidate_pool_share_v{VERSION}"]
        ),
        "v71_improving_omitted_columns_v364": int(
            row[f"v71_improving_omitted_columns_v{VERSION}"]
        ),
        "best_source_exact_cvar_gap_vs_cap_v364": float(
            row[f"best_source_exact_cvar_gap_vs_cap_v{VERSION}"]
        ),
        "valid_full_v55_dual_bound_certificate_v364": False,
        "full_universe_integer_optimality_claim_allowed_v364": False,
        "working_champion_claim_allowed_v364": False,
        "paper1_promotion_allowed_v364": False,
        "paper4_working_champion_changed_v364": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v364": int(len(blockers)),
        "claim_matrix_rows_v364": int(len(claim_matrix)),
        "next_artifact_v364": NEXT_ARTIFACT,
        "claim_boundary": (
            "v364 records executable pending work and a goal prompt; full dual-bound, "
            "champion and promotion claims remain blocked"
        ),
    }
    GOAL_PROMPT.write_text(_goal_prompt(status), encoding="utf-8")
    _update_claim_boundaries()
    _update_backlog()
    write_json(STATUS_DIR / "paper4_v364_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v364": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
