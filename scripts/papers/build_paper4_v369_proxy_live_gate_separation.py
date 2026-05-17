#!/usr/bin/env python3
"""Build Paper 4 v369 proxy/live gate-separation artifacts."""

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

VERSION = 369
PRIOR_SCOPE_VERSION = 368
PRIOR_ROUTE_VERSION = 367
PRIOR_CHUNK_VERSION = 366
PRIOR_DYNAMIC_PROXY_VERSION = 297
PRIOR_ONLINE_GATE_VERSION = 298
PRIOR_HOLDOUT_VERSION = 67
NEXT_VERSION = 370
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_future_execution_backlog_refresh.csv"


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v369 separates offline proxy evidence from live deployment gates.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v369_proxy_live_gate_separation.csv"
                ),
                "boundary": "Separation matrix only; no live deployability claim.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v369 allows proxy-only and offline evidence statements with explicit labels.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v369_offline_evidence_inventory.csv"
                ),
                "boundary": "Each statement must carry offline/proxy/live/final gate status.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v369 authorizes strict live deployability.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v369_gate_requirement_matrix.csv"
                ),
                "boundary": "External holdout, live monitoring and deployment gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v369 turns IFRS9, fairness or online monitoring proxies into production controls.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v369_claim_blockers.csv"
                ),
                "boundary": "Proxy evidence is not contractual/legal/production validation.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v369 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v369_claim_blockers.csv"
                ),
                "boundary": "No final promotion artifact, champion replacement or deployment gate is created.",
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
                "lane": "Execution Planning",
                "executable_item": (
                    "v369 separates offline/proxy evidence from live deployment and "
                    "final-promotion gates, then routes the lab to refresh the next backlog."
                ),
                "status": "proxy_live_gate_separation_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v370 converts the v365-v369 results into the next executable backlog "
                    "without overclaiming"
                ),
                "last_wave": "v369",
                "execution_result": (
                    "offline_proxy_live_deployment_and_promotion_gates_separated"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v369")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V369_PROXY_LIVE_GATE_SEPARATION_START -->"
    end = "<!-- V369_PROXY_LIVE_GATE_SEPARATION_END -->"
    block = f"""
{start}

## Wave v369: Proxy/Live Gate Separation

Generated: {status["generated_at_utc"]}

### Objective

v368 defined the strongest bounded publishable claim. v369 separates evidence
that is safe for offline/publication language from evidence that remains
proxy-only, live-gated, contractual/legal-gated or final-promotion-gated.

### Results

- Separation rows:
  `{status["separation_rows_v369"]}`.
- Offline evidence inventory rows:
  `{status["offline_evidence_inventory_rows_v369"]}`.
- Gate requirement rows:
  `{status["gate_requirement_rows_v369"]}`.
- Gate requirements met:
  `{status["gate_requirements_met_v369"]}`.
- Dynamic proxy trace rows from v297:
  `{status["dynamic_proxy_trace_rows_v369"]}`.
- External live pass rows from v298:
  `{status["external_live_pass_rows_v369"]}`.
- IFRS9 proxy uncovered loan rows from v298:
  `{status["ifrs9_proxy_uncovered_loan_rows_v369"]}`.
- Strict live deployability claim allowed:
  `{status["strict_live_deployability_claim_allowed_v369"]}`.
- Final promotion allowed:
  `{status["final_promotion_allowed_v369"]}`.
- Next artifact:
  `{status["next_artifact_v369"]}`.

### Interpretation

v369 is a governance cleanup wave. It lets the paper use offline and proxy
evidence honestly while keeping live deployment, contractual IFRS9, legal
fairness, global solver and final-promotion claims blocked.

### Claim Impact

- Allowed: offline/proxy evidence statements with explicit labels.
- Still prohibited: strict live deployability, contractual/legal production
  controls, final champion replacement and Paper 4 final promotion.

### Quarto Promotion Decision

Keep v369 in the living notebook. v370 should refresh the executable backlog
after v365-v369 and choose the next useful experimental lane.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v67_status = json.loads((STATUS_DIR / "paper4_v67_status.json").read_text(encoding="utf-8"))
    v297_status = json.loads((STATUS_DIR / "paper4_v297_status.json").read_text(encoding="utf-8"))
    v298_status = json.loads((STATUS_DIR / "paper4_v298_status.json").read_text(encoding="utf-8"))
    v366_status = json.loads((STATUS_DIR / "paper4_v366_status.json").read_text(encoding="utf-8"))
    v368_status = json.loads((STATUS_DIR / "paper4_v368_status.json").read_text(encoding="utf-8"))
    if bool(v368_status["working_champion_claim_allowed_v368"]):
        raise RuntimeError("v369 expects v368 to keep working champion claims blocked.")

    separation = pd.DataFrame(
        [
            {
                f"lane_id_v{VERSION}": "publishable_bounded_living_lab",
                f"evidence_tier_v{VERSION}": "offline_publishable",
                f"source_artifact_v{VERSION}": "paper4_v368_publishable_claim_scope_update.md",
                f"evidence_count_v{VERSION}": int(v368_status["allowed_publishable_claim_rows_v368"]),
                f"offline_publishable_claim_allowed_v{VERSION}": True,
                f"proxy_only_claim_allowed_v{VERSION}": False,
                f"strict_live_deployability_claim_allowed_v{VERSION}": False,
                f"final_promotion_allowed_v{VERSION}": False,
                f"missing_gate_v{VERSION}": "none for bounded wording; cite limitations",
                f"claim_boundary_v{VERSION}": "publication language only",
            },
            {
                f"lane_id_v{VERSION}": "bounded_solver_gap",
                f"evidence_tier_v{VERSION}": "offline_solver_evidence",
                f"source_artifact_v{VERSION}": (
                    "paper4_v361_v353_fourth_order_or_full_dual_bound.csv;"
                    "paper4_v363_v353_full_dual_bound_or_gap_certificate.csv;"
                    "paper4_v366_v353_full_v55_pricing_chunk_prototype.csv"
                ),
                f"evidence_count_v{VERSION}": int(
                    v368_status["v361_ordered_fourth_order_rows_v368"]
                ),
                f"offline_publishable_claim_allowed_v{VERSION}": True,
                f"proxy_only_claim_allowed_v{VERSION}": False,
                f"strict_live_deployability_claim_allowed_v{VERSION}": False,
                f"final_promotion_allowed_v{VERSION}": False,
                f"missing_gate_v{VERSION}": "full-v55 termination and integer certificate",
                f"claim_boundary_v{VERSION}": "bounded/gap evidence only",
            },
            {
                f"lane_id_v{VERSION}": "dynamic_proxy_replay",
                f"evidence_tier_v{VERSION}": "offline_proxy",
                f"source_artifact_v{VERSION}": "paper4_v297_dynamic_proxy_trace.parquet",
                f"evidence_count_v{VERSION}": int(v297_status["dynamic_proxy_trace_rows_v297"]),
                f"offline_publishable_claim_allowed_v{VERSION}": True,
                f"proxy_only_claim_allowed_v{VERSION}": True,
                f"strict_live_deployability_claim_allowed_v{VERSION}": bool(
                    v297_status["live_deployment_claim_allowed_v297"]
                ),
                f"final_promotion_allowed_v{VERSION}": False,
                f"missing_gate_v{VERSION}": "validated live replay and deployment monitoring",
                f"claim_boundary_v{VERSION}": "proxy replay only",
            },
            {
                f"lane_id_v{VERSION}": "online_external_holdout",
                f"evidence_tier_v{VERSION}": "live_gate_blocked",
                f"source_artifact_v{VERSION}": (
                    "paper4_v67_external_holdout_scorecard.csv;"
                    "paper4_v298_online_gate_transfer_audit.csv"
                ),
                f"evidence_count_v{VERSION}": int(
                    v298_status["online_external_live_pass_rows_v298"]
                ),
                f"offline_publishable_claim_allowed_v{VERSION}": False,
                f"proxy_only_claim_allowed_v{VERSION}": False,
                f"strict_live_deployability_claim_allowed_v{VERSION}": False,
                f"final_promotion_allowed_v{VERSION}": False,
                f"missing_gate_v{VERSION}": "external/future holdout data and pass rows",
                f"claim_boundary_v{VERSION}": "live claim blocked",
            },
            {
                f"lane_id_v{VERSION}": "ifrs9_proxy_contractual",
                f"evidence_tier_v{VERSION}": "proxy_contractual_blocked",
                f"source_artifact_v{VERSION}": "paper4_v298_ifrs9_v295_proxy_coverage.csv",
                f"evidence_count_v{VERSION}": int(
                    v298_status["ifrs9_proxy_covered_loan_rows_v298"]
                ),
                f"offline_publishable_claim_allowed_v{VERSION}": True,
                f"proxy_only_claim_allowed_v{VERSION}": True,
                f"strict_live_deployability_claim_allowed_v{VERSION}": bool(
                    v298_status["strict_live_deployability_claim_allowed_v298"]
                ),
                f"final_promotion_allowed_v{VERSION}": False,
                f"missing_gate_v{VERSION}": "contractual IFRS9 data coverage and approval",
                f"claim_boundary_v{VERSION}": "IFRS9-inspired proxy only",
            },
            {
                f"lane_id_v{VERSION}": "deployment_final_promotion",
                f"evidence_tier_v{VERSION}": "promotion_blocked",
                f"source_artifact_v{VERSION}": "paper4_final_promotion.json",
                f"evidence_count_v{VERSION}": 0,
                f"offline_publishable_claim_allowed_v{VERSION}": False,
                f"proxy_only_claim_allowed_v{VERSION}": False,
                f"strict_live_deployability_claim_allowed_v{VERSION}": False,
                f"final_promotion_allowed_v{VERSION}": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"missing_gate_v{VERSION}": "approved final-promotion gate",
                f"claim_boundary_v{VERSION}": "final promotion forbidden",
            },
        ]
    )
    requirements = pd.DataFrame(
        [
            {
                f"requirement_id_v{VERSION}": "bounded_claim_scope_update_exists",
                f"gate_tier_v{VERSION}": "offline_publishable",
                f"met_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v368_status["allowed_publishable_claim_rows_v368"]),
                f"claim_enabled_v{VERSION}": "bounded living-lab publication wording",
            },
            {
                f"requirement_id_v{VERSION}": "dynamic_proxy_replay_executed",
                f"gate_tier_v{VERSION}": "offline_proxy",
                f"met_v{VERSION}": bool(v297_status["dynamic_proxy_replay_executed_v297"]),
                f"evidence_count_v{VERSION}": int(v297_status["dynamic_proxy_trace_rows_v297"]),
                f"claim_enabled_v{VERSION}": "proxy replay evidence, not live deployment",
            },
            {
                f"requirement_id_v{VERSION}": "external_online_holdout_available",
                f"gate_tier_v{VERSION}": "live_deployment",
                f"met_v{VERSION}": bool(v67_status["holdout_data_available_v67"]),
                f"evidence_count_v{VERSION}": int(v67_status["scorecard_rows_v67"]),
                f"claim_enabled_v{VERSION}": "none until external holdout exists",
            },
            {
                f"requirement_id_v{VERSION}": "external_online_holdout_passes",
                f"gate_tier_v{VERSION}": "live_deployment",
                f"met_v{VERSION}": bool(v67_status["passing_methods_v67"]),
                f"evidence_count_v{VERSION}": int(v67_status["passing_methods_v67"]),
                f"claim_enabled_v{VERSION}": "none until holdout passes",
            },
            {
                f"requirement_id_v{VERSION}": "ifrs9_contractual_coverage_complete",
                f"gate_tier_v{VERSION}": "contractual_ifrs9",
                f"met_v{VERSION}": bool(v298_status["contractual_ifrs9_claim_allowed_v298"]),
                f"evidence_count_v{VERSION}": int(
                    v298_status["ifrs9_proxy_uncovered_loan_rows_v298"]
                ),
                f"claim_enabled_v{VERSION}": "none; uncovered proxy rows remain",
            },
            {
                f"requirement_id_v{VERSION}": "formal_spo_dla_claim_review_passed",
                f"gate_tier_v{VERSION}": "formal_method",
                f"met_v{VERSION}": bool(v298_status["formal_spo_dla_claim_allowed_v298"]),
                f"evidence_count_v{VERSION}": int(v298_status["spo_dla_audit_rows_v298"]),
                f"claim_enabled_v{VERSION}": "none; historical audit only",
            },
            {
                f"requirement_id_v{VERSION}": "full_v55_certificate_available",
                f"gate_tier_v{VERSION}": "global_solver",
                f"met_v{VERSION}": bool(v368_status["valid_full_v55_dual_bound_certificate_v368"]),
                f"evidence_count_v{VERSION}": int(v368_status["remaining_unpriced_chunks_v368"]),
                f"claim_enabled_v{VERSION}": "none; global proof remains open",
            },
            {
                f"requirement_id_v{VERSION}": "source_exact_full_v55_chunk_evidence_available",
                f"gate_tier_v{VERSION}": "source_governance",
                f"met_v{VERSION}": bool(v366_status["source_exact_rows_v366"]),
                f"evidence_count_v{VERSION}": int(v366_status["source_exact_rows_v366"]),
                f"claim_enabled_v{VERSION}": "none; first chunk found zero source-exact rows",
            },
            {
                f"requirement_id_v{VERSION}": "deployment_monitoring_runbook_exists",
                f"gate_tier_v{VERSION}": "live_deployment",
                f"met_v{VERSION}": False,
                f"evidence_count_v{VERSION}": 0,
                f"claim_enabled_v{VERSION}": "none; no deployment runbook",
            },
            {
                f"requirement_id_v{VERSION}": "final_promotion_gate_created",
                f"gate_tier_v{VERSION}": "final_promotion",
                f"met_v{VERSION}": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"evidence_count_v{VERSION}": 0,
                f"claim_enabled_v{VERSION}": "none; final promotion remains forbidden",
            },
        ]
    )
    evidence_inventory = pd.DataFrame(
        [
            {
                f"inventory_id_v{VERSION}": "v67_external_holdout_scorer",
                f"source_version_v{VERSION}": 67,
                f"evidence_count_v{VERSION}": int(v67_status["scorecard_rows_v67"]),
                f"evidence_class_v{VERSION}": "blocked_holdout_scorer",
                f"live_claim_allowed_v{VERSION}": bool(
                    v67_status["strict_live_deployability_claim_allowed_v67"]
                ),
            },
            {
                f"inventory_id_v{VERSION}": "v297_dynamic_proxy_replay",
                f"source_version_v{VERSION}": 297,
                f"evidence_count_v{VERSION}": int(v297_status["dynamic_proxy_trace_rows_v297"]),
                f"evidence_class_v{VERSION}": "offline_proxy_replay",
                f"live_claim_allowed_v{VERSION}": bool(
                    v297_status["live_deployment_claim_allowed_v297"]
                ),
            },
            {
                f"inventory_id_v{VERSION}": "v298_online_ifrs9_gate_expansion",
                f"source_version_v{VERSION}": 298,
                f"evidence_count_v{VERSION}": int(v298_status["online_internal_pass_rows_v298"]),
                f"evidence_class_v{VERSION}": "historical_gate_transfer_audit",
                f"live_claim_allowed_v{VERSION}": bool(
                    v298_status["strict_live_deployability_claim_allowed_v298"]
                ),
            },
            {
                f"inventory_id_v{VERSION}": "v361_bounded_solver_no_entry",
                f"source_version_v{VERSION}": 361,
                f"evidence_count_v{VERSION}": int(
                    v368_status["v361_ordered_fourth_order_rows_v368"]
                ),
                f"evidence_class_v{VERSION}": "bounded_solver_screen",
                f"live_claim_allowed_v{VERSION}": False,
            },
            {
                f"inventory_id_v{VERSION}": "v363_full_dual_bound_gap",
                f"source_version_v{VERSION}": 363,
                f"evidence_count_v{VERSION}": int(
                    v368_status["v71_improving_omitted_columns_v368"]
                ),
                f"evidence_class_v{VERSION}": "global_gap_disclosure",
                f"live_claim_allowed_v{VERSION}": False,
            },
            {
                f"inventory_id_v{VERSION}": "v366_full_v55_chunk_source_blocker",
                f"source_version_v{VERSION}": 366,
                f"evidence_count_v{VERSION}": int(v366_status["ordered_one_swap_rows_v366"]),
                f"evidence_class_v{VERSION}": "single_chunk_source_governance_blocker",
                f"live_claim_allowed_v{VERSION}": False,
            },
            {
                f"inventory_id_v{VERSION}": "v368_claim_scope_update",
                f"source_version_v{VERSION}": 368,
                f"evidence_count_v{VERSION}": int(v368_status["allowed_publishable_claim_rows_v368"]),
                f"evidence_class_v{VERSION}": "publishable_scope_boundary",
                f"live_claim_allowed_v{VERSION}": False,
            },
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "external_holdout_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v67_status["scorecard_rows_v67"]),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "live deployment requires external/future holdout",
            },
            {
                f"blocker_id_v{VERSION}": "ifrs9_proxy_uncovered_rows",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    v298_status["ifrs9_proxy_uncovered_loan_rows_v298"]
                ),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "contractual IFRS9 remains blocked",
            },
            {
                f"blocker_id_v{VERSION}": "full_solver_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v368_status["remaining_unpriced_chunks_v368"]),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "global optimality remains blocked",
            },
            {
                f"blocker_id_v{VERSION}": "source_exact_chunk_rows_zero",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v366_status["source_exact_rows_v366"]),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "source governance blocks full-v55 chunk evidence",
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
                "claim_id": "v369_proxy_live_gate_separation_created",
                "allowed": True,
                "artifact": "paper4_v369_proxy_live_gate_separation.csv",
                "boundary": "separation matrix only",
            },
            {
                "claim_id": "v369_offline_proxy_labels_allowed",
                "allowed": True,
                "artifact": "paper4_v369_offline_evidence_inventory.csv",
                "boundary": "offline/proxy labels required",
            },
            {
                "claim_id": "v369_strict_live_deployability",
                "allowed": False,
                "artifact": "paper4_v369_gate_requirement_matrix.csv",
                "boundary": "live gates missing",
            },
            {
                "claim_id": "v369_contractual_or_legal_production_controls",
                "allowed": False,
                "artifact": "paper4_v369_claim_blockers.csv",
                "boundary": "proxy evidence only",
            },
            {
                "claim_id": "v369_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v369_claim_blockers.csv",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v369_proxy_live_gate_separation.csv", separation)
    write_csv(TABLE_DIR / "paper4_v369_gate_requirement_matrix.csv", requirements)
    write_csv(TABLE_DIR / "paper4_v369_offline_evidence_inventory.csv", evidence_inventory)
    write_csv(TABLE_DIR / "paper4_v369_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v369_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v369_proxy_live_gate_separation",
        "schema_version": "2026-05-17.369",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_scope_version_v369": PRIOR_SCOPE_VERSION,
        "prior_route_version_v369": PRIOR_ROUTE_VERSION,
        "prior_chunk_version_v369": PRIOR_CHUNK_VERSION,
        "prior_dynamic_proxy_version_v369": PRIOR_DYNAMIC_PROXY_VERSION,
        "prior_online_gate_version_v369": PRIOR_ONLINE_GATE_VERSION,
        "prior_holdout_version_v369": PRIOR_HOLDOUT_VERSION,
        "separation_rows_v369": int(len(separation)),
        "offline_evidence_inventory_rows_v369": int(len(evidence_inventory)),
        "gate_requirement_rows_v369": int(len(requirements)),
        "gate_requirements_met_v369": int(requirements[f"met_v{VERSION}"].astype(bool).sum()),
        "offline_publishable_allowed_rows_v369": int(
            separation[f"offline_publishable_claim_allowed_v{VERSION}"].astype(bool).sum()
        ),
        "proxy_only_allowed_rows_v369": int(
            separation[f"proxy_only_claim_allowed_v{VERSION}"].astype(bool).sum()
        ),
        "dynamic_proxy_trace_rows_v369": int(v297_status["dynamic_proxy_trace_rows_v297"]),
        "external_live_pass_rows_v369": int(v298_status["online_external_live_pass_rows_v298"]),
        "online_internal_pass_rows_v369": int(v298_status["online_internal_pass_rows_v298"]),
        "holdout_data_available_v369": bool(v67_status["holdout_data_available_v67"]),
        "ifrs9_proxy_covered_loan_rows_v369": int(
            v298_status["ifrs9_proxy_covered_loan_rows_v298"]
        ),
        "ifrs9_proxy_uncovered_loan_rows_v369": int(
            v298_status["ifrs9_proxy_uncovered_loan_rows_v298"]
        ),
        "v366_source_exact_rows_v369": int(v366_status["source_exact_rows_v366"]),
        "remaining_unpriced_chunks_v369": int(v368_status["remaining_unpriced_chunks_v368"]),
        "claim_blocker_rows_v369": int(len(blockers)),
        "claim_matrix_rows_v369": int(len(claim_matrix)),
        "strict_live_deployability_claim_allowed_v369": False,
        "contractual_or_legal_production_claim_allowed_v369": False,
        "full_universe_integer_optimality_claim_allowed_v369": False,
        "working_champion_claim_allowed_v369": False,
        "paper1_promotion_allowed_v369": False,
        "paper4_working_champion_changed_v369": False,
        "final_promotion_allowed_v369": FORBIDDEN_FINAL_PROMOTION.exists(),
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v369": NEXT_ARTIFACT,
        "claim_boundary": (
            "v369 allows labeled offline/proxy evidence only; live, contractual/legal, "
            "global solver and final-promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v369_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v369": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
