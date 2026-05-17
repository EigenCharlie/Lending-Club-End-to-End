#!/usr/bin/env python3
"""Build Paper 4 v466 executable domain backlog refocus artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.papers.paper4_one_swap_living_lab import (
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    PAPER4_ROOT,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    now,
    write_csv,
    write_json,
)

VERSION = 466
PRIOR_CITATION_INTEGRATION_VERSION = 465
NEXT_ARTIFACT = "paper4_v467_cvar_tail_risk_frontier_probe.md"
REFOCUS_MD = NOTEBOOK.parent / "paper4_v466_domain_execution_backlog_refocus.md"
REQUIRED_LANES = {
    "cvar_tail_risk",
    "source_governance",
    "dynamic_replay",
    "online_monitoring",
    "spo_dla",
    "ifrs9_proxy",
}


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _artifact_exists(name: str) -> bool:
    for base in (TABLE_DIR, NOTEBOOK.parent, STATUS_DIR, PAPER4_ROOT / "references"):
        if (base / name).exists():
            return True
    return Path(name).exists()


def _domain_lanes() -> pd.DataFrame:
    rows = [
        {
            "domain_lane_v466": "cvar_tail_risk",
            "lane_order_v466": 1,
            "executable_now_v466": True,
            "anchor_artifacts_v466": (
                "paper4_v353_v347_apply_expanded_branch_price_candidate.csv;"
                "paper4_v357_second_order_branch_price_stage_summary.csv;"
                "paper4_v53_cvar_expected_loss_active_constraints_tail_first.csv"
            ),
            "implementation_action_v466": (
                "Compare current return/CVaR frontier evidence against open tail-risk gates."
            ),
            "next_artifact_v466": NEXT_ARTIFACT,
            "success_condition_v466": (
                "v467 produces a bounded CVaR frontier probe with blocker-aware claims."
            ),
            "claim_boundary_v466": "tail-risk diagnostic only; no champion authorization",
        },
        {
            "domain_lane_v466": "source_governance",
            "lane_order_v466": 2,
            "executable_now_v466": True,
            "anchor_artifacts_v466": (
                "paper4_v383_source_governance_audit_plan.csv;"
                "paper4_v371_source_governance_blocker_diagnostic.csv;"
                "paper4_v371_tight_source_blockers.csv"
            ),
            "implementation_action_v466": (
                "Refresh source-governance blockers and identify actionable cap/slack probes."
            ),
            "next_artifact_v466": "paper4_v468_source_governance_refresh.md",
            "success_condition_v466": (
                "v468 separates cap-design evidence from source-relief experiment choices."
            ),
            "claim_boundary_v466": "governance diagnostic only; no cap relaxation approval",
        },
        {
            "domain_lane_v466": "dynamic_replay",
            "lane_order_v466": 3,
            "executable_now_v466": True,
            "anchor_artifacts_v466": (
                "paper4_v340_dynamic_proxy_summary.csv;"
                "paper4_v340_dynamic_proxy_or_global_bound_after_v338.csv;"
                "paper4_v297_global_dynamic_gate_summary.csv"
            ),
            "implementation_action_v466": (
                "Audit dynamic proxy replay gaps against current static frontier candidates."
            ),
            "next_artifact_v466": "paper4_v469_dynamic_replay_reproducibility_probe.md",
            "success_condition_v466": (
                "v469 records reproducible dynamic-replay inputs and unresolved live gates."
            ),
            "claim_boundary_v466": "proxy replay only; no live deployment claim",
        },
        {
            "domain_lane_v466": "online_monitoring",
            "lane_order_v466": 4,
            "executable_now_v466": True,
            "anchor_artifacts_v466": (
                "paper4_v9_online_efficiency_frontier.csv;"
                "paper4_v323_v320_online_temporal_summary.csv;"
                "paper4_v341_v338_online_temporal_summary.csv"
            ),
            "implementation_action_v466": (
                "Summarize online monitoring evidence and current internal-holdout limits."
            ),
            "next_artifact_v466": "paper4_v470_online_conformal_monitoring_proxy.md",
            "success_condition_v466": (
                "v470 distinguishes defended internal monitoring from external deployment."
            ),
            "claim_boundary_v466": "internal online proxy only; no production monitoring claim",
        },
        {
            "domain_lane_v466": "spo_dla",
            "lane_order_v466": 5,
            "executable_now_v466": True,
            "anchor_artifacts_v466": (
                "paper4_v384_formal_spo_dla_review_packet.csv;"
                "paper4_v298_spo_dla_claim_audit.csv;"
                "paper4_v57_spo_dependency_probe.csv"
            ),
            "implementation_action_v466": (
                "Extract remaining SPO-DLA proof/review dependencies into an executable audit."
            ),
            "next_artifact_v466": "paper4_v471_spo_dla_boundary_probe.md",
            "success_condition_v466": (
                "v471 states which SPO-DLA claims are historical, surrogate or blocked."
            ),
            "claim_boundary_v466": "formal boundary audit only; no theorem or approval claim",
        },
        {
            "domain_lane_v466": "ifrs9_proxy",
            "lane_order_v466": 6,
            "executable_now_v466": True,
            "anchor_artifacts_v466": (
                "paper4_v341_v338_cashflow_proxy_summary.csv;"
                "paper4_v323_v320_cashflow_online_ifrs9_gate.csv;"
                "paper4_v47_ifrs9_proxy_panel_v45.parquet"
            ),
            "implementation_action_v466": (
                "Audit proxy coverage, imputation and contractual IFRS9 blockers."
            ),
            "next_artifact_v466": "paper4_v472_ifrs9_proxy_boundary_probe.md",
            "success_condition_v466": (
                "v472 separates IFRS9-inspired proxy evidence from contractual accounting claims."
            ),
            "claim_boundary_v466": "proxy boundary only; no contractual IFRS9 claim",
        },
    ]
    out = pd.DataFrame(rows)
    out["anchor_artifact_count_v466"] = out["anchor_artifacts_v466"].str.split(";").str.len()
    out["all_anchor_artifacts_present_v466"] = [
        all(_artifact_exists(item) for item in str(raw).split(";"))
        for raw in out["anchor_artifacts_v466"]
    ]
    return out


def _dependency_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dependency_id_v466": "cvar_before_champion_language",
                "source_lane_v466": "cvar_tail_risk",
                "blocked_claim_v466": "working champion or global optimality",
                "required_evidence_v466": (
                    "tail-risk frontier plus global/branch-price blocker inventory"
                ),
                "next_artifact_v466": NEXT_ARTIFACT,
                "claim_boundary_v466": "frontier diagnostic before any champion wording",
            },
            {
                "dependency_id_v466": "source_before_cap_policy",
                "source_lane_v466": "source_governance",
                "blocked_claim_v466": "source cap approval or relaxation",
                "required_evidence_v466": "source blocker and slack refresh",
                "next_artifact_v466": "paper4_v468_source_governance_refresh.md",
                "claim_boundary_v466": "cap-design evidence remains diagnostic",
            },
            {
                "dependency_id_v466": "dynamic_before_live_language",
                "source_lane_v466": "dynamic_replay",
                "blocked_claim_v466": "live sequential deployment",
                "required_evidence_v466": "reproducible dynamic replay under current candidates",
                "next_artifact_v466": "paper4_v469_dynamic_replay_reproducibility_probe.md",
                "claim_boundary_v466": "historical/proxy replay only",
            },
            {
                "dependency_id_v466": "online_before_monitoring_language",
                "source_lane_v466": "online_monitoring",
                "blocked_claim_v466": "production monitoring or external holdout",
                "required_evidence_v466": "internal monitoring summary plus external-gap caveat",
                "next_artifact_v466": "paper4_v470_online_conformal_monitoring_proxy.md",
                "claim_boundary_v466": "internal conformal monitoring proxy only",
            },
            {
                "dependency_id_v466": "spo_dla_before_formal_claims",
                "source_lane_v466": "spo_dla",
                "blocked_claim_v466": "formal SPO/DLA theorem, approval or CRC guarantee",
                "required_evidence_v466": "proof and review dependency audit",
                "next_artifact_v466": "paper4_v471_spo_dla_boundary_probe.md",
                "claim_boundary_v466": "historical/surrogate boundary only",
            },
            {
                "dependency_id_v466": "ifrs9_before_accounting_language",
                "source_lane_v466": "ifrs9_proxy",
                "blocked_claim_v466": "contractual IFRS9 compliance or accounting validation",
                "required_evidence_v466": "proxy coverage and imputation blocker audit",
                "next_artifact_v466": "paper4_v472_ifrs9_proxy_boundary_probe.md",
                "claim_boundary_v466": "IFRS9-inspired proxy only",
            },
        ]
    )


def _remaining_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v466": "cvar_frontier_probe_pending",
                "blocking_v466": True,
                "evidence_count_v466": 0,
                "required_next_artifact_v466": NEXT_ARTIFACT,
                "claim_boundary_v466": "must probe tail-risk frontier before stronger claims",
            },
            {
                "blocker_id_v466": "domain_lanes_not_yet_executed",
                "blocking_v466": True,
                "evidence_count_v466": 6,
                "required_next_artifact_v466": "v467_to_v472_domain_wave_sequence",
                "claim_boundary_v466": "v466 schedules lanes but does not execute them",
            },
            {
                "blocker_id_v466": "external_holdout_and_live_validation_missing",
                "blocking_v466": True,
                "evidence_count_v466": 0,
                "required_next_artifact_v466": "future_external_validation_design",
                "claim_boundary_v466": "no live or external deployment claim",
            },
            {
                "blocker_id_v466": "contractual_ifrs9_missing",
                "blocking_v466": True,
                "evidence_count_v466": 0,
                "required_next_artifact_v466": "future_contractual_cashflow_validation",
                "claim_boundary_v466": "IFRS9 evidence remains proxy only",
            },
            {
                "blocker_id_v466": "paper4_final_promotion_forbidden",
                "blocking_v466": True,
                "evidence_count_v466": 1,
                "required_next_artifact_v466": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v466": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v466_domain_backlog_refocus_created",
                "allowed": True,
                "artifact": "paper4_v466_domain_execution_backlog_refocus.md",
                "boundary": "execution-plan artifact only",
            },
            {
                "claim_id": "v466_six_domain_lanes_mapped",
                "allowed": True,
                "artifact": "paper4_v466_domain_lane_backlog.csv",
                "boundary": "lane mapping with existing anchors",
            },
            {
                "claim_id": "v466_cvar_next_lane_selected",
                "allowed": True,
                "artifact": "paper4_v466_domain_dependency_matrix.csv",
                "boundary": "next local wave selection only",
            },
            {
                "claim_id": "v466_domain_lanes_executed_or_resolved",
                "allowed": False,
                "artifact": "paper4_v466_remaining_blockers.csv",
                "boundary": "v466 schedules execution; v467+ must run it",
            },
            {
                "claim_id": "v466_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "no champion, deployment or final promotion claim",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v466 refocuses Paper 4 around six executable domain lanes.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v466_domain_lane_backlog.csv"
                ),
                "boundary": "Backlog refocus only; domain lanes still need execution.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v466 selects CVaR tail-risk as the next executable wave.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v466_domain_dependency_matrix.csv"
                ),
                "boundary": "Next-wave selection only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v466 executes and resolves all domain lanes.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v466_remaining_blockers.csv"
                ),
                "boundary": "v466 schedules v467-v472; it does not run those probes.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v466 authorizes a Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v466_remaining_blockers.csv"
                ),
                "boundary": "Global, proxy, dynamic, online and deployment gates remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v466 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v466_remaining_blockers.csv"
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
                "lane": "Execution Planning",
                "executable_item": "v466 refocuses domain execution lanes.",
                "status": "domain_backlog_refocus_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v467 probes the current CVaR tail-risk frontier without champion claims"
                ),
                "last_wave": "v466",
                "execution_result": "six_domain_lanes_mapped_cvar_selected_next",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v466")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _refocus_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Domain Execution Backlog Refocus v466

Generated: {status["generated_at_utc"]}

## Result

v466 converts the post-manuscript Paper 4 backlog into six immediately
executable domain lanes: CVaR tail risk, source governance, dynamic replay,
online monitoring, SPO-DLA boundaries and IFRS9 proxy evidence.

## Counts

- Domain lanes: `{status["domain_lane_count_v466"]}`.
- Executable-now lanes: `{status["executable_now_count_v466"]}`.
- Lanes with anchors present: `{status["lanes_with_all_anchors_present_v466"]}`.
- Selected next lane: `{status["selected_next_lane_v466"]}`.
- Next artifact: `{status["next_artifact_v466"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v466 is an execution refocus artifact. It does not execute the domain probes,
authorize a Paper 4 working champion, modify Quarto sources, replace Paper
Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V466_DOMAIN_EXECUTION_BACKLOG_REFOCUS_START -->"
    end = "<!-- V466_DOMAIN_EXECUTION_BACKLOG_REFOCUS_END -->"
    block = f"""
{start}

## Wave v466: Domain Execution Backlog Refocus

Generated: {status["generated_at_utc"]}

### Objective

v466 turns the post-citation backlog into executable domain lanes for CVaR,
source governance, dynamic replay, online monitoring, SPO-DLA and IFRS9 proxy
work.

### Results

- Domain lanes:
  `{status["domain_lane_count_v466"]}`.
- Executable-now lanes:
  `{status["executable_now_count_v466"]}`.
- Lanes with all anchors present:
  `{status["lanes_with_all_anchors_present_v466"]}`.
- Selected next lane:
  `{status["selected_next_lane_v466"]}`.
- CVaR lane selected:
  `{status["cvar_lane_selected_v466"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v466"]}`.

### Interpretation

The living lab is back in execution mode. The highest-value next move is the
bounded CVaR tail-risk frontier probe, because it touches the strongest
performance claim pressure while preserving the open global, proxy, dynamic and
online blockers.

### Claim Impact

- Allowed: six-lane executable backlog refocus and CVaR-next selection.
- Still prohibited: claims that the lanes are resolved, working-champion
  authorization, deployment readiness, Paper Estrella replacement and final
  Paper 4 promotion.

### Quarto Promotion Decision

Keep v466 in the living notebook. v467 should generate the CVaR tail-risk
frontier probe without changing book references or Quarto sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v465 = _read_status(PRIOR_CITATION_INTEGRATION_VERSION)
    if v465["next_artifact_v465"] != "paper4_v466_domain_execution_backlog_refocus.md":
        raise RuntimeError("v466 expects v465 to route to domain backlog refocus.")
    if v465["citation_integration_dry_run_created_v465"] is not True:
        raise RuntimeError("v466 expects v465 citation integration dry-run.")
    if v465["paper4_final_promotion_created"] is not False:
        raise RuntimeError("v466 expects no final Paper 4 promotion.")

    lanes = _domain_lanes()
    lane_ids = set(lanes["domain_lane_v466"].astype(str))
    if lane_ids != REQUIRED_LANES:
        raise RuntimeError(f"Unexpected v466 domain lanes: {sorted(lane_ids)}")
    if not lanes["executable_now_v466"].astype(bool).all():
        raise RuntimeError("v466 expects all refocused lanes to be executable now.")

    dependencies = _dependency_matrix()
    blockers = _remaining_blockers()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v466_domain_lane_backlog.csv", lanes)
    write_csv(TABLE_DIR / "paper4_v466_domain_dependency_matrix.csv", dependencies)
    write_csv(TABLE_DIR / "paper4_v466_remaining_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v466_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v466_domain_execution_backlog_refocus",
        "schema_version": "2026-05-17.466",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_citation_integration_version_v466": PRIOR_CITATION_INTEGRATION_VERSION,
        "domain_lane_count_v466": len(lanes),
        "executable_now_count_v466": int(lanes["executable_now_v466"].astype(bool).sum()),
        "lanes_with_all_anchors_present_v466": int(
            lanes["all_anchor_artifacts_present_v466"].astype(bool).sum()
        ),
        "required_domain_lanes_present_v466": lane_ids == REQUIRED_LANES,
        "domain_backlog_refocus_created_v466": True,
        "cvar_lane_selected_v466": True,
        "source_governance_lane_present_v466": "source_governance" in lane_ids,
        "dynamic_replay_lane_present_v466": "dynamic_replay" in lane_ids,
        "online_monitoring_lane_present_v466": "online_monitoring" in lane_ids,
        "spo_dla_lane_present_v466": "spo_dla" in lane_ids,
        "ifrs9_proxy_lane_present_v466": "ifrs9_proxy" in lane_ids,
        "selected_next_lane_v466": "cvar_tail_risk",
        "domain_lanes_executed_v466": False,
        "working_champion_claim_allowed_v466": False,
        "paper1_promotion_allowed_v466": False,
        "paper4_working_champion_changed_v466": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v466": NEXT_ARTIFACT,
        "claim_boundary": (
            "v466 refocuses execution lanes only; domain resolution, deployment, "
            "champion authorization and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v466 must not create final Paper 4 promotion.")

    REFOCUS_MD.write_text(_refocus_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v466": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
