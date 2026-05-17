#!/usr/bin/env python3
"""Build Paper 4 v375 live-gate data-contract artifacts."""

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

VERSION = 375
PRIOR_GATE_VERSION = 369
PRIOR_STOP_RULE_VERSION = 373
PRIOR_CLAIM_LANGUAGE_VERSION = 374
NEXT_VERSION = 376
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_publication_integration_patch.md"


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": (
                    "v375 specifies data-contract requirements for live, contractual/legal "
                    "and final-promotion claims."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v375_live_gate_data_contract.csv"
                ),
                "boundary": "Requirements artifact only; no blocked claim is activated.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v375 separates available offline/proxy data from missing live-gate data."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v375_gate_readiness_summary.csv"
                ),
                "boundary": "Offline/proxy labels stay separate from live deployment claims.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v375 authorizes strict live deployment, contractual/legal claims or "
                    "production monitoring."
                ),
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v375_claim_permission_register.csv"
                ),
                "boundary": "Live, legal/contractual and production gates remain unmet.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v375 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v375_claim_blockers.csv"
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
                "lane": "Publishability/Scope",
                "executable_item": (
                    "v375 converts proxy/live/final blockers into a data contract, then "
                    "routes the next wave to publication integration."
                ),
                "status": "live_gate_data_contract_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v376 maps v374 language and v375 data gates into manuscript sections "
                    "without changing claim permissions"
                ),
                "last_wave": "v375",
                "execution_result": "live_contract_blocks_live_legal_global_and_final_claims",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v375")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V375_LIVE_GATE_DATA_CONTRACT_START -->"
    end = "<!-- V375_LIVE_GATE_DATA_CONTRACT_END -->"
    block = f"""
{start}

## Wave v375: Live-Gate Data Contract

Generated: {status["generated_at_utc"]}

### Objective

v374 produced bounded paper language. v375 turns the remaining live,
contractual/legal, global-solver and final-promotion blockers into a concrete
data contract that future waves can satisfy or cite as missing.

### Results

- Contract rows:
  `{status["contract_rows_v375"]}`.
- Gate readiness rows:
  `{status["gate_readiness_rows_v375"]}`.
- Claim permission rows:
  `{status["claim_permission_rows_v375"]}`.
- Data assets currently available:
  `{status["data_asset_available_rows_v375"]}`.
- Claim gates currently met:
  `{status["claim_gate_met_rows_v375"]}`.
- Live deployment gates met:
  `{status["live_deployment_gate_met_rows_v375"]}`.
- Contractual/legal gates met:
  `{status["contractual_or_legal_gate_met_rows_v375"]}`.
- Global solver gates met:
  `{status["global_solver_gate_met_rows_v375"]}`.
- Final promotion gates met:
  `{status["final_promotion_gate_met_rows_v375"]}`.
- Strict live deployment language allowed:
  `{status["strict_live_deployment_language_allowed_v375"]}`.
- Next artifact:
  `{status["next_artifact_v375"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.

### Interpretation

The lab now has a reusable contract for what would have to exist before Paper 4
could say anything stronger than bounded offline/proxy evidence. The contract
keeps the paper useful while making overclaiming mechanically visible.

### Claim Impact

- Allowed: bounded manuscript language and labeled offline/proxy replay.
- Still prohibited: strict live deployment, contractual/legal IFRS9 or fairness
  claims, global optimality, working-champion replacement and final promotion.

### Quarto Promotion Decision

Keep v375 in the living notebook. v376 should map the v374/v375 language into a
publication integration patch without changing claim permissions.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def _gate_summary(contract: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tier, group in contract.groupby(f"gate_tier_v{VERSION}", sort=True):
        rows.append(
            {
                f"gate_tier_v{VERSION}": tier,
                f"contract_rows_v{VERSION}": int(len(group)),
                f"data_asset_available_rows_v{VERSION}": int(
                    group[f"data_asset_available_v{VERSION}"].astype(bool).sum()
                ),
                f"claim_gate_met_rows_v{VERSION}": int(
                    group[f"claim_gate_met_v{VERSION}"].astype(bool).sum()
                ),
                f"blocked_rows_v{VERSION}": int(
                    (~group[f"claim_gate_met_v{VERSION}"].astype(bool)).sum()
                ),
                f"current_evidence_count_v{VERSION}": int(
                    group[f"current_evidence_count_v{VERSION}"].astype(int).sum()
                ),
                f"claim_boundary_v{VERSION}": (
                    "tier ready only if every required contract row is claim-gate met"
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary[f"tier_claim_ready_v{VERSION}"] = (
        summary[f"contract_rows_v{VERSION}"] == summary[f"claim_gate_met_rows_v{VERSION}"]
    )
    return summary


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v363_status = json.loads((STATUS_DIR / "paper4_v363_status.json").read_text(encoding="utf-8"))
    v369_status = json.loads((STATUS_DIR / "paper4_v369_status.json").read_text(encoding="utf-8"))
    v373_status = json.loads((STATUS_DIR / "paper4_v373_status.json").read_text(encoding="utf-8"))
    v374_status = json.loads((STATUS_DIR / "paper4_v374_status.json").read_text(encoding="utf-8"))
    if v374_status["next_artifact_v374"] != "paper4_v375_live_gate_data_contract.csv":
        raise RuntimeError("v375 expects v374 to route to the live-gate data contract.")

    gate_requirements = read_csv("paper4_v369_gate_requirement_matrix.csv")
    if gate_requirements.empty:
        raise RuntimeError("Missing v369 gate requirement matrix.")
    v374_sections = read_csv("paper4_v374_claim_language_section_draft.csv")
    v374_citations = read_csv("paper4_v374_evidence_citation_map.csv")
    if v374_sections.empty or v374_citations.empty:
        raise RuntimeError("Missing v374 claim language or citation map.")

    contract = pd.DataFrame(
        [
            {
                f"contract_id_v{VERSION}": "bounded_publication_language_pack",
                f"gate_tier_v{VERSION}": "offline_publishable",
                f"required_data_asset_v{VERSION}": "bounded manuscript language with evidence citations",
                f"required_schema_v{VERSION}": "section_id; draft_text; citation_keys; claim_boundary",
                f"minimum_required_evidence_count_v{VERSION}": 5,
                f"current_evidence_count_v{VERSION}": int(len(v374_sections)),
                f"current_source_artifact_v{VERSION}": (
                    "paper4_v374_claim_language_section_draft.csv;"
                    "paper4_v374_evidence_citation_map.csv"
                ),
                f"data_asset_available_v{VERSION}": True,
                f"claim_gate_met_v{VERSION}": True,
                f"claim_language_unlocked_v{VERSION}": "bounded living-lab manuscript language",
                f"claim_language_blocked_v{VERSION}": "working champion; live; legal; final",
                f"blocking_evidence_v{VERSION}": "prohibited-language register remains active",
                f"next_validation_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "offline manuscript wording only",
            },
            {
                f"contract_id_v{VERSION}": "dynamic_proxy_trace_dataset",
                f"gate_tier_v{VERSION}": "offline_proxy",
                f"required_data_asset_v{VERSION}": "dynamic proxy replay trace",
                f"required_schema_v{VERSION}": "policy_id; period; proxy decision; trace metric; claim label",
                f"minimum_required_evidence_count_v{VERSION}": 1,
                f"current_evidence_count_v{VERSION}": int(v369_status["dynamic_proxy_trace_rows_v369"]),
                f"current_source_artifact_v{VERSION}": "paper4_v297_dynamic_proxy_trace.parquet",
                f"data_asset_available_v{VERSION}": True,
                f"claim_gate_met_v{VERSION}": True,
                f"claim_language_unlocked_v{VERSION}": "offline proxy replay evidence",
                f"claim_language_blocked_v{VERSION}": "strict live deployment",
                f"blocking_evidence_v{VERSION}": "proxy trace is not external/future live data",
                f"next_validation_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "proxy replay only",
            },
            {
                f"contract_id_v{VERSION}": "external_future_holdout_panel",
                f"gate_tier_v{VERSION}": "live_deployment",
                f"required_data_asset_v{VERSION}": "external or future holdout decision panel",
                f"required_schema_v{VERSION}": (
                    "loan_id; decision_timestamp; policy_id; funded_flag; realized_outcome"
                ),
                f"minimum_required_evidence_count_v{VERSION}": 1,
                f"current_evidence_count_v{VERSION}": int(v369_status["external_live_pass_rows_v369"]),
                f"current_source_artifact_v{VERSION}": "paper4_v369_gate_requirement_matrix.csv",
                f"data_asset_available_v{VERSION}": bool(v369_status["holdout_data_available_v369"]),
                f"claim_gate_met_v{VERSION}": False,
                f"claim_language_unlocked_v{VERSION}": "none",
                f"claim_language_blocked_v{VERSION}": "strict live deployment",
                f"blocking_evidence_v{VERSION}": "external live pass rows are zero",
                f"next_validation_artifact_v{VERSION}": "external_holdout_panel_not_available",
                f"claim_boundary_v{VERSION}": "live claim blocked",
            },
            {
                f"contract_id_v{VERSION}": "online_shadow_monitoring_log",
                f"gate_tier_v{VERSION}": "live_deployment",
                f"required_data_asset_v{VERSION}": "shadow deployment monitoring log",
                f"required_schema_v{VERSION}": "timestamp; policy_id; prediction; outcome; alert_state",
                f"minimum_required_evidence_count_v{VERSION}": 1,
                f"current_evidence_count_v{VERSION}": 0,
                f"current_source_artifact_v{VERSION}": "not_created",
                f"data_asset_available_v{VERSION}": False,
                f"claim_gate_met_v{VERSION}": False,
                f"claim_language_unlocked_v{VERSION}": "none",
                f"claim_language_blocked_v{VERSION}": "production monitoring readiness",
                f"blocking_evidence_v{VERSION}": "no shadow deployment log exists",
                f"next_validation_artifact_v{VERSION}": "shadow_monitoring_log_not_created",
                f"claim_boundary_v{VERSION}": "monitoring claim blocked",
            },
            {
                f"contract_id_v{VERSION}": "deployment_monitoring_runbook",
                f"gate_tier_v{VERSION}": "live_deployment",
                f"required_data_asset_v{VERSION}": "deployment monitoring and rollback runbook",
                f"required_schema_v{VERSION}": "owner; threshold; rollback action; audit cadence",
                f"minimum_required_evidence_count_v{VERSION}": 1,
                f"current_evidence_count_v{VERSION}": 0,
                f"current_source_artifact_v{VERSION}": "not_created",
                f"data_asset_available_v{VERSION}": False,
                f"claim_gate_met_v{VERSION}": False,
                f"claim_language_unlocked_v{VERSION}": "none",
                f"claim_language_blocked_v{VERSION}": "production deployment readiness",
                f"blocking_evidence_v{VERSION}": "no deployment runbook exists",
                f"next_validation_artifact_v{VERSION}": "deployment_runbook_not_created",
                f"claim_boundary_v{VERSION}": "deployment claim blocked",
            },
            {
                f"contract_id_v{VERSION}": "ifrs9_contractual_coverage_dataset",
                f"gate_tier_v{VERSION}": "contractual_ifrs9",
                f"required_data_asset_v{VERSION}": "complete contractual IFRS9 input coverage",
                f"required_schema_v{VERSION}": "loan_id; stage; ead; pd; lgd; scenario; approval_status",
                f"minimum_required_evidence_count_v{VERSION}": 0,
                f"current_evidence_count_v{VERSION}": int(
                    v369_status["ifrs9_proxy_uncovered_loan_rows_v369"]
                ),
                f"current_source_artifact_v{VERSION}": "paper4_v298_ifrs9_v295_proxy_coverage.csv",
                f"data_asset_available_v{VERSION}": False,
                f"claim_gate_met_v{VERSION}": False,
                f"claim_language_unlocked_v{VERSION}": "none",
                f"claim_language_blocked_v{VERSION}": "contractual IFRS9 claim",
                f"blocking_evidence_v{VERSION}": "76 proxy rows remain uncovered",
                f"next_validation_artifact_v{VERSION}": "ifrs9_contractual_coverage_not_complete",
                f"claim_boundary_v{VERSION}": "IFRS9-inspired proxy only",
            },
            {
                f"contract_id_v{VERSION}": "legal_fairness_attribute_review_file",
                f"gate_tier_v{VERSION}": "legal_fairness",
                f"required_data_asset_v{VERSION}": "approved legal fairness attribute review",
                f"required_schema_v{VERSION}": "attribute; legal basis; consent status; reviewer; decision",
                f"minimum_required_evidence_count_v{VERSION}": 1,
                f"current_evidence_count_v{VERSION}": 0,
                f"current_source_artifact_v{VERSION}": "not_created",
                f"data_asset_available_v{VERSION}": False,
                f"claim_gate_met_v{VERSION}": False,
                f"claim_language_unlocked_v{VERSION}": "none",
                f"claim_language_blocked_v{VERSION}": "legal fairness compliance",
                f"blocking_evidence_v{VERSION}": "fair lending legal review is not in the lab",
                f"next_validation_artifact_v{VERSION}": "legal_fairness_review_not_created",
                f"claim_boundary_v{VERSION}": "fairness proxy diagnostics only",
            },
            {
                f"contract_id_v{VERSION}": "formal_spo_dla_review_packet",
                f"gate_tier_v{VERSION}": "formal_method",
                f"required_data_asset_v{VERSION}": "approved SPO/DLA formal claim review",
                f"required_schema_v{VERSION}": "claim_id; theorem_or_proof; reviewer; approval_status",
                f"minimum_required_evidence_count_v{VERSION}": 1,
                f"current_evidence_count_v{VERSION}": 2,
                f"current_source_artifact_v{VERSION}": "paper4_v369_gate_requirement_matrix.csv",
                f"data_asset_available_v{VERSION}": True,
                f"claim_gate_met_v{VERSION}": False,
                f"claim_language_unlocked_v{VERSION}": "historical audit only",
                f"claim_language_blocked_v{VERSION}": "formal SPO/DLA method claim",
                f"blocking_evidence_v{VERSION}": "v369 records historical audit rows, not approval",
                f"next_validation_artifact_v{VERSION}": "formal_claim_review_not_approved",
                f"claim_boundary_v{VERSION}": "formal claim blocked",
            },
            {
                f"contract_id_v{VERSION}": "full_v55_dual_bound_certificate",
                f"gate_tier_v{VERSION}": "global_solver",
                f"required_data_asset_v{VERSION}": "full-v55 reduced-cost dual-bound certificate",
                f"required_schema_v{VERSION}": "column_id; reduced_cost; dual_bound; termination_flag",
                f"minimum_required_evidence_count_v{VERSION}": 0,
                f"current_evidence_count_v{VERSION}": int(
                    v363_status["v71_improving_omitted_columns_v363"]
                ),
                f"current_source_artifact_v{VERSION}": (
                    "paper4_v363_v353_full_dual_bound_or_gap_certificate.csv"
                ),
                f"data_asset_available_v{VERSION}": False,
                f"claim_gate_met_v{VERSION}": False,
                f"claim_language_unlocked_v{VERSION}": "none",
                f"claim_language_blocked_v{VERSION}": "full-v55 global optimality",
                f"blocking_evidence_v{VERSION}": "5738 improving omitted columns remain",
                f"next_validation_artifact_v{VERSION}": "full_v55_certificate_not_available",
                f"claim_boundary_v{VERSION}": "bounded/gap evidence only",
            },
            {
                f"contract_id_v{VERSION}": "source_governance_full_v55_chunk_screen",
                f"gate_tier_v{VERSION}": "source_governance",
                f"required_data_asset_v{VERSION}": "source-exact full-v55 chunk evidence",
                f"required_schema_v{VERSION}": "chunk_id; source_exact_rows; blocker_family; pass_rows",
                f"minimum_required_evidence_count_v{VERSION}": 1,
                f"current_evidence_count_v{VERSION}": int(
                    v373_status["sampled_total_source_exact_rows_v373"]
                ),
                f"current_source_artifact_v{VERSION}": "paper4_v373_sampled_chunk_source_screen.csv",
                f"data_asset_available_v{VERSION}": False,
                f"claim_gate_met_v{VERSION}": False,
                f"claim_language_unlocked_v{VERSION}": "none",
                f"claim_language_blocked_v{VERSION}": "full-v55 source-governance proof",
                f"blocking_evidence_v{VERSION}": "sampled source-exact rows are zero",
                f"next_validation_artifact_v{VERSION}": "source_exact_chunk_evidence_not_available",
                f"claim_boundary_v{VERSION}": "source-governance blocker remains",
            },
            {
                f"contract_id_v{VERSION}": "final_promotion_approval_gate",
                f"gate_tier_v{VERSION}": "final_promotion",
                f"required_data_asset_v{VERSION}": "approved Paper 4 final-promotion artifact",
                f"required_schema_v{VERSION}": "champion_id; approval_id; reviewer; generated_at",
                f"minimum_required_evidence_count_v{VERSION}": 1,
                f"current_evidence_count_v{VERSION}": 0,
                f"current_source_artifact_v{VERSION}": "paper4_final_promotion.json",
                f"data_asset_available_v{VERSION}": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"claim_gate_met_v{VERSION}": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"claim_language_unlocked_v{VERSION}": "none",
                f"claim_language_blocked_v{VERSION}": "Paper 4 final promotion",
                f"blocking_evidence_v{VERSION}": "final promotion artifact is absent by design",
                f"next_validation_artifact_v{VERSION}": "paper4_final_promotion_gate_not_created",
                f"claim_boundary_v{VERSION}": "final promotion forbidden",
            },
        ]
    )
    gate_readiness = _gate_summary(contract)
    claim_permissions = pd.DataFrame(
        [
            {
                f"claim_id_v{VERSION}": "bounded_living_lab_manuscript_language",
                f"allowed_v{VERSION}": True,
                f"supporting_contract_id_v{VERSION}": "bounded_publication_language_pack",
                f"evidence_count_v{VERSION}": int(len(v374_sections)),
                f"claim_boundary_v{VERSION}": "must cite v361-v374 evidence",
            },
            {
                f"claim_id_v{VERSION}": "offline_proxy_replay_language",
                f"allowed_v{VERSION}": True,
                f"supporting_contract_id_v{VERSION}": "dynamic_proxy_trace_dataset",
                f"evidence_count_v{VERSION}": int(v369_status["dynamic_proxy_trace_rows_v369"]),
                f"claim_boundary_v{VERSION}": "proxy replay only",
            },
            {
                f"claim_id_v{VERSION}": "strict_live_deployment_language",
                f"allowed_v{VERSION}": False,
                f"supporting_contract_id_v{VERSION}": "external_future_holdout_panel",
                f"evidence_count_v{VERSION}": int(v369_status["external_live_pass_rows_v369"]),
                f"claim_boundary_v{VERSION}": "external holdout and monitoring gates unmet",
            },
            {
                f"claim_id_v{VERSION}": "contractual_ifrs9_or_legal_language",
                f"allowed_v{VERSION}": False,
                f"supporting_contract_id_v{VERSION}": "ifrs9_contractual_coverage_dataset",
                f"evidence_count_v{VERSION}": int(
                    v369_status["ifrs9_proxy_uncovered_loan_rows_v369"]
                ),
                f"claim_boundary_v{VERSION}": "contractual/legal evidence absent",
            },
            {
                f"claim_id_v{VERSION}": "full_v55_global_optimality_language",
                f"allowed_v{VERSION}": False,
                f"supporting_contract_id_v{VERSION}": "full_v55_dual_bound_certificate",
                f"evidence_count_v{VERSION}": int(
                    v363_status["v71_improving_omitted_columns_v363"]
                ),
                f"claim_boundary_v{VERSION}": "dual-bound certificate absent",
            },
            {
                f"claim_id_v{VERSION}": "working_champion_or_paper_estrella_replacement",
                f"allowed_v{VERSION}": False,
                f"supporting_contract_id_v{VERSION}": "final_promotion_approval_gate",
                f"evidence_count_v{VERSION}": 0,
                f"claim_boundary_v{VERSION}": "Paper Estrella remains protected",
            },
            {
                f"claim_id_v{VERSION}": "final_paper4_promotion",
                f"allowed_v{VERSION}": False,
                f"supporting_contract_id_v{VERSION}": "final_promotion_approval_gate",
                f"evidence_count_v{VERSION}": 0,
                f"claim_boundary_v{VERSION}": "final promotion forbidden",
            },
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "external_holdout_contract_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v369_status["external_live_pass_rows_v369"]),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "live deployment requires external/future holdout",
            },
            {
                f"blocker_id_v{VERSION}": "deployment_monitoring_runbook_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 0,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "production monitoring readiness remains blocked",
            },
            {
                f"blocker_id_v{VERSION}": "ifrs9_contractual_uncovered_rows",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    v369_status["ifrs9_proxy_uncovered_loan_rows_v369"]
                ),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "contractual IFRS9 remains blocked",
            },
            {
                f"blocker_id_v{VERSION}": "legal_fairness_review_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 0,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "legal fairness compliance remains blocked",
            },
            {
                f"blocker_id_v{VERSION}": "full_v55_dual_bound_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    v363_status["v71_improving_omitted_columns_v363"]
                ),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "global optimality remains blocked",
            },
            {
                f"blocker_id_v{VERSION}": "source_governance_source_exact_zero",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    v373_status["sampled_total_source_exact_rows_v373"]
                ),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "sampled source-exact rows are zero",
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
                "claim_id": "v375_live_gate_data_contract_created",
                "allowed": True,
                "artifact": "paper4_v375_live_gate_data_contract.csv",
                "boundary": "requirements matrix only",
            },
            {
                "claim_id": "v375_offline_proxy_available_data_labeled",
                "allowed": True,
                "artifact": "paper4_v375_claim_permission_register.csv",
                "boundary": "offline/proxy labels only",
            },
            {
                "claim_id": "v375_strict_live_deployment_or_contractual_claim",
                "allowed": False,
                "artifact": "paper4_v375_claim_blockers.csv",
                "boundary": "live and contractual/legal gates unmet",
            },
            {
                "claim_id": "v375_full_v55_global_optimality_claim",
                "allowed": False,
                "artifact": "paper4_v375_claim_blockers.csv",
                "boundary": "global certificate absent",
            },
            {
                "claim_id": "v375_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v375_claim_blockers.csv",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v375_live_gate_data_contract.csv", contract)
    write_csv(TABLE_DIR / "paper4_v375_gate_readiness_summary.csv", gate_readiness)
    write_csv(TABLE_DIR / "paper4_v375_claim_permission_register.csv", claim_permissions)
    write_csv(TABLE_DIR / "paper4_v375_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v375_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    live_gate_met_rows = int(
        contract.loc[
            contract[f"gate_tier_v{VERSION}"].eq("live_deployment"),
            f"claim_gate_met_v{VERSION}",
        ]
        .astype(bool)
        .sum()
    )
    contractual_or_legal_gate_met_rows = int(
        contract.loc[
            contract[f"gate_tier_v{VERSION}"].isin(["contractual_ifrs9", "legal_fairness"]),
            f"claim_gate_met_v{VERSION}",
        ]
        .astype(bool)
        .sum()
    )
    global_solver_gate_met_rows = int(
        contract.loc[
            contract[f"gate_tier_v{VERSION}"].eq("global_solver"),
            f"claim_gate_met_v{VERSION}",
        ]
        .astype(bool)
        .sum()
    )
    final_promotion_gate_met_rows = int(
        contract.loc[
            contract[f"gate_tier_v{VERSION}"].eq("final_promotion"),
            f"claim_gate_met_v{VERSION}",
        ]
        .astype(bool)
        .sum()
    )
    status = {
        "phase": "v375_live_gate_data_contract",
        "schema_version": "2026-05-17.375",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_gate_version_v375": PRIOR_GATE_VERSION,
        "prior_stop_rule_version_v375": PRIOR_STOP_RULE_VERSION,
        "prior_claim_language_version_v375": PRIOR_CLAIM_LANGUAGE_VERSION,
        "prior_gate_requirement_rows_v375": int(v369_status["gate_requirement_rows_v369"]),
        "prior_gate_requirements_met_v375": int(v369_status["gate_requirements_met_v369"]),
        "prior_v374_draft_section_rows_v375": int(v374_status["draft_section_rows_v374"]),
        "contract_rows_v375": int(len(contract)),
        "gate_readiness_rows_v375": int(len(gate_readiness)),
        "claim_permission_rows_v375": int(len(claim_permissions)),
        "claim_blocker_rows_v375": int(len(blockers)),
        "claim_matrix_rows_v375": int(len(claim_matrix)),
        "data_asset_available_rows_v375": int(
            contract[f"data_asset_available_v{VERSION}"].astype(bool).sum()
        ),
        "claim_gate_met_rows_v375": int(contract[f"claim_gate_met_v{VERSION}"].astype(bool).sum()),
        "live_deployment_gate_met_rows_v375": live_gate_met_rows,
        "contractual_or_legal_gate_met_rows_v375": contractual_or_legal_gate_met_rows,
        "global_solver_gate_met_rows_v375": global_solver_gate_met_rows,
        "final_promotion_gate_met_rows_v375": final_promotion_gate_met_rows,
        "strict_live_deployment_language_allowed_v375": live_gate_met_rows > 0,
        "contractual_or_legal_language_allowed_v375": contractual_or_legal_gate_met_rows > 0,
        "global_optimality_language_allowed_v375": global_solver_gate_met_rows > 0,
        "bounded_living_lab_language_allowed_v375": True,
        "offline_proxy_language_allowed_v375": True,
        "working_champion_claim_allowed_v375": False,
        "paper1_promotion_allowed_v375": False,
        "paper4_working_champion_changed_v375": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v375": NEXT_ARTIFACT,
        "claim_boundary": (
            "v375 creates a data contract; live, contractual/legal, global solver and "
            "final-promotion language remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v375_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v375": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
