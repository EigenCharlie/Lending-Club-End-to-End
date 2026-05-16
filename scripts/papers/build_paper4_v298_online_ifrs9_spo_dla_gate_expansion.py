#!/usr/bin/env python3
"""Build Paper 4 v298 online/IFRS9/SPO-DLA gate-expansion artifacts."""

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
    read_parquet,
    write_csv,
    write_json,
)

VERSION = 298
SOURCE_CANDIDATE_VERSION = 295
GLOBAL_DYNAMIC_GATE_VERSION = 297
NEXT_VERSION = 299


def _online_gate_transfer() -> pd.DataFrame:
    winners = read_csv("paper4_v65_online_margin_repair_winners.csv")
    scorecard = read_csv("paper4_v67_external_holdout_scorecard.csv")
    rows: list[dict[str, Any]] = []
    for _, row in winners.iterrows():
        rows.append(
            {
                f"gate_lane_v{VERSION}": "online_internal_margin_repair",
                "source_family": str(row["source_family"]),
                f"source_artifact_v{VERSION}": "paper4_v65_online_margin_repair_winners.csv",
                f"gate_available_v{VERSION}": bool(row["all_splits_gate_pass_v65"]),
                f"gate_pass_v{VERSION}": bool(row["all_splits_gate_pass_v65"]),
                f"external_holdout_available_v{VERSION}": False,
                f"strict_live_claim_allowed_v{VERSION}": False,
                f"claim_boundary_v{VERSION}": (
                    "internal historical pseudo-unseen pass only; not transferred to live v295"
                ),
            }
        )
    for _, row in scorecard.iterrows():
        rows.append(
            {
                f"gate_lane_v{VERSION}": "online_external_holdout",
                "source_family": str(row["source_family"]),
                f"source_artifact_v{VERSION}": "paper4_v67_external_holdout_scorecard.csv",
                f"gate_available_v{VERSION}": bool(row["holdout_data_available_v67"]),
                f"gate_pass_v{VERSION}": bool(row["all_gates_pass_v67"]),
                f"external_holdout_available_v{VERSION}": bool(row["holdout_data_available_v67"]),
                f"strict_live_claim_allowed_v{VERSION}": bool(
                    row["strict_live_deployability_claim_allowed"]
                ),
                f"claim_boundary_v{VERSION}": (
                    "external holdout gate remains blocked by missing future data"
                ),
            }
        )
    return pd.DataFrame(rows)


def _ifrs9_proxy_coverage(v295: pd.DataFrame) -> pd.DataFrame:
    panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet")
    v295_ids = set(v295["loan_id"].astype(str))
    panel = panel.loc[panel["loan_id"].astype(str).isin(v295_ids)].copy()
    panel["loan_id"] = panel["loan_id"].astype(str)
    dedup = panel.drop_duplicates(["loan_id", "month_index", "scenario"]).copy()
    covered_ids = set(dedup["loan_id"].astype(str))
    selected_rows = int(len(v295_ids))
    covered_rows = int(len(covered_ids))
    coverage_share = covered_rows / max(selected_rows, 1)
    return pd.DataFrame(
        [
            {
                f"candidate_version_v{VERSION}": SOURCE_CANDIDATE_VERSION,
                f"selected_rows_v{VERSION}": selected_rows,
                f"ifrs9_proxy_covered_loan_rows_v{VERSION}": covered_rows,
                f"ifrs9_proxy_uncovered_loan_rows_v{VERSION}": selected_rows - covered_rows,
                f"ifrs9_proxy_coverage_share_v{VERSION}": coverage_share,
                f"ifrs9_proxy_panel_rows_v{VERSION}": int(len(dedup)),
                f"ifrs9_proxy_scenarios_v{VERSION}": int(dedup["scenario"].nunique())
                if not dedup.empty
                else 0,
                f"ifrs9_proxy_months_v{VERSION}": int(dedup["month_index"].nunique())
                if not dedup.empty
                else 0,
                f"covered_ecl_proxy_total_v{VERSION}": float(dedup["ecl_proxy_v29"].sum())
                if "ecl_proxy_v29" in dedup
                else 0.0,
                f"covered_net_cash_proxy_total_v{VERSION}": float(dedup["net_cash_proxy_v47"].sum())
                if "net_cash_proxy_v47" in dedup
                else 0.0,
                f"contractual_ifrs9_claim_allowed_v{VERSION}": False,
                f"claim_boundary_v{VERSION}": (
                    "partial loan-id proxy coverage only; contractual IFRS9 remains blocked"
                ),
            }
        ]
    )


def _spo_dla_claim_audit() -> pd.DataFrame:
    v46 = json.loads((STATUS_DIR / "paper4_v46_status.json").read_text(encoding="utf-8"))
    return pd.DataFrame(
        [
            {
                f"gate_lane_v{VERSION}": "spo_dependency_audit",
                f"source_artifact_v{VERSION}": "paper4_v46_status.json",
                f"evidence_rows_v{VERSION}": int(v46["spo_dependency_rows_v46"]),
                f"gate_pass_v{VERSION}": True,
                f"formal_claim_allowed_v{VERSION}": bool(
                    v46["formal_differentiable_spo_claim_allowed"]
                ),
                f"claim_boundary_v{VERSION}": (
                    "SPO dependencies audited historically; no formal differentiable SPO claim"
                ),
            },
            {
                f"gate_lane_v{VERSION}": "dla_common_path_audit",
                f"source_artifact_v{VERSION}": "paper4_v46_status.json",
                f"evidence_rows_v{VERSION}": int(v46["dla_common_path_rows_v46"]),
                f"gate_pass_v{VERSION}": True,
                f"formal_claim_allowed_v{VERSION}": bool(v46["bellman_exact_claim_allowed"]),
                f"claim_boundary_v{VERSION}": (
                    "DLA common-path replay exists historically; no exact Bellman claim"
                ),
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v298 online/IFRS9/SPO-DLA gate expansion for v295.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v298_online_ifrs9_spo_dla_gate_expansion.csv"
                ),
                "boundary": "Gate audit and partial proxy coverage only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v298 transfers historical online gates into live deployability.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v298_claim_blockers.csv"
                ),
                "boundary": "External holdout data is still missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v298 implements contractual IFRS9 for v295.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v298_claim_blockers.csv"
                ),
                "boundary": "Only partial IFRS9-inspired proxy coverage exists.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v298 proves formal differentiable SPO or exact Bellman DLA.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v298_claim_blockers.csv"
                ),
                "boundary": "Historical audits exist but formal claims remain blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v298 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v298_claim_blockers.csv"
                ),
                "boundary": "No final promotion, dynamic validation or deployment gate is created.",
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
                "lane": "Online/IFRS9/SPO-DLA",
                "executable_item": (
                    "v298 audits transferability of historical online, IFRS9 proxy and "
                    "SPO-DLA gates to the v295/v296 candidate."
                ),
                "status": "gate_expansion_executed_claims_remain_blocked",
                "next_artifact": f"paper4_v{NEXT_VERSION}_v295_cashflow_or_online_holdout_rerun.csv",
                "success_condition": (
                    "build actual v295-specific cashflow/online holdout reruns or keep claims blocked"
                ),
                "last_wave": "v298",
                "execution_result": "partial_ifrs9_proxy_coverage_online_external_spo_dla_blocked",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v298")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V298_ONLINE_IFRS9_SPO_DLA_GATE_EXPANSION_START -->"
    end = "<!-- V298_ONLINE_IFRS9_SPO_DLA_GATE_EXPANSION_END -->"
    block = f"""
{start}

## Wave v298: Online / IFRS9 / SPO-DLA Gate Expansion

Generated: {status["generated_at_utc"]}

### Objective

v297 showed that v295 has a favorable dynamic proxy replay but still lacks
online/source, IFRS9 proxy and SPO-DLA gate transfer. v298 audits those lanes
explicitly for the v295/v296 candidate.

### Results

- Online transfer rows: `{status["online_transfer_rows_v298"]}`.
- Internal online pass rows: `{status["online_internal_pass_rows_v298"]}`.
- External online live pass rows: `{status["online_external_live_pass_rows_v298"]}`.
- IFRS9 proxy covered loans: `{status["ifrs9_proxy_covered_loan_rows_v298"]}`.
- IFRS9 proxy coverage share: `{status["ifrs9_proxy_coverage_share_v298"]}`.
- SPO/DLA audit rows: `{status["spo_dla_audit_rows_v298"]}`.
- Strict live deployability claim allowed:
  `{status["strict_live_deployability_claim_allowed_v298"]}`.
- Contractual IFRS9 claim allowed:
  `{status["contractual_ifrs9_claim_allowed_v298"]}`.
- Formal SPO/DLA claim allowed:
  `{status["formal_spo_dla_claim_allowed_v298"]}`.

### Interpretation

v298 prevents an easy but invalid shortcut: historical gates are useful context,
but they do not automatically transfer to the v295 candidate. Online external
holdout is still missing, IFRS9 coverage is partial and proxy-only, and SPO/DLA
formal claims remain blocked.

### Claim Impact

- Allowed: gate expansion audit and partial IFRS9 proxy coverage diagnostic.
- Still prohibited: live deployability, contractual IFRS9, formal SPO/DLA,
  working champion replacement, Paper Estrella replacement and final promotion.

### Quarto Promotion Decision

Keep v298 in the living notebook. Promotion remains blocked.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    v295 = read_parquet("paper4_v295_broader_multi_swap_allocations.parquet")
    v297_status = json.loads((STATUS_DIR / "paper4_v297_status.json").read_text(encoding="utf-8"))
    if v295.empty:
        raise RuntimeError("Missing v295 allocation input for v298.")
    if not bool(v297_status["v295_dynamic_proxy_beats_v293_v297"]):
        raise RuntimeError("v298 expects v297 to identify v295 as the proxy replay candidate.")

    online = _online_gate_transfer()
    ifrs9 = _ifrs9_proxy_coverage(v295)
    spo_dla = _spo_dla_claim_audit()
    internal_online_pass_rows = int(
        online.loc[online[f"gate_lane_v{VERSION}"].eq("online_internal_margin_repair")][
            f"gate_pass_v{VERSION}"
        ].sum()
    )
    external_live_pass_rows = int(
        online.loc[online[f"gate_lane_v{VERSION}"].eq("online_external_holdout")][
            f"strict_live_claim_allowed_v{VERSION}"
        ].sum()
    )
    contractual_ifrs9_allowed = bool(ifrs9[f"contractual_ifrs9_claim_allowed_v{VERSION}"].any())
    formal_spo_dla_allowed = bool(spo_dla[f"formal_claim_allowed_v{VERSION}"].any())
    strict_live_allowed = bool(online[f"strict_live_claim_allowed_v{VERSION}"].any())
    ifrs9_row = ifrs9.iloc[0]
    summary = pd.DataFrame(
        [
            {
                f"gate_id_v{VERSION}": "v298_online_ifrs9_spo_dla_gate_expansion",
                f"source_candidate_version_v{VERSION}": SOURCE_CANDIDATE_VERSION,
                f"global_dynamic_gate_version_v{VERSION}": GLOBAL_DYNAMIC_GATE_VERSION,
                f"online_transfer_rows_v{VERSION}": int(len(online)),
                f"online_internal_pass_rows_v{VERSION}": internal_online_pass_rows,
                f"online_external_live_pass_rows_v{VERSION}": external_live_pass_rows,
                f"ifrs9_proxy_covered_loan_rows_v{VERSION}": int(
                    ifrs9_row[f"ifrs9_proxy_covered_loan_rows_v{VERSION}"]
                ),
                f"ifrs9_proxy_uncovered_loan_rows_v{VERSION}": int(
                    ifrs9_row[f"ifrs9_proxy_uncovered_loan_rows_v{VERSION}"]
                ),
                f"ifrs9_proxy_coverage_share_v{VERSION}": float(
                    ifrs9_row[f"ifrs9_proxy_coverage_share_v{VERSION}"]
                ),
                f"spo_dla_audit_rows_v{VERSION}": int(len(spo_dla)),
                f"strict_live_deployability_claim_allowed_v{VERSION}": strict_live_allowed,
                f"contractual_ifrs9_claim_allowed_v{VERSION}": contractual_ifrs9_allowed,
                f"formal_spo_dla_claim_allowed_v{VERSION}": formal_spo_dla_allowed,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_v295_cashflow_or_online_holdout_rerun.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "gate expansion only; historical gates do not transfer to promotion"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "external_online_holdout_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": external_live_pass_rows,
                f"required_next_artifact_v{VERSION}": "future_external_holdout_or_temporal_rerun",
                f"claim_boundary_v{VERSION}": "no external online live validation for v295",
            },
            {
                f"blocker_id_v{VERSION}": "ifrs9_proxy_partial_coverage_only",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    ifrs9_row[f"ifrs9_proxy_uncovered_loan_rows_v{VERSION}"]
                ),
                f"required_next_artifact_v{VERSION}": "future_v295_ifrs9_cashflow_panel",
                f"claim_boundary_v{VERSION}": "contractual IFRS9 remains data-blocked",
            },
            {
                f"blocker_id_v{VERSION}": "formal_spo_dla_claims_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(spo_dla)),
                f"required_next_artifact_v{VERSION}": "future_formal_spo_dla_protocol",
                f"claim_boundary_v{VERSION}": "historical SPO/DLA audits are not formal claims",
            },
            {
                f"blocker_id_v{VERSION}": "paper4_working_champion_gate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_global_dynamic_promotion_gate",
                f"claim_boundary_v{VERSION}": "working champion replacement remains blocked",
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
                "claim_id": "v298_gate_expansion_executed",
                "allowed": True,
                "artifact": "paper4_v298_online_ifrs9_spo_dla_gate_expansion.csv",
                "boundary": "audit only",
            },
            {
                "claim_id": "v298_strict_live_online_deployability",
                "allowed": False,
                "artifact": "paper4_v298_claim_blockers.csv",
                "boundary": "external holdout missing",
            },
            {
                "claim_id": "v298_contractual_ifrs9",
                "allowed": False,
                "artifact": "paper4_v298_claim_blockers.csv",
                "boundary": "partial proxy coverage only",
            },
            {
                "claim_id": "v298_formal_spo_dla",
                "allowed": False,
                "artifact": "paper4_v298_claim_blockers.csv",
                "boundary": "formal SPO/DLA claims missing",
            },
            {
                "claim_id": "v298_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v298_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v298_online_ifrs9_spo_dla_gate_expansion.csv", summary)
    write_csv(TABLE_DIR / "paper4_v298_online_gate_transfer_audit.csv", online)
    write_csv(TABLE_DIR / "paper4_v298_ifrs9_v295_proxy_coverage.csv", ifrs9)
    write_csv(TABLE_DIR / "paper4_v298_spo_dla_claim_audit.csv", spo_dla)
    write_csv(TABLE_DIR / "paper4_v298_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v298_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = summary.iloc[0]
    status = {
        "phase": "v298_online_ifrs9_spo_dla_gate_expansion",
        "schema_version": "2026-05-15.298",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "source_candidate_version_v298": SOURCE_CANDIDATE_VERSION,
        "global_dynamic_gate_version_v298": GLOBAL_DYNAMIC_GATE_VERSION,
        "online_transfer_rows_v298": int(row[f"online_transfer_rows_v{VERSION}"]),
        "online_internal_pass_rows_v298": int(row[f"online_internal_pass_rows_v{VERSION}"]),
        "online_external_live_pass_rows_v298": int(
            row[f"online_external_live_pass_rows_v{VERSION}"]
        ),
        "ifrs9_proxy_covered_loan_rows_v298": int(row[f"ifrs9_proxy_covered_loan_rows_v{VERSION}"]),
        "ifrs9_proxy_uncovered_loan_rows_v298": int(
            row[f"ifrs9_proxy_uncovered_loan_rows_v{VERSION}"]
        ),
        "ifrs9_proxy_coverage_share_v298": float(row[f"ifrs9_proxy_coverage_share_v{VERSION}"]),
        "spo_dla_audit_rows_v298": int(row[f"spo_dla_audit_rows_v{VERSION}"]),
        "strict_live_deployability_claim_allowed_v298": bool(
            row[f"strict_live_deployability_claim_allowed_v{VERSION}"]
        ),
        "contractual_ifrs9_claim_allowed_v298": bool(
            row[f"contractual_ifrs9_claim_allowed_v{VERSION}"]
        ),
        "formal_spo_dla_claim_allowed_v298": bool(row[f"formal_spo_dla_claim_allowed_v{VERSION}"]),
        "working_champion_claim_allowed_v298": False,
        "paper1_promotion_allowed_v298": False,
        "paper4_working_champion_changed_v298": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v298": int(len(blockers)),
        "claim_matrix_rows_v298": int(len(claim_matrix)),
        "next_artifact_v298": row[f"next_artifact_v{VERSION}"],
        "claim_boundary": (
            "v298 audits online/IFRS9/SPO-DLA gates for v295; live deployment, "
            "contractual IFRS9, formal SPO-DLA and promotion remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v298_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v298": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
