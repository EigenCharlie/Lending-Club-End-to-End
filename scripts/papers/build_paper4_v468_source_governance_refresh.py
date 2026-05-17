#!/usr/bin/env python3
"""Build Paper 4 v468 source-governance refresh artifacts."""

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

VERSION = 468
PRIOR_CVAR_FRONTIER_VERSION = 467
NEXT_ARTIFACT = "paper4_v469_dynamic_replay_reproducibility_probe.md"
REFRESH_MD = NOTEBOOK.parent / "paper4_v468_source_governance_refresh.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _first_row(name: str) -> pd.Series:
    data = pd.read_csv(TABLE_DIR / name)
    if data.empty:
        raise RuntimeError(f"Expected non-empty artifact: {name}")
    return data.iloc[0]


def _tight_source_rankings() -> pd.DataFrame:
    tight = pd.read_csv(TABLE_DIR / "paper4_v371_tight_source_blockers.csv").copy()
    retention = pd.read_csv(TABLE_DIR / "paper4_v371_source_family_retention.csv").copy()
    flows = pd.read_csv(TABLE_DIR / "paper4_v371_source_pair_flow_diagnostics.csv").copy()
    tight["source_family_v371"] = tight["source_family_v371"].astype(str)
    tight["source_id_v371"] = tight["source_id_v371"].astype(str)
    rows: list[dict[str, Any]] = []
    for _, row in tight.iterrows():
        family = str(row["source_family_v371"])
        source_id = str(row["source_id_v371"])
        family_row = retention.loc[retention["source_family_v371"].astype(str).eq(family)]
        pressure = flows.loc[
            flows["source_family_v371"].astype(str).eq(family)
            & flows["source_id_v371"].astype(str).eq(source_id)
            & flows["flow_category_v371"].eq("source_pressure_drop_other_add_tight")
        ]
        relief = flows.loc[
            flows["source_family_v371"].astype(str).eq(family)
            & flows["source_id_v371"].astype(str).eq(source_id)
            & flows["flow_category_v371"].eq("source_relief_drop_tight_add_other")
        ]
        rows.append(
            {
                "source_family_v468": family,
                "source_id_v468": source_id,
                "source_slack_v468": float(row["source_slack_v371"]),
                "source_slack_rank_v468": int(row["source_slack_rank_v371"]),
                "v356_positive_return_candidate_rows_v468": int(
                    row["v356_positive_return_candidate_rows_v371"]
                ),
                "tight_source_pass_rows_v468": int(row["tight_source_pass_rows_v371"]),
                "primary_blocker_v468": bool(row["primary_blocker_v371"]),
                "family_retention_share_v468": (
                    float(family_row["family_retention_share_v371"].iloc[0])
                    if not family_row.empty
                    else 0.0
                ),
                "source_pressure_budget_return_rows_v468": (
                    int(pressure["budget_return_rows_v371"].iloc[0])
                    if not pressure.empty
                    else 0
                ),
                "source_pressure_share_v468": (
                    float(pressure["share_of_budget_return_rows_v371"].iloc[0])
                    if not pressure.empty
                    else 0.0
                ),
                "source_relief_budget_return_rows_v468": (
                    int(relief["budget_return_rows_v371"].iloc[0]) if not relief.empty else 0
                ),
                "claim_boundary_v468": "source-governance diagnostic only",
            }
        )
    return pd.DataFrame(rows)


def _refresh_summary(rankings: pd.DataFrame) -> pd.DataFrame:
    diagnostic = _first_row("paper4_v371_source_governance_blocker_diagnostic.csv")
    relief = _first_row("paper4_v372_grade_a_source_relief_prefilter.csv")
    stop_rule = _first_row("paper4_v373_full_v55_chunk_002_or_stop_rule.csv")
    audit_status = _read_status(383)
    primary = rankings.loc[rankings["primary_blocker_v468"]].iloc[0]
    return pd.DataFrame(
        [
            {
                "summary_id_v468": "source_governance_refresh",
                "primary_blocker_family_v468": str(diagnostic["primary_blocker_family_v371"]),
                "primary_blocker_source_id_v468": str(
                    diagnostic["primary_blocker_source_id_v371"]
                ),
                "primary_blocker_pass_rows_v468": int(
                    diagnostic["primary_blocker_pass_rows_v371"]
                ),
                "secondary_blocker_family_v468": str(
                    diagnostic["secondary_blocker_family_v371"]
                ),
                "secondary_blocker_pass_rows_v468": int(
                    diagnostic["secondary_blocker_pass_rows_v371"]
                ),
                "fully_nonbinding_family_count_v468": int(
                    diagnostic["fully_nonbinding_family_count_v371"]
                ),
                "grade_a_pressure_budget_return_rows_v468": int(
                    primary["source_pressure_budget_return_rows_v468"]
                ),
                "grade_a_pressure_share_v468": float(primary["source_pressure_share_v468"]),
                "grade_a_relief_return_improving_rows_v468": int(
                    relief["grade_a_relief_return_improving_rows_v372"]
                ),
                "best_grade_a_relief_budget_return_delta_v468": float(
                    relief["best_grade_a_relief_budget_return_delta_v372"]
                ),
                "sampled_chunk_count_v468": int(stop_rule["sampled_chunk_count_v373"]),
                "sampled_total_budget_return_rows_v468": int(
                    stop_rule["sampled_total_budget_return_rows_v373"]
                ),
                "sampled_total_source_exact_rows_v468": int(
                    stop_rule["sampled_total_source_exact_rows_v373"]
                ),
                "audit_plan_rows_v468": int(audit_status["audit_plan_rows_v383"]),
                "source_cap_relaxation_authorized_v468": False,
                "blind_chunking_restarted_v468": False,
                "claim_boundary_v468": (
                    "source blockers refreshed; no cap relaxation, candidate apply or "
                    "full-v55 proof"
                ),
            }
        ]
    )


def _execution_decision() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id_v468": "keep_grade_a_as_primary_blocker",
                "recommended_v468": True,
                "evidence_v468": (
                    "grade=A has zero pass rows in v371 and zero return-improving "
                    "relief in v372"
                ),
                "next_artifact_v468": NEXT_ARTIFACT,
                "claim_boundary_v468": "use as claim caveat, not cap relaxation",
            },
            {
                "decision_id_v468": "do_not_restart_blind_chunking",
                "recommended_v468": True,
                "evidence_v468": "v373 sampled eight chunks with zero source-exact rows",
                "next_artifact_v468": NEXT_ARTIFACT,
                "claim_boundary_v468": "stop-rule remains diagnostic",
            },
            {
                "decision_id_v468": "do_not_relax_source_caps",
                "recommended_v468": True,
                "evidence_v468": "v383 audit plan explicitly lacks cap approval",
                "next_artifact_v468": "future_source_cap_approval_or_counterfactual",
                "claim_boundary_v468": "no governance approval or policy change",
            },
            {
                "decision_id_v468": "route_to_dynamic_replay_lane",
                "recommended_v468": True,
                "evidence_v468": "source blockers are now documented enough for bounded claims",
                "next_artifact_v468": NEXT_ARTIFACT,
                "claim_boundary_v468": "routing decision only",
            },
        ]
    )


def _blocker_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v468": "grade_a_primary_source_blocker",
                "blocking_v468": True,
                "evidence_count_v468": 0,
                "required_next_artifact_v468": "future_grade_a_source_counterfactual",
                "claim_boundary_v468": "grade=A pass rows remain zero",
            },
            {
                "blocker_id_v468": "source_cap_relaxation_not_authorized",
                "blocking_v468": True,
                "evidence_count_v468": 1,
                "required_next_artifact_v468": "future_source_cap_approval_or_counterfactual",
                "claim_boundary_v468": "no source cap change authorized",
            },
            {
                "blocker_id_v468": "blind_chunking_stop_rule_active",
                "blocking_v468": True,
                "evidence_count_v468": 8,
                "required_next_artifact_v468": "future_targeted_source_aware_pricing",
                "claim_boundary_v468": "sampled chunks produced zero source-exact rows",
            },
            {
                "blocker_id_v468": "global_solver_claims_still_blocked",
                "blocking_v468": True,
                "evidence_count_v468": 1,
                "required_next_artifact_v468": "future_full_v55_certificate_pack",
                "claim_boundary_v468": "source refresh is not a solver certificate",
            },
            {
                "blocker_id_v468": "paper4_final_promotion_forbidden",
                "blocking_v468": True,
                "evidence_count_v468": 1,
                "required_next_artifact_v468": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v468": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v468_source_governance_refresh_created",
                "allowed": True,
                "artifact": "paper4_v468_source_governance_refresh.csv",
                "boundary": "diagnostic refresh only",
            },
            {
                "claim_id": "v468_grade_a_primary_blocker_documented",
                "allowed": True,
                "artifact": "paper4_v468_tight_source_rankings.csv",
                "boundary": "source blocker statement only",
            },
            {
                "claim_id": "v468_blind_chunking_stop_rule_reaffirmed",
                "allowed": True,
                "artifact": "paper4_v468_source_execution_decision.csv",
                "boundary": "execution routing only",
            },
            {
                "claim_id": "v468_source_caps_relaxed_or_approved",
                "allowed": False,
                "artifact": "paper4_v468_source_blocker_register.csv",
                "boundary": "no approval or cap mutation",
            },
            {
                "claim_id": "v468_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "no champion, global proof or final promotion",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v468 documents grade=A as the primary source-governance blocker.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v468_tight_source_rankings.csv"
                ),
                "boundary": "Diagnostic source-blocker statement only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v468 reaffirms the stop rule against blind full-v55 chunking.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v468_source_execution_decision.csv"
                ),
                "boundary": "Execution routing only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v468 relaxes or approves source caps.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v468_source_blocker_register.csv"
                ),
                "boundary": "No cap approval, cap mutation or policy relaxation.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v468 proves global solver optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v468_source_blocker_register.csv"
                ),
                "boundary": "Source refresh is not a solver certificate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v468 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v468_source_blocker_register.csv"
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
                "lane": "Source Governance",
                "executable_item": "v468 refreshes source-governance blockers.",
                "status": "source_governance_refresh_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v469 audits dynamic replay reproducibility with source blockers preserved"
                ),
                "last_wave": "v468",
                "execution_result": "grade_a_primary_blocker_and_blind_chunk_stop_rule_reaffirmed",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v468")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _refresh_markdown(status: dict[str, Any]) -> str:
    primary_blocker = (
        f"{status['primary_blocker_family_v468']}="
        f"{status['primary_blocker_source_id_v468']}"
    )
    return f"""# Paper 4 Source-Governance Refresh v468

Generated: {status["generated_at_utc"]}

## Result

v468 refreshes the source-governance lane around the current CVaR frontier. The
primary bottleneck remains grade=A: it has zero pass rows in the source blocker
diagnostic, zero return-improving relief rows in the grade-A prefilter, and the
sampled full-v55 chunks still produce zero source-exact rows.

## Counts

- Primary blocker: `{primary_blocker}`.
- Primary blocker pass rows: `{status["primary_blocker_pass_rows_v468"]}`.
- Grade-A pressure rows: `{status["grade_a_pressure_budget_return_rows_v468"]}`.
- Grade-A relief return-improving rows: `{status["grade_a_relief_return_improving_rows_v468"]}`.
- Sampled chunks: `{status["sampled_chunk_count_v468"]}`.
- Sampled source-exact rows: `{status["sampled_total_source_exact_rows_v468"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v468 documents source-governance blockers. It does not relax source caps, approve
policy changes, restart blind chunking, prove global optimality, authorize a
working champion, replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V468_SOURCE_GOVERNANCE_REFRESH_START -->"
    end = "<!-- V468_SOURCE_GOVERNANCE_REFRESH_END -->"
    block = f"""
{start}

## Wave v468: Source-Governance Refresh

Generated: {status["generated_at_utc"]}

### Objective

v468 refreshes the source-governance blocker surface after the v467 CVaR
frontier probe.

### Results

- Primary blocker:
  `{status["primary_blocker_family_v468"]}={status["primary_blocker_source_id_v468"]}`.
- Primary blocker pass rows:
  `{status["primary_blocker_pass_rows_v468"]}`.
- Secondary blocker pass rows:
  `{status["secondary_blocker_pass_rows_v468"]}`.
- Fully nonbinding source families:
  `{status["fully_nonbinding_family_count_v468"]}`.
- Grade-A pressure rows:
  `{status["grade_a_pressure_budget_return_rows_v468"]}`.
- Grade-A relief return-improving rows:
  `{status["grade_a_relief_return_improving_rows_v468"]}`.
- Sampled source-exact rows:
  `{status["sampled_total_source_exact_rows_v468"]}`.
- Source cap relaxation authorized:
  `{status["source_cap_relaxation_authorized_v468"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v468"]}`.

### Interpretation

The source-governance blocker is concentrated enough to be a clear claim
caveat: grade=A is the primary bottleneck, score_decile=0 is secondary, and
blind chunking is not a high-value next move without a source-aware route.

### Claim Impact

- Allowed: grade-A blocker language and blind-chunking stop-rule language.
- Still prohibited: source-cap relaxation, global solver optimality,
  working-champion language, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v468 in the living notebook. v469 should move to dynamic replay
reproducibility while preserving the source-governance caveat.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v467 = _read_status(PRIOR_CVAR_FRONTIER_VERSION)
    if v467["next_artifact_v467"] != "paper4_v468_source_governance_refresh.md":
        raise RuntimeError("v468 expects v467 to route to source governance refresh.")
    if v467["local_frontier_candidate_v467"] != "v353":
        raise RuntimeError("v468 expects v467 to identify v353 as local frontier.")

    rankings = _tight_source_rankings()
    summary = _refresh_summary(rankings)
    decision = _execution_decision()
    blockers = _blocker_register()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v468_tight_source_rankings.csv", rankings)
    write_csv(TABLE_DIR / "paper4_v468_source_governance_refresh.csv", summary)
    write_csv(TABLE_DIR / "paper4_v468_source_execution_decision.csv", decision)
    write_csv(TABLE_DIR / "paper4_v468_source_blocker_register.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v468_claim_matrix_delta.csv", claim_matrix)

    row = summary.iloc[0]
    status = {
        "phase": "v468_source_governance_refresh",
        "schema_version": "2026-05-17.468",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_cvar_frontier_version_v468": PRIOR_CVAR_FRONTIER_VERSION,
        "source_governance_refresh_created_v468": True,
        "tight_source_ranking_rows_v468": len(rankings),
        "primary_blocker_family_v468": str(row["primary_blocker_family_v468"]),
        "primary_blocker_source_id_v468": str(row["primary_blocker_source_id_v468"]),
        "primary_blocker_pass_rows_v468": int(row["primary_blocker_pass_rows_v468"]),
        "secondary_blocker_family_v468": str(row["secondary_blocker_family_v468"]),
        "secondary_blocker_pass_rows_v468": int(row["secondary_blocker_pass_rows_v468"]),
        "fully_nonbinding_family_count_v468": int(row["fully_nonbinding_family_count_v468"]),
        "grade_a_pressure_budget_return_rows_v468": int(
            row["grade_a_pressure_budget_return_rows_v468"]
        ),
        "grade_a_pressure_share_v468": float(row["grade_a_pressure_share_v468"]),
        "grade_a_relief_return_improving_rows_v468": int(
            row["grade_a_relief_return_improving_rows_v468"]
        ),
        "sampled_chunk_count_v468": int(row["sampled_chunk_count_v468"]),
        "sampled_total_source_exact_rows_v468": int(
            row["sampled_total_source_exact_rows_v468"]
        ),
        "audit_plan_rows_v468": int(row["audit_plan_rows_v468"]),
        "source_cap_relaxation_authorized_v468": False,
        "blind_chunking_restarted_v468": False,
        "global_solver_claim_allowed_v468": False,
        "working_champion_claim_allowed_v468": False,
        "paper1_promotion_allowed_v468": False,
        "paper4_working_champion_changed_v468": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v468": NEXT_ARTIFACT,
        "claim_boundary": (
            "v468 refreshes source-governance blockers only; cap relaxation, "
            "global solver, champion and final promotion claims remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v468 must not create final Paper 4 promotion.")

    REFRESH_MD.write_text(_refresh_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v468": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
