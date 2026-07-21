"""Build Paper 4 v34 DLA redesign and ADP failure analysis artifacts."""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers.build_paper4_extended_experiments import _safe_read_csv
from scripts.papers.build_paper4_v6_priority_resolution import (
    TABLE_DIR,
    _write_csv,
    _write_json,
    _write_note,
)
from scripts.papers.build_paper4_v10_resolution_wave import PAPER1_PROMOTION, PAPER4_FINAL_PROMOTION

SCHEMA_VERSION = "2026-05-15.34"


def _state_schema() -> pd.DataFrame:
    rows = [
        (
            "S_t",
            "cash",
            "available cash before monthly funding decision",
            "observed/replayed",
            "state",
        ),
        (
            "S_t",
            "outstanding_principal",
            "principal remaining from previously funded loans",
            "proxy from dynamic trace",
            "state",
        ),
        ("S_t", "budget_remaining", "remaining capital budget", "dynamic engine", "state"),
        ("S_t", "stage_mix", "Stage 1/2/3 proxy composition", "IFRS9 proxy panel", "state"),
        (
            "S_t",
            "coverage_state",
            "online conformal gate state",
            "v9/v10 online artifacts",
            "state",
        ),
        (
            "x_t",
            "loan_selection",
            "loans funded this month",
            "policy adapter or endogenous DLA",
            "decision",
        ),
        (
            "S_t^x",
            "funded_exposure",
            "post-decision exposure before outcomes",
            "dynamic engine",
            "post_decision_state",
        ),
        (
            "W_{t+1}",
            "default_prepay_recovery_shock",
            "common random path outcomes",
            "sample paths",
            "exogenous",
        ),
        ("S_{t+1}", "wealth", "cash plus outstanding net of losses", "dynamic trace", "next_state"),
    ]
    return pd.DataFrame(
        rows, columns=["sdam_symbol", "variable", "definition", "artifact_source", "role"]
    )


def _redesign_options() -> pd.DataFrame:
    rows = [
        (
            "rollout_depth_2_tail_source",
            "ADP rollout",
            "add tail/source penalty and two-step lookahead proxy",
            "implemented_as_design_candidate",
            False,
        ),
        (
            "cash_deployment_guard",
            "DLA heuristic",
            "avoid underdeployment while preserving ECL/source caps",
            "implemented_as_design_candidate",
            False,
        ),
        (
            "bellman_fvi_exact",
            "exact Bellman",
            "solve full state/action Bellman recursion",
            "implementation_blocked_state_space",
            False,
        ),
        (
            "risk_constrained_rollout",
            "ADP rollout",
            "include CVaR and source cap penalties in value proxy",
            "implemented_as_design_candidate",
            False,
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "design_id",
            "policy_class",
            "design_change",
            "status_v34",
            "bellman_exact_claim_allowed",
        ],
    )


def _failure_memo() -> pd.DataFrame:
    under = _safe_read_csv(TABLE_DIR / "paper4_v28_dla_fvi_underperformance_diagnosis.csv")
    combined = _safe_read_csv(TABLE_DIR / "paper4_v28_dynamic_combined_summary.csv")
    rows = []
    if not under.empty:
        for _, row in under.iterrows():
            rows.append(
                {
                    "policy_id": row.get("policy_id", ""),
                    "failure_axis": "wealth_gap_tail_or_adapter",
                    "wealth_gap_vs_champion": row.get("wealth_gap_vs_champion", np.nan),
                    "loss_p95_gap_vs_champion": row.get("loss_p95_gap_vs_champion", np.nan),
                    "diagnosis_v34": row.get(
                        "diagnosis_v28", "ADP/DLA needs stronger value function and constraints"
                    ),
                    "action_v34": "retain as learning lane; compare under redesigned rollout before promotion",
                    "claim_boundary_v34": "failure analysis, not optimality proof",
                }
            )
    if not combined.empty and "policy_id" in combined:
        adp = combined[
            combined["policy_id"].astype(str).str.contains("fvi|adp|dla", case=False, na=False)
        ].copy()
        for _, row in adp.head(10).iterrows():
            rows.append(
                {
                    "policy_id": row.get("policy_id", ""),
                    "failure_axis": "capital_deployment_reinvestment_source_tail",
                    "wealth_gap_vs_champion": np.nan,
                    "loss_p95_gap_vs_champion": np.nan,
                    "diagnosis_v34": "dynamic summary shows DLA comparator needs endogenous decision discipline before champion claim",
                    "action_v34": "park as serious method lane unless paired robustness improves",
                    "claim_boundary_v34": "diagnostic only",
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["policy_id", "failure_axis"])


def _case_studies() -> pd.DataFrame:
    cases = _safe_read_csv(TABLE_DIR / "paper4_v28_champion_case_studies.csv")
    if cases.empty:
        return pd.DataFrame()
    out = cases.copy()
    out["version_v34"] = "champion_dla_failure_case_studies"
    out["interpretation_v34"] = np.where(
        out.get("selection_bucket", "")
        .astype(str)
        .str.contains("challenger_only", case=False, na=False),
        "loan bucket selected by challenger but not champion; inspect if DLA takes extra tail/source risk",
        "loan bucket overlap or champion-only exposure; inspect return/reinvestment mechanism",
    )
    return out


def build_v34() -> dict[str, Any]:
    start = time.time()
    schema = _state_schema()
    redesign = _redesign_options()
    failure = _failure_memo()
    cases = _case_studies()
    summary = _safe_read_csv(TABLE_DIR / "paper4_v28_dla_adp_dynamic_summary.csv")
    if not summary.empty:
        summary = summary.copy()
        summary["version_v34"] = "dla_adp_reaudit"
        summary["bellman_exact_claim_allowed"] = False
        summary["claim_boundary_v34"] = "ADP/FVI/rollout approximation only"

    _write_csv("paper4_v34_dla_state_transition_schema.csv", schema)
    _write_csv("paper4_v34_dla_redesign_options.csv", redesign)
    _write_csv("paper4_v34_dla_adp_failure_memo.csv", failure)
    _write_csv("paper4_v34_dla_case_studies.csv", cases)
    _write_csv("paper4_v34_dla_adp_dynamic_summary_reaudit.csv", summary)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v34_dla_redesign_failure_analysis",
        "state_variables_v34": int(len(schema)),
        "redesign_options_v34": int(len(redesign)),
        "failure_rows_v34": int(len(failure)),
        "bellman_exact_claim_allowed": False,
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "DLA redesigned as formal SDAM lane, but exact Bellman optimality remains false",
        "runtime_seconds": round(time.time() - start, 3),
    }
    _write_json("paper4_v34_status.json", status)
    _write_note(
        "paper4_v34_dla_redesign_failure_analysis.md",
        "\n".join(
            [
                "# Paper 4 v34 DLA Redesign and Failure Analysis",
                "",
                f"- State variables/roles: `{status['state_variables_v34']}`.",
                f"- Redesign options: `{status['redesign_options_v34']}`.",
                "- Exact Bellman optimality claim remains false.",
            ]
        ),
    )
    print(pd.Series(status).to_json(indent=2))
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    build_v34()


if __name__ == "__main__":
    main()
