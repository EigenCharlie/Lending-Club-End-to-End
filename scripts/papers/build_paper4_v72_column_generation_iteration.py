#!/usr/bin/env python3
"""Build Paper 4 v72 column-generation iteration-1 artifacts.

v72 consumes v71 negative reduced-cost columns, appends them to the v69
restricted master for each policy/regime where they exist, and resolves the
continuous LP.  This is one executable column-generation iteration, not a
termination certificate; the new solution must be re-priced in v73.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.papers import build_paper4_v70_restricted_master_solver as v70  # noqa: E402

PAPER4_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = PAPER4_ROOT / "tables"
STATUS_DIR = PAPER4_ROOT / "status"
NOTE_DIR = PAPER4_ROOT / "notes"
NOTEBOOK = NOTE_DIR / "paper4_living_lab_notebook.md"
FORBIDDEN_FINAL_PROMOTION = STATUS_DIR / "paper4_final_promotion.json"
V71_ROLE = "v71_negative_reduced_cost_column"


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_csv(name: str, directory: Path = TABLE_DIR) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_parquet(
    name: str | Path, directory: Path = TABLE_DIR, columns: list[str] | None = None
) -> pd.DataFrame:
    path = Path(name)
    if not path.is_absolute():
        path = directory / path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, columns=columns)


def write_csv(path: Path, df: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(df) if isinstance(df, list) else df
    out.to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rename_v70_to_v72(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={col: col.replace("_v70", "_v72") for col in df.columns})


def _enriched_base_master(universe: pd.DataFrame) -> pd.DataFrame:
    master = read_parquet("paper4_v69_expanded_restricted_master.parquet")
    universe_cols = [
        "loan_id",
        "issue_month",
        "pd_high_alpha01",
        "lgd_proxy_v55",
        "base_return_vec",
        "qhat_v4",
        "weak_source_proxy",
    ]
    return master.merge(
        universe[universe_cols].assign(loan_id=lambda df: df["loan_id"].astype(str)),
        on="loan_id",
        how="left",
    )


def _candidate_columns(universe: pd.DataFrame, reduced_costs: pd.DataFrame) -> pd.DataFrame:
    negative = reduced_costs.loc[reduced_costs["improving_column_v71"].astype(bool)].copy()
    if negative.empty:
        return pd.DataFrame()
    keep_universe = [
        "loan_index_v55",
        "loan_id",
        "loan_amnt",
        "term_months",
        "int_rate_decimal",
        "installment",
        "grade",
        "sub_grade",
        "issue_d",
        "issue_month",
        "period",
        "loan_status",
        "addr_state",
        "state_top20",
        "annual_inc",
        "income_band",
        "dti",
        "dti_band",
        "fico_score",
        "score_decile",
        "y_true",
        "pd_point",
        "pd_high_alpha01",
        "pd_low_90",
        "pd_high_90",
        "width_90",
        "qhat_v4",
        "weak_source_proxy",
        "lgd_proxy_v55",
        "base_return_vec",
        "next_pymnt_d",
    ]
    candidates = negative[
        [
            "policy_id",
            "regime_v71",
            "loan_id",
            "minimization_reduced_cost_v71",
            "return_improvement_signal_v71",
        ]
    ].merge(
        universe[keep_universe].assign(loan_id=lambda df: df["loan_id"].astype(str)),
        on="loan_id",
        how="left",
    )
    candidates["policy_id_v69"] = candidates["policy_id"]
    candidates["source_policy_id_v63"] = candidates["policy_id"]
    candidates["master_role_v69"] = V71_ROLE
    candidates["candidate_rank_v68"] = np.nan
    candidates["pricing_screen_score_v69"] = np.nan
    candidates["restricted_master_scope_v69"] = (
        "v72 iteration-1 negative reduced-cost columns appended to v69 master"
    )
    candidates["exact_column_generation_certificate_v69"] = False
    candidates["claim_boundary_v69"] = (
        "v72 iteration candidate from v71 reduced-cost screen; not termination proof"
    )
    return candidates.sort_values(
        ["policy_id", "regime_v71", "minimization_reduced_cost_v71"]
    ).reset_index(drop=True)


def _pool_for_iteration(
    base_master: pd.DataFrame,
    candidates: pd.DataFrame,
    policy_id: str,
    regime: str,
) -> pd.DataFrame:
    base = base_master.loc[base_master["policy_id_v69"].astype(str).eq(policy_id)].copy()
    extra = candidates.loc[
        candidates["policy_id"].astype(str).eq(policy_id)
        & candidates["regime_v71"].astype(str).eq(regime)
    ].copy()
    combined = pd.concat([base, extra], ignore_index=True, sort=False)
    combined = combined.drop_duplicates("loan_id", keep="first").copy()
    combined["policy_id_v72"] = policy_id
    combined["regime_v72"] = regime
    return combined


def _status_to_v72(
    status: dict[str, Any],
    policy_id: str,
    regime: str,
    pool: pd.DataFrame,
    candidate_count: int,
    v70_frontier: pd.DataFrame,
    allocations: pd.DataFrame,
) -> dict[str, Any]:
    row = {key.replace("_v70", "_v72"): value for key, value in status.items()}
    row["policy_id"] = policy_id
    row["regime_v72"] = regime
    row["iteration_v72"] = 1
    row["negative_reduced_cost_candidates_added_v72"] = candidate_count
    row["expanded_master_rows_v72"] = int(len(pool))
    row["v71_candidate_allocated_exposure_v72"] = (
        float(
            allocations.loc[
                allocations["master_role_v69"].astype(str).eq(V71_ROLE),
                "allocated_exposure_v70",
            ].sum()
        )
        if not allocations.empty and "master_role_v69" in allocations
        else 0.0
    )
    row["v71_candidate_allocated_rows_v72"] = (
        int(allocations["master_role_v69"].astype(str).eq(V71_ROLE).sum())
        if not allocations.empty and "master_role_v69" in allocations
        else 0
    )
    baseline = v70_frontier.loc[
        v70_frontier["policy_id"].astype(str).eq(policy_id)
        & v70_frontier["regime_v70"].astype(str).eq(regime)
    ]
    if not baseline.empty and "objective_return_v70" in baseline:
        row["v70_objective_return_baseline_v72"] = float(baseline["objective_return_v70"].iloc[0])
        row["delta_return_vs_v70_iteration_v72"] = float(
            row.get("objective_return_v72", np.nan) - baseline["objective_return_v70"].iloc[0]
        )
        row["delta_cvar90_vs_v70_iteration_v72"] = float(
            row.get("scenario_loss_cvar90_v72", np.nan)
            - baseline["scenario_loss_cvar90_v70"].iloc[0]
        )
    else:
        row["v70_objective_return_baseline_v72"] = np.nan
        row["delta_return_vs_v70_iteration_v72"] = np.nan
        row["delta_cvar90_vs_v70_iteration_v72"] = np.nan
    row["reprice_after_iteration_performed_v72"] = False
    row["column_generation_termination_claim_allowed_v72"] = False
    row["exact_full_universe_cvar_claim_allowed_v72"] = False
    row["paper1_promotion_allowed_v72"] = False
    row["paper4_working_champion_changed_v72"] = False
    row["claim_boundary_v72"] = (
        "iteration-1 restricted-master LP after adding v71 columns; re-pricing still required"
    )
    return row


def build_v72_iteration() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    reduced_costs = read_parquet("paper4_v71_full_universe_reduced_costs.parquet")
    concentration = read_csv("paper4_v63_source_repair_concentration.csv")
    v70_frontier = read_csv("paper4_v70_restricted_master_solver_frontier.csv")
    if universe.empty or reduced_costs.empty or concentration.empty or v70_frontier.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    base_master = _enriched_base_master(universe)
    candidates = _candidate_columns(universe, reduced_costs)
    frontier_rows: list[dict[str, Any]] = []
    allocation_frames: list[pd.DataFrame] = []
    scenario_frames: list[pd.DataFrame] = []
    active_frames: list[pd.DataFrame] = []
    candidate_groups = candidates.groupby(["policy_id", "regime_v71"], dropna=False)
    for (policy_id, regime), local_candidates in candidate_groups:
        policy_id = str(policy_id)
        regime = str(regime)
        pool = _pool_for_iteration(base_master, candidates, policy_id, regime)
        source_map = v70._policy_source_map(concentration, policy_id)
        status, allocations, scenarios, active = v70._solve_policy_regime(
            policy_id, pool, source_map, regime
        )
        frontier_rows.append(
            _status_to_v72(
                status,
                policy_id,
                regime,
                pool,
                int(len(local_candidates)),
                v70_frontier,
                allocations,
            )
        )
        if not allocations.empty:
            alloc = _rename_v70_to_v72(allocations)
            alloc["iteration_v72"] = 1
            alloc["master_role_v72"] = alloc["master_role_v69"]
            alloc["claim_boundary_v72"] = (
                "v72 iteration-1 restricted-master allocation; re-pricing still required"
            )
            allocation_frames.append(alloc)
        if not scenarios.empty:
            scen = _rename_v70_to_v72(scenarios)
            scen["iteration_v72"] = 1
            scen["claim_boundary_v72"] = "v72 iteration scenario evaluation only"
            scenario_frames.append(scen)
        if not active.empty:
            act = _rename_v70_to_v72(active)
            act["iteration_v72"] = 1
            act["claim_boundary_v72"] = "v72 iteration active-constraint diagnostic only"
            active_frames.append(act)

    frontier = pd.DataFrame(frontier_rows)
    allocations = (
        pd.concat(allocation_frames, ignore_index=True) if allocation_frames else pd.DataFrame()
    )
    scenarios = pd.concat(scenario_frames, ignore_index=True) if scenario_frames else pd.DataFrame()
    active = pd.concat(active_frames, ignore_index=True) if active_frames else pd.DataFrame()
    return candidates, frontier, allocations, scenarios, active


def _comparison(frontier: pd.DataFrame) -> pd.DataFrame:
    if frontier.empty:
        return pd.DataFrame()
    keep = [
        "policy_id",
        "regime_v72",
        "objective_return_v72",
        "v70_objective_return_baseline_v72",
        "delta_return_vs_v70_iteration_v72",
        "scenario_loss_cvar90_v72",
        "delta_cvar90_vs_v70_iteration_v72",
        "v71_candidate_allocated_exposure_v72",
        "v71_candidate_allocated_rows_v72",
        "column_generation_termination_claim_allowed_v72",
        "claim_boundary_v72",
    ]
    return frontier[[col for col in keep if col in frontier.columns]].copy()


def _claim_blockers(frontier: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v72": "post_iteration_repricing_missing",
                "blocking_v72": True,
                "evidence_count_v72": int(len(frontier)),
                "required_next_artifact_v72": "paper4_v73_reprice_after_iteration_1.parquet",
                "claim_boundary_v72": "after adding columns, v72 must be re-priced before termination claims",
            },
            {
                "blocker_id_v72": "source_constraint_scope_needs_reaudit",
                "blocking_v72": True,
                "evidence_count_v72": int(len(frontier)),
                "required_next_artifact_v72": "paper4_v73_source_scope_after_iteration.csv",
                "claim_boundary_v72": "new columns change source IDs and active source constraints",
            },
            {
                "blocker_id_v72": "continuous_relaxation_not_whole_loan_milp",
                "blocking_v72": True,
                "evidence_count_v72": 1,
                "required_next_artifact_v72": "paper4_v73_integrality_gap_or_milp_probe.csv",
                "claim_boundary_v72": "iteration uses continuous LP, not whole-loan integer allocations",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v72 first column-generation iteration over negative v71 columns.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v72_iteration_1_frontier.csv"
                ),
                "boundary": "One restricted-master iteration only; re-pricing still required.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v72 proves column-generation convergence.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v72_claim_blockers.csv"
                ),
                "boundary": "Blocked until v72 solution is re-priced and no improving columns remain.",
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
                "lane": "CVaR/OCE",
                "executable_item": (
                    "v72 adds v71 negative reduced-cost columns and resolves iteration-1 "
                    "restricted-master LPs."
                ),
                "status": "ready_for_repricing",
                "next_artifact": "paper4_v73_reprice_after_iteration_1.parquet",
                "success_condition": "post-iteration reduced-cost screen has no improving omitted columns",
                "last_wave": "v72",
                "execution_result": "column_generation_iteration_1_completed",
                "quarto_promotion_decision": "living_notebook_only",
            },
            {
                "horizon": "short",
                "lane": "Source governance",
                "executable_item": "Reaudit source-scope coverage after v72 adds new reduced-cost columns.",
                "status": "gated",
                "next_artifact": "paper4_v73_source_scope_after_iteration.csv",
                "success_condition": "source constraints cover all active and omitted full-universe source IDs",
                "last_wave": "v72",
                "execution_result": "source_scope_reaudit_queued",
                "quarto_promotion_decision": "not_promoted_to_quarto",
            },
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    key_cols = ["last_wave", "lane", "next_artifact"]
    merged_keys = set(map(tuple, additions[key_cols].astype(str).to_numpy()))
    keep = [tuple(row) not in merged_keys for row in current[key_cols].astype(str).to_numpy()]
    write_csv(path, pd.concat([current.loc[keep].copy(), additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    existing = NOTEBOOK.read_text(encoding="utf-8") if NOTEBOOK.exists() else ""
    start = "<!-- V72_COLUMN_GENERATION_ITERATION_START -->"
    end = "<!-- V72_COLUMN_GENERATION_ITERATION_END -->"
    block = f"""
{start}

## Wave v72: Column-Generation Iteration 1

Generated: {status["generated_at_utc"]}

### Objective

Use the v71 negative reduced-cost columns as actual master columns and resolve
the restricted-master continuous LP. This converts the v71 pricing blocker into
an executable first column-generation iteration.

### Results

- Candidate rows added: `{status["candidate_rows_v72"]}`.
- Iteration frontier rows: `{status["frontier_rows_v72"]}`.
- Successful LP rows: `{status["successful_iteration_rows_v72"]}`.
- Allocation rows: `{status["allocation_rows_v72"]}`.
- Scenario rows: `{status["scenario_rows_v72"]}`.
- Best return delta vs v70: `{status["best_delta_return_vs_v70_iteration_v72"]}`.
- Re-price after iteration performed: `{status["reprice_after_iteration_performed_v72"]}`.

### Interpretation

v72 confirms that v71 was not just a diagnostic: the negative reduced-cost
columns can be inserted into the restricted master and solved. The result still
cannot claim convergence because the new solution must be priced again.

### Claim Impact

- Allowed: first restricted-master column-generation iteration completed.
- Still prohibited: convergence, exact full-universe CVaR optimality, MILP
  whole-loan optimality, Paper Estrella replacement, final Paper 4 promotion
  and live deployment.

### Quarto Promotion Decision

Keep v72 in the living notebook. Promote only after post-iteration re-pricing
and integrality/source-scope review pass.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v72() -> dict[str, Any]:
    started = datetime.now(UTC)
    candidates, frontier, allocations, scenarios, active = build_v72_iteration()
    comparison = _comparison(frontier)
    blockers = _claim_blockers(frontier)

    candidates.to_parquet(
        TABLE_DIR / "paper4_v72_iteration_1_candidates.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v72_iteration_1_frontier.csv", frontier)
    allocations.to_parquet(
        TABLE_DIR / "paper4_v72_iteration_1_allocations.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v72_iteration_1_scenario_losses.csv", scenarios)
    write_csv(TABLE_DIR / "paper4_v72_iteration_1_active_constraints.csv", active)
    write_csv(TABLE_DIR / "paper4_v72_iteration_1_comparison.csv", comparison)
    write_csv(TABLE_DIR / "paper4_v72_claim_blockers.csv", blockers)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v72_column_generation_iteration_1_completed",
                "allowed": True,
                "artifact": "paper4_v72_iteration_1_frontier.csv",
                "boundary": "one restricted-master iteration only",
            },
            {
                "claim_id": "v72_column_generation_converged",
                "allowed": False,
                "artifact": "paper4_v72_claim_blockers.csv",
                "boundary": "requires post-iteration re-pricing with no improving columns",
            },
            {
                "claim_id": "v72_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v72_iteration_1_comparison.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v72_claim_matrix_delta.csv", claim_matrix)

    successful = (
        frontier.loc[frontier["solver_success_v72"].astype(bool)].copy()
        if not frontier.empty and "solver_success_v72" in frontier
        else pd.DataFrame()
    )
    status = {
        "phase": "v72_column_generation_iteration_1",
        "schema_version": "2026-05-15.72",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "candidate_rows_v72": int(len(candidates)),
        "frontier_rows_v72": int(len(frontier)),
        "successful_iteration_rows_v72": int(len(successful)),
        "allocation_rows_v72": int(len(allocations)),
        "scenario_rows_v72": int(len(scenarios)),
        "active_constraint_rows_v72": int(len(active)),
        "comparison_rows_v72": int(len(comparison)),
        "best_delta_return_vs_v70_iteration_v72": float(
            successful["delta_return_vs_v70_iteration_v72"].max()
        )
        if not successful.empty
        else 0.0,
        "v71_candidate_allocated_exposure_v72": float(
            successful["v71_candidate_allocated_exposure_v72"].sum()
        )
        if not successful.empty
        else 0.0,
        "reprice_after_iteration_performed_v72": False,
        "column_generation_termination_claim_allowed_v72": False,
        "exact_full_universe_cvar_claim_allowed_v72": False,
        "paper1_promotion_allowed_v72": False,
        "paper4_working_champion_changed_v72": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v72 completes one restricted-master iteration; post-iteration re-pricing "
            "and integrality evidence remain missing"
        ),
    }
    write_json(STATUS_DIR / "paper4_v72_status.json", status)
    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v72": build_v72()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
