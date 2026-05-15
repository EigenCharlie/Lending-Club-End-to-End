#!/usr/bin/env python3
"""Build Paper 4 v76 column-generation iteration-3 artifacts.

v76 consumes the remaining negative reduced-cost columns found by the v75
post-iteration-2 pricing screen, appends them to the v74 restricted master, and
resolves the affected continuous LPs. It is a focused third iteration, not a
termination certificate; the new solution must be re-priced in v77.
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
V73_ROLE = "v73_negative_reduced_cost_column"
V75_ROLE = "v75_negative_reduced_cost_column"


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


def _rename_v70_to_v76(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={col: col.replace("_v70", "_v76") for col in df.columns})


def _universe_join_columns() -> list[str]:
    return [
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


def _candidate_columns(universe: pd.DataFrame, repriced: pd.DataFrame) -> pd.DataFrame:
    negative = repriced.loc[repriced["improving_column_v75"].astype(bool)].copy()
    if negative.empty:
        return pd.DataFrame()
    candidates = negative[
        [
            "policy_id",
            "regime_v75",
            "loan_id",
            "minimization_reduced_cost_v75",
            "return_improvement_signal_v75",
        ]
    ].merge(
        universe[_universe_join_columns()].assign(loan_id=lambda df: df["loan_id"].astype(str)),
        on="loan_id",
        how="left",
    )
    candidates["policy_id_v69"] = candidates["policy_id"]
    candidates["source_policy_id_v63"] = candidates["policy_id"]
    candidates["master_role_v69"] = V75_ROLE
    candidates["candidate_rank_v68"] = np.nan
    candidates["pricing_screen_score_v69"] = np.nan
    candidates["restricted_master_scope_v69"] = (
        "v76 iteration-3 negative reduced-cost columns appended to v74 master"
    )
    candidates["exact_column_generation_certificate_v69"] = False
    candidates["claim_boundary_v69"] = (
        "v76 iteration candidate from v75 reduced-cost screen; not termination proof"
    )
    return candidates.sort_values(
        ["policy_id", "regime_v75", "minimization_reduced_cost_v75"]
    ).reset_index(drop=True)


def _pool_for_iteration(
    base_master: pd.DataFrame,
    iteration_1_candidates: pd.DataFrame,
    iteration_2_candidates: pd.DataFrame,
    iteration_3_candidates: pd.DataFrame,
    policy_id: str,
    regime: str,
) -> pd.DataFrame:
    base = base_master.loc[base_master["policy_id_v69"].astype(str).eq(policy_id)].copy()
    first = iteration_1_candidates.loc[
        iteration_1_candidates["policy_id"].astype(str).eq(policy_id)
        & iteration_1_candidates["regime_v71"].astype(str).eq(regime)
    ].copy()
    second = iteration_2_candidates.loc[
        iteration_2_candidates["policy_id"].astype(str).eq(policy_id)
        & iteration_2_candidates["regime_v73"].astype(str).eq(regime)
    ].copy()
    third = iteration_3_candidates.loc[
        iteration_3_candidates["policy_id"].astype(str).eq(policy_id)
        & iteration_3_candidates["regime_v75"].astype(str).eq(regime)
    ].copy()
    combined = pd.concat([base, first, second, third], ignore_index=True, sort=False)
    combined = combined.drop_duplicates("loan_id", keep="first").copy()
    combined["policy_id_v76"] = policy_id
    combined["regime_v76"] = regime
    return combined


def _role_exposure(allocations: pd.DataFrame, role: str) -> float:
    if allocations.empty or "master_role_v69" not in allocations:
        return 0.0
    return float(
        allocations.loc[
            allocations["master_role_v69"].astype(str).eq(role),
            "allocated_exposure_v70",
        ].sum()
    )


def _role_rows(allocations: pd.DataFrame, role: str) -> int:
    if allocations.empty or "master_role_v69" not in allocations:
        return 0
    return int(allocations["master_role_v69"].astype(str).eq(role).sum())


def _status_to_v76(
    status: dict[str, Any],
    policy_id: str,
    regime: str,
    pool: pd.DataFrame,
    candidate_count: int,
    v74_frontier: pd.DataFrame,
    allocations: pd.DataFrame,
) -> dict[str, Any]:
    row = {key.replace("_v70", "_v76"): value for key, value in status.items()}
    row["policy_id"] = policy_id
    row["regime_v76"] = regime
    row["iteration_v76"] = 3
    row["negative_reduced_cost_candidates_added_v76"] = candidate_count
    row["expanded_master_rows_v76"] = int(len(pool))
    row["v75_candidate_allocated_exposure_v76"] = _role_exposure(allocations, V75_ROLE)
    row["v75_candidate_allocated_rows_v76"] = _role_rows(allocations, V75_ROLE)
    row["v73_previous_candidate_allocated_exposure_v76"] = _role_exposure(allocations, V73_ROLE)
    row["v73_previous_candidate_allocated_rows_v76"] = _role_rows(allocations, V73_ROLE)
    row["v71_previous_candidate_allocated_exposure_v76"] = _role_exposure(allocations, V71_ROLE)
    row["v71_previous_candidate_allocated_rows_v76"] = _role_rows(allocations, V71_ROLE)
    baseline = v74_frontier.loc[
        v74_frontier["policy_id"].astype(str).eq(policy_id)
        & v74_frontier["regime_v74"].astype(str).eq(regime)
    ]
    if not baseline.empty and "objective_return_v74" in baseline:
        row["v74_objective_return_baseline_v76"] = float(baseline["objective_return_v74"].iloc[0])
        row["delta_return_vs_v74_iteration_v76"] = float(
            row.get("objective_return_v76", np.nan) - baseline["objective_return_v74"].iloc[0]
        )
        row["delta_cvar90_vs_v74_iteration_v76"] = float(
            row.get("scenario_loss_cvar90_v76", np.nan)
            - baseline["scenario_loss_cvar90_v74"].iloc[0]
        )
    else:
        row["v74_objective_return_baseline_v76"] = np.nan
        row["delta_return_vs_v74_iteration_v76"] = np.nan
        row["delta_cvar90_vs_v74_iteration_v76"] = np.nan
    row["reprice_after_iteration_performed_v76"] = False
    row["column_generation_termination_claim_allowed_v76"] = False
    row["exact_full_universe_cvar_claim_allowed_v76"] = False
    row["paper1_promotion_allowed_v76"] = False
    row["paper4_working_champion_changed_v76"] = False
    row["claim_boundary_v76"] = (
        "iteration-3 restricted-master LP after adding v75 columns; re-pricing still required"
    )
    return row


def build_v76_iteration() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    repriced = read_parquet("paper4_v75_reprice_after_iteration_2.parquet")
    iteration_1_candidates = read_parquet("paper4_v72_iteration_1_candidates.parquet")
    iteration_2_candidates = read_parquet("paper4_v74_iteration_2_candidates.parquet")
    concentration = read_csv("paper4_v63_source_repair_concentration.csv")
    v74_frontier = read_csv("paper4_v74_iteration_2_frontier.csv")
    if (
        universe.empty
        or repriced.empty
        or iteration_1_candidates.empty
        or iteration_2_candidates.empty
        or concentration.empty
        or v74_frontier.empty
    ):
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    base_master = _enriched_base_master(universe)
    candidates = _candidate_columns(universe, repriced)
    frontier_rows: list[dict[str, Any]] = []
    allocation_frames: list[pd.DataFrame] = []
    scenario_frames: list[pd.DataFrame] = []
    active_frames: list[pd.DataFrame] = []
    candidate_groups = candidates.groupby(["policy_id", "regime_v75"], dropna=False)
    for (policy_id, regime), local_candidates in candidate_groups:
        policy_id = str(policy_id)
        regime = str(regime)
        pool = _pool_for_iteration(
            base_master,
            iteration_1_candidates,
            iteration_2_candidates,
            candidates,
            policy_id,
            regime,
        )
        source_map = v70._policy_source_map(concentration, policy_id)
        status, allocations, scenarios, active = v70._solve_policy_regime(
            policy_id, pool, source_map, regime
        )
        frontier_rows.append(
            _status_to_v76(
                status,
                policy_id,
                regime,
                pool,
                int(len(local_candidates)),
                v74_frontier,
                allocations,
            )
        )
        if not allocations.empty:
            alloc = _rename_v70_to_v76(allocations)
            alloc["iteration_v76"] = 3
            alloc["master_role_v76"] = alloc["master_role_v69"]
            alloc["claim_boundary_v76"] = (
                "v76 iteration-3 restricted-master allocation; re-pricing still required"
            )
            allocation_frames.append(alloc)
        if not scenarios.empty:
            scen = _rename_v70_to_v76(scenarios)
            scen["iteration_v76"] = 3
            scen["claim_boundary_v76"] = "v76 iteration scenario evaluation only"
            scenario_frames.append(scen)
        if not active.empty:
            act = _rename_v70_to_v76(active)
            act["iteration_v76"] = 3
            act["claim_boundary_v76"] = "v76 iteration active-constraint diagnostic only"
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
        "regime_v76",
        "objective_return_v76",
        "v74_objective_return_baseline_v76",
        "delta_return_vs_v74_iteration_v76",
        "scenario_loss_cvar90_v76",
        "delta_cvar90_vs_v74_iteration_v76",
        "v75_candidate_allocated_exposure_v76",
        "v75_candidate_allocated_rows_v76",
        "v73_previous_candidate_allocated_exposure_v76",
        "v71_previous_candidate_allocated_exposure_v76",
        "column_generation_termination_claim_allowed_v76",
        "claim_boundary_v76",
    ]
    return frontier[[col for col in keep if col in frontier.columns]].copy()


def _claim_blockers(frontier: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v76": "post_iteration_3_repricing_missing",
                "blocking_v76": True,
                "evidence_count_v76": int(len(frontier)),
                "required_next_artifact_v76": "paper4_v77_reprice_after_iteration_3.parquet",
                "claim_boundary_v76": "after adding v75 columns, v76 must be re-priced",
            },
            {
                "blocker_id_v76": "source_constraint_scope_needs_reaudit_after_iteration_3",
                "blocking_v76": True,
                "evidence_count_v76": int(len(frontier)),
                "required_next_artifact_v76": "paper4_v77_source_scope_after_iteration_3.csv",
                "claim_boundary_v76": (
                    "third-iteration columns change source IDs and active constraints"
                ),
            },
            {
                "blocker_id_v76": "continuous_relaxation_not_whole_loan_milp",
                "blocking_v76": True,
                "evidence_count_v76": 1,
                "required_next_artifact_v76": "paper4_v77_integrality_gap_or_milp_probe.csv",
                "claim_boundary_v76": "iteration uses continuous LP, not whole-loan integers",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": (
                    "Paper 4 has a v76 third column-generation iteration over remaining "
                    "negative v75 columns."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v76_iteration_3_frontier.csv"
                ),
                "boundary": "Third restricted-master iteration only; re-pricing still required.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v76 proves column-generation convergence.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v76_claim_blockers.csv"
                ),
                "boundary": "Blocked until v76 solution is re-priced and no blockers remain.",
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
                    "v76 adds remaining v75 negative reduced-cost columns and resolves "
                    "iteration-3 restricted-master LPs."
                ),
                "status": "ready_for_repricing",
                "next_artifact": "paper4_v77_reprice_after_iteration_3.parquet",
                "success_condition": (
                    "third post-iteration reduced-cost screen has no improving omitted columns"
                ),
                "last_wave": "v76",
                "execution_result": "column_generation_iteration_3_completed",
                "quarto_promotion_decision": "living_notebook_only",
            },
            {
                "horizon": "short",
                "lane": "Source governance",
                "executable_item": "Reaudit source-scope coverage after v76 adds v75 columns.",
                "status": "gated",
                "next_artifact": "paper4_v77_source_scope_after_iteration_3.csv",
                "success_condition": "source constraints cover all active and omitted source IDs",
                "last_wave": "v76",
                "execution_result": "source_scope_reaudit_after_iteration_3_queued",
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
    start = "<!-- V76_COLUMN_GENERATION_ITERATION_3_START -->"
    end = "<!-- V76_COLUMN_GENERATION_ITERATION_3_END -->"
    block = f"""
{start}

## Wave v76: Column-Generation Iteration 3

Generated: {status["generated_at_utc"]}

### Objective

Use the remaining v75 negative reduced-cost columns as actual master columns.
Because v75 found improving columns in only one policy/regime pair, v76 is a
focused third restricted-master LP iteration.

### Results

- Candidate rows added: `{status["candidate_rows_v76"]}`.
- Iteration frontier rows: `{status["frontier_rows_v76"]}`.
- Successful LP rows: `{status["successful_iteration_rows_v76"]}`.
- Allocation rows: `{status["allocation_rows_v76"]}`.
- Scenario rows: `{status["scenario_rows_v76"]}`.
- Best return delta vs v74: `{status["best_delta_return_vs_v74_iteration_v76"]}`.
- v75 candidate allocated exposure: `{status["v75_candidate_allocated_exposure_v76"]}`.
- Re-price after iteration performed: `{status["reprice_after_iteration_performed_v76"]}`.

### Interpretation

v76 shows the column-generation loop can narrow from broad exploration to a
single active regime. It remains a restricted-master continuous LP iteration,
so post-iteration-3 re-pricing is required before any convergence statement.

### Claim Impact

- Allowed: third restricted-master column-generation iteration completed.
- Still prohibited: convergence, exact full-universe CVaR optimality, MILP
  whole-loan optimality, Paper Estrella replacement, final Paper 4 promotion
  and live deployment.

### Quarto Promotion Decision

Keep v76 in the living notebook. Promote only after iteration-3 re-pricing,
source-scope coverage and integrality review pass.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v76() -> dict[str, Any]:
    started = datetime.now(UTC)
    candidates, frontier, allocations, scenarios, active = build_v76_iteration()
    comparison = _comparison(frontier)
    blockers = _claim_blockers(frontier)

    candidates.to_parquet(
        TABLE_DIR / "paper4_v76_iteration_3_candidates.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v76_iteration_3_frontier.csv", frontier)
    allocations.to_parquet(
        TABLE_DIR / "paper4_v76_iteration_3_allocations.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v76_iteration_3_scenario_losses.csv", scenarios)
    write_csv(TABLE_DIR / "paper4_v76_iteration_3_active_constraints.csv", active)
    write_csv(TABLE_DIR / "paper4_v76_iteration_3_comparison.csv", comparison)
    write_csv(TABLE_DIR / "paper4_v76_claim_blockers.csv", blockers)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v76_column_generation_iteration_3_completed",
                "allowed": True,
                "artifact": "paper4_v76_iteration_3_frontier.csv",
                "boundary": "third restricted-master iteration only",
            },
            {
                "claim_id": "v76_column_generation_converged",
                "allowed": False,
                "artifact": "paper4_v76_claim_blockers.csv",
                "boundary": "requires post-iteration-3 re-pricing with no improving columns",
            },
            {
                "claim_id": "v76_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v76_iteration_3_comparison.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v76_claim_matrix_delta.csv", claim_matrix)

    successful = (
        frontier.loc[frontier["solver_success_v76"].astype(bool)].copy()
        if not frontier.empty and "solver_success_v76" in frontier
        else pd.DataFrame()
    )
    status = {
        "phase": "v76_column_generation_iteration_3",
        "schema_version": "2026-05-15.76",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "candidate_rows_v76": int(len(candidates)),
        "frontier_rows_v76": int(len(frontier)),
        "successful_iteration_rows_v76": int(len(successful)),
        "allocation_rows_v76": int(len(allocations)),
        "scenario_rows_v76": int(len(scenarios)),
        "active_constraint_rows_v76": int(len(active)),
        "comparison_rows_v76": int(len(comparison)),
        "best_delta_return_vs_v74_iteration_v76": float(
            successful["delta_return_vs_v74_iteration_v76"].max()
        )
        if not successful.empty
        else 0.0,
        "v75_candidate_allocated_exposure_v76": float(
            successful["v75_candidate_allocated_exposure_v76"].sum()
        )
        if not successful.empty
        else 0.0,
        "v73_previous_candidate_allocated_exposure_v76": float(
            successful["v73_previous_candidate_allocated_exposure_v76"].sum()
        )
        if not successful.empty
        else 0.0,
        "v71_previous_candidate_allocated_exposure_v76": float(
            successful["v71_previous_candidate_allocated_exposure_v76"].sum()
        )
        if not successful.empty
        else 0.0,
        "reprice_after_iteration_performed_v76": False,
        "column_generation_termination_claim_allowed_v76": False,
        "exact_full_universe_cvar_claim_allowed_v76": False,
        "paper1_promotion_allowed_v76": False,
        "paper4_working_champion_changed_v76": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v76 completes a focused third restricted-master iteration; "
            "post-iteration-3 re-pricing and integrality evidence remain missing"
        ),
    }
    write_json(STATUS_DIR / "paper4_v76_status.json", status)
    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v76": build_v76()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
