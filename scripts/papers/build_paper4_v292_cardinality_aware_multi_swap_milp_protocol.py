#!/usr/bin/env python3
"""Build Paper 4 v292 cardinality-aware multi-swap MILP artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from scripts.papers import build_paper4_v70_restricted_master_solver as v70
from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71
from scripts.papers.paper4_one_swap_living_lab import (
    FAMILIES,
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

VERSION = 292
PREVIOUS_PROTOCOL_VERSION = 291
REPAIR_VERSION = 289
BASE_REPAIR_VERSION = 279
NEXT_VERSION = 293
POOL_CANDIDATE_LIMIT = 15000
TARGET_SELECTED_ROWS = 171
MILP_TIME_LIMIT_SECONDS = 180.0
MIP_REL_GAP = 1e-6


def _source_cap_map(source_caps: pd.DataFrame, family: str) -> dict[str, float]:
    family_caps = source_caps.loc[source_caps["source_family"].astype(str).eq(family)].copy()
    family_caps["source_id"] = family_caps["source_id"].astype(str)
    return family_caps.set_index("source_id")["cap_share_v80"].astype(float).to_dict()


def _build_pool(
    *,
    universe: pd.DataFrame,
    selected: pd.DataFrame,
    mean_returns: np.ndarray,
) -> pd.DataFrame:
    selected_ids = set(selected["loan_id"].astype(str))
    omitted = universe.loc[~universe["loan_id"].astype(str).isin(selected_ids)].copy()
    omitted[f"mean_return_v{VERSION}"] = mean_returns[omitted.index.to_numpy()]
    top_candidates = omitted.sort_values(f"mean_return_v{VERSION}", ascending=False).head(
        POOL_CANDIDATE_LIMIT
    )

    selected_pool = selected[["loan_id", "loan_amnt", *FAMILIES]].copy()
    selected_pool["pool_role_v292"] = "current_v289_selected"
    candidate_pool = top_candidates[["loan_id", "loan_amnt", *FAMILIES]].copy()
    candidate_pool["pool_role_v292"] = "top15000_omitted_by_mean_return"
    pool = pd.concat([selected_pool, candidate_pool], ignore_index=True)
    pool["loan_id"] = pool["loan_id"].astype(str)
    return pool.drop_duplicates("loan_id", keep="first").reset_index(drop=True)


def _source_constraints(
    *,
    pool: pd.DataFrame,
    amounts: np.ndarray,
    source_caps: pd.DataFrame,
    var_count: int,
) -> tuple[list[np.ndarray], list[float], list[float], list[dict[str, Any]]]:
    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []
    meta: list[dict[str, Any]] = []
    for family in FAMILIES:
        caps = _source_cap_map(source_caps, family)
        source_values = pool[family].astype(str)
        for source_id in sorted(source_values.dropna().unique()):
            cap = float(caps.get(source_id, 1.0))
            indicator = source_values.eq(source_id).to_numpy(float)
            row = np.zeros(var_count)
            row[: len(pool)] = amounts * (indicator - cap)
            rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)
            meta.append(
                {
                    f"constraint_type_v{VERSION}": "source_share",
                    "source_family": family,
                    "source_id": source_id,
                    f"cap_share_v{VERSION}": cap,
                }
            )
    return rows, lb, ub, meta


def _solve_cardinality_milp(
    *,
    pool: pd.DataFrame,
    losses_pool: np.ndarray,
    mean_returns_pool: np.ndarray,
    source_caps: pd.DataFrame,
    exposure_min: float,
    exposure_max: float,
    cvar_cap: float,
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    n = len(pool)
    scenario_count = losses_pool.shape[0]
    var_count = n + 1 + scenario_count
    amounts = pool["loan_amnt"].to_numpy(float)
    c = np.zeros(var_count)
    c[:n] = -mean_returns_pool

    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []
    meta: list[dict[str, Any]] = []

    budget = np.zeros(var_count)
    budget[:n] = amounts
    rows.append(budget)
    lb.append(exposure_min)
    ub.append(exposure_max)
    meta.append({f"constraint_type_v{VERSION}": "budget_range"})

    cardinality = np.zeros(var_count)
    cardinality[:n] = 1.0
    rows.append(cardinality)
    lb.append(float(TARGET_SELECTED_ROWS))
    ub.append(float(TARGET_SELECTED_ROWS))
    meta.append({f"constraint_type_v{VERSION}": "selected_row_cardinality"})

    source_rows, source_lb, source_ub, source_meta = _source_constraints(
        pool=pool,
        amounts=amounts,
        source_caps=source_caps,
        var_count=var_count,
    )
    rows.extend(source_rows)
    lb.extend(source_lb)
    ub.extend(source_ub)
    meta.extend(source_meta)

    cvar_row = np.zeros(var_count)
    cvar_row[n] = 1.0
    cvar_row[n + 1 :] = 1.0 / ((1.0 - v70.ALPHA) * scenario_count)
    rows.append(cvar_row)
    lb.append(-np.inf)
    ub.append(cvar_cap)
    meta.append({f"constraint_type_v{VERSION}": "cvar_cap"})

    for scenario_idx in range(scenario_count):
        row = np.zeros(var_count)
        row[:n] = losses_pool[scenario_idx, :]
        row[n] = -1.0
        row[n + 1 + scenario_idx] = -1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)
        meta.append({f"constraint_type_v{VERSION}": "cvar_path_excess"})

    result = milp(
        c,
        integrality=np.r_[np.ones(n), np.zeros(1 + scenario_count)],
        bounds=Bounds(
            np.r_[np.zeros(n), 0.0, np.zeros(scenario_count)],
            np.r_[np.ones(n), np.inf, np.full(scenario_count, np.inf)],
        ),
        constraints=LinearConstraint(np.vstack(rows), np.array(lb), np.array(ub)),
        options={"time_limit": MILP_TIME_LIMIT_SECONDS, "mip_rel_gap": MIP_REL_GAP},
    )
    x = np.zeros(n, dtype=bool)
    incumbent_available = result.x is not None
    if incumbent_available:
        x = np.rint(np.clip(result.x[:n], 0, 1)).astype(bool)

    diagnostics = {
        f"milp_success_v{VERSION}": bool(result.success),
        f"milp_incumbent_available_v{VERSION}": bool(incumbent_available),
        f"milp_status_v{VERSION}": int(result.status),
        f"milp_message_v{VERSION}": str(result.message),
        f"milp_fun_v{VERSION}": float(result.fun) if result.fun is not None else np.nan,
        f"milp_dual_bound_v{VERSION}": float(getattr(result, "mip_dual_bound", np.nan)),
        f"milp_gap_v{VERSION}": float(getattr(result, "mip_gap", np.nan)),
        f"milp_node_count_v{VERSION}": int(getattr(result, "mip_node_count", -1)),
        f"milp_time_limit_seconds_v{VERSION}": MILP_TIME_LIMIT_SECONDS,
        f"milp_mip_rel_gap_v{VERSION}": MIP_REL_GAP,
        f"milp_variable_count_v{VERSION}": int(var_count),
        f"milp_binary_variable_count_v{VERSION}": int(n),
        f"milp_constraint_rows_v{VERSION}": int(len(rows)),
    }
    return diagnostics, x, pd.DataFrame(meta)


def _source_summary(
    *,
    universe: pd.DataFrame,
    portfolio: pd.DataFrame,
    source_caps: pd.DataFrame,
) -> pd.DataFrame:
    exposure = float(portfolio["loan_amnt"].sum())
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        caps = _source_cap_map(source_caps, family)
        by_source = portfolio.groupby(family, dropna=False)["loan_amnt"].sum()
        for source_id in sorted(universe[family].dropna().astype(str).unique()):
            source_exposure = float(by_source.get(source_id, 0.0))
            share = source_exposure / max(exposure, 1.0)
            cap = float(caps.get(source_id, 1.0))
            rows.append(
                {
                    f"portfolio_label_v{VERSION}": "bounded_top15000_cardinality_milp",
                    "source_family": family,
                    "source_id": source_id,
                    f"cap_share_v{VERSION}": cap,
                    f"source_exposure_v{VERSION}": source_exposure,
                    f"source_share_v{VERSION}": share,
                    f"source_slack_v{VERSION}": cap - share,
                    f"source_cap_violated_v{VERSION}": share > cap + 1e-7,
                    f"claim_boundary_v{VERSION}": (
                        "v292 bounded cardinality MILP source diagnostic only"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _update_claim_boundaries(*, cardinality_restored: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v292 bounded cardinality-aware MILP restoration probe.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v292_cardinality_aware_multi_swap_milp_protocol.csv"
                ),
                "boundary": (
                    "Top-15000 omitted-return candidate pool plus v289 selected loans; "
                    "not a full-universe proof."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v292 restores 171-row cardinality inside the bounded top-15000 pool.",
                "allowed": bool(cardinality_restored),
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v292_cardinality_milp_action.csv"
                ),
                "boundary": (
                    "Bounded-pool feasibility claim only; return gap and global evidence remain open."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v292 is a new Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v292_claim_blockers.csv"
                ),
                "boundary": "The cardinality-restored portfolio trails v289 and v279 return.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v292 proves global full-universe integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v292_claim_blockers.csv"
                ),
                "boundary": "Bounded-pool MILP only; full universe and branch-price bound missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v292 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v292_claim_blockers.csv"
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
                "lane": "CVaR/OCE",
                "executable_item": (
                    "v292 solves a cardinality-aware MILP over v289 selected loans plus "
                    "the top-15000 omitted loans by expected return."
                ),
                "status": "bounded_cardinality_restored_return_gap_remains",
                "next_artifact": (
                    "paper4_v293_cardinality_return_gap_decomposition_or_diverse_pool_probe.csv"
                ),
                "success_condition": (
                    "explain or close the 3.031 return gap to v289, or find a "
                    "cardinality-restored bounded-pool portfolio that beats v289"
                ),
                "last_wave": "v292",
                "execution_result": "top15000_milp_restores_cardinality_but_trails_v289_return",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v292")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V292_CARDINALITY_AWARE_MULTI_SWAP_MILP_START -->"
    end = "<!-- V292_CARDINALITY_AWARE_MULTI_SWAP_MILP_END -->"
    block = f"""
{start}

## Wave v292: Cardinality-Aware Multi-Swap MILP

Generated: {status["generated_at_utc"]}

### Objective

v291 showed that restoring the v289 repair from 168 to 171 rows is impossible
with add-only moves. v292 escalates to a bounded multi-swap MILP over the v289
selected loans plus the top-15000 omitted loans by expected return, with exact
171-row cardinality, source caps, budget and the v289 CVaR cap.

### Results

- Pool rows: `{status["pool_rows_v292"]}`.
- Candidate pool limit: `{status["candidate_pool_limit_v292"]}`.
- MILP success: `{status["milp_success_v292"]}`.
- MILP gap: `{status["milp_gap_v292"]}`.
- Selected rows: `{status["selected_rows_v292"]}`.
- Added rows vs v289: `{status["added_rows_v292"]}`.
- Dropped rows vs v289: `{status["dropped_rows_v292"]}`.
- Portfolio exposure: `{status["portfolio_exposure_v292"]}`.
- Objective return: `{status["objective_return_v292"]}`.
- Delta return vs v289: `{status["delta_return_vs_v289_v292"]}`.
- Delta return vs v279: `{status["delta_return_vs_v279_v292"]}`.
- CVaR90: `{status["scenario_loss_cvar90_v292"]}`.
- Delta CVaR90 vs v289: `{status["delta_cvar90_vs_v289_v292"]}`.
- Source cap violations: `{status["source_cap_violations_v292"]}`.
- Cardinality restored: `{status["cardinality_restored_v292"]}`.

### Interpretation

v292 resolves one blocker and preserves another. The bounded MILP can restore
171 rows while remaining budget/source/CVaR feasible, but the best bounded
solution trails v289 by 3.031 return units and trails v279 by 2.328. That means
cardinality restoration is feasible, but not yet economically dominant.

### Claim Impact

- Allowed: bounded top-15000 cardinality-aware MILP executed; 171-row
  cardinality restored inside that bounded pool.
- Still prohibited: new working champion, full-universe optimality, Paper
  Estrella replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v292 in the living notebook. The next live-lab step is decomposing the
return gap or expanding the pool/design, not promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v289_exact_relief_repair_allocations.parquet").reset_index(
        drop=True
    )
    v289_summary = read_csv("paper4_v289_exact_relief_repair_summary.csv")
    v279_summary = read_csv("paper4_v279_restricted_pool_milp_repair_summary.csv")
    source_caps = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_caps = source_caps.loc[
        source_caps["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    if universe.empty or selected.empty or v289_summary.empty or v279_summary.empty:
        raise RuntimeError("Missing v55, v289, or v279 inputs for v292.")
    if source_caps.empty:
        raise RuntimeError("Missing focused source caps for v292.")

    v289_row = v289_summary.iloc[0]
    v279_row = v279_summary.iloc[0]
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    universe["loan_id"] = universe["loan_id"].astype(str)
    selected["loan_id"] = selected["loan_id"].astype(str)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    pool = _build_pool(universe=universe, selected=selected, mean_returns=mean_returns)
    pool_idx = idx_by_id.loc[pool["loan_id"].astype(str)].to_numpy()
    losses_pool = losses[:, pool_idx]
    mean_returns_pool = mean_returns[pool_idx]
    pool[f"mean_return_v{VERSION}"] = mean_returns_pool

    exposure_min = float(v279_row[f"exposure_min_v{BASE_REPAIR_VERSION}"])
    exposure_max = float(v279_row[f"exposure_max_v{BASE_REPAIR_VERSION}"])
    cvar_cap = float(v289_row[f"scenario_loss_cvar90_v{REPAIR_VERSION}"])
    diagnostics, selected_mask, constraints = _solve_cardinality_milp(
        pool=pool,
        losses_pool=losses_pool,
        mean_returns_pool=mean_returns_pool,
        source_caps=source_caps,
        exposure_min=exposure_min,
        exposure_max=exposure_max,
        cvar_cap=cvar_cap,
    )

    solution = pool.loc[selected_mask].copy()
    solution_idx = idx_by_id.loc[solution["loan_id"].astype(str)].to_numpy()
    solution_losses = losses[:, solution_idx].sum(axis=1)
    selected_ids = set(selected["loan_id"].astype(str))
    solution_ids = set(solution["loan_id"].astype(str))
    added_ids = sorted(solution_ids - selected_ids)
    dropped_ids = sorted(selected_ids - solution_ids)
    kept_current = len(solution_ids & selected_ids)
    objective_return = float(mean_returns[solution_idx].sum())
    exposure = float(solution["loan_amnt"].sum())
    cvar90 = v70._tail_cvar(solution_losses)
    source_summary = _source_summary(
        universe=universe,
        portfolio=solution,
        source_caps=source_caps,
    )
    source_violations = int(source_summary[f"source_cap_violated_v{VERSION}"].sum())
    cardinality_restored = int(len(solution)) == TARGET_SELECTED_ROWS
    return_beats_v289 = objective_return > float(v289_row[f"objective_return_v{REPAIR_VERSION}"])
    return_beats_v279 = objective_return > float(
        v279_row[f"objective_return_v{BASE_REPAIR_VERSION}"]
    )
    cvar_feasible = cvar90 <= cvar_cap + 1e-7
    budget_feasible = exposure_min - 1e-7 <= exposure <= exposure_max + 1e-7
    source_feasible = source_violations == 0

    solution["selected_v292"] = True
    solution["portfolio_label_v292"] = "bounded_top15000_cardinality_milp"
    solution["repair_action_v292"] = np.select(
        [
            solution["loan_id"].astype(str).isin(added_ids),
            solution["loan_id"].astype(str).isin(selected_ids),
        ],
        ["added_by_v292_milp", "kept_from_v289"],
        default="selected_by_v292_milp",
    )
    solution["claim_boundary_v292"] = (
        "bounded top-15000 cardinality-aware MILP allocation; not global proof"
    )

    pool_summary = pd.DataFrame(
        [
            {
                "pool_role_v292": role,
                "pool_rows_v292": int(len(group)),
                "selected_rows_v292": int(group["loan_id"].astype(str).isin(solution_ids).sum()),
                "claim_boundary_v292": "v292 bounded MILP pool composition only",
            }
            for role, group in pool.groupby("pool_role_v292", dropna=False)
        ]
    )
    constraints_summary = (
        constraints.groupby(f"constraint_type_v{VERSION}", dropna=False)
        .size()
        .reset_index(name=f"constraint_rows_v{VERSION}")
    )
    constraints_summary[f"claim_boundary_v{VERSION}"] = "v292 bounded MILP constraint count only"

    action = pd.DataFrame(
        [
            {
                f"action_id_v{VERSION}": "bounded_top15000_cardinality_milp_action",
                f"added_loan_ids_v{VERSION}": "|".join(added_ids),
                f"dropped_loan_ids_v{VERSION}": "|".join(dropped_ids),
                f"kept_current_rows_v{VERSION}": kept_current,
                f"added_rows_v{VERSION}": int(len(added_ids)),
                f"dropped_rows_v{VERSION}": int(len(dropped_ids)),
                f"selected_rows_v{VERSION}": int(len(solution)),
                f"cardinality_restored_v{VERSION}": cardinality_restored,
                f"return_beats_v289_v{VERSION}": return_beats_v289,
                f"return_beats_v279_v{VERSION}": return_beats_v279,
                f"claim_boundary_v{VERSION}": (
                    "cardinality restoration action only; return gap and global proof remain"
                ),
            }
        ]
    )
    protocol = pd.DataFrame(
        [
            {
                f"protocol_id_v{VERSION}": "bounded_top15000_cardinality_aware_milp",
                f"previous_protocol_version_v{VERSION}": PREVIOUS_PROTOCOL_VERSION,
                f"repair_version_v{VERSION}": REPAIR_VERSION,
                f"base_repair_version_v{VERSION}": BASE_REPAIR_VERSION,
                f"candidate_pool_limit_v{VERSION}": POOL_CANDIDATE_LIMIT,
                f"pool_rows_v{VERSION}": int(len(pool)),
                f"selected_rows_v{VERSION}": int(len(solution)),
                f"target_selected_rows_v{VERSION}": TARGET_SELECTED_ROWS,
                f"cardinality_restored_v{VERSION}": cardinality_restored,
                f"kept_current_rows_v{VERSION}": kept_current,
                f"added_rows_v{VERSION}": int(len(added_ids)),
                f"dropped_rows_v{VERSION}": int(len(dropped_ids)),
                f"portfolio_exposure_v{VERSION}": exposure,
                f"exposure_min_v{VERSION}": exposure_min,
                f"exposure_max_v{VERSION}": exposure_max,
                f"objective_return_v{VERSION}": objective_return,
                f"delta_return_vs_v289_v{VERSION}": objective_return
                - float(v289_row[f"objective_return_v{REPAIR_VERSION}"]),
                f"delta_return_vs_v279_v{VERSION}": objective_return
                - float(v279_row[f"objective_return_v{BASE_REPAIR_VERSION}"]),
                f"scenario_loss_mean_v{VERSION}": float(solution_losses.mean()),
                f"scenario_loss_cvar90_v{VERSION}": cvar90,
                f"delta_cvar90_vs_v289_v{VERSION}": cvar90 - cvar_cap,
                f"source_cap_violations_v{VERSION}": source_violations,
                f"budget_feasible_v{VERSION}": budget_feasible,
                f"cvar_feasible_v{VERSION}": cvar_feasible,
                f"source_feasible_v{VERSION}": source_feasible,
                f"return_beats_v289_v{VERSION}": return_beats_v289,
                f"return_beats_v279_v{VERSION}": return_beats_v279,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"bounded_pool_optimality_claim_allowed_v{VERSION}": bool(
                    diagnostics[f"milp_success_v{VERSION}"]
                ),
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_cardinality_return_gap_decomposition_or_"
                    "diverse_pool_probe.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "bounded cardinality-aware MILP restores rows but trails v289/v279 return"
                ),
                **diagnostics,
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "cardinality_restored_but_return_gap_vs_v289",
                f"blocking_v{VERSION}": not return_beats_v289,
                f"evidence_count_v{VERSION}": abs(
                    objective_return - float(v289_row[f"objective_return_v{REPAIR_VERSION}"])
                ),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_cardinality_return_gap_decomposition_or_"
                    "diverse_pool_probe.csv"
                ),
                f"claim_boundary_v{VERSION}": "v292 restores rows but does not beat v289 return",
            },
            {
                f"blocker_id_v{VERSION}": "bounded_pool_not_full_universe",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    len(universe) - len(selected) - POOL_CANDIDATE_LIMIT
                ),
                f"required_next_artifact_v{VERSION}": "future_full_universe_branch_price_bound",
                f"claim_boundary_v{VERSION}": "v292 uses top-15000 omitted candidates only",
            },
            {
                f"blocker_id_v{VERSION}": "dynamic_replay_and_deployment_gates_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_dynamic_replay_validation",
                f"claim_boundary_v{VERSION}": "no online or deployment validation created",
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
                "claim_id": "v292_cardinality_aware_milp_protocol_executed",
                "allowed": True,
                "artifact": "paper4_v292_cardinality_aware_multi_swap_milp_protocol.csv",
                "boundary": "bounded top-15000 pool",
            },
            {
                "claim_id": "v292_cardinality_restored_in_bounded_pool",
                "allowed": cardinality_restored,
                "artifact": "paper4_v292_cardinality_milp_action.csv",
                "boundary": "bounded top-15000 pool feasibility only",
            },
            {
                "claim_id": "v292_bounded_pool_milp_optimality",
                "allowed": bool(diagnostics[f"milp_success_v{VERSION}"]),
                "artifact": "paper4_v292_cardinality_aware_multi_swap_milp_protocol.csv",
                "boundary": "bounded-pool solver optimality, not full universe",
            },
            {
                "claim_id": "v292_return_improves_vs_v289",
                "allowed": return_beats_v289,
                "artifact": "paper4_v292_claim_blockers.csv",
                "boundary": "false because v292 trails v289 return",
            },
            {
                "claim_id": "v292_working_champion",
                "allowed": False,
                "artifact": "paper4_v292_claim_blockers.csv",
                "boundary": "return and global evidence missing",
            },
            {
                "claim_id": "v292_global_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v292_claim_blockers.csv",
                "boundary": "global certificate missing",
            },
            {
                "claim_id": "v292_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v292_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v292_cardinality_aware_multi_swap_milp_protocol.csv", protocol)
    write_csv(TABLE_DIR / "paper4_v292_cardinality_milp_action.csv", action)
    write_csv(TABLE_DIR / "paper4_v292_cardinality_milp_pool_summary.csv", pool_summary)
    write_csv(
        TABLE_DIR / "paper4_v292_cardinality_milp_constraint_summary.csv", constraints_summary
    )
    write_csv(TABLE_DIR / "paper4_v292_cardinality_milp_source_summary.csv", source_summary)
    write_csv(TABLE_DIR / "paper4_v292_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v292_claim_matrix_delta.csv", claim_matrix)
    solution.to_parquet(TABLE_DIR / "paper4_v292_cardinality_milp_allocations.parquet", index=False)
    _update_claim_boundaries(cardinality_restored=cardinality_restored)
    _update_backlog()

    row = protocol.iloc[0]
    status = {
        "phase": "v292_cardinality_aware_multi_swap_milp_protocol",
        "schema_version": "2026-05-15.292",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "previous_protocol_version_v292": PREVIOUS_PROTOCOL_VERSION,
        "repair_version_v292": REPAIR_VERSION,
        "base_repair_version_v292": BASE_REPAIR_VERSION,
        "candidate_pool_limit_v292": POOL_CANDIDATE_LIMIT,
        "pool_rows_v292": int(row[f"pool_rows_v{VERSION}"]),
        "selected_rows_v292": int(row[f"selected_rows_v{VERSION}"]),
        "target_selected_rows_v292": TARGET_SELECTED_ROWS,
        "cardinality_restored_v292": bool(row[f"cardinality_restored_v{VERSION}"]),
        "kept_current_rows_v292": int(row[f"kept_current_rows_v{VERSION}"]),
        "added_rows_v292": int(row[f"added_rows_v{VERSION}"]),
        "dropped_rows_v292": int(row[f"dropped_rows_v{VERSION}"]),
        "portfolio_exposure_v292": float(row[f"portfolio_exposure_v{VERSION}"]),
        "objective_return_v292": float(row[f"objective_return_v{VERSION}"]),
        "delta_return_vs_v289_v292": float(row[f"delta_return_vs_v289_v{VERSION}"]),
        "delta_return_vs_v279_v292": float(row[f"delta_return_vs_v279_v{VERSION}"]),
        "scenario_loss_cvar90_v292": float(row[f"scenario_loss_cvar90_v{VERSION}"]),
        "delta_cvar90_vs_v289_v292": float(row[f"delta_cvar90_vs_v289_v{VERSION}"]),
        "source_cap_violations_v292": source_violations,
        "budget_feasible_v292": bool(row[f"budget_feasible_v{VERSION}"]),
        "cvar_feasible_v292": bool(row[f"cvar_feasible_v{VERSION}"]),
        "source_feasible_v292": bool(row[f"source_feasible_v{VERSION}"]),
        "return_beats_v289_v292": bool(row[f"return_beats_v289_v{VERSION}"]),
        "return_beats_v279_v292": bool(row[f"return_beats_v279_v{VERSION}"]),
        "milp_success_v292": bool(row[f"milp_success_v{VERSION}"]),
        "milp_status_v292": int(row[f"milp_status_v{VERSION}"]),
        "milp_gap_v292": float(row[f"milp_gap_v{VERSION}"]),
        "milp_node_count_v292": int(row[f"milp_node_count_v{VERSION}"]),
        "milp_variable_count_v292": int(row[f"milp_variable_count_v{VERSION}"]),
        "milp_constraint_rows_v292": int(row[f"milp_constraint_rows_v{VERSION}"]),
        "pool_summary_rows_v292": int(len(pool_summary)),
        "constraint_summary_rows_v292": int(len(constraints_summary)),
        "source_summary_rows_v292": int(len(source_summary)),
        "claim_blocker_rows_v292": int(len(blockers)),
        "claim_matrix_rows_v292": int(len(claim_matrix)),
        "working_champion_claim_allowed_v292": False,
        "bounded_pool_optimality_claim_allowed_v292": bool(
            row[f"bounded_pool_optimality_claim_allowed_v{VERSION}"]
        ),
        "full_universe_integer_optimality_claim_allowed_v292": False,
        "paper1_promotion_allowed_v292": False,
        "paper4_working_champion_changed_v292": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v292": row[f"next_artifact_v{VERSION}"],
        "claim_boundary": (
            "v292 restores cardinality inside a bounded top-15000 MILP, but trails "
            "v289/v279 return and cannot promote"
        ),
    }
    write_json(STATUS_DIR / "paper4_v292_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v292": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
