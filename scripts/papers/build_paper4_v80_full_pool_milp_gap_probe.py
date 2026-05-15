#!/usr/bin/env python3
"""Build Paper 4 v80 full-pool binary MILP/gap probe artifacts.

v80 moves beyond the v79 active-support probe and attempts a binary whole-loan
MILP over the full focused restricted pool. This is still not a full-universe
certificate: the pool is the current generated-column pool, not every v55 loan,
and integer pricing/global optimality remains a stronger claim than this probe.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.papers import build_paper4_v70_restricted_master_solver as v70  # noqa: E402
from scripts.papers import build_paper4_v78_source_scope_expanded_reprice as v78  # noqa: E402

PAPER4_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = PAPER4_ROOT / "tables"
STATUS_DIR = PAPER4_ROOT / "status"
NOTE_DIR = PAPER4_ROOT / "notes"
NOTEBOOK = NOTE_DIR / "paper4_living_lab_notebook.md"
FORBIDDEN_FINAL_PROMOTION = STATUS_DIR / "paper4_final_promotion.json"
MILP_TIME_LIMIT_SECONDS = 300.0


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


def _focused_pool() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    iteration_1_candidates = read_parquet("paper4_v72_iteration_1_candidates.parquet")
    iteration_2_candidates = read_parquet("paper4_v74_iteration_2_candidates.parquet")
    iteration_3_candidates = read_parquet("paper4_v76_iteration_3_candidates.parquet")
    frontier = read_csv("paper4_v76_iteration_3_frontier.csv")
    concentration = read_csv("paper4_v63_source_repair_concentration.csv")
    if (
        universe.empty
        or iteration_1_candidates.empty
        or iteration_2_candidates.empty
        or iteration_3_candidates.empty
        or frontier.empty
        or concentration.empty
    ):
        return pd.DataFrame(), universe, concentration, "", ""
    policy_id = str(frontier["policy_id"].iloc[0])
    regime = str(frontier["regime_v76"].iloc[0])
    base_master = v78._enriched_base_master(universe)
    pool = v78._pool_for_reprice(
        base_master,
        iteration_1_candidates,
        iteration_2_candidates,
        iteration_3_candidates,
        policy_id,
        regime,
    )
    allocations = read_parquet("paper4_v76_iteration_3_allocations.parquet")
    allocation_cols = [
        "loan_id",
        "allocation_fraction_v76",
        "allocated_exposure_v76",
        "master_role_v76",
    ]
    pool = pool.merge(
        allocations[allocation_cols].assign(loan_id=lambda df: df["loan_id"].astype(str)),
        on="loan_id",
        how="left",
    )
    pool["allocation_fraction_v76"] = pool["allocation_fraction_v76"].fillna(0.0)
    pool["allocated_exposure_v76"] = pool["allocated_exposure_v76"].fillna(0.0)
    return pool, universe, concentration, policy_id, regime


def _tail_cvar(values: np.ndarray) -> float:
    return v70._tail_cvar(values)


def _source_diagnostics(
    pool: pd.DataFrame,
    universe: pd.DataFrame,
    x: np.ndarray,
    source_map: pd.DataFrame,
    regime: str,
) -> pd.DataFrame:
    exposure = float(pool["loan_amnt"].to_numpy(float) @ x)
    rows: list[dict[str, Any]] = []
    for family in v78.FAMILIES:
        cap = v70._cap_share(source_map, family, regime)
        for source_id in sorted(universe[family].dropna().astype(str).unique()):
            indicator = pool[family].astype(str).eq(source_id).to_numpy(float)
            source_exposure = float((pool["loan_amnt"].to_numpy(float) * indicator) @ x)
            share = source_exposure / max(exposure, 1.0)
            rows.append(
                {
                    "source_family": family,
                    "source_id": source_id,
                    "cap_share_v80": cap,
                    "source_exposure_v80": source_exposure,
                    "source_share_v80": share,
                    "source_slack_v80": cap - share,
                    "source_cap_violated_v80": share > cap + 1e-7,
                }
            )
    return pd.DataFrame(rows)


def _portfolio_metrics(
    label: str,
    pool: pd.DataFrame,
    x: np.ndarray,
    losses: np.ndarray,
    returns_by_path: np.ndarray,
    source_map: pd.DataFrame,
    regime: str,
    universe: pd.DataFrame,
    solver_success: bool,
    solver_message: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    amounts = pool["loan_amnt"].to_numpy(float)
    scenario_losses = losses @ x
    scenario_returns = returns_by_path @ x
    source_diag = _source_diagnostics(pool, universe, x, source_map, regime)
    row = {
        "portfolio_label_v80": label,
        "solver_success_v80": solver_success,
        "solver_message_v80": solver_message,
        "pool_rows_v80": int(len(pool)),
        "selected_rows_v80": int((x > 1e-7).sum()),
        "fractional_rows_v80": int(((x > 1e-7) & (x < 0.999999)).sum()),
        "portfolio_exposure_v80": float(amounts @ x),
        "objective_return_v80": float(scenario_returns.mean()),
        "scenario_loss_mean_v80": float(scenario_losses.mean()),
        "scenario_loss_cvar90_v80": _tail_cvar(scenario_losses),
        "source_cap_violations_v80": int(source_diag["source_cap_violated_v80"].sum()),
        "max_source_share_v80": float(source_diag["source_share_v80"].max()),
        "min_source_slack_v80": float(source_diag["source_slack_v80"].min()),
        "claim_boundary_v80": (
            "focused restricted-pool MILP/gap probe; not full-universe whole-loan certificate"
        ),
    }
    source_diag["portfolio_label_v80"] = label
    return row, source_diag


def _full_pool_milp(
    pool: pd.DataFrame,
    universe: pd.DataFrame,
    source_map: pd.DataFrame,
    regime: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    losses, returns_by_path, path_ids = v70._expected_by_path_matrix(pool)
    n = len(pool)
    amounts = pool["loan_amnt"].to_numpy(float)
    frontier = read_csv("paper4_v76_iteration_3_frontier.csv")
    frontier_row = frontier.loc[frontier["regime_v76"].astype(str).eq(regime)].head(1)
    if frontier_row.empty:
        exposure_min = float(pool["allocated_exposure_v76"].sum())
        exposure_max = max(v70.TARGET_EXPOSURE, exposure_min)
        cvar_cap = float(_tail_cvar(losses @ pool["allocation_fraction_v76"].to_numpy(float)))
    else:
        exposure_min = float(frontier_row["portfolio_exposure_v76"].iloc[0])
        exposure_max = max(
            v70.TARGET_EXPOSURE, float(frontier_row["incumbent_exposure_v76"].iloc[0])
        )
        cvar_cap = float(frontier_row["cvar_cap_v76"].iloc[0])

    s_count = len(path_ids)
    var_count = n + 1 + s_count
    c = np.zeros(var_count)
    c[:n] = -returns_by_path.mean(axis=0)

    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []
    constraint_meta: list[dict[str, Any]] = []

    budget = np.zeros(var_count)
    budget[:n] = amounts
    rows.append(budget)
    lb.append(exposure_min)
    ub.append(exposure_max)
    constraint_meta.append({"constraint_type_v80": "budget_range"})

    source_rows, source_bounds, source_meta = v78._source_constraint_rows_full_scope(
        pool, universe, amounts, source_map, regime
    )
    for row, bound, meta in zip(source_rows, source_bounds, source_meta, strict=False):
        full = np.zeros(var_count)
        full[:n] = row
        rows.append(full)
        lb.append(-np.inf)
        ub.append(bound)
        constraint_meta.append(
            {
                "constraint_type_v80": "source_share",
                "source_family": meta.get("source_family"),
                "source_id": meta.get("source_id"),
                "cap_share_v80": meta.get("cap_share_v71"),
            }
        )

    cvar_row = np.zeros(var_count)
    cvar_row[n] = 1.0
    cvar_row[n + 1 :] = 1.0 / ((1.0 - v70.ALPHA) * s_count)
    rows.append(cvar_row)
    lb.append(-np.inf)
    ub.append(cvar_cap)
    constraint_meta.append({"constraint_type_v80": "cvar_cap"})

    for s_idx in range(s_count):
        row = np.zeros(var_count)
        row[:n] = losses[s_idx, :]
        row[n] = -1.0
        row[n + 1 + s_idx] = -1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)
        constraint_meta.append(
            {"constraint_type_v80": "cvar_path_excess", "path_id": path_ids[s_idx]}
        )

    a = np.vstack(rows)
    result = milp(
        c,
        integrality=np.r_[np.ones(n), np.zeros(1 + s_count)],
        bounds=Bounds(
            np.r_[np.zeros(n), 0.0, np.zeros(s_count)],
            np.r_[np.ones(n), np.inf, np.full(s_count, np.inf)],
        ),
        constraints=LinearConstraint(a, np.array(lb), np.array(ub)),
        options={"time_limit": MILP_TIME_LIMIT_SECONDS, "mip_rel_gap": 1e-6},
    )
    x = np.zeros(n)
    incumbent_available = result.x is not None
    if incumbent_available:
        x = np.rint(np.clip(result.x[:n], 0, 1)).astype(float)
    diagnostics = {
        "milp_success_v80": bool(result.success),
        "milp_incumbent_available_v80": bool(incumbent_available),
        "milp_status_v80": int(result.status),
        "milp_message_v80": str(result.message),
        "milp_fun_v80": float(result.fun) if result.fun is not None else np.nan,
        "milp_dual_bound_v80": float(getattr(result, "mip_dual_bound", np.nan)),
        "milp_gap_v80": float(getattr(result, "mip_gap", np.nan)),
        "milp_node_count_v80": int(getattr(result, "mip_node_count", -1)),
        "time_limit_seconds_v80": MILP_TIME_LIMIT_SECONDS,
        "exposure_min_v80": exposure_min,
        "exposure_max_v80": exposure_max,
        "cvar_cap_v80": cvar_cap,
        "constraint_rows_v80": int(len(rows)),
        "binary_variables_v80": int(n),
        "continuous_variables_v80": int(1 + s_count),
    }
    constraint_diagnostics = pd.DataFrame(constraint_meta)
    constraint_diagnostics["constraint_row_v80"] = np.arange(len(constraint_diagnostics))
    return diagnostics, x, losses, returns_by_path, constraint_diagnostics


def build_full_pool_probe() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    pool, universe, concentration, policy_id, regime = _focused_pool()
    if pool.empty or universe.empty or concentration.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty
    source_map = v70._policy_source_map(concentration, policy_id)
    milp_info, milp_x, losses, returns_by_path, constraint_diagnostics = _full_pool_milp(
        pool, universe, source_map, regime
    )
    lp_x = pool["allocation_fraction_v76"].to_numpy(float)

    rows: list[dict[str, Any]] = []
    source_frames: list[pd.DataFrame] = []
    lp_row, lp_source = _portfolio_metrics(
        "continuous_lp_reference",
        pool,
        lp_x,
        losses,
        returns_by_path,
        source_map,
        regime,
        universe,
        True,
        "v76 continuous LP reference",
    )
    rows.append(lp_row)
    source_frames.append(lp_source)

    milp_row, milp_source = _portfolio_metrics(
        "focused_full_pool_binary_milp",
        pool,
        milp_x,
        losses,
        returns_by_path,
        source_map,
        regime,
        universe,
        bool(milp_info["milp_success_v80"]),
        str(milp_info["milp_message_v80"]),
    )
    milp_row.update(milp_info)
    rows.append(milp_row)
    source_frames.append(milp_source)

    summary = pd.DataFrame(rows)
    lp_return = float(
        summary.loc[
            summary["portfolio_label_v80"].eq("continuous_lp_reference"), "objective_return_v80"
        ].iloc[0]
    )
    lp_cvar = float(
        summary.loc[
            summary["portfolio_label_v80"].eq("continuous_lp_reference"),
            "scenario_loss_cvar90_v80",
        ].iloc[0]
    )
    summary["delta_return_vs_lp_v80"] = summary["objective_return_v80"] - lp_return
    summary["delta_cvar90_vs_lp_v80"] = summary["scenario_loss_cvar90_v80"] - lp_cvar
    summary["policy_id"] = policy_id
    summary["regime_v80"] = regime

    allocations = pool[
        [
            "loan_id",
            "loan_amnt",
            "master_role_v69",
            "allocation_fraction_v76",
            "allocated_exposure_v76",
            "grade",
            "score_decile",
            "income_band",
            "dti_band",
            "period",
            "state_top20",
        ]
    ].copy()
    allocations["policy_id"] = policy_id
    allocations["regime_v80"] = regime
    allocations["focused_full_pool_binary_selected_v80"] = milp_x
    allocations["focused_full_pool_binary_exposure_v80"] = (
        allocations["loan_amnt"].astype(float)
        * allocations["focused_full_pool_binary_selected_v80"]
    )
    allocations["claim_boundary_v80"] = "focused restricted-pool binary allocation only"

    source_summary = pd.concat(source_frames, ignore_index=True)
    blockers = pd.DataFrame(
        [
            {
                "blocker_id_v80": "focused_full_pool_milp_gap_recorded",
                "blocking_v80": False,
                "evidence_count_v80": int(len(pool)),
                "required_next_artifact_v80": "paper4_v80_full_pool_milp_gap_summary.csv",
                "claim_boundary_v80": "focused restricted-pool MILP/gap evidence only",
            },
            {
                "blocker_id_v80": "full_universe_integer_pricing_missing",
                "blocking_v80": True,
                "evidence_count_v80": 257954,
                "required_next_artifact_v80": "paper4_v81_integer_pricing_or_global_gap_protocol.csv",
                "claim_boundary_v80": "binary MILP over generated pool is not full-universe proof",
            },
            {
                "blocker_id_v80": "paper_estrella_or_final_promotion_not_allowed",
                "blocking_v80": True,
                "evidence_count_v80": 1,
                "required_next_artifact_v80": "paper4_future_promotion_protocol.csv",
                "claim_boundary_v80": "MILP/gap probe does not replace Paper Estrella",
            },
        ]
    )
    return summary, allocations, source_summary, constraint_diagnostics, blockers


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v80 focused full-pool binary MILP/gap probe.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v80_full_pool_milp_gap_summary.csv"
                ),
                "boundary": "Generated focused restricted pool only; not full-universe proof.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v80 proves whole-loan full-universe optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v80_claim_blockers.csv"
                ),
                "boundary": "Requires integer pricing/global gap evidence over omitted universe loans.",
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
                    "v80 attempts a binary MILP over the focused generated full pool and records gap."
                ),
                "status": "integer_pricing_gated",
                "next_artifact": "paper4_v81_integer_pricing_or_global_gap_protocol.csv",
                "success_condition": (
                    "integer pricing/global gap protocol covers omitted full-universe loans"
                ),
                "last_wave": "v80",
                "execution_result": "focused_full_pool_milp_gap_probe_completed",
                "quarto_promotion_decision": "living_notebook_only",
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
    start = "<!-- V80_FULL_POOL_MILP_GAP_START -->"
    end = "<!-- V80_FULL_POOL_MILP_GAP_END -->"
    block = f"""
{start}

## Wave v80: Focused Full-Pool MILP Gap Probe

Generated: {status["generated_at_utc"]}

### Objective

Attempt a binary whole-loan MILP over the focused generated restricted pool
after pricing and source-scope diagnostics cleared. This expands v79 from the
active support to the full generated pool, but it still does not cover every
omitted v55 universe loan as integer alternatives.

### Results

- Pool rows: `{status["pool_rows_v80"]}`.
- Constraint rows: `{status["constraint_rows_v80"]}`.
- MILP success: `{status["milp_solver_success_v80"]}`.
- MILP incumbent available: `{status["milp_incumbent_available_v80"]}`.
- MILP gap: `{status["milp_gap_v80"]}`.
- Selected rows: `{status["milp_selected_rows_v80"]}`.
- Return delta vs LP: `{status["milp_delta_return_vs_lp_v80"]}`.
- CVaR delta vs LP: `{status["milp_delta_cvar90_vs_lp_v80"]}`.
- Source cap violations: `{status["milp_source_cap_violations_v80"]}`.
- Whole-loan full-universe claim allowed: `{status["whole_loan_full_universe_claim_allowed_v80"]}`.

### Interpretation

v80 strengthens the integrality story from active support to the generated
focused pool. The remaining blocker is not this pool solve itself; it is the
absence of an integer-pricing or global-gap protocol for omitted full-universe
loans outside the generated pool.

### Claim Impact

- Allowed: focused full-pool binary MILP/gap probe completed.
- Still prohibited: whole-loan full-universe optimality, Paper Estrella
  replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v80 in the living notebook. Promote only after omitted-universe integer
pricing or an equivalent global gap certificate exists.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v80() -> dict[str, Any]:
    started = datetime.now(UTC)
    summary, allocations, source_summary, constraints, blockers = build_full_pool_probe()
    write_csv(TABLE_DIR / "paper4_v80_full_pool_milp_gap_summary.csv", summary)
    allocations.to_parquet(
        TABLE_DIR / "paper4_v80_full_pool_milp_gap_allocations.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v80_full_pool_milp_gap_source_summary.csv", source_summary)
    write_csv(TABLE_DIR / "paper4_v80_full_pool_milp_gap_constraints.csv", constraints)
    write_csv(TABLE_DIR / "paper4_v80_claim_blockers.csv", blockers)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v80_focused_full_pool_milp_gap_probe_executed",
                "allowed": True,
                "artifact": "paper4_v80_full_pool_milp_gap_summary.csv",
                "boundary": "focused generated restricted pool only",
            },
            {
                "claim_id": "v80_whole_loan_full_universe_optimality",
                "allowed": False,
                "artifact": "paper4_v80_claim_blockers.csv",
                "boundary": "requires omitted-universe integer pricing/global gap evidence",
            },
            {
                "claim_id": "v80_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v80_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v80_claim_matrix_delta.csv", claim_matrix)

    lp = summary.loc[summary["portfolio_label_v80"].eq("continuous_lp_reference")].iloc[0]
    milp_row = summary.loc[summary["portfolio_label_v80"].eq("focused_full_pool_binary_milp")].iloc[
        0
    ]
    status = {
        "phase": "v80_focused_full_pool_milp_gap_probe",
        "schema_version": "2026-05-15.80",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "summary_rows_v80": int(len(summary)),
        "allocation_rows_v80": int(len(allocations)),
        "source_summary_rows_v80": int(len(source_summary)),
        "constraint_rows_v80": int(milp_row.get("constraint_rows_v80", len(constraints))),
        "claim_blocker_rows_v80": int(len(blockers)),
        "pool_rows_v80": int(lp["pool_rows_v80"]),
        "lp_fractional_rows_v80": int(lp["fractional_rows_v80"]),
        "milp_solver_success_v80": bool(milp_row["solver_success_v80"]),
        "milp_incumbent_available_v80": bool(milp_row.get("milp_incumbent_available_v80", False)),
        "milp_status_v80": int(milp_row.get("milp_status_v80", -1)),
        "milp_gap_v80": float(milp_row.get("milp_gap_v80", np.nan)),
        "milp_node_count_v80": int(milp_row.get("milp_node_count_v80", -1)),
        "milp_selected_rows_v80": int(milp_row["selected_rows_v80"]),
        "milp_delta_return_vs_lp_v80": float(milp_row["delta_return_vs_lp_v80"]),
        "milp_delta_cvar90_vs_lp_v80": float(milp_row["delta_cvar90_vs_lp_v80"]),
        "milp_source_cap_violations_v80": int(milp_row["source_cap_violations_v80"]),
        "whole_loan_full_universe_claim_allowed_v80": False,
        "paper1_promotion_allowed_v80": False,
        "paper4_working_champion_changed_v80": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v80 solves or probes a focused generated-pool binary MILP; omitted-universe "
            "integer pricing/global gap evidence remains missing"
        ),
    }
    write_json(STATUS_DIR / "paper4_v80_status.json", status)
    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v80": build_v80()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
