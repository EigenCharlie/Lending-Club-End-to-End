#!/usr/bin/env python3
"""Build Paper 4 v79 support-restricted integrality probe artifacts.

v79 solves a small binary MILP over the active v76 support to quantify the
continuous-to-whole-loan gap. This is deliberately not a full-pool or
full-universe MILP certificate; it only tests whether the focused LP solution's
active support has a nearby whole-loan feasible counterpart.
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
SUPPORT_TIME_LIMIT_SECONDS = 120.0


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


def _support_pool() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
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
    full_pool = v78._pool_for_reprice(
        base_master,
        iteration_1_candidates,
        iteration_2_candidates,
        iteration_3_candidates,
        policy_id,
        regime,
    )
    allocations = read_parquet("paper4_v76_iteration_3_allocations.parquet")
    support_ids = set(allocations["loan_id"].astype(str))
    support = full_pool.loc[full_pool["loan_id"].astype(str).isin(support_ids)].copy()
    support = support.merge(
        allocations[
            [
                "loan_id",
                "allocation_fraction_v76",
                "allocated_exposure_v76",
                "master_role_v76",
            ]
        ].assign(loan_id=lambda df: df["loan_id"].astype(str)),
        on="loan_id",
        how="left",
    )
    return support, universe, concentration, policy_id, regime


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
                    "cap_share_v79": cap,
                    "source_exposure_v79": source_exposure,
                    "source_share_v79": share,
                    "source_slack_v79": cap - share,
                    "source_cap_violated_v79": share > cap + 1e-7,
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
        "portfolio_label_v79": label,
        "solver_success_v79": solver_success,
        "solver_message_v79": solver_message,
        "support_rows_v79": int(len(pool)),
        "selected_rows_v79": int((x > 1e-7).sum()),
        "fractional_rows_v79": int(((x > 1e-7) & (x < 0.999999)).sum()),
        "portfolio_exposure_v79": float(amounts @ x),
        "objective_return_v79": float(scenario_returns.mean()),
        "scenario_loss_mean_v79": float(scenario_losses.mean()),
        "scenario_loss_cvar90_v79": _tail_cvar(scenario_losses),
        "source_cap_violations_v79": int(source_diag["source_cap_violated_v79"].sum()),
        "max_source_share_v79": float(source_diag["source_share_v79"].max()),
        "min_source_slack_v79": float(source_diag["source_slack_v79"].min()),
        "claim_boundary_v79": (
            "support-restricted integrality probe only; not full-pool or full-universe MILP"
        ),
    }
    source_diag["portfolio_label_v79"] = label
    return row, source_diag


def _milp_support_solution(
    pool: pd.DataFrame,
    universe: pd.DataFrame,
    source_map: pd.DataFrame,
    regime: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
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

    budget = np.zeros(var_count)
    budget[:n] = amounts
    rows.append(budget)
    lb.append(exposure_min)
    ub.append(exposure_max)

    source_rows, source_bounds, _source_meta = v78._source_constraint_rows_full_scope(
        pool, universe, amounts, source_map, regime
    )
    for row, bound in zip(source_rows, source_bounds, strict=False):
        full = np.zeros(var_count)
        full[:n] = row
        rows.append(full)
        lb.append(-np.inf)
        ub.append(bound)

    cvar_row = np.zeros(var_count)
    cvar_row[n] = 1.0
    cvar_row[n + 1 :] = 1.0 / ((1.0 - v70.ALPHA) * s_count)
    rows.append(cvar_row)
    lb.append(-np.inf)
    ub.append(cvar_cap)

    for s_idx in range(s_count):
        row = np.zeros(var_count)
        row[:n] = losses[s_idx, :]
        row[n] = -1.0
        row[n + 1 + s_idx] = -1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)

    constraints = LinearConstraint(np.vstack(rows), np.array(lb), np.array(ub))
    result = milp(
        c,
        integrality=np.r_[np.ones(n), np.zeros(1 + s_count)],
        bounds=Bounds(
            np.r_[np.zeros(n), 0.0, np.zeros(s_count)],
            np.r_[np.ones(n), np.inf, np.full(s_count, np.inf)],
        ),
        constraints=constraints,
        options={"time_limit": SUPPORT_TIME_LIMIT_SECONDS, "mip_rel_gap": 1e-9},
    )
    x = np.zeros(n)
    if result.x is not None:
        x = np.rint(np.clip(result.x[:n], 0, 1)).astype(float)
    info = {
        "milp_success_v79": bool(result.success),
        "milp_status_v79": int(result.status),
        "milp_message_v79": str(result.message),
        "milp_fun_v79": float(result.fun) if result.fun is not None else np.nan,
        "exposure_min_v79": exposure_min,
        "exposure_max_v79": exposure_max,
        "cvar_cap_v79": cvar_cap,
    }
    return info, x, losses, returns_by_path


def build_integrality_probe() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool, universe, concentration, policy_id, regime = _support_pool()
    if pool.empty or universe.empty or concentration.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    source_map = v70._policy_source_map(concentration, policy_id)
    milp_info, milp_x, losses, returns_by_path = _milp_support_solution(
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
        "support_binary_milp",
        pool,
        milp_x,
        losses,
        returns_by_path,
        source_map,
        regime,
        universe,
        bool(milp_info["milp_success_v79"]),
        str(milp_info["milp_message_v79"]),
    )
    milp_row.update(milp_info)
    rows.append(milp_row)
    source_frames.append(milp_source)

    summary = pd.DataFrame(rows)
    lp_return = float(
        summary.loc[
            summary["portfolio_label_v79"].eq("continuous_lp_reference"), "objective_return_v79"
        ].iloc[0]
    )
    lp_cvar = float(
        summary.loc[
            summary["portfolio_label_v79"].eq("continuous_lp_reference"), "scenario_loss_cvar90_v79"
        ].iloc[0]
    )
    summary["delta_return_vs_lp_v79"] = summary["objective_return_v79"] - lp_return
    summary["delta_cvar90_vs_lp_v79"] = summary["scenario_loss_cvar90_v79"] - lp_cvar
    summary["policy_id"] = policy_id
    summary["regime_v79"] = regime

    allocation_out = pool[
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
    allocation_out["policy_id"] = policy_id
    allocation_out["regime_v79"] = regime
    allocation_out["support_binary_selected_v79"] = milp_x
    allocation_out["support_binary_exposure_v79"] = (
        allocation_out["loan_amnt"].astype(float) * allocation_out["support_binary_selected_v79"]
    )
    allocation_out["claim_boundary_v79"] = "support-restricted binary allocation only"

    source_summary = pd.concat(source_frames, ignore_index=True)
    blockers = pd.DataFrame(
        [
            {
                "blocker_id_v79": "support_integrality_gap_quantified",
                "blocking_v79": False,
                "evidence_count_v79": 1,
                "required_next_artifact_v79": "paper4_v79_integrality_probe_summary.csv",
                "claim_boundary_v79": "support binary MILP solved only over active LP rows",
            },
            {
                "blocker_id_v79": "full_pool_or_full_universe_milp_missing",
                "blocking_v79": True,
                "evidence_count_v79": int(len(pool)),
                "required_next_artifact_v79": "paper4_v80_full_pool_milp_or_gap_certificate.csv",
                "claim_boundary_v79": "support MILP is not full-pool or full-universe optimality",
            },
            {
                "blocker_id_v79": "paper_estrella_or_final_promotion_not_allowed",
                "blocking_v79": True,
                "evidence_count_v79": 1,
                "required_next_artifact_v79": "paper4_future_promotion_protocol.csv",
                "claim_boundary_v79": "integrality probe does not replace Paper Estrella",
            },
        ]
    )
    return summary, allocation_out, source_summary, blockers


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v79 support-restricted binary MILP integrality probe.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v79_integrality_probe_summary.csv"
                ),
                "boundary": "Active support only; not full-pool or full-universe MILP.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v79 proves whole-loan full-universe optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v79_claim_blockers.csv"
                ),
                "boundary": "Requires full-pool/full-universe MILP or valid global gap certificate.",
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
                    "v79 solves a binary MILP over the active v76 support to quantify integrality."
                ),
                "status": "global_milp_gated",
                "next_artifact": "paper4_v80_full_pool_milp_or_gap_certificate.csv",
                "success_condition": (
                    "full-pool or full-universe MILP/gap certificate replaces support-only probe"
                ),
                "last_wave": "v79",
                "execution_result": "support_integrality_probe_completed",
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
    start = "<!-- V79_INTEGRALITY_PROBE_START -->"
    end = "<!-- V79_INTEGRALITY_PROBE_END -->"
    block = f"""
{start}

## Wave v79: Support Integrality Probe

Generated: {status["generated_at_utc"]}

### Objective

Solve a binary MILP over the active v76 support to quantify how much the
focused continuous LP depends on fractional loan allocations. This is an
active-support probe only, not a full-pool or full-universe MILP certificate.

### Results

- Support rows: `{status["support_rows_v79"]}`.
- LP fractional rows: `{status["lp_fractional_rows_v79"]}`.
- MILP solver success: `{status["milp_solver_success_v79"]}`.
- MILP selected rows: `{status["milp_selected_rows_v79"]}`.
- Return delta vs LP: `{status["milp_delta_return_vs_lp_v79"]}`.
- CVaR delta vs LP: `{status["milp_delta_cvar90_vs_lp_v79"]}`.
- Full-universe MILP claim allowed: `{status["whole_loan_full_universe_claim_allowed_v79"]}`.

### Interpretation

v79 is useful because it makes the remaining blocker concrete. It can say
whether the active support has a nearby whole-loan solution, but it cannot
certify global whole-loan optimality because unselected pool and universe loans
are outside the MILP.

### Claim Impact

- Allowed: support-restricted binary integrality probe completed.
- Still prohibited: full-pool/full-universe MILP optimality, Paper Estrella
  replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v79 in the living notebook. Promote only after a global MILP/gap protocol
passes.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v79() -> dict[str, Any]:
    started = datetime.now(UTC)
    summary, allocations, source_summary, blockers = build_integrality_probe()
    write_csv(TABLE_DIR / "paper4_v79_integrality_probe_summary.csv", summary)
    allocations.to_parquet(
        TABLE_DIR / "paper4_v79_integrality_probe_allocations.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v79_integrality_probe_source_summary.csv", source_summary)
    write_csv(TABLE_DIR / "paper4_v79_claim_blockers.csv", blockers)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v79_support_integrality_probe_executed",
                "allowed": True,
                "artifact": "paper4_v79_integrality_probe_summary.csv",
                "boundary": "support-restricted binary MILP only",
            },
            {
                "claim_id": "v79_whole_loan_full_universe_optimality",
                "allowed": False,
                "artifact": "paper4_v79_claim_blockers.csv",
                "boundary": "requires full-pool/full-universe MILP or global gap certificate",
            },
            {
                "claim_id": "v79_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v79_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v79_claim_matrix_delta.csv", claim_matrix)

    lp = summary.loc[summary["portfolio_label_v79"].eq("continuous_lp_reference")].iloc[0]
    milp_row = summary.loc[summary["portfolio_label_v79"].eq("support_binary_milp")].iloc[0]
    status = {
        "phase": "v79_support_integrality_probe",
        "schema_version": "2026-05-15.79",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "summary_rows_v79": int(len(summary)),
        "allocation_rows_v79": int(len(allocations)),
        "source_summary_rows_v79": int(len(source_summary)),
        "claim_blocker_rows_v79": int(len(blockers)),
        "support_rows_v79": int(lp["support_rows_v79"]),
        "lp_fractional_rows_v79": int(lp["fractional_rows_v79"]),
        "milp_solver_success_v79": bool(milp_row["solver_success_v79"]),
        "milp_selected_rows_v79": int(milp_row["selected_rows_v79"]),
        "milp_delta_return_vs_lp_v79": float(milp_row["delta_return_vs_lp_v79"]),
        "milp_delta_cvar90_vs_lp_v79": float(milp_row["delta_cvar90_vs_lp_v79"]),
        "milp_source_cap_violations_v79": int(milp_row["source_cap_violations_v79"]),
        "whole_loan_full_universe_claim_allowed_v79": False,
        "paper1_promotion_allowed_v79": False,
        "paper4_working_champion_changed_v79": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v79 solves a support-restricted binary MILP only; full-pool/full-universe "
            "whole-loan optimality remains unproven"
        ),
    }
    write_json(STATUS_DIR / "paper4_v79_status.json", status)
    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v79": build_v79()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
