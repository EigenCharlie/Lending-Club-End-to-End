#!/usr/bin/env python3
"""Solve Paper 4 v70 restricted-master LPs.

v70 is the first solver pass over the v69 expanded restricted master.  It uses
continuous allocation variables over the v63 incumbent books plus v68 candidate
columns, with budget, source-share and CVaR constraints.  The result is exact
for this restricted master and this proxy scenario model, but it is not an
exact full-universe column-generation certificate.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAPER4_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = PAPER4_ROOT / "tables"
STATUS_DIR = PAPER4_ROOT / "status"
NOTE_DIR = PAPER4_ROOT / "notes"
NOTEBOOK = NOTE_DIR / "paper4_living_lab_notebook.md"
FORBIDDEN_FINAL_PROMOTION = STATUS_DIR / "paper4_final_promotion.json"
TARGET_EXPOSURE = 850_000.0
ALPHA = 0.90
N_PATHS = 128
FAMILIES = ["grade", "score_decile", "income_band", "dti_band", "period", "state_top20"]


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


def _tail_cvar(values: np.ndarray, alpha: float = ALPHA) -> float:
    if values.size == 0:
        return 0.0
    tail_n = max(1, int(np.ceil((1.0 - alpha) * values.size)))
    return float(np.sort(values)[-tail_n:].mean())


def _scenario_factors(n_paths: int = N_PATHS) -> pd.DataFrame:
    paths = read_parquet("paper4_v31_sample_paths.parquet")
    if paths.empty:
        return pd.DataFrame()
    path_ids = sorted(paths["path_id"].drop_duplicates().astype(int).tolist())[:n_paths]
    p = paths.loc[paths["path_id"].isin(path_ids)].copy()
    p["issue_month"] = pd.to_datetime(p["month"], errors="coerce").dt.to_period("M").astype(str)
    keep = [
        "path_id",
        "issue_month",
        "macro_regime_v15",
        "path_family_v19",
        "default_factor_v15",
        "lgd_factor_v15",
        "prepay_factor_v15",
    ]
    p = p[keep].drop_duplicates(["path_id", "issue_month"])
    fallback = (
        p.sort_values("issue_month")
        .groupby("path_id", dropna=False)
        .head(1)
        .assign(issue_month="__fallback__")
    )
    return pd.concat([p, fallback], ignore_index=True)


def _expected_by_path_matrix(pool: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int]]:
    factors = _scenario_factors()
    path_ids = sorted(factors["path_id"].drop_duplicates().astype(int).tolist())[:N_PATHS]
    losses = np.zeros((len(path_ids), len(pool)), dtype=np.float64)
    returns = np.zeros((len(path_ids), len(pool)), dtype=np.float64)
    fallback = factors.loc[factors["issue_month"].eq("__fallback__")].copy()
    base = pool[
        [
            "loan_id",
            "issue_month",
            "loan_amnt",
            "pd_high_alpha01",
            "lgd_proxy_v55",
            "base_return_vec",
        ]
    ].copy()
    for row_idx, path_id in enumerate(path_ids):
        f = factors.loc[factors["path_id"].eq(path_id)].drop(columns=["path_id"])
        merged = base.merge(f, on="issue_month", how="left")
        missing = merged["default_factor_v15"].isna()
        if missing.any():
            fb = fallback.loc[fallback["path_id"].eq(path_id)].head(1)
            for col in [
                "macro_regime_v15",
                "path_family_v19",
                "default_factor_v15",
                "lgd_factor_v15",
                "prepay_factor_v15",
            ]:
                merged.loc[missing, col] = fb[col].iloc[0] if not fb.empty else 1.0
        dp = (
            pd.to_numeric(merged["pd_high_alpha01"], errors="coerce").fillna(0.0)
            * pd.to_numeric(merged["default_factor_v15"], errors="coerce").fillna(1.0)
        ).clip(0, 0.95)
        lgd_factor = (
            pd.to_numeric(merged["lgd_factor_v15"], errors="coerce").fillna(1.0).clip(0.25, 2.5)
        )
        prepay_factor = (
            pd.to_numeric(merged["prepay_factor_v15"], errors="coerce").fillna(1.0).clip(0.25, 2.5)
        )
        expected_loss = (
            pd.to_numeric(merged["loan_amnt"], errors="coerce").fillna(0.0)
            * pd.to_numeric(merged["lgd_proxy_v55"], errors="coerce").fillna(0.45)
            * dp
            * lgd_factor
        )
        prepay_drag = (
            pd.to_numeric(merged["loan_amnt"], errors="coerce").fillna(0.0)
            * 0.012
            * (1 - dp)
            * prepay_factor
        )
        losses[row_idx, :] = expected_loss.to_numpy(float)
        returns[row_idx, :] = (
            pd.to_numeric(merged["base_return_vec"], errors="coerce").fillna(0.0)
            - expected_loss
            - prepay_drag
        ).to_numpy(float)
    return losses, returns, path_ids


def _policy_source_map(concentration: pd.DataFrame, policy_id: str) -> pd.DataFrame:
    return concentration.loc[concentration["policy_id"].astype(str).eq(policy_id)].copy()


def _cap_share(source_map: pd.DataFrame, family: str, regime: str) -> float:
    row = source_map.loc[source_map["source_family"].astype(str).eq(family)]
    if row.empty:
        return 1.0
    target = float(row["target_cap_v63"].iloc[0])
    incumbent_top = float(row["top_exposure_share_v63"].iloc[0])
    if regime == "target_source_cap_probe_lp":
        return target
    return max(target, incumbent_top)


def _source_constraint_rows(
    pool: pd.DataFrame, amounts: np.ndarray, source_map: pd.DataFrame, regime: str
) -> tuple[list[np.ndarray], list[float], list[dict[str, Any]]]:
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    meta: list[dict[str, Any]] = []
    for family in FAMILIES:
        if family not in pool:
            continue
        cap = _cap_share(source_map, family, regime)
        for source_id in sorted(pool[family].dropna().astype(str).unique()):
            indicator = pool[family].astype(str).eq(source_id).to_numpy(float)
            rows.append(amounts * (indicator - cap))
            bounds.append(0.0)
            meta.append(
                {
                    "constraint_type_v70": "source_share",
                    "source_family": family,
                    "source_id": source_id,
                    "cap_share_v70": cap,
                    "cap_mode_v70": regime,
                }
            )
    return rows, bounds, meta


def _allocated_concentration(
    pool: pd.DataFrame, x: np.ndarray, source_map: pd.DataFrame
) -> tuple[pd.DataFrame, float, float]:
    allocated = pool.assign(allocated_exposure_v70=pool["loan_amnt"].to_numpy(float) * x)
    exposure = float(allocated["allocated_exposure_v70"].sum())
    rows: list[dict[str, Any]] = []
    max_share = 0.0
    max_target_slack = 0.0
    for family in FAMILIES:
        if family not in allocated:
            continue
        target_cap = _cap_share(source_map, family, "target_source_cap_probe_lp")
        by_source = allocated.groupby(family, dropna=False)["allocated_exposure_v70"].sum()
        if by_source.empty:
            continue
        top_source = str(by_source.idxmax())
        top_share = float(by_source.max() / max(exposure, 1.0))
        target_slack = max(0.0, top_share - target_cap)
        max_share = max(max_share, top_share)
        max_target_slack = max(max_target_slack, target_slack)
        rows.append(
            {
                "source_family": family,
                "top_source_id_v70": top_source,
                "top_exposure_share_v70": top_share,
                "target_cap_v70": target_cap,
                "target_cap_slack_v70": target_slack,
            }
        )
    return pd.DataFrame(rows), max_share, max_target_slack


def _solve_policy_regime(
    policy_id: str,
    pool: pd.DataFrame,
    source_map: pd.DataFrame,
    regime: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(pool)
    losses, returns_by_path, path_ids = _expected_by_path_matrix(pool)
    if n == 0 or losses.size == 0:
        return (
            {
                "policy_id": policy_id,
                "regime_v70": regime,
                "solver_success_v70": False,
                "solver_status_v70": "missing_pool_or_scenarios",
                "claim_boundary_v70": "no solver claim",
            },
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    amounts = pool["loan_amnt"].to_numpy(float)
    incumbent_mask = pool["master_role_v69"].astype(str).eq("incumbent_v63_book").to_numpy(bool)
    incumbent_x = incumbent_mask.astype(float)
    incumbent_exposure = float(amounts @ incumbent_x)
    incumbent_losses = losses @ incumbent_x
    incumbent_returns = returns_by_path @ incumbent_x
    incumbent_cvar = _tail_cvar(incumbent_losses)
    incumbent_return = float(incumbent_returns.mean())

    exposure_min_multiplier = 0.995 if regime == "incumbent_cvar_relaxed_source_lp" else 0.950
    cvar_multiplier = 1.0001 if regime == "incumbent_cvar_relaxed_source_lp" else 1.050
    exposure_min = incumbent_exposure * exposure_min_multiplier
    exposure_max = max(TARGET_EXPOSURE, incumbent_exposure)
    cvar_cap = incumbent_cvar * cvar_multiplier

    s_count = len(path_ids)
    var_count = n + 1 + s_count
    c = np.zeros(var_count)
    c[:n] = -returns_by_path.mean(axis=0)

    a_rows: list[np.ndarray] = []
    b_rows: list[float] = []
    active_meta: list[dict[str, Any]] = []

    upper = np.zeros(var_count)
    upper[:n] = amounts
    a_rows.append(upper)
    b_rows.append(exposure_max)
    active_meta.append({"constraint_type_v70": "budget_upper"})

    lower = np.zeros(var_count)
    lower[:n] = -amounts
    a_rows.append(lower)
    b_rows.append(-exposure_min)
    active_meta.append({"constraint_type_v70": "budget_lower"})

    source_rows, source_bounds, source_meta = _source_constraint_rows(
        pool, amounts, source_map, regime
    )
    for row, bound, meta in zip(source_rows, source_bounds, source_meta, strict=False):
        full = np.zeros(var_count)
        full[:n] = row
        a_rows.append(full)
        b_rows.append(bound)
        active_meta.append(meta)

    cvar_row = np.zeros(var_count)
    cvar_row[n] = 1.0
    cvar_row[n + 1 :] = 1.0 / ((1.0 - ALPHA) * s_count)
    a_rows.append(cvar_row)
    b_rows.append(cvar_cap)
    active_meta.append({"constraint_type_v70": "cvar_cap"})

    for s_idx in range(s_count):
        row = np.zeros(var_count)
        row[:n] = losses[s_idx, :]
        row[n] = -1.0
        row[n + 1 + s_idx] = -1.0
        a_rows.append(row)
        b_rows.append(0.0)
        active_meta.append({"constraint_type_v70": "cvar_path_excess", "path_id": path_ids[s_idx]})

    bounds = [(0.0, 1.0)] * n + [(0.0, None)] + [(0.0, None)] * s_count
    result = linprog(
        c,
        A_ub=np.vstack(a_rows),
        b_ub=np.array(b_rows),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return (
            {
                "policy_id": policy_id,
                "regime_v70": regime,
                "solver_success_v70": False,
                "solver_status_v70": str(result.message),
                "incumbent_exposure_v70": incumbent_exposure,
                "incumbent_return_v70": incumbent_return,
                "incumbent_cvar90_v70": incumbent_cvar,
                "exposure_min_v70": exposure_min,
                "cvar_cap_v70": cvar_cap,
                "exact_restricted_master_lp_v70": False,
                "exact_full_universe_cvar_claim_allowed_v70": False,
                "claim_boundary_v70": "restricted-master LP failed; no optimizer or full-universe claim",
            },
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    x = np.clip(result.x[:n], 0.0, 1.0)
    scenario_losses = losses @ x
    scenario_returns = returns_by_path @ x
    exposure = float(amounts @ x)
    concentration, max_source_share, max_target_slack = _allocated_concentration(
        pool, x, source_map
    )
    positive = pool.loc[x > 1e-7].copy()
    positive["allocation_fraction_v70"] = x[x > 1e-7]
    positive["allocated_exposure_v70"] = (
        positive["loan_amnt"].astype(float) * positive["allocation_fraction_v70"]
    )
    positive["policy_id"] = policy_id
    positive["regime_v70"] = regime
    positive["claim_boundary_v70"] = (
        "exact continuous LP over v69 restricted master; not full-universe column generation"
    )
    alloc_keep = [
        "policy_id",
        "regime_v70",
        "loan_id",
        "master_role_v69",
        "candidate_rank_v68",
        "loan_amnt",
        "allocation_fraction_v70",
        "allocated_exposure_v70",
        "pricing_screen_score_v69",
        "grade",
        "score_decile",
        "income_band",
        "dti_band",
        "period",
        "state_top20",
        "claim_boundary_v70",
    ]
    allocations = positive[[col for col in alloc_keep if col in positive.columns]]

    scenario = pd.DataFrame(
        {
            "policy_id": policy_id,
            "regime_v70": regime,
            "path_id": path_ids,
            "scenario_loss_v70": scenario_losses,
            "scenario_return_v70": scenario_returns,
            "claim_boundary_v70": "restricted-master scenario evaluation only",
        }
    )
    lhs = np.vstack(a_rows) @ result.x
    active_rows: list[dict[str, Any]] = []
    for meta, lhs_value, rhs_value in zip(active_meta, lhs, b_rows, strict=False):
        slack = float(rhs_value - lhs_value)
        row = {
            "policy_id": policy_id,
            "regime_v70": regime,
            "lhs_v70": float(lhs_value),
            "rhs_v70": float(rhs_value),
            "slack_v70": slack,
            "binding_v70": abs(slack) <= 1e-5 * max(1.0, abs(float(rhs_value))),
            "claim_boundary_v70": "restricted-master LP diagnostic only",
        }
        row.update(meta)
        active_rows.append(row)
    active = pd.DataFrame(active_rows)

    candidate_exposure = float(
        allocations.loc[
            allocations["master_role_v69"].astype(str).eq("v68_pricing_candidate"),
            "allocated_exposure_v70",
        ].sum()
    )
    fractional_rows = int(
        (
            (allocations["allocation_fraction_v70"] > 1e-7)
            & (allocations["allocation_fraction_v70"] < 0.999999)
        ).sum()
    )
    status = {
        "policy_id": policy_id,
        "regime_v70": regime,
        "solver_success_v70": True,
        "solver_status_v70": str(result.message),
        "objective_return_v70": float(scenario_returns.mean()),
        "incumbent_return_v70": incumbent_return,
        "delta_return_vs_incumbent_v70": float(scenario_returns.mean() - incumbent_return),
        "portfolio_exposure_v70": exposure,
        "incumbent_exposure_v70": incumbent_exposure,
        "scenario_loss_mean_v70": float(scenario_losses.mean()),
        "scenario_loss_p95_v70": float(np.quantile(scenario_losses, 0.95)),
        "scenario_loss_cvar90_v70": _tail_cvar(scenario_losses),
        "incumbent_cvar90_v70": incumbent_cvar,
        "cvar_cap_v70": cvar_cap,
        "max_source_share_v70": max_source_share,
        "max_source_slack_to_target_v70": max_target_slack,
        "candidate_allocated_exposure_v70": candidate_exposure,
        "candidate_allocation_share_v70": candidate_exposure / max(exposure, 1.0),
        "allocated_rows_v70": int(len(allocations)),
        "fractional_allocation_rows_v70": fractional_rows,
        "exact_restricted_master_lp_v70": True,
        "exact_full_universe_cvar_claim_allowed_v70": False,
        "claim_boundary_v70": (
            "exact continuous LP over v69 restricted master; no omitted-column pricing "
            "or full-universe termination certificate"
        ),
    }
    return status, allocations, scenario, active


def build_v70_solver() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master = read_parquet("paper4_v69_expanded_restricted_master.parquet")
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    concentration = read_csv("paper4_v63_source_repair_concentration.csv")
    if master.empty or universe.empty or concentration.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    universe_cols = [
        "loan_id",
        "issue_month",
        "pd_high_alpha01",
        "lgd_proxy_v55",
        "base_return_vec",
        "qhat_v4",
        "weak_source_proxy",
    ]
    enriched = master.merge(
        universe[universe_cols].assign(loan_id=lambda df: df["loan_id"].astype(str)),
        on="loan_id",
        how="left",
    )
    frontier_rows: list[dict[str, Any]] = []
    allocations: list[pd.DataFrame] = []
    scenarios: list[pd.DataFrame] = []
    active_constraints: list[pd.DataFrame] = []
    regimes = ["incumbent_cvar_relaxed_source_lp", "target_source_cap_probe_lp"]
    for policy_id in sorted(enriched["policy_id_v69"].dropna().astype(str).unique()):
        pool = enriched.loc[enriched["policy_id_v69"].astype(str).eq(policy_id)].copy()
        source_map = _policy_source_map(concentration, policy_id)
        for regime in regimes:
            row, alloc, scenario, active = _solve_policy_regime(policy_id, pool, source_map, regime)
            frontier_rows.append(row)
            if not alloc.empty:
                allocations.append(alloc)
            if not scenario.empty:
                scenarios.append(scenario)
            if not active.empty:
                active_constraints.append(active)

    frontier = pd.DataFrame(frontier_rows)
    allocation_out = pd.concat(allocations, ignore_index=True) if allocations else pd.DataFrame()
    scenario_out = pd.concat(scenarios, ignore_index=True) if scenarios else pd.DataFrame()
    active_out = (
        pd.concat(active_constraints, ignore_index=True) if active_constraints else pd.DataFrame()
    )
    return frontier, allocation_out, scenario_out, active_out


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has exact continuous LP solves over the v69 restricted master.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v70_restricted_master_solver_frontier.csv"
                ),
                "boundary": "Restricted-master continuous LP only; not MILP and not full-universe pricing.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v70 proves exact full-universe CVaR optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v70_solver_active_constraints.csv"
                ),
                "boundary": "Requires omitted-column reduced costs and column-generation termination.",
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
                    "v70 solves continuous LPs over the v69 restricted master and records "
                    "active constraints."
                ),
                "status": "near_resolved_with_plateau",
                "next_artifact": "paper4_v71_full_universe_reduced_costs.parquet",
                "success_condition": "all omitted v55 columns receive exact reduced-cost pricing",
                "last_wave": "v70",
                "execution_result": "restricted_master_lp_completed",
                "quarto_promotion_decision": "living_notebook_only",
            },
            {
                "horizon": "short",
                "lane": "Source governance",
                "executable_item": (
                    "Use v70 active constraints to decide whether strict source caps need "
                    "MILP, relaxed CVaR or new candidate columns."
                ),
                "status": "gated",
                "next_artifact": "paper4_v71_source_cap_dual_diagnostics.csv",
                "success_condition": "source cap blockers are tied to duals or infeasibility evidence",
                "last_wave": "v70",
                "execution_result": "source_constraint_activity_recorded",
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
    start = "<!-- V70_RESTRICTED_MASTER_SOLVER_START -->"
    end = "<!-- V70_RESTRICTED_MASTER_SOLVER_END -->"
    block = f"""
{start}

## Wave v70: Restricted-Master Continuous LP Solver

Generated: {status["generated_at_utc"]}

### Objective

Use the v69 expanded restricted master in an exact continuous LP with budget,
source-share and CVaR constraints. This tests whether the candidate columns can
improve the restricted master before spending effort on full-universe reduced
cost pricing.

### Results

- Frontier rows: `{status["frontier_rows_v70"]}`.
- Successful LP rows: `{status["successful_lp_rows_v70"]}`.
- Allocation rows: `{status["allocation_rows_v70"]}`.
- Scenario rows: `{status["scenario_rows_v70"]}`.
- Active constraint rows: `{status["active_constraint_rows_v70"]}`.
- Best return delta vs incumbent: `{status["best_delta_return_vs_incumbent_v70"]}`.
- Exact full-universe CVaR claim allowed: `{status["exact_full_universe_cvar_claim_allowed_v70"]}`.

### Interpretation

v70 is the first real optimization over the v68-v69 candidate queue. It is
valuable even when it remains bounded: it produces restricted-master primal
allocations and active-constraint evidence, while making the missing next step
explicit: full-universe reduced-cost pricing over omitted columns.

### Claim Impact

- Allowed: exact continuous LP solves over the v69 restricted master.
- Still prohibited: exact full-universe optimality, MILP whole-loan optimality,
  Paper Estrella replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v70 in the living notebook. Promote only after omitted-column pricing and
claim-boundary review are complete.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v70() -> dict[str, Any]:
    started = datetime.now(UTC)
    frontier, allocations, scenarios, active = build_v70_solver()
    write_csv(TABLE_DIR / "paper4_v70_restricted_master_solver_frontier.csv", frontier)
    allocations.to_parquet(
        TABLE_DIR / "paper4_v70_restricted_master_allocations.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v70_restricted_master_scenario_losses.csv", scenarios)
    write_csv(TABLE_DIR / "paper4_v70_solver_active_constraints.csv", active)
    blocker = pd.DataFrame(
        [
            {
                "blocker_id_v70": "omitted_column_reduced_costs_missing",
                "status_v70": "blocking_full_universe_claim",
                "required_next_artifact_v70": "paper4_v71_full_universe_reduced_costs.parquet",
                "claim_boundary_v70": (
                    "restricted-master LP has no full-universe termination certificate"
                ),
            },
            {
                "blocker_id_v70": "continuous_relaxation_not_whole_loan_milp",
                "status_v70": "blocking_whole_loan_optimality_claim",
                "required_next_artifact_v70": "paper4_v71_integrality_gap_or_milp_probe.csv",
                "claim_boundary_v70": (
                    "fractional LP allocations cannot be described as whole-loan optimality"
                ),
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v70_solver_claim_blockers.csv", blocker)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v70_exact_restricted_master_continuous_lp",
                "allowed": bool(
                    not frontier.empty
                    and frontier.get("solver_success_v70", pd.Series(False)).astype(bool).any()
                ),
                "artifact": "paper4_v70_restricted_master_solver_frontier.csv",
                "boundary": "restricted-master continuous LP only",
            },
            {
                "claim_id": "v70_exact_full_universe_cvar_optimality",
                "allowed": False,
                "artifact": "paper4_v70_solver_claim_blockers.csv",
                "boundary": "requires omitted-column reduced-cost termination",
            },
            {
                "claim_id": "v70_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v70_solver_active_constraints.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v70_claim_matrix_delta.csv", claim_matrix)

    successful = (
        frontier.loc[frontier["solver_success_v70"].astype(bool)].copy()
        if not frontier.empty and "solver_success_v70" in frontier
        else pd.DataFrame()
    )
    status = {
        "phase": "v70_restricted_master_continuous_lp_solver",
        "schema_version": "2026-05-15.70",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "frontier_rows_v70": int(len(frontier)),
        "successful_lp_rows_v70": int(len(successful)),
        "allocation_rows_v70": int(len(allocations)),
        "scenario_rows_v70": int(len(scenarios)),
        "active_constraint_rows_v70": int(len(active)),
        "best_delta_return_vs_incumbent_v70": float(
            successful["delta_return_vs_incumbent_v70"].max()
        )
        if not successful.empty
        else 0.0,
        "policies_with_successful_lp_v70": int(successful["policy_id"].nunique())
        if not successful.empty
        else 0,
        "exact_restricted_master_lp_claim_allowed_v70": bool(not successful.empty),
        "exact_full_universe_cvar_claim_allowed_v70": False,
        "paper1_promotion_allowed_v70": False,
        "paper4_working_champion_changed_v70": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v70 solves restricted-master continuous LPs only; omitted-column pricing "
            "and integrality evidence remain missing"
        ),
    }
    write_json(STATUS_DIR / "paper4_v70_status.json", status)
    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v70": build_v70()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
