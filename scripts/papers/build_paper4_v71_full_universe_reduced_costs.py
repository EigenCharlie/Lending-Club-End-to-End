#!/usr/bin/env python3
"""Build Paper 4 v71 dual-pricing and reduced-cost artifacts.

v71 reruns the v70 restricted-master continuous LPs so HiGHS marginals can be
persisted.  Those marginals are then applied to all v55 comparable-universe
loans omitted from each v69 master, producing a full-universe reduced-cost
screen under the v70 restricted-master dual system.

This is not a full-universe termination certificate.  It is a pricing screen
with two explicit limits: source-share rows come from the v69 restricted master
and the underlying optimization is a continuous relaxation, not a whole-loan
MILP.
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

from scripts.papers import build_paper4_v70_restricted_master_solver as v70  # noqa: E402

PAPER4_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = PAPER4_ROOT / "tables"
STATUS_DIR = PAPER4_ROOT / "status"
NOTE_DIR = PAPER4_ROOT / "notes"
NOTEBOOK = NOTE_DIR / "paper4_living_lab_notebook.md"
FORBIDDEN_FINAL_PROMOTION = STATUS_DIR / "paper4_final_promotion.json"
FAMILIES = ["grade", "score_decile", "income_band", "dti_band", "period", "state_top20"]
REGIMES = ["incumbent_cvar_relaxed_source_lp", "target_source_cap_probe_lp"]


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


def _full_universe_loss_and_return_mean(
    universe: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return path losses and mean path returns for the full comparable universe."""

    factors = v70._scenario_factors()
    path_ids = sorted(factors["path_id"].drop_duplicates().astype(int).tolist())[: v70.N_PATHS]
    losses = np.zeros((len(path_ids), len(universe)), dtype=np.float64)
    return_sum = np.zeros(len(universe), dtype=np.float64)
    fallback = factors.loc[factors["issue_month"].eq("__fallback__")].copy()
    base = universe[
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
        path_return = (
            pd.to_numeric(merged["base_return_vec"], errors="coerce").fillna(0.0)
            - expected_loss
            - prepay_drag
        ).to_numpy(float)
        losses[row_idx, :] = expected_loss.to_numpy(float)
        return_sum += path_return
    return losses, return_sum / max(len(path_ids), 1), path_ids


def _enriched_master_and_universe() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master = read_parquet("paper4_v69_expanded_restricted_master.parquet")
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    concentration = read_csv("paper4_v63_source_repair_concentration.csv")
    if master.empty or universe.empty or concentration.empty:
        return master, universe, concentration
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
    return enriched, universe, concentration


def _solve_policy_with_duals(
    policy_id: str,
    pool: pd.DataFrame,
    source_map: pd.DataFrame,
    regime: str,
) -> dict[str, Any]:
    n = len(pool)
    losses, returns_by_path, path_ids = v70._expected_by_path_matrix(pool)
    if n == 0 or losses.size == 0:
        return {
            "policy_id": policy_id,
            "regime_v71": regime,
            "success_v71": False,
            "message_v71": "missing_pool_or_scenarios",
        }

    amounts = pool["loan_amnt"].to_numpy(float)
    incumbent_mask = pool["master_role_v69"].astype(str).eq("incumbent_v63_book").to_numpy(bool)
    incumbent_x = incumbent_mask.astype(float)
    incumbent_exposure = float(amounts @ incumbent_x)
    incumbent_losses = losses @ incumbent_x
    incumbent_cvar = v70._tail_cvar(incumbent_losses)

    exposure_min_multiplier = 0.995 if regime == "incumbent_cvar_relaxed_source_lp" else 0.950
    cvar_multiplier = 1.0001 if regime == "incumbent_cvar_relaxed_source_lp" else 1.050
    exposure_min = incumbent_exposure * exposure_min_multiplier
    exposure_max = max(v70.TARGET_EXPOSURE, incumbent_exposure)
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
    active_meta.append({"constraint_type_v71": "budget_upper"})

    lower = np.zeros(var_count)
    lower[:n] = -amounts
    a_rows.append(lower)
    b_rows.append(-exposure_min)
    active_meta.append({"constraint_type_v71": "budget_lower"})

    source_rows, source_bounds, source_meta = v70._source_constraint_rows(
        pool, amounts, source_map, regime
    )
    for row, bound, meta in zip(source_rows, source_bounds, source_meta, strict=False):
        full = np.zeros(var_count)
        full[:n] = row
        converted = {
            "constraint_type_v71": meta.get("constraint_type_v70", "source_share"),
            "source_family": meta.get("source_family"),
            "source_id": meta.get("source_id"),
            "cap_share_v71": meta.get("cap_share_v70"),
            "cap_mode_v71": meta.get("cap_mode_v70"),
        }
        a_rows.append(full)
        b_rows.append(bound)
        active_meta.append(converted)

    cvar_row = np.zeros(var_count)
    cvar_row[n] = 1.0
    cvar_row[n + 1 :] = 1.0 / ((1.0 - v70.ALPHA) * s_count)
    a_rows.append(cvar_row)
    b_rows.append(cvar_cap)
    active_meta.append({"constraint_type_v71": "cvar_cap"})

    for s_idx in range(s_count):
        row = np.zeros(var_count)
        row[:n] = losses[s_idx, :]
        row[n] = -1.0
        row[n + 1 + s_idx] = -1.0
        a_rows.append(row)
        b_rows.append(0.0)
        active_meta.append({"constraint_type_v71": "cvar_path_excess", "path_id": path_ids[s_idx]})

    bounds = [(0.0, 1.0)] * n + [(0.0, None)] + [(0.0, None)] * s_count
    a_ub = np.vstack(a_rows)
    b_ub = np.array(b_rows)
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success:
        return {
            "policy_id": policy_id,
            "regime_v71": regime,
            "success_v71": False,
            "message_v71": str(result.message),
            "incumbent_exposure_v71": incumbent_exposure,
            "incumbent_cvar90_v71": incumbent_cvar,
        }

    lhs = a_ub @ result.x
    marginals = np.asarray(result.ineqlin.marginals, dtype=float)
    dual_rows: list[dict[str, Any]] = []
    for idx, (meta, lhs_value, rhs_value, marginal) in enumerate(
        zip(active_meta, lhs, b_rows, marginals, strict=False)
    ):
        slack = float(rhs_value - lhs_value)
        row = {
            "policy_id": policy_id,
            "regime_v71": regime,
            "constraint_index_v71": idx,
            "lhs_v71": float(lhs_value),
            "rhs_v71": float(rhs_value),
            "slack_v71": slack,
            "marginal_v71": float(marginal),
            "binding_v71": abs(slack) <= 1e-5 * max(1.0, abs(float(rhs_value))),
            "claim_boundary_v71": (
                "HiGHS marginal from v70 restricted-master LP; not full-universe certificate"
            ),
        }
        row.update(meta)
        dual_rows.append(row)
    return {
        "policy_id": policy_id,
        "regime_v71": regime,
        "success_v71": True,
        "message_v71": str(result.message),
        "pool": pool,
        "source_map": source_map,
        "path_ids": path_ids,
        "marginals": marginals,
        "active_meta": active_meta,
        "dual_rows": pd.DataFrame(dual_rows),
        "objective_min_v71": float(result.fun),
        "objective_return_v71": float(-result.fun),
        "incumbent_exposure_v71": incumbent_exposure,
        "incumbent_cvar90_v71": incumbent_cvar,
        "exposure_min_v71": exposure_min,
        "exposure_max_v71": exposure_max,
        "cvar_cap_v71": cvar_cap,
    }


def _source_dual_contribution(
    universe: pd.DataFrame,
    amounts: np.ndarray,
    active_meta: list[dict[str, Any]],
    marginals: np.ndarray,
) -> tuple[np.ndarray, int]:
    contribution = np.zeros(len(universe), dtype=np.float64)
    source_rows = 0
    for meta, marginal in zip(active_meta, marginals, strict=False):
        if meta.get("constraint_type_v71") != "source_share":
            continue
        family = str(meta.get("source_family"))
        if family not in universe:
            continue
        source_rows += 1
        source_id = str(meta.get("source_id"))
        cap = float(meta.get("cap_share_v71", 1.0))
        indicator = universe[family].astype(str).eq(source_id).to_numpy(float)
        contribution += amounts * (indicator - cap) * float(marginal)
    return contribution, source_rows


def _price_universe(
    universe: pd.DataFrame,
    losses: np.ndarray,
    mean_returns: np.ndarray,
    solution: dict[str, Any],
) -> pd.DataFrame:
    policy_id = str(solution["policy_id"])
    regime = str(solution["regime_v71"])
    pool = solution["pool"]
    active_meta = solution["active_meta"]
    marginals = solution["marginals"]
    master_ids = set(pool["loan_id"].astype(str))
    amounts = universe["loan_amnt"].to_numpy(float)

    reduced_cost = -mean_returns.copy()
    reduced_cost += amounts * float(marginals[0])
    reduced_cost += -amounts * float(marginals[1])
    source_contrib, source_rows = _source_dual_contribution(
        universe, amounts, active_meta, marginals
    )
    reduced_cost += source_contrib

    path_indices = [
        idx
        for idx, meta in enumerate(active_meta)
        if meta.get("constraint_type_v71") == "cvar_path_excess"
    ]
    path_marginals = marginals[path_indices]
    reduced_cost += losses.T @ path_marginals

    omitted_mask = ~universe["loan_id"].astype(str).isin(master_ids).to_numpy()
    out = universe.loc[
        omitted_mask,
        [
            "loan_index_v55",
            "loan_id",
            "loan_amnt",
            "grade",
            "score_decile",
            "income_band",
            "dti_band",
            "period",
            "state_top20",
        ],
    ].copy()
    omitted_reduced = reduced_cost[omitted_mask]
    out["policy_id"] = policy_id
    out["regime_v71"] = regime
    out["minimization_reduced_cost_v71"] = omitted_reduced.astype("float32")
    out["return_improvement_signal_v71"] = (-omitted_reduced).astype("float32")
    out["improving_column_v71"] = omitted_reduced < -1e-7
    out["source_dual_rows_used_v71"] = source_rows
    out["pricing_scope_v71"] = "v70_restricted_master_duals_applied_to_omitted_v55_columns"
    out["claim_boundary_v71"] = (
        "reduced-cost screen under restricted-master duals; not full-universe termination"
    )
    return out[
        [
            "policy_id",
            "regime_v71",
            "loan_index_v55",
            "loan_id",
            "loan_amnt",
            "grade",
            "score_decile",
            "income_band",
            "dti_band",
            "period",
            "state_top20",
            "minimization_reduced_cost_v71",
            "return_improvement_signal_v71",
            "improving_column_v71",
            "source_dual_rows_used_v71",
            "pricing_scope_v71",
            "claim_boundary_v71",
        ]
    ]


def _source_cap_diagnostics(
    universe: pd.DataFrame, solution: dict[str, Any], dual_rows: pd.DataFrame
) -> pd.DataFrame:
    pool = solution["pool"]
    rows: list[dict[str, Any]] = []
    source_duals = dual_rows.loc[dual_rows["constraint_type_v71"].eq("source_share")].copy()
    for family in FAMILIES:
        if family not in universe or family not in pool:
            continue
        universe_sources = set(universe[family].dropna().astype(str))
        master_sources = set(pool[family].dropna().astype(str))
        missing_sources = sorted(universe_sources - master_sources)
        local = source_duals.loc[source_duals["source_family"].astype(str).eq(family)]
        rows.append(
            {
                "policy_id": solution["policy_id"],
                "regime_v71": solution["regime_v71"],
                "source_family": family,
                "universe_source_ids_v71": len(universe_sources),
                "master_source_ids_v71": len(master_sources),
                "missing_source_ids_v71": len(missing_sources),
                "binding_source_constraints_v71": int(local["binding_v71"].astype(bool).sum())
                if not local.empty
                else 0,
                "nonzero_source_marginals_v71": int(local["marginal_v71"].abs().gt(1e-9).sum())
                if not local.empty
                else 0,
                "max_abs_source_marginal_v71": float(local["marginal_v71"].abs().max())
                if not local.empty
                else 0.0,
                "source_constraint_scope_complete_v71": len(missing_sources) == 0,
                "missing_source_examples_v71": "|".join(missing_sources[:8]),
                "claim_boundary_v71": (
                    "source duals inherit v69 restricted-master source IDs; missing IDs block full certificate"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_v71_pricing() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master, universe, concentration = _enriched_master_and_universe()
    if master.empty or universe.empty or concentration.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    losses, mean_returns, _path_ids = _full_universe_loss_and_return_mean(universe)
    reduced_frames: list[pd.DataFrame] = []
    dual_frames: list[pd.DataFrame] = []
    source_frames: list[pd.DataFrame] = []
    solution_rows: list[dict[str, Any]] = []
    for policy_id in sorted(master["policy_id_v69"].dropna().astype(str).unique()):
        pool = master.loc[master["policy_id_v69"].astype(str).eq(policy_id)].copy()
        source_map = v70._policy_source_map(concentration, policy_id)
        for regime in REGIMES:
            solution = _solve_policy_with_duals(policy_id, pool, source_map, regime)
            solution_rows.append(
                {
                    "policy_id": policy_id,
                    "regime_v71": regime,
                    "solver_success_v71": bool(solution.get("success_v71", False)),
                    "solver_message_v71": solution.get("message_v71", ""),
                    "objective_return_v71": solution.get("objective_return_v71", np.nan),
                    "incumbent_exposure_v71": solution.get("incumbent_exposure_v71", np.nan),
                    "incumbent_cvar90_v71": solution.get("incumbent_cvar90_v71", np.nan),
                    "cvar_cap_v71": solution.get("cvar_cap_v71", np.nan),
                    "claim_boundary_v71": (
                        "restricted-master LP rerun for marginals; not full-universe proof"
                    ),
                }
            )
            if not solution.get("success_v71", False):
                continue
            dual_rows = solution["dual_rows"]
            dual_frames.append(dual_rows)
            source_frames.append(_source_cap_diagnostics(universe, solution, dual_rows))
            reduced_frames.append(_price_universe(universe, losses, mean_returns, solution))

    reduced_costs = (
        pd.concat(reduced_frames, ignore_index=True) if reduced_frames else pd.DataFrame()
    )
    duals = pd.concat(dual_frames, ignore_index=True) if dual_frames else pd.DataFrame()
    source_diag = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame()
    solution_summary = pd.DataFrame(solution_rows)
    return reduced_costs, duals, source_diag, solution_summary


def _summary_from_reduced_costs(
    reduced_costs: pd.DataFrame, solution_summary: pd.DataFrame
) -> pd.DataFrame:
    if reduced_costs.empty:
        return pd.DataFrame()
    grouped = (
        reduced_costs.groupby(["policy_id", "regime_v71"], dropna=False)
        .agg(
            omitted_rows_priced_v71=("loan_id", "count"),
            improving_columns_v71=("improving_column_v71", "sum"),
            min_reduced_cost_v71=("minimization_reduced_cost_v71", "min"),
            best_return_improvement_signal_v71=("return_improvement_signal_v71", "max"),
            median_reduced_cost_v71=("minimization_reduced_cost_v71", "median"),
        )
        .reset_index()
    )
    grouped["negative_reduced_cost_detected_v71"] = grouped["improving_columns_v71"].gt(0)
    grouped["column_generation_termination_certificate_v71"] = False
    grouped["exact_full_universe_cvar_claim_allowed_v71"] = False
    grouped["claim_boundary_v71"] = (
        "improving omitted columns or scope blockers prevent termination claim"
    )
    return grouped.merge(solution_summary, on=["policy_id", "regime_v71"], how="left")


def _claim_blockers(summary: pd.DataFrame, source_diag: pd.DataFrame) -> pd.DataFrame:
    improving = int(summary["improving_columns_v71"].sum()) if not summary.empty else 0
    missing_source = (
        int(source_diag["missing_source_ids_v71"].sum()) if not source_diag.empty else 0
    )
    return pd.DataFrame(
        [
            {
                "blocker_id_v71": "negative_reduced_cost_columns_detected",
                "blocking_v71": improving > 0,
                "evidence_count_v71": improving,
                "required_next_artifact_v71": "paper4_v72_column_generation_iteration_1.csv",
                "claim_boundary_v71": (
                    "negative reduced-cost omitted columns mean the v70 master has not terminated"
                ),
            },
            {
                "blocker_id_v71": "source_constraint_scope_not_full_universe",
                "blocking_v71": missing_source > 0,
                "evidence_count_v71": missing_source,
                "required_next_artifact_v71": "paper4_v72_source_constraint_expansion.csv",
                "claim_boundary_v71": (
                    "restricted-master source rows may omit source IDs present in the full universe"
                ),
            },
            {
                "blocker_id_v71": "continuous_relaxation_not_whole_loan_milp",
                "blocking_v71": True,
                "evidence_count_v71": 1,
                "required_next_artifact_v71": "paper4_v72_integrality_gap_or_milp_probe.csv",
                "claim_boundary_v71": (
                    "dual pricing is for continuous LP variables, not whole-loan integer allocations"
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
                "claim": "Paper 4 has v71 reduced-cost pricing for omitted v55 columns under v70 duals.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v71_full_universe_reduced_costs.parquet"
                ),
                "boundary": (
                    "Reduced-cost screen under restricted-master duals; not termination certificate."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v71 proves full-universe column-generation termination.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v71_claim_blockers.csv"
                ),
                "boundary": (
                    "Blocked by improving omitted columns, source-scope limits or integrality limits."
                ),
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
                    "v71 prices omitted v55 columns with v70 restricted-master duals and "
                    "detects whether the current master has terminated."
                ),
                "status": "ready_for_column_generation_iteration",
                "next_artifact": "paper4_v72_column_generation_iteration_1.csv",
                "success_condition": "new negative reduced-cost columns are added and the LP is re-solved",
                "last_wave": "v71",
                "execution_result": "dual_pricing_screen_completed",
                "quarto_promotion_decision": "living_notebook_only",
            },
            {
                "horizon": "short",
                "lane": "Source governance",
                "executable_item": (
                    "Expand source-share rows beyond v69 master IDs where v71 found missing "
                    "full-universe source IDs."
                ),
                "status": "gated",
                "next_artifact": "paper4_v72_source_constraint_expansion.csv",
                "success_condition": "source constraint set covers all v55 source IDs or documents why not",
                "last_wave": "v71",
                "execution_result": "source_scope_blocker_quantified",
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
    start = "<!-- V71_FULL_UNIVERSE_REDUCED_COSTS_START -->"
    end = "<!-- V71_FULL_UNIVERSE_REDUCED_COSTS_END -->"
    block = f"""
{start}

## Wave v71: Full-Universe Reduced-Cost Screen

Generated: {status["generated_at_utc"]}

### Objective

Persist v70 restricted-master LP duals and apply them to all omitted v55
comparable-universe columns. The goal is to test whether the v70 master has
terminated under its own dual system, not to promote a full-universe claim.

### Results

- Dual rows: `{status["dual_rows_v71"]}`.
- Reduced-cost rows: `{status["reduced_cost_rows_v71"]}`.
- Summary rows: `{status["summary_rows_v71"]}`.
- Improving omitted columns: `{status["improving_omitted_columns_v71"]}`.
- Source-scope diagnostic rows: `{status["source_cap_dual_rows_v71"]}`.
- Full-universe termination claim allowed: `{status["full_universe_termination_claim_allowed_v71"]}`.

### Interpretation

v71 turns v70's active-constraint diagnostics into a concrete pricing screen.
If negative reduced-cost omitted columns exist, the restricted master has not
terminated. Even if pricing improves later, the source-constraint scope and
continuous-relaxation blockers must be resolved before any exact full-universe
or whole-loan claim.

### Claim Impact

- Allowed: v71 prices omitted v55 columns under v70 restricted-master duals.
- Still prohibited: full-universe column-generation termination, exact
  full-universe CVaR optimality, MILP whole-loan optimality, Paper Estrella
  replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v71 in the living notebook. Promote only after column-generation
iterations converge, source-scope coverage is complete and claim review passes.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v71() -> dict[str, Any]:
    started = datetime.now(UTC)
    reduced_costs, duals, source_diag, solution_summary = build_v71_pricing()
    summary = _summary_from_reduced_costs(reduced_costs, solution_summary)
    blockers = _claim_blockers(summary, source_diag)

    reduced_costs.to_parquet(
        TABLE_DIR / "paper4_v71_full_universe_reduced_costs.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v71_restricted_master_duals.csv", duals)
    write_csv(TABLE_DIR / "paper4_v71_reduced_cost_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v71_source_cap_dual_diagnostics.csv", source_diag)
    write_csv(TABLE_DIR / "paper4_v71_claim_blockers.csv", blockers)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v71_omitted_column_reduced_cost_screen",
                "allowed": True,
                "artifact": "paper4_v71_full_universe_reduced_costs.parquet",
                "boundary": "restricted-master dual pricing screen only",
            },
            {
                "claim_id": "v71_full_universe_column_generation_termination",
                "allowed": False,
                "artifact": "paper4_v71_claim_blockers.csv",
                "boundary": "blocked until no improving omitted columns and source/integrality blockers clear",
            },
            {
                "claim_id": "v71_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v71_reduced_cost_summary.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v71_claim_matrix_delta.csv", claim_matrix)

    improving = int(summary["improving_columns_v71"].sum()) if not summary.empty else 0
    status = {
        "phase": "v71_full_universe_reduced_cost_screen",
        "schema_version": "2026-05-15.71",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "dual_rows_v71": int(len(duals)),
        "reduced_cost_rows_v71": int(len(reduced_costs)),
        "summary_rows_v71": int(len(summary)),
        "source_cap_dual_rows_v71": int(len(source_diag)),
        "claim_blocker_rows_v71": int(len(blockers)),
        "improving_omitted_columns_v71": improving,
        "policies_priced_v71": int(summary["policy_id"].nunique()) if not summary.empty else 0,
        "regime_rows_priced_v71": int(len(summary)),
        "negative_reduced_cost_detected_v71": bool(improving > 0),
        "full_universe_termination_claim_allowed_v71": False,
        "exact_full_universe_cvar_claim_allowed_v71": False,
        "paper1_promotion_allowed_v71": False,
        "paper4_working_champion_changed_v71": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v71 is restricted-master dual pricing over omitted v55 columns; full-universe "
            "termination remains blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v71_status.json", status)
    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v71": build_v71()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
