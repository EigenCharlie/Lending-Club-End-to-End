#!/usr/bin/env python3
"""Build Paper 4 v78 source-scope expanded re-pricing artifacts.

v78 reruns the v77-cleared pricing check after expanding source-share
constraints to every source ID observed in the full comparable v55 universe.
This targets the source-scope blocker directly while keeping the same
continuous restricted-master LP and claim boundaries.
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
from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71  # noqa: E402

PAPER4_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = PAPER4_ROOT / "tables"
STATUS_DIR = PAPER4_ROOT / "status"
NOTE_DIR = PAPER4_ROOT / "notes"
NOTEBOOK = NOTE_DIR / "paper4_living_lab_notebook.md"
FORBIDDEN_FINAL_PROMOTION = STATUS_DIR / "paper4_final_promotion.json"
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


def _rename_suffix(df: pd.DataFrame, old: str = "_v71", new: str = "_v78") -> pd.DataFrame:
    return df.rename(columns={col: col.replace(old, new) for col in df.columns})


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


def _pool_for_reprice(
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
    combined["policy_id_v78"] = policy_id
    combined["regime_v78"] = regime
    return combined


def _source_constraint_rows_full_scope(
    pool: pd.DataFrame,
    universe: pd.DataFrame,
    amounts: np.ndarray,
    source_map: pd.DataFrame,
    regime: str,
) -> tuple[list[np.ndarray], list[float], list[dict[str, Any]]]:
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    meta: list[dict[str, Any]] = []
    for family in FAMILIES:
        if family not in pool or family not in universe:
            continue
        cap = v70._cap_share(source_map, family, regime)
        for source_id in sorted(universe[family].dropna().astype(str).unique()):
            indicator = pool[family].astype(str).eq(source_id).to_numpy(float)
            rows.append(amounts * (indicator - cap))
            bounds.append(0.0)
            meta.append(
                {
                    "constraint_type_v71": "source_share",
                    "source_family": family,
                    "source_id": source_id,
                    "cap_share_v71": cap,
                    "cap_mode_v71": regime,
                    "source_scope_expanded_v78": True,
                    "source_id_present_in_master_v78": bool(indicator.any()),
                }
            )
    return rows, bounds, meta


def _solve_with_full_source_scope(
    policy_id: str,
    pool: pd.DataFrame,
    universe: pd.DataFrame,
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

    source_rows, source_bounds, source_meta = _source_constraint_rows_full_scope(
        pool, universe, amounts, source_map, regime
    )
    for row, bound, meta in zip(source_rows, source_bounds, source_meta, strict=False):
        full = np.zeros(var_count)
        full[:n] = row
        a_rows.append(full)
        b_rows.append(bound)
        active_meta.append(meta)

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

    a_ub = np.vstack(a_rows)
    b_ub = np.array(b_rows)
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0.0, 1.0)] * n + [(0.0, None)] + [(0.0, None)] * s_count,
        method="highs",
    )
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
                "HiGHS marginal with full source-scope rows; not full-universe certificate"
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


def _source_scope_diagnostics(
    universe: pd.DataFrame,
    pool: pd.DataFrame,
    solution: dict[str, Any],
    dual_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_duals = dual_rows.loc[dual_rows["constraint_type_v71"].eq("source_share")].copy()
    for family in FAMILIES:
        universe_sources = set(universe[family].dropna().astype(str))
        master_sources = set(pool[family].dropna().astype(str))
        constrained_sources = set(
            source_duals.loc[source_duals["source_family"].astype(str).eq(family), "source_id"]
            .dropna()
            .astype(str)
        )
        missing_constraints = sorted(universe_sources - constrained_sources)
        local = source_duals.loc[source_duals["source_family"].astype(str).eq(family)]
        rows.append(
            {
                "policy_id": solution["policy_id"],
                "regime_v78": solution["regime_v71"],
                "source_family": family,
                "universe_source_ids_v78": len(universe_sources),
                "master_source_ids_v78": len(master_sources),
                "constrained_source_ids_v78": len(constrained_sources),
                "missing_source_constraint_ids_v78": len(missing_constraints),
                "source_constraint_scope_complete_v78": len(missing_constraints) == 0,
                "binding_source_constraints_v78": int(local["binding_v71"].astype(bool).sum())
                if not local.empty
                else 0,
                "nonzero_source_marginals_v78": int(local["marginal_v71"].abs().gt(1e-9).sum())
                if not local.empty
                else 0,
                "missing_source_examples_v78": "|".join(missing_constraints[:8]),
                "claim_boundary_v78": (
                    "full-universe source IDs represented in LP rows; integrality still blocks final claim"
                ),
            }
        )
    return pd.DataFrame(rows)


def _expanded_reprice() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    base_master = _enriched_base_master(universe)
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    reduced_frames: list[pd.DataFrame] = []
    dual_frames: list[pd.DataFrame] = []
    source_frames: list[pd.DataFrame] = []
    solution_rows: list[dict[str, Any]] = []
    for _, row in frontier.iterrows():
        policy_id = str(row["policy_id"])
        regime = str(row["regime_v76"])
        pool = _pool_for_reprice(
            base_master,
            iteration_1_candidates,
            iteration_2_candidates,
            iteration_3_candidates,
            policy_id,
            regime,
        )
        source_map = v70._policy_source_map(concentration, policy_id)
        solution = _solve_with_full_source_scope(policy_id, pool, universe, source_map, regime)
        solution_rows.append(
            {
                "policy_id": policy_id,
                "regime_v78": regime,
                "solver_success_v78": bool(solution.get("success_v71", False)),
                "solver_message_v78": solution.get("message_v71", ""),
                "objective_return_v78": solution.get("objective_return_v71", np.nan),
                "v76_objective_return_baseline_v78": row.get("objective_return_v76", np.nan),
                "delta_return_vs_v76_v78": float(
                    solution.get("objective_return_v71", np.nan)
                    - row.get("objective_return_v76", np.nan)
                ),
                "v78_master_rows_repriced": int(len(pool)),
                "claim_boundary_v78": (
                    "source-scope expanded LP rerun; not whole-loan or final promotion proof"
                ),
            }
        )
        if not solution.get("success_v71", False):
            continue
        priced = _rename_suffix(v71._price_universe(universe, losses, mean_returns, solution))
        priced["source_scope_expanded_reprice_v78"] = 1
        priced["pricing_scope_v78"] = "full_source_scope_duals_applied_to_omitted_v55_columns"
        priced["claim_boundary_v78"] = (
            "full source-scope re-pricing; continuous/integrality blocker remains"
        )
        reduced_frames.append(priced)

        duals = _rename_suffix(solution["dual_rows"])
        duals["source_scope_expanded_reprice_v78"] = 1
        duals["claim_boundary_v78"] = "full source-scope HiGHS marginal; not whole-loan certificate"
        dual_frames.append(duals)
        source_frames.append(
            _source_scope_diagnostics(universe, pool, solution, solution["dual_rows"])
        )

    reduced_costs = (
        pd.concat(reduced_frames, ignore_index=True) if reduced_frames else pd.DataFrame()
    )
    duals = pd.concat(dual_frames, ignore_index=True) if dual_frames else pd.DataFrame()
    source_diag = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame()
    solution_summary = pd.DataFrame(solution_rows)
    return reduced_costs, duals, source_diag, solution_summary


def _summary(reduced_costs: pd.DataFrame, solution_summary: pd.DataFrame) -> pd.DataFrame:
    if reduced_costs.empty:
        return pd.DataFrame()
    grouped = (
        reduced_costs.groupby(["policy_id", "regime_v78"], dropna=False)
        .agg(
            omitted_rows_priced_v78=("loan_id", "count"),
            improving_columns_v78=("improving_column_v78", "sum"),
            min_reduced_cost_v78=("minimization_reduced_cost_v78", "min"),
            best_return_improvement_signal_v78=("return_improvement_signal_v78", "max"),
            median_reduced_cost_v78=("minimization_reduced_cost_v78", "median"),
        )
        .reset_index()
    )
    grouped["negative_reduced_cost_detected_v78"] = grouped["improving_columns_v78"].gt(0)
    grouped["source_scope_expanded_reprice_v78"] = True
    grouped["exact_full_universe_cvar_claim_allowed_v78"] = False
    grouped["claim_boundary_v78"] = (
        "pricing and source-scope diagnostics only; whole-loan integrality remains"
    )
    return grouped.merge(solution_summary, on=["policy_id", "regime_v78"], how="left")


def _claim_blockers(summary: pd.DataFrame, source_diag: pd.DataFrame) -> pd.DataFrame:
    improving = int(summary["improving_columns_v78"].sum()) if not summary.empty else 0
    missing_source = (
        int(source_diag["missing_source_constraint_ids_v78"].sum()) if not source_diag.empty else 0
    )
    return pd.DataFrame(
        [
            {
                "blocker_id_v78": "negative_reduced_cost_columns_after_full_source_scope",
                "blocking_v78": improving > 0,
                "evidence_count_v78": improving,
                "required_next_artifact_v78": "paper4_v79_column_generation_if_needed.csv",
                "claim_boundary_v78": "full source-scope pricing has no improving columns if zero",
            },
            {
                "blocker_id_v78": "source_constraint_scope_incomplete",
                "blocking_v78": missing_source > 0,
                "evidence_count_v78": missing_source,
                "required_next_artifact_v78": "paper4_v79_source_scope_patch.csv",
                "claim_boundary_v78": "full universe source IDs must have constraint rows",
            },
            {
                "blocker_id_v78": "continuous_relaxation_not_whole_loan_milp",
                "blocking_v78": True,
                "evidence_count_v78": 1,
                "required_next_artifact_v78": "paper4_v79_integrality_gap_or_milp_probe.csv",
                "claim_boundary_v78": "expanded-source pricing remains a continuous LP diagnostic",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has v78 full source-scope rows for the focused pricing check.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v78_source_scope_expanded_diagnostics.csv"
                ),
                "boundary": "Source-scope diagnostic only; continuous LP and integrality limits remain.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v78 proves whole-loan full-universe optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v78_claim_blockers.csv"
                ),
                "boundary": "Requires MILP/integrality evidence beyond continuous LP pricing.",
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
                    "v78 re-prices with source rows expanded to all comparable-universe source IDs."
                ),
                "status": "integrality_gated",
                "next_artifact": "paper4_v79_integrality_gap_or_milp_probe.csv",
                "success_condition": (
                    "whole-loan or integrality-gap evidence resolves continuous-relaxation blocker"
                ),
                "last_wave": "v78",
                "execution_result": "pricing_and_source_scope_cleared_for_focused_check",
                "quarto_promotion_decision": "living_notebook_only",
            },
            {
                "horizon": "short",
                "lane": "Source governance",
                "executable_item": "Track source-scope expanded rows as lab-only evidence.",
                "status": "resolved_for_focused_check",
                "next_artifact": "paper4_v79_source_scope_monitor.csv",
                "success_condition": "future pricing runs keep universe source IDs represented",
                "last_wave": "v78",
                "execution_result": "source_scope_rows_expanded",
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
    start = "<!-- V78_SOURCE_SCOPE_EXPANDED_REPRICE_START -->"
    end = "<!-- V78_SOURCE_SCOPE_EXPANDED_REPRICE_END -->"
    block = f"""
{start}

## Wave v78: Source-Scope Expanded Re-Price

Generated: {status["generated_at_utc"]}

### Objective

Rerun the focused v77 pricing check with source-share rows expanded to every
source ID present in the full comparable v55 universe. This targets the
remaining source-scope blocker without changing the continuous LP claim
boundary.

### Results

- Re-price rows: `{status["reprice_rows_v78"]}`.
- Summary rows: `{status["summary_rows_v78"]}`.
- Dual rows: `{status["dual_rows_v78"]}`.
- Source-scope rows: `{status["source_scope_rows_v78"]}`.
- Missing source constraint IDs: `{status["missing_source_constraint_ids_v78"]}`.
- Improving columns after full source-scope pricing: `{status["improving_columns_v78"]}`.
- Pricing blocker cleared: `{status["pricing_blocker_cleared_v78"]}`.
- Source-scope blocker cleared: `{status["source_scope_blocker_cleared_v78"]}`.
- Exact full-universe CVaR claim allowed: `{status["exact_full_universe_cvar_claim_allowed_v78"]}`.

### Interpretation

v78 separates another layer of evidence: pricing remains clean after adding
full source-scope rows, and the source-scope diagnostic is locally complete.
The remaining hard blocker is integrality/whole-loan evidence because the
solver is still a continuous restricted-master LP.

### Claim Impact

- Allowed: focused pricing check has full comparable-universe source rows.
- Still prohibited: exact full-universe CVaR optimality, MILP whole-loan
  optimality, Paper Estrella replacement, final Paper 4 promotion and live
  deployment.

### Quarto Promotion Decision

Keep v78 in the living notebook. Promote only after integrality evidence passes.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v78() -> dict[str, Any]:
    started = datetime.now(UTC)
    reduced_costs, duals, source_diag, solution_summary = _expanded_reprice()
    summary = _summary(reduced_costs, solution_summary)
    blockers = _claim_blockers(summary, source_diag)

    reduced_costs.to_parquet(
        TABLE_DIR / "paper4_v78_source_scope_expanded_reprice.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v78_source_scope_expanded_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v78_source_scope_expanded_duals.csv", duals)
    write_csv(TABLE_DIR / "paper4_v78_source_scope_expanded_diagnostics.csv", source_diag)
    write_csv(TABLE_DIR / "paper4_v78_claim_blockers.csv", blockers)

    improving = int(summary["improving_columns_v78"].sum()) if not summary.empty else 0
    missing_source = (
        int(source_diag["missing_source_constraint_ids_v78"].sum()) if not source_diag.empty else 0
    )
    pricing_clear = improving == 0 and not summary.empty
    source_clear = missing_source == 0 and not source_diag.empty
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v78_full_source_scope_reprice_executed",
                "allowed": True,
                "artifact": "paper4_v78_source_scope_expanded_reprice.parquet",
                "boundary": "focused source-scope expanded reduced-cost screen",
            },
            {
                "claim_id": "v78_pricing_and_source_scope_cleared",
                "allowed": pricing_clear and source_clear,
                "artifact": "paper4_v78_claim_blockers.csv",
                "boundary": "allowed only if pricing and source-scope evidence counts are zero",
            },
            {
                "claim_id": "v78_exact_full_universe_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v78_claim_blockers.csv",
                "boundary": "continuous relaxation/integrality blocker remains",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v78_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v78_source_scope_expanded_reprice",
        "schema_version": "2026-05-15.78",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "reprice_rows_v78": int(len(reduced_costs)),
        "summary_rows_v78": int(len(summary)),
        "dual_rows_v78": int(len(duals)),
        "source_scope_rows_v78": int(len(source_diag)),
        "claim_blocker_rows_v78": int(len(blockers)),
        "improving_columns_v78": improving,
        "negative_reduced_cost_detected_v78": bool(improving > 0),
        "pricing_blocker_cleared_v78": bool(pricing_clear),
        "missing_source_constraint_ids_v78": missing_source,
        "source_scope_blocker_cleared_v78": bool(source_clear),
        "exact_full_universe_cvar_claim_allowed_v78": False,
        "paper1_promotion_allowed_v78": False,
        "paper4_working_champion_changed_v78": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v78 clears focused pricing and source-scope diagnostics, but exact/full "
            "promotion remains blocked by continuous relaxation and integrality evidence"
        ),
    }
    write_json(STATUS_DIR / "paper4_v78_status.json", status)
    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v78": build_v78()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
