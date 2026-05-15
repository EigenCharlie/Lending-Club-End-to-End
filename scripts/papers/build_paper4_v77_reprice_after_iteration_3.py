#!/usr/bin/env python3
"""Build Paper 4 v77 post-iteration-3 re-pricing artifacts.

v77 reruns dual pricing after the focused v76 iteration-3 columns have been
added to the restricted master. It verifies whether any omitted v55 columns
still have negative reduced cost in the affected policy/regime pair. Even if
pricing clears, source-scope and continuous-relaxation blockers remain.
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
from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71  # noqa: E402

PAPER4_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = PAPER4_ROOT / "tables"
STATUS_DIR = PAPER4_ROOT / "status"
NOTE_DIR = PAPER4_ROOT / "notes"
NOTEBOOK = NOTE_DIR / "paper4_living_lab_notebook.md"
FORBIDDEN_FINAL_PROMOTION = STATUS_DIR / "paper4_final_promotion.json"


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


def _rename_suffix(df: pd.DataFrame, old: str = "_v71", new: str = "_v77") -> pd.DataFrame:
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
    combined["policy_id_v77"] = policy_id
    combined["regime_v77"] = regime
    return combined


def _price_after_iteration_3() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        solution = v71._solve_policy_with_duals(policy_id, pool, source_map, regime)
        solution_rows.append(
            {
                "policy_id": policy_id,
                "regime_v77": regime,
                "solver_success_v77": bool(solution.get("success_v71", False)),
                "solver_message_v77": solution.get("message_v71", ""),
                "objective_return_v77": solution.get("objective_return_v71", np.nan),
                "v76_objective_return_baseline_v77": row.get("objective_return_v76", np.nan),
                "delta_return_vs_v76_v77": float(
                    solution.get("objective_return_v71", np.nan)
                    - row.get("objective_return_v76", np.nan)
                ),
                "incumbent_exposure_v77": solution.get("incumbent_exposure_v71", np.nan),
                "incumbent_cvar90_v77": solution.get("incumbent_cvar90_v71", np.nan),
                "cvar_cap_v77": solution.get("cvar_cap_v71", np.nan),
                "v76_master_rows_repriced_v77": int(len(pool)),
                "claim_boundary_v77": (
                    "post-v76 restricted-master LP rerun for marginals; not final proof"
                ),
            }
        )
        if not solution.get("success_v71", False):
            continue
        priced = _rename_suffix(v71._price_universe(universe, losses, mean_returns, solution))
        priced["post_iteration_reprice_v77"] = 1
        priced["pricing_scope_v77"] = "v76_iteration_3_duals_applied_to_omitted_v55_columns"
        priced["claim_boundary_v77"] = (
            "post-iteration-3 reduced-cost screen; source and integrality blockers still apply"
        )
        reduced_frames.append(priced)

        duals = _rename_suffix(solution["dual_rows"])
        duals["post_iteration_reprice_v77"] = 1
        duals["claim_boundary_v77"] = (
            "HiGHS marginal from v76 iteration LP; not full-universe certificate"
        )
        dual_frames.append(duals)

        source = _rename_suffix(
            v71._source_cap_diagnostics(universe, solution, solution["dual_rows"])
        )
        source["post_iteration_reprice_v77"] = 1
        source["claim_boundary_v77"] = (
            "source scope after v76 iteration; incomplete scope blocks full certificate"
        )
        source_frames.append(source)

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
        reduced_costs.groupby(["policy_id", "regime_v77"], dropna=False)
        .agg(
            omitted_rows_priced_v77=("loan_id", "count"),
            improving_columns_v77=("improving_column_v77", "sum"),
            min_reduced_cost_v77=("minimization_reduced_cost_v77", "min"),
            best_return_improvement_signal_v77=("return_improvement_signal_v77", "max"),
            median_reduced_cost_v77=("minimization_reduced_cost_v77", "median"),
        )
        .reset_index()
    )
    grouped["negative_reduced_cost_detected_v77"] = grouped["improving_columns_v77"].gt(0)
    grouped["post_iteration_reprice_performed_v77"] = True
    grouped["column_generation_termination_claim_allowed_v77"] = False
    grouped["exact_full_universe_cvar_claim_allowed_v77"] = False
    grouped["claim_boundary_v77"] = (
        "post-v76 re-pricing screen; source-scope and integrality blockers remain"
    )
    return grouped.merge(solution_summary, on=["policy_id", "regime_v77"], how="left")


def _claim_blockers(summary: pd.DataFrame, source_diag: pd.DataFrame) -> pd.DataFrame:
    improving = int(summary["improving_columns_v77"].sum()) if not summary.empty else 0
    missing_source = (
        int(source_diag["missing_source_ids_v77"].sum()) if not source_diag.empty else 0
    )
    return pd.DataFrame(
        [
            {
                "blocker_id_v77": "negative_reduced_cost_columns_after_iteration_3",
                "blocking_v77": improving > 0,
                "evidence_count_v77": improving,
                "required_next_artifact_v77": "paper4_v78_column_generation_iteration_4.csv",
                "claim_boundary_v77": ("post-v76 re-pricing finds no improving columns if zero"),
            },
            {
                "blocker_id_v77": "source_scope_after_iteration_3_incomplete",
                "blocking_v77": missing_source > 0,
                "evidence_count_v77": missing_source,
                "required_next_artifact_v77": "paper4_v78_source_scope_expansion_v4.csv",
                "claim_boundary_v77": (
                    "source constraints after v76 still do not cover all full-universe IDs"
                ),
            },
            {
                "blocker_id_v77": "continuous_relaxation_not_whole_loan_milp",
                "blocking_v77": True,
                "evidence_count_v77": 1,
                "required_next_artifact_v77": "paper4_v78_integrality_gap_or_milp_probe.csv",
                "claim_boundary_v77": (
                    "post-iteration-3 pricing remains a continuous LP diagnostic"
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
                "claim": "Paper 4 has v77 post-iteration-3 re-pricing after v76 column generation.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v77_reprice_after_iteration_3.parquet"
                ),
                "boundary": "Post-iteration-3 reduced-cost screen only; blockers decide claims.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v77 proves exact full-universe CVaR optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v77_claim_blockers.csv"
                ),
                "boundary": "Pricing may clear, but source-scope and integrality review remain.",
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


def _update_backlog(negative_detected: bool) -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    status = "ready_for_iteration_or_stop_decision" if negative_detected else "scope_gated"
    next_artifact = (
        "paper4_v78_column_generation_iteration_4_or_stop_memo.csv"
        if negative_detected
        else "paper4_v78_source_integrality_convergence_review.csv"
    )
    success_condition = (
        "post-iteration-3 pricing either terminates or queues iteration 4"
        if negative_detected
        else "source-scope and integrality blockers clear after pricing has no improving columns"
    )
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "CVaR/OCE",
                "executable_item": (
                    "v77 re-prices the omitted v55 universe after the v76 iteration-3 solve."
                ),
                "status": status,
                "next_artifact": next_artifact,
                "success_condition": success_condition,
                "last_wave": "v77",
                "execution_result": "post_iteration_3_reprice_completed",
                "quarto_promotion_decision": "living_notebook_only",
            },
            {
                "horizon": "short",
                "lane": "Source governance",
                "executable_item": (
                    "Resolve source-scope blockers after v77 post-iteration-3 pricing."
                ),
                "status": "gated",
                "next_artifact": "paper4_v78_source_scope_expansion_v4.csv",
                "success_condition": "source rows cover all full-universe IDs used by pricing",
                "last_wave": "v77",
                "execution_result": "post_iteration_3_source_scope_quantified",
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
    start = "<!-- V77_REPRICE_AFTER_ITERATION_3_START -->"
    end = "<!-- V77_REPRICE_AFTER_ITERATION_3_END -->"
    block = f"""
{start}

## Wave v77: Re-Price After Column-Generation Iteration 3

Generated: {status["generated_at_utc"]}

### Objective

Re-price omitted v55 columns after the focused v76 iteration-3 restricted-master
solve. This tests whether adding the remaining v75 negative columns clears the
pricing blocker in the affected regime.

### Results

- Re-price rows: `{status["reprice_rows_v77"]}`.
- Summary rows: `{status["summary_rows_v77"]}`.
- Dual rows: `{status["dual_rows_v77"]}`.
- Improving columns after iteration 3: `{status["improving_columns_after_iteration_3_v77"]}`.
- Source-scope rows: `{status["source_scope_rows_v77"]}`.
- Source-scope missing IDs: `{status["source_scope_missing_ids_v77"]}`.
- Pricing blocker cleared: `{status["pricing_blocker_cleared_v77"]}`.
- Exact full-universe CVaR claim allowed: `{status["exact_full_universe_cvar_claim_allowed_v77"]}`.

### Interpretation

v77 separates the pricing story from the governance story. If no improving
omitted columns remain, the column-generation pricing loop has locally cleared
for the affected regime, but source-scope and continuous-relaxation blockers
still prevent exact full-universe or whole-loan claims.

### Claim Impact

- Allowed: post-v76 re-pricing was executed and documented.
- Still prohibited: exact full-universe CVaR optimality, MILP whole-loan
  optimality, Paper Estrella replacement, final Paper 4 promotion and live
  deployment.

### Quarto Promotion Decision

Keep v77 in the living notebook. Promote only after source-scope coverage and
integrality review pass.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v77() -> dict[str, Any]:
    started = datetime.now(UTC)
    reduced_costs, duals, source_diag, solution_summary = _price_after_iteration_3()
    summary = _summary(reduced_costs, solution_summary)
    blockers = _claim_blockers(summary, source_diag)

    reduced_costs.to_parquet(
        TABLE_DIR / "paper4_v77_reprice_after_iteration_3.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v77_reprice_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v77_restricted_master_duals.csv", duals)
    write_csv(TABLE_DIR / "paper4_v77_source_scope_after_iteration_3.csv", source_diag)
    write_csv(TABLE_DIR / "paper4_v77_claim_blockers.csv", blockers)

    improving = int(summary["improving_columns_v77"].sum()) if not summary.empty else 0
    source_missing = (
        int(source_diag["missing_source_ids_v77"].sum()) if not source_diag.empty else 0
    )
    pricing_clear = improving == 0 and not summary.empty
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v77_post_iteration_3_reprice_executed",
                "allowed": True,
                "artifact": "paper4_v77_reprice_after_iteration_3.parquet",
                "boundary": "post-iteration-3 reduced-cost screen only",
            },
            {
                "claim_id": "v77_pricing_blocker_cleared",
                "allowed": pricing_clear,
                "artifact": "paper4_v77_claim_blockers.csv",
                "boundary": "allowed only if improving-column evidence count is zero",
            },
            {
                "claim_id": "v77_exact_full_universe_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v77_claim_blockers.csv",
                "boundary": "source-scope and integrality blockers remain",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v77_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v77_reprice_after_column_generation_iteration_3",
        "schema_version": "2026-05-15.77",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "reprice_rows_v77": int(len(reduced_costs)),
        "summary_rows_v77": int(len(summary)),
        "dual_rows_v77": int(len(duals)),
        "source_scope_rows_v77": int(len(source_diag)),
        "claim_blocker_rows_v77": int(len(blockers)),
        "improving_columns_after_iteration_3_v77": improving,
        "negative_reduced_cost_detected_v77": bool(improving > 0),
        "pricing_blocker_cleared_v77": bool(pricing_clear),
        "source_scope_missing_ids_v77": source_missing,
        "post_iteration_reprice_performed_v77": True,
        "column_generation_termination_claim_allowed_v77": False,
        "exact_full_universe_cvar_claim_allowed_v77": False,
        "paper1_promotion_allowed_v77": False,
        "paper4_working_champion_changed_v77": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v77 re-prices after v76 iteration; pricing may clear, but source-scope "
            "and integrality blockers still control exact/full promotion claims"
        ),
    }
    write_json(STATUS_DIR / "paper4_v77_status.json", status)
    _update_claim_boundaries()
    _update_backlog(bool(improving > 0))
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v77": build_v77()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
