#!/usr/bin/env python3
"""Build Paper 4 v75 post-iteration-2 re-pricing artifacts.

v75 reruns dual pricing after the v74 iteration-2 columns have been added to
the restricted master. It asks whether the second column-generation iteration
removed all negative reduced-cost omitted columns. It still does not promote
full-universe, MILP, Paper Estrella or live-deployment claims.
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


def _rename_suffix(df: pd.DataFrame, old: str = "_v71", new: str = "_v75") -> pd.DataFrame:
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
    combined = pd.concat([base, first, second], ignore_index=True, sort=False)
    combined = combined.drop_duplicates("loan_id", keep="first").copy()
    combined["policy_id_v75"] = policy_id
    combined["regime_v75"] = regime
    return combined


def _price_after_iteration_2() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    iteration_1_candidates = read_parquet("paper4_v72_iteration_1_candidates.parquet")
    iteration_2_candidates = read_parquet("paper4_v74_iteration_2_candidates.parquet")
    frontier = read_csv("paper4_v74_iteration_2_frontier.csv")
    concentration = read_csv("paper4_v63_source_repair_concentration.csv")
    if (
        universe.empty
        or iteration_1_candidates.empty
        or iteration_2_candidates.empty
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
        regime = str(row["regime_v74"])
        pool = _pool_for_reprice(
            base_master,
            iteration_1_candidates,
            iteration_2_candidates,
            policy_id,
            regime,
        )
        source_map = v70._policy_source_map(concentration, policy_id)
        solution = v71._solve_policy_with_duals(policy_id, pool, source_map, regime)
        solution_rows.append(
            {
                "policy_id": policy_id,
                "regime_v75": regime,
                "solver_success_v75": bool(solution.get("success_v71", False)),
                "solver_message_v75": solution.get("message_v71", ""),
                "objective_return_v75": solution.get("objective_return_v71", np.nan),
                "v74_objective_return_baseline_v75": row.get("objective_return_v74", np.nan),
                "delta_return_vs_v74_v75": float(
                    solution.get("objective_return_v71", np.nan)
                    - row.get("objective_return_v74", np.nan)
                ),
                "incumbent_exposure_v75": solution.get("incumbent_exposure_v71", np.nan),
                "incumbent_cvar90_v75": solution.get("incumbent_cvar90_v71", np.nan),
                "cvar_cap_v75": solution.get("cvar_cap_v71", np.nan),
                "v74_master_rows_repriced_v75": int(len(pool)),
                "claim_boundary_v75": (
                    "post-v74 restricted-master LP rerun for marginals; not convergence proof"
                ),
            }
        )
        if not solution.get("success_v71", False):
            continue
        priced = _rename_suffix(v71._price_universe(universe, losses, mean_returns, solution))
        priced["post_iteration_reprice_v75"] = 1
        priced["pricing_scope_v75"] = "v74_iteration_2_duals_applied_to_omitted_v55_columns"
        priced["claim_boundary_v75"] = (
            "post-iteration-2 reduced-cost screen; termination allowed only if no blockers remain"
        )
        reduced_frames.append(priced)

        duals = _rename_suffix(solution["dual_rows"])
        duals["post_iteration_reprice_v75"] = 1
        duals["claim_boundary_v75"] = (
            "HiGHS marginal from v74 iteration LP; not full-universe certificate"
        )
        dual_frames.append(duals)

        source = _rename_suffix(
            v71._source_cap_diagnostics(universe, solution, solution["dual_rows"])
        )
        source["post_iteration_reprice_v75"] = 1
        source["claim_boundary_v75"] = (
            "source scope after v74 iteration; incomplete scope blocks full certificate"
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
        reduced_costs.groupby(["policy_id", "regime_v75"], dropna=False)
        .agg(
            omitted_rows_priced_v75=("loan_id", "count"),
            improving_columns_v75=("improving_column_v75", "sum"),
            min_reduced_cost_v75=("minimization_reduced_cost_v75", "min"),
            best_return_improvement_signal_v75=("return_improvement_signal_v75", "max"),
            median_reduced_cost_v75=("minimization_reduced_cost_v75", "median"),
        )
        .reset_index()
    )
    grouped["negative_reduced_cost_detected_v75"] = grouped["improving_columns_v75"].gt(0)
    grouped["post_iteration_reprice_performed_v75"] = True
    grouped["column_generation_termination_claim_allowed_v75"] = False
    grouped["exact_full_universe_cvar_claim_allowed_v75"] = False
    grouped["claim_boundary_v75"] = (
        "post-v74 re-pricing screen; termination requires no improving columns and no scope blockers"
    )
    return grouped.merge(solution_summary, on=["policy_id", "regime_v75"], how="left")


def _claim_blockers(summary: pd.DataFrame, source_diag: pd.DataFrame) -> pd.DataFrame:
    improving = int(summary["improving_columns_v75"].sum()) if not summary.empty else 0
    missing_source = (
        int(source_diag["missing_source_ids_v75"].sum()) if not source_diag.empty else 0
    )
    return pd.DataFrame(
        [
            {
                "blocker_id_v75": "negative_reduced_cost_columns_after_iteration_2",
                "blocking_v75": improving > 0,
                "evidence_count_v75": improving,
                "required_next_artifact_v75": "paper4_v76_column_generation_iteration_3.csv",
                "claim_boundary_v75": ("post-v74 re-pricing still finds improving omitted columns"),
            },
            {
                "blocker_id_v75": "source_scope_after_iteration_2_incomplete",
                "blocking_v75": missing_source > 0,
                "evidence_count_v75": missing_source,
                "required_next_artifact_v75": "paper4_v76_source_scope_expansion_v3.csv",
                "claim_boundary_v75": (
                    "source constraints after v74 still do not cover all full-universe IDs"
                ),
            },
            {
                "blocker_id_v75": "continuous_relaxation_not_whole_loan_milp",
                "blocking_v75": True,
                "evidence_count_v75": 1,
                "required_next_artifact_v75": "paper4_v76_integrality_gap_or_milp_probe.csv",
                "claim_boundary_v75": (
                    "post-iteration-2 pricing remains a continuous LP diagnostic"
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
                "claim": "Paper 4 has v75 post-iteration-2 re-pricing after v74 column generation.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v75_reprice_after_iteration_2.parquet"
                ),
                "boundary": "Post-iteration-2 reduced-cost screen only; blockers decide next step.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v75 proves column-generation convergence.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v75_claim_blockers.csv"
                ),
                "boundary": "Requires no improving columns, complete source scope and integrality review.",
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
        "paper4_v76_column_generation_iteration_3_or_stop_memo.csv"
        if negative_detected
        else "paper4_v76_source_integrality_convergence_review.csv"
    )
    success_condition = (
        "post-iteration-2 pricing either terminates or queues iteration 3"
        if negative_detected
        else "source-scope and integrality blockers clear after pricing has no improving columns"
    )
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "CVaR/OCE",
                "executable_item": (
                    "v75 re-prices the omitted v55 universe after the v74 iteration-2 solve."
                ),
                "status": status,
                "next_artifact": next_artifact,
                "success_condition": success_condition,
                "last_wave": "v75",
                "execution_result": "post_iteration_2_reprice_completed",
                "quarto_promotion_decision": "living_notebook_only",
            },
            {
                "horizon": "short",
                "lane": "Source governance",
                "executable_item": (
                    "Resolve source-scope blockers discovered after v75 post-iteration-2 pricing."
                ),
                "status": "gated",
                "next_artifact": "paper4_v76_source_scope_expansion_v3.csv",
                "success_condition": "source rows cover all full-universe IDs used by pricing",
                "last_wave": "v75",
                "execution_result": "post_iteration_2_source_scope_quantified",
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
    start = "<!-- V75_REPRICE_AFTER_ITERATION_2_START -->"
    end = "<!-- V75_REPRICE_AFTER_ITERATION_2_END -->"
    block = f"""
{start}

## Wave v75: Re-Price After Column-Generation Iteration 2

Generated: {status["generated_at_utc"]}

### Objective

Re-price omitted v55 columns after the v74 iteration-2 restricted-master solve.
This tests whether the second iteration removed the negative reduced-cost
columns found in v73.

### Results

- Re-price rows: `{status["reprice_rows_v75"]}`.
- Summary rows: `{status["summary_rows_v75"]}`.
- Dual rows: `{status["dual_rows_v75"]}`.
- Improving columns after iteration 2: `{status["improving_columns_after_iteration_2_v75"]}`.
- Source-scope rows: `{status["source_scope_rows_v75"]}`.
- Source-scope missing IDs: `{status["source_scope_missing_ids_v75"]}`.
- Termination claim allowed: `{status["column_generation_termination_claim_allowed_v75"]}`.

### Interpretation

v75 is the stop-or-continue check after the second restricted-master iteration.
If improving omitted columns remain, the laboratory should continue to
iteration 3. If pricing clears, source-scope and integrality blockers still
need to clear before stronger claims are allowed.

### Claim Impact

- Allowed: post-v74 re-pricing was executed and documented.
- Still prohibited: convergence unless all blockers clear, exact full-universe
  CVaR optimality, MILP whole-loan optimality, Paper Estrella replacement,
  final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v75 in the living notebook. Promote only after pricing, source-scope
coverage and integrality review all pass.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v75() -> dict[str, Any]:
    started = datetime.now(UTC)
    reduced_costs, duals, source_diag, solution_summary = _price_after_iteration_2()
    summary = _summary(reduced_costs, solution_summary)
    blockers = _claim_blockers(summary, source_diag)

    reduced_costs.to_parquet(
        TABLE_DIR / "paper4_v75_reprice_after_iteration_2.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v75_reprice_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v75_restricted_master_duals.csv", duals)
    write_csv(TABLE_DIR / "paper4_v75_source_scope_after_iteration_2.csv", source_diag)
    write_csv(TABLE_DIR / "paper4_v75_claim_blockers.csv", blockers)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v75_post_iteration_2_reprice_executed",
                "allowed": True,
                "artifact": "paper4_v75_reprice_after_iteration_2.parquet",
                "boundary": "post-iteration-2 reduced-cost screen only",
            },
            {
                "claim_id": "v75_column_generation_converged",
                "allowed": False,
                "artifact": "paper4_v75_claim_blockers.csv",
                "boundary": "requires no improving columns, source-scope coverage and integrality review",
            },
            {
                "claim_id": "v75_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v75_reprice_summary.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v75_claim_matrix_delta.csv", claim_matrix)

    improving = int(summary["improving_columns_v75"].sum()) if not summary.empty else 0
    source_missing = (
        int(source_diag["missing_source_ids_v75"].sum()) if not source_diag.empty else 0
    )
    status = {
        "phase": "v75_reprice_after_column_generation_iteration_2",
        "schema_version": "2026-05-15.75",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "reprice_rows_v75": int(len(reduced_costs)),
        "summary_rows_v75": int(len(summary)),
        "dual_rows_v75": int(len(duals)),
        "source_scope_rows_v75": int(len(source_diag)),
        "claim_blocker_rows_v75": int(len(blockers)),
        "improving_columns_after_iteration_2_v75": improving,
        "negative_reduced_cost_detected_v75": bool(improving > 0),
        "source_scope_missing_ids_v75": source_missing,
        "post_iteration_reprice_performed_v75": True,
        "column_generation_termination_claim_allowed_v75": False,
        "exact_full_universe_cvar_claim_allowed_v75": False,
        "paper1_promotion_allowed_v75": False,
        "paper4_working_champion_changed_v75": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v75 re-prices after v74 iteration; convergence remains blocked unless "
            "pricing, source-scope and integrality blockers all clear"
        ),
    }
    write_json(STATUS_DIR / "paper4_v75_status.json", status)
    _update_claim_boundaries()
    _update_backlog(bool(improving > 0))
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v75": build_v75()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
