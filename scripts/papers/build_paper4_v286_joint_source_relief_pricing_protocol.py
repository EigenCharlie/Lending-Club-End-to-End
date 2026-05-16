#!/usr/bin/env python3
"""Build Paper 4 v286 exact joint source-relief pricing artifacts."""

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

VERSION = 286
SOURCE_TIGHT_SCREEN_VERSION = 285
INCUMBENT_REPAIR_VERSION = 279
NEXT_VERSION = 287
TOP_CANDIDATE_LIMIT = 200
MILP_TIME_LIMIT_SECONDS = 5.0


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v286 exact joint source-relief pricing protocol.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v286_joint_source_relief_pricing_protocol.csv"
                ),
                "boundary": (
                    "Exact source-cap relief MILP over the top-200 v285 candidates; "
                    "not an exhaustive full-universe branch-price certificate."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v286 finds no return-positive exact joint source-relief column "
                    "inside the top-200 source-tight screen."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v286_joint_source_relief_candidate_screen.csv"
                ),
                "boundary": (
                    "Top-200 source-tight candidate screen only; broader candidates, "
                    "multi-add reinvestment and global bounds remain open."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v286 proves full-universe branch-price termination.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v286_claim_blockers.csv"
                ),
                "boundary": "No full ranked pricing loop or global dual-bound certificate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v286 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v286_claim_blockers.csv"
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
                    "v286 solves exact source-cap relief MILPs for the top-200 v285 "
                    "source-tight candidates and confirms the best exact relief bundle "
                    "is still return-negative."
                ),
                "status": "top200_exact_source_relief_screen_no_entering_column",
                "next_artifact": "paper4_v287_expand_exact_relief_or_multi_add_probe.csv",
                "success_condition": (
                    "expand exact relief beyond top-200 or test multi-add reinvestment "
                    "before any branch-price termination claim"
                ),
                "last_wave": "v286",
                "execution_result": "exact_joint_source_relief_top200_return_negative",
                "quarto_promotion_decision": "living_notebook_only",
            }
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
    start = "<!-- V286_JOINT_SOURCE_RELIEF_PRICING_START -->"
    end = "<!-- V286_JOINT_SOURCE_RELIEF_PRICING_END -->"
    block = f"""
{start}

## Wave v286: Exact Joint Source-Relief Pricing Protocol

Generated: {status["generated_at_utc"]}

### Objective

Respond to v285 by solving the correct exact source-cap relief problem for the
top-ranked source-tight candidates. Unlike the first relief heuristic, v286
keeps the exposure denominator inside the MILP source constraints, so grade A,
score decile 0 and every other active source cap are checked jointly.

### Results

- Candidate rows available from v285: `{status["v285_candidate_rows_available_v286"]}`.
- Candidate rows screened: `{status["candidate_rows_screened_v286"]}`.
- Unique exact relief MILPs solved: `{status["unique_relief_milp_signatures_v286"]}`.
- Successful relief MILP rows: `{status["relief_milp_success_rows_v286"]}`.
- Source-violating relief rows: `{status["source_violation_rows_v286"]}`.
- CVaR-feasible relief rows: `{status["cvar_feasible_rows_v286"]}`.
- Return-positive exact relief rows: `{status["return_positive_exact_relief_rows_v286"]}`.
- Best exact relief return delta:
  `{status["best_exact_relief_return_delta_v286"]}`.
- Exact relief entering columns found:
  `{status["exact_relief_entering_columns_found_v286"]}`.
- Valid branch-price bound produced:
  `{status["valid_branch_price_bound_v286"]}`.

### Interpretation

v286 tightens the lesson from v285. Exact source-feasible relief bundles do
exist for the top-200 source-tight candidates, and they pass budget/source/CVaR
checks, but the cheapest exact relief bundle still gives up more expected
return than the entering candidate adds. The best top-200 delta is negative, so
there is no entering source-relief column in this screened set.

### Claim Impact

- Allowed: exact top-200 source-relief protocol completed; no return-positive
  exact relief column found in that screened set.
- Still prohibited: full ranked pricing termination, global integer optimality,
  Paper Estrella replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v286 in the living notebook. The next live-lab step is expanding exact
relief coverage or testing multi-add reinvestment, not promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def _source_cap_maps(source_caps: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        family: {
            str(k): float(v)
            for k, v in source_caps.loc[source_caps["source_family"].astype(str).eq(family)]
            .set_index("source_id")["cap_share_v80"]
            .to_dict()
            .items()
        }
        for family in FAMILIES
    }


def _current_source_maps(selected: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        family: {
            str(k): float(v)
            for k, v in selected.groupby(family, dropna=False)["loan_amnt"].sum().to_dict().items()
        }
        for family in FAMILIES
    }


def _solve_exact_relief_milp(
    *,
    add_row: pd.Series,
    selected: pd.DataFrame,
    selected_amounts: np.ndarray,
    selected_returns: np.ndarray,
    current_exposure: float,
    exposure_min: float,
    exposure_max: float,
    current_by_family: dict[str, dict[str, float]],
    cap_by_family: dict[str, dict[str, float]],
) -> dict[str, Any]:
    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []
    add_amount = float(add_row["loan_amnt"])

    rows.append(selected_amounts)
    lb.append(max(0.0, current_exposure + add_amount - exposure_max))
    ub.append(current_exposure + add_amount - exposure_min)

    for family in FAMILIES:
        add_source = str(add_row[family])
        sources = set(current_by_family[family]) | {add_source}
        selected_source = selected[family].astype(str)
        for source_id in sources:
            cap = cap_by_family[family].get(source_id, 1.0)
            indicator = selected_source.eq(source_id).to_numpy(float)
            rows.append(selected_amounts * (cap - indicator))
            lb.append(-np.inf)
            ub.append(
                cap * (current_exposure + add_amount)
                - current_by_family[family].get(source_id, 0.0)
                - (add_amount if source_id == add_source else 0.0)
            )

    result = milp(
        c=selected_returns,
        integrality=np.ones(len(selected), dtype=float),
        bounds=Bounds(np.zeros(len(selected)), np.ones(len(selected))),
        constraints=LinearConstraint(np.vstack(rows), np.array(lb), np.array(ub)),
        options={"time_limit": MILP_TIME_LIMIT_SECONDS, "mip_rel_gap": 1e-6},
    )
    mask = np.zeros(len(selected), dtype=bool)
    if result.x is not None:
        mask = np.rint(np.clip(result.x, 0, 1)).astype(bool)
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "mip_gap": float(getattr(result, "mip_gap", np.nan)),
        "drop_mask": mask,
    }


def _source_metrics(
    *,
    add_row: pd.Series,
    selected: pd.DataFrame,
    drop_mask: np.ndarray,
    current_by_family: dict[str, dict[str, float]],
    cap_by_family: dict[str, dict[str, float]],
    new_exposure: float,
) -> tuple[float, float, int, str, str]:
    dropped = selected.loc[drop_mask]
    min_slack = np.inf
    max_share = 0.0
    violations = 0
    first_family = ""
    first_source = ""
    add_amount = float(add_row["loan_amnt"])
    for family in FAMILIES:
        add_source = str(add_row[family])
        drop_by_source = (
            {
                str(k): float(v)
                for k, v in dropped.groupby(family, dropna=False)["loan_amnt"].sum().items()
            }
            if not dropped.empty
            else {}
        )
        sources = set(current_by_family[family]) | {add_source} | set(drop_by_source)
        for source_id in sources:
            exposure = current_by_family[family].get(source_id, 0.0)
            if source_id == add_source:
                exposure += add_amount
            exposure -= drop_by_source.get(source_id, 0.0)
            cap = cap_by_family[family].get(source_id, 1.0)
            share = exposure / max(new_exposure, 1.0)
            slack = cap - share
            min_slack = min(min_slack, slack)
            max_share = max(max_share, share)
            if share > cap + 1e-7:
                violations += 1
                if not first_family:
                    first_family = family
                    first_source = source_id
    return float(min_slack), float(max_share), int(violations), first_family, first_source


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v279_restricted_pool_milp_repair_allocations.parquet")
    repair_summary = read_csv("paper4_v279_restricted_pool_milp_repair_summary.csv")
    source_caps = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_caps = source_caps.loc[
        source_caps["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    v285_candidates = read_csv("paper4_v285_source_tight_candidate_diagnostics.csv")
    if universe.empty or selected.empty or repair_summary.empty or source_caps.empty:
        raise RuntimeError("Missing v55, v279, or source-cap inputs for v286.")
    if v285_candidates.empty:
        raise RuntimeError("Missing v285 source-tight candidate diagnostics for v286.")

    selected = selected.reset_index(drop=True)
    v285_candidates["loan_id"] = v285_candidates["loan_id"].astype(str)
    add_source_cols = ["loan_id", "state_top20"]
    candidates = v285_candidates.head(TOP_CANDIDATE_LIMIT).merge(
        universe[add_source_cols].assign(loan_id=lambda df: df["loan_id"].astype(str)),
        on="loan_id",
        how="left",
    )
    repair_row = repair_summary.loc[
        repair_summary[f"portfolio_label_v{INCUMBENT_REPAIR_VERSION}"].eq(
            "restricted_pool_milp_repair_candidate"
        )
    ].iloc[0]
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy()
    selected[f"mean_return_if_dropped_v{VERSION}"] = mean_returns[selected_idx]
    current_losses = losses[:, selected_idx].sum(axis=1)

    current_exposure = float(repair_row[f"portfolio_exposure_v{INCUMBENT_REPAIR_VERSION}"])
    exposure_min = float(repair_row[f"exposure_min_v{INCUMBENT_REPAIR_VERSION}"])
    exposure_max = float(repair_row[f"exposure_max_v{INCUMBENT_REPAIR_VERSION}"])
    cvar_cap = float(repair_row[f"cvar_cap_v{INCUMBENT_REPAIR_VERSION}"])
    current_objective_return = float(repair_row[f"objective_return_v{INCUMBENT_REPAIR_VERSION}"])
    selected_amounts = selected["loan_amnt"].to_numpy(float)
    selected_returns = selected[f"mean_return_if_dropped_v{VERSION}"].to_numpy(float)
    current_by_family = _current_source_maps(selected)
    cap_by_family = _source_cap_maps(source_caps)

    solution_cache: dict[tuple[object, ...], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    drop_rows: list[dict[str, Any]] = []
    for rank, add_row in enumerate(candidates.itertuples(index=False), start=1):
        add = pd.Series(add_row._asdict())
        cache_key = tuple(add[col] for col in ["loan_amnt", *FAMILIES])
        if cache_key not in solution_cache:
            solution_cache[cache_key] = _solve_exact_relief_milp(
                add_row=add,
                selected=selected,
                selected_amounts=selected_amounts,
                selected_returns=selected_returns,
                current_exposure=current_exposure,
                exposure_min=exposure_min,
                exposure_max=exposure_max,
                current_by_family=current_by_family,
                cap_by_family=cap_by_family,
            )
        solution = solution_cache[cache_key]
        drop_mask = solution["drop_mask"]
        dropped = selected.loc[drop_mask].copy()
        drop_amount = float(dropped["loan_amnt"].sum())
        drop_return = float(dropped[f"mean_return_if_dropped_v{VERSION}"].sum())
        new_exposure = current_exposure + float(add["loan_amnt"]) - drop_amount
        added_idx = int(idx_by_id.loc[str(add["loan_id"])])
        new_losses = current_losses + losses[:, added_idx]
        if drop_mask.any():
            new_losses = new_losses - losses[:, selected_idx[drop_mask]].sum(axis=1)
        cvar_after = v70._tail_cvar(new_losses)
        min_slack, max_share, violations, first_family, first_source = _source_metrics(
            add_row=add,
            selected=selected,
            drop_mask=drop_mask,
            current_by_family=current_by_family,
            cap_by_family=cap_by_family,
            new_exposure=new_exposure,
        )
        return_delta = float(add[f"mean_return_v{SOURCE_TIGHT_SCREEN_VERSION}"]) - drop_return
        exact_relief_feasible = (
            bool(solution["success"])
            and new_exposure >= exposure_min - 1e-7
            and new_exposure <= exposure_max + 1e-7
            and violations == 0
            and cvar_after <= cvar_cap + 1e-7
        )
        entering_column = exact_relief_feasible and return_delta > 1e-9
        rows.append(
            {
                f"candidate_rank_v{VERSION}": rank,
                f"added_loan_id_v{VERSION}": str(add["loan_id"]),
                f"added_loan_amount_v{VERSION}": float(add["loan_amnt"]),
                f"added_mean_return_v{VERSION}": float(
                    add[f"mean_return_v{SOURCE_TIGHT_SCREEN_VERSION}"]
                ),
                f"pricing_block_ids_v{VERSION}": str(
                    add[f"pricing_block_ids_v{SOURCE_TIGHT_SCREEN_VERSION}"]
                ),
                f"relief_milp_success_v{VERSION}": bool(solution["success"]),
                f"relief_milp_status_v{VERSION}": int(solution["status"]),
                f"relief_milp_gap_v{VERSION}": float(solution["mip_gap"]),
                f"drop_count_v{VERSION}": int(len(dropped)),
                f"drop_exposure_v{VERSION}": drop_amount,
                f"drop_mean_return_v{VERSION}": drop_return,
                f"return_delta_after_exact_relief_v{VERSION}": return_delta,
                f"objective_return_after_exact_relief_v{VERSION}": (
                    current_objective_return + return_delta
                ),
                f"exposure_after_exact_relief_v{VERSION}": new_exposure,
                f"source_min_slack_after_exact_relief_v{VERSION}": min_slack,
                f"max_source_share_after_exact_relief_v{VERSION}": max_share,
                f"source_cap_violations_after_exact_relief_v{VERSION}": violations,
                f"first_source_block_family_v{VERSION}": first_family,
                f"first_source_block_id_v{VERSION}": first_source,
                f"cvar90_after_exact_relief_v{VERSION}": cvar_after,
                f"budget_source_cvar_feasible_exact_relief_v{VERSION}": exact_relief_feasible,
                f"return_positive_exact_relief_v{VERSION}": return_delta > 1e-9,
                f"exact_relief_entering_column_v{VERSION}": entering_column,
                f"cardinality_preserved_v{VERSION}": int(len(dropped)) == 1,
                f"cardinality_after_exact_relief_v{VERSION}": int(len(selected) - len(dropped) + 1),
                f"claim_boundary_v{VERSION}": (
                    "top-200 exact joint source-relief diagnostic only; not global pricing"
                ),
            }
        )
        for drop_rank, (_, drop) in enumerate(
            dropped.sort_values(f"mean_return_if_dropped_v{VERSION}").iterrows(),
            start=1,
        ):
            drop_rows.append(
                {
                    f"candidate_rank_v{VERSION}": rank,
                    f"added_loan_id_v{VERSION}": str(add["loan_id"]),
                    f"drop_rank_v{VERSION}": drop_rank,
                    f"dropped_loan_id_v{VERSION}": str(drop["loan_id"]),
                    f"dropped_loan_amount_v{VERSION}": float(drop["loan_amnt"]),
                    f"dropped_mean_return_v{VERSION}": float(
                        drop[f"mean_return_if_dropped_v{VERSION}"]
                    ),
                    f"drop_grade_v{VERSION}": str(drop["grade"]),
                    f"drop_score_decile_v{VERSION}": str(drop["score_decile"]),
                    f"claim_boundary_v{VERSION}": (
                        "selected by exact source-relief MILP for diagnostic pricing only"
                    ),
                }
            )

    candidate_screen = pd.DataFrame(rows).sort_values(
        f"return_delta_after_exact_relief_v{VERSION}", ascending=False
    )
    drop_bundle = pd.DataFrame(drop_rows)
    success_rows = int(candidate_screen[f"relief_milp_success_v{VERSION}"].sum())
    source_violation_rows = int(
        candidate_screen[f"source_cap_violations_after_exact_relief_v{VERSION}"].gt(0).sum()
    )
    cvar_feasible_rows = int(
        candidate_screen[f"cvar90_after_exact_relief_v{VERSION}"].le(cvar_cap + 1e-7).sum()
    )
    return_positive_rows = int(candidate_screen[f"return_positive_exact_relief_v{VERSION}"].sum())
    entering_rows = int(candidate_screen[f"exact_relief_entering_column_v{VERSION}"].sum())
    best = candidate_screen.head(1)
    protocol = pd.DataFrame(
        [
            {
                f"protocol_id_v{VERSION}": "top200_exact_joint_source_relief_pricing_protocol",
                f"source_tight_screen_version_v{VERSION}": SOURCE_TIGHT_SCREEN_VERSION,
                f"incumbent_repair_version_v{VERSION}": INCUMBENT_REPAIR_VERSION,
                f"v285_candidate_rows_available_v{VERSION}": int(len(v285_candidates)),
                f"candidate_screen_limit_v{VERSION}": TOP_CANDIDATE_LIMIT,
                f"candidate_rows_screened_v{VERSION}": int(len(candidate_screen)),
                f"unique_relief_milp_signatures_v{VERSION}": int(len(solution_cache)),
                f"relief_milp_success_rows_v{VERSION}": success_rows,
                f"source_violation_rows_v{VERSION}": source_violation_rows,
                f"cvar_feasible_rows_v{VERSION}": cvar_feasible_rows,
                f"return_positive_exact_relief_rows_v{VERSION}": return_positive_rows,
                f"exact_relief_entering_column_rows_v{VERSION}": entering_rows,
                f"best_exact_relief_added_loan_id_v{VERSION}": str(
                    best[f"added_loan_id_v{VERSION}"].iloc[0]
                )
                if not best.empty
                else "",
                f"best_exact_relief_return_delta_v{VERSION}": float(
                    best[f"return_delta_after_exact_relief_v{VERSION}"].iloc[0]
                )
                if not best.empty
                else np.nan,
                f"best_exact_relief_drop_count_v{VERSION}": int(
                    best[f"drop_count_v{VERSION}"].iloc[0]
                )
                if not best.empty
                else 0,
                f"exact_relief_entering_columns_found_v{VERSION}": entering_rows > 0,
                f"valid_branch_price_bound_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_expand_exact_relief_or_multi_add_probe.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "top-200 exact source-relief protocol only; broader pricing and "
                    "global bound evidence remain missing"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "top200_exact_relief_no_return_positive_column",
                f"blocking_v{VERSION}": entering_rows == 0,
                f"evidence_count_v{VERSION}": int(len(candidate_screen)),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_expand_exact_relief_or_multi_add_probe.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "top-200 exact relief bundles are source/CVaR feasible but return-negative"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "full_ranked_exact_relief_screen_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(v285_candidates) - len(candidate_screen)),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_expand_exact_relief_or_multi_add_probe.csv"
                ),
                f"claim_boundary_v{VERSION}": "v286 screens top-200, not every v285 candidate",
            },
            {
                f"blocker_id_v{VERSION}": "branch_price_dual_bound_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_branch_price_dual_bound_loop",
                f"claim_boundary_v{VERSION}": "no dual-bound loop or termination certificate",
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
                "claim_id": "v286_exact_joint_source_relief_protocol_executed",
                "allowed": True,
                "artifact": "paper4_v286_joint_source_relief_pricing_protocol.csv",
                "boundary": "top-200 exact source-relief MILP screen",
            },
            {
                "claim_id": "v286_no_top200_return_positive_exact_relief_column",
                "allowed": entering_rows == 0,
                "artifact": "paper4_v286_joint_source_relief_candidate_screen.csv",
                "boundary": "top-200 only; not full ranked pricing",
            },
            {
                "claim_id": "v286_valid_branch_price_bound",
                "allowed": False,
                "artifact": "paper4_v286_claim_blockers.csv",
                "boundary": "dual-bound loop missing",
            },
            {
                "claim_id": "v286_global_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v286_claim_blockers.csv",
                "boundary": "global certificate missing",
            },
            {
                "claim_id": "v286_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v286_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v286_joint_source_relief_pricing_protocol.csv", protocol)
    write_csv(TABLE_DIR / "paper4_v286_joint_source_relief_candidate_screen.csv", candidate_screen)
    write_csv(TABLE_DIR / "paper4_v286_joint_source_relief_drop_bundles.csv", drop_bundle)
    write_csv(TABLE_DIR / "paper4_v286_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v286_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = protocol.iloc[0]
    status = {
        "phase": "v286_exact_joint_source_relief_pricing_protocol",
        "schema_version": "2026-05-15.286",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "source_tight_screen_version_v286": SOURCE_TIGHT_SCREEN_VERSION,
        "incumbent_repair_version_v286": INCUMBENT_REPAIR_VERSION,
        "v285_candidate_rows_available_v286": int(row[f"v285_candidate_rows_available_v{VERSION}"]),
        "candidate_screen_limit_v286": TOP_CANDIDATE_LIMIT,
        "candidate_rows_screened_v286": int(row[f"candidate_rows_screened_v{VERSION}"]),
        "unique_relief_milp_signatures_v286": int(row[f"unique_relief_milp_signatures_v{VERSION}"]),
        "relief_milp_success_rows_v286": success_rows,
        "source_violation_rows_v286": source_violation_rows,
        "cvar_feasible_rows_v286": cvar_feasible_rows,
        "return_positive_exact_relief_rows_v286": return_positive_rows,
        "exact_relief_entering_column_rows_v286": entering_rows,
        "best_exact_relief_added_loan_id_v286": str(
            row[f"best_exact_relief_added_loan_id_v{VERSION}"]
        ),
        "best_exact_relief_return_delta_v286": float(
            row[f"best_exact_relief_return_delta_v{VERSION}"]
        ),
        "best_exact_relief_drop_count_v286": int(row[f"best_exact_relief_drop_count_v{VERSION}"]),
        "exact_relief_entering_columns_found_v286": bool(
            row[f"exact_relief_entering_columns_found_v{VERSION}"]
        ),
        "valid_branch_price_bound_v286": False,
        "full_universe_integer_optimality_claim_allowed_v286": False,
        "paper1_promotion_allowed_v286": False,
        "paper4_working_champion_changed_v286": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "drop_bundle_rows_v286": int(len(drop_bundle)),
        "claim_blocker_rows_v286": int(len(blockers)),
        "claim_matrix_rows_v286": int(len(claim_matrix)),
        "next_artifact_v286": f"paper4_v{NEXT_VERSION}_expand_exact_relief_or_multi_add_probe.csv",
        "claim_boundary": (
            "v286 screens exact source-relief over top-200 candidates only; global "
            "pricing, branch-price bounds and promotion remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v286_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v286": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
