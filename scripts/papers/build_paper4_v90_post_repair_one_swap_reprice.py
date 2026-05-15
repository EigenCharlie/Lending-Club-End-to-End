#!/usr/bin/env python3
"""Build Paper 4 v90 post-v89 one-swap repricing artifacts."""

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
from scripts.papers import build_paper4_v82_one_swap_integer_pricing_probe as v82  # noqa: E402

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


def read_parquet(name: str | Path, directory: Path = TABLE_DIR) -> pd.DataFrame:
    path = Path(name)
    if not path.is_absolute():
        path = directory / path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def write_csv(path: Path, df: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(df) if isinstance(df, list) else df
    out.to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _post_repair_context() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    selected = read_parquet("paper4_v89_next_one_swap_repair_allocations.parquet").copy()
    source_summary = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_summary = source_summary.loc[
        source_summary["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    summary = read_csv("paper4_v89_next_one_swap_repair_summary.csv")
    repair_summary = summary.loc[
        summary["portfolio_label_v89"].eq("next_one_swap_repair_candidate")
    ].iloc[0]
    selected_ids = set(selected["loan_id"].astype(str))
    candidates = universe.loc[~universe["loan_id"].astype(str).isin(selected_ids)].copy()
    return selected, source_summary, repair_summary, candidates


def _candidate_pairs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    selected, source_summary, repair_summary, candidates = _post_repair_context()
    action = read_csv("paper4_v89_next_one_swap_repair_action.csv")
    if universe.empty or selected.empty or source_summary.empty or candidates.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy()
    candidate_idx = idx_by_id.loc[candidates["loan_id"].astype(str)].to_numpy()
    selected = selected.reset_index(drop=True)
    candidates = candidates.reset_index(drop=True)
    selected["mean_return_if_dropped_v90"] = mean_returns[selected_idx]
    candidates["mean_return_if_added_v90"] = mean_returns[candidate_idx]

    policy_id = str(action["policy_id"].iloc[0]) if not action.empty else "v89_repair_candidate"
    regime = str(action["regime_v89"].iloc[0]) if not action.empty else "post_v89_repair"
    current_exposure = float(repair_summary["portfolio_exposure_v89"])
    exposure_min = float(repair_summary["exposure_min_v89"])
    exposure_max = float(repair_summary["exposure_max_v89"])
    cvar_cap = float(repair_summary["cvar_cap_v89"])
    current_objective_return = float(repair_summary["objective_return_v89"])
    current_losses = losses[:, selected_idx].sum(axis=1)

    add_amount = candidates["loan_amnt"].to_numpy(float)
    drop_amount = selected["loan_amnt"].to_numpy(float)
    add_return = candidates["mean_return_if_added_v90"].to_numpy(float)
    drop_return = selected["mean_return_if_dropped_v90"].to_numpy(float)
    return_mask = add_return[:, None] - drop_return[None, :] > 1e-9
    total_pairs = int(return_mask.size)
    return_pairs = int(return_mask.sum())
    exposure_after = current_exposure + add_amount[:, None] - drop_amount[None, :]
    budget_return_mask = (
        return_mask
        & (exposure_after >= exposure_min - 1e-7)
        & (exposure_after <= exposure_max + 1e-7)
    )
    budget_return_pairs = int(budget_return_mask.sum())
    source_prefilter_mask = v82._source_prefilter_mask(
        budget_return_mask, candidates, selected, source_summary, current_exposure
    )
    source_prefilter_pairs = int(source_prefilter_mask.sum())

    current_by_family, cap_by_family = v82._source_maps(selected, source_summary)
    rows: list[dict[str, Any]] = []
    for candidate_pos, selected_pos in np.argwhere(source_prefilter_mask):
        add_row = candidates.iloc[int(candidate_pos)]
        drop_row = selected.iloc[int(selected_pos)]
        new_total = float(current_exposure + add_row["loan_amnt"] - drop_row["loan_amnt"])
        source_ok, min_slack, max_share, violations, first_family, first_source = (
            v82._exact_source_metrics(
                add_row, drop_row, current_by_family, cap_by_family, new_total
            )
        )
        if not source_ok:
            continue
        swapped_losses = (
            current_losses
            + losses[:, candidate_idx[int(candidate_pos)]]
            - losses[:, selected_idx[int(selected_pos)]]
        )
        cvar_after = v70._tail_cvar(swapped_losses)
        return_delta = float(
            add_row["mean_return_if_added_v90"] - drop_row["mean_return_if_dropped_v90"]
        )
        row: dict[str, Any] = {
            "policy_id": policy_id,
            "regime_v90": regime,
            "added_loan_id_v90": str(add_row["loan_id"]),
            "dropped_loan_id_v90": str(drop_row["loan_id"]),
            "added_loan_amount_v90": float(add_row["loan_amnt"]),
            "dropped_loan_amount_v90": float(drop_row["loan_amnt"]),
            "added_mean_return_v90": float(add_row["mean_return_if_added_v90"]),
            "dropped_mean_return_v90": float(drop_row["mean_return_if_dropped_v90"]),
            "return_delta_v90": return_delta,
            "objective_return_after_swap_v90": current_objective_return + return_delta,
            "exposure_after_swap_v90": new_total,
            "budget_swap_feasible_v90": True,
            "source_swap_feasible_v90": True,
            "source_min_slack_after_swap_v90": min_slack,
            "max_source_share_after_swap_v90": max_share,
            "source_cap_violations_after_swap_v90": violations,
            "first_source_block_family_v90": first_family,
            "first_source_block_id_v90": first_source,
            "loss_mean_after_swap_v90": float(swapped_losses.mean()),
            "cvar90_after_swap_v90": cvar_after,
            "cvar_swap_feasible_v90": cvar_after <= cvar_cap + 1e-7,
            "one_swap_improves_return_v90": return_delta > 1e-9 and cvar_after <= cvar_cap + 1e-7,
            "integer_screen_scope_v90": "post_v89_one_drop_one_add_whole_loan_swap",
            "claim_boundary_v90": (
                "post-v89 one-swap pricing only; not multi-swap or global proof"
            ),
        }
        for family in FAMILIES:
            row[f"added_{family}_v90"] = str(add_row[family])
            row[f"dropped_{family}_v90"] = str(drop_row[family])
        rows.append(row)

    pairs = pd.DataFrame(rows)
    cvar_feasible_pairs = int(pairs["cvar_swap_feasible_v90"].sum()) if not pairs.empty else 0
    improving_pairs = int(pairs["one_swap_improves_return_v90"].sum()) if not pairs.empty else 0
    best = pairs.sort_values("return_delta_v90", ascending=False).head(1)
    summary = pd.DataFrame(
        [
            {
                "policy_id": policy_id,
                "regime_v90": regime,
                "selected_rows_v90": int(len(selected)),
                "candidate_add_rows_v90": int(len(candidates)),
                "total_pair_rows_screened_v90": total_pairs,
                "return_improving_pair_rows_v90": return_pairs,
                "budget_return_feasible_pair_rows_v90": budget_return_pairs,
                "source_prefilter_pair_rows_v90": source_prefilter_pairs,
                "source_exact_pair_rows_v90": int(len(pairs)),
                "cvar_feasible_pair_rows_v90": cvar_feasible_pairs,
                "one_swap_improving_rows_v90": improving_pairs,
                "best_one_swap_return_delta_v90": float(best["return_delta_v90"].iloc[0])
                if not best.empty
                else np.nan,
                "best_one_swap_cvar90_after_v90": float(best["cvar90_after_swap_v90"].iloc[0])
                if not best.empty
                else np.nan,
                "current_exposure_v90": current_exposure,
                "exposure_min_v90": exposure_min,
                "exposure_max_v90": exposure_max,
                "cvar_cap_v90": cvar_cap,
                "current_objective_return_v90": current_objective_return,
                "post_repair_one_swap_local_optimality_cleared_v90": improving_pairs == 0,
                "full_universe_integer_optimality_claim_allowed_v90": False,
                "claim_boundary_v90": (
                    "post-v89 one-swap screen only; repeat repair/repricing if improvements remain"
                ),
            }
        ]
    )
    stage_summary = pd.DataFrame(
        [
            {"stage_v90": "all_pairs", "pair_rows_v90": total_pairs},
            {"stage_v90": "return_improving", "pair_rows_v90": return_pairs},
            {"stage_v90": "budget_return_feasible", "pair_rows_v90": budget_return_pairs},
            {"stage_v90": "source_prefilter_feasible", "pair_rows_v90": source_prefilter_pairs},
            {"stage_v90": "source_exact_feasible", "pair_rows_v90": int(len(pairs))},
            {"stage_v90": "cvar_feasible_improving", "pair_rows_v90": cvar_feasible_pairs},
        ]
    )
    stage_summary["claim_boundary_v90"] = "post-v89 one-swap screen stage count only"
    return pairs, summary, stage_summary


def _claim_blockers(summary: pd.DataFrame) -> pd.DataFrame:
    improving = int(summary["one_swap_improving_rows_v90"].iloc[0]) if not summary.empty else 0
    return pd.DataFrame(
        [
            {
                "blocker_id_v90": "post_v89_one_swap_improvement_found",
                "blocking_v90": improving > 0,
                "evidence_count_v90": improving,
                "required_next_artifact_v90": "paper4_v91_apply_next_swap_or_iterate.csv",
                "claim_boundary_v90": (
                    "feasible improving post-v89 one-swaps block local optimality"
                ),
            },
            {
                "blocker_id_v90": "multi_swap_integer_pricing_missing",
                "blocking_v90": True,
                "evidence_count_v90": 1,
                "required_next_artifact_v90": "paper4_v91_iterated_swap_or_milp_repair.csv",
                "claim_boundary_v90": "one-swap screen does not cover multi-loan exchanges",
            },
            {
                "blocker_id_v90": "global_integer_gap_certificate_missing",
                "blocking_v90": True,
                "evidence_count_v90": 1,
                "required_next_artifact_v90": "paper4_v91_global_integer_gap_protocol.csv",
                "claim_boundary_v90": "no branch-and-price/global full-universe integer certificate",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v90 post-v89 one-swap pricing screen.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v90_post_repair_one_swap_reprice.csv"
                ),
                "boundary": "One-swap local screen after v89 repair; not global integer proof.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v90 proves the v89 repaired portfolio is locally optimal.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v90_claim_blockers.csv"
                ),
                "boundary": "Allowed only if no feasible improving post-v89 one-swaps remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v90 proves full-universe integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v90_claim_blockers.csv"
                ),
                "boundary": "Requires iterated repair, multi-swap/branch-and-price or global gap certificate.",
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
                    "v90 reruns one-swap pricing after the v89 repair over all non-selected "
                    "comparable loans."
                ),
                "status": "iterate_if_improving_swaps_remain",
                "next_artifact": "paper4_v91_apply_next_swap_or_iterated_repair.csv",
                "success_condition": "no feasible improving post-v89 one-swaps remain",
                "last_wave": "v90",
                "execution_result": "post_v89_one_swap_reprice_completed",
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
    start = "<!-- V90_POST_V89_ONE_SWAP_REPRICE_START -->"
    end = "<!-- V90_POST_V89_ONE_SWAP_REPRICE_END -->"
    block = f"""
{start}

## Wave v90: Post-v89 One-Swap Repricing

Generated: {status["generated_at_utc"]}

### Objective

Rerun one-drop/one-add integer pricing after the v89 fourth repair, using all
non-selected loans from the comparable v55 universe as possible additions. This
tests whether the v89 candidate is one-swap locally optimal.

### Results

- Pair rows screened: `{status["total_pair_rows_screened_v90"]}`.
- Candidate add rows: `{status["candidate_add_rows_v90"]}`.
- Return-improving pairs: `{status["return_improving_pair_rows_v90"]}`.
- Exact source-feasible pairs: `{status["source_exact_pair_rows_v90"]}`.
- CVaR-feasible improving one-swaps: `{status["one_swap_improving_rows_v90"]}`.
- Best post-v89 one-swap return delta:
  `{status["best_one_swap_return_delta_v90"]}`.
- Post-v89 local optimality cleared:
  `{status["post_repair_one_swap_local_optimality_cleared_v90"]}`.

### Interpretation

v90 is the required re-pricing after v89 changed the portfolio. If additional
feasible improving one-swaps remain, the lab should continue the repair/reprice
loop; if it clears, the next blocker would still be multi-swap/global integer
evidence.

### Claim Impact

- Allowed: post-v89 one-swap pricing screen completed.
- Still prohibited: full-universe integer optimality, Paper Estrella
  replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v90 in the living notebook. Promote only after the repair loop terminates
and stronger integer/dynamic/promotion gates pass.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v90() -> dict[str, Any]:
    started = datetime.now(UTC)
    pairs, summary, stage_summary = _candidate_pairs()
    top_candidates = pairs.sort_values(
        ["one_swap_improves_return_v90", "return_delta_v90"],
        ascending=[False, False],
    ).head(200)
    blockers = _claim_blockers(summary)

    write_csv(TABLE_DIR / "paper4_v90_post_repair_one_swap_reprice.csv", pairs)
    write_csv(TABLE_DIR / "paper4_v90_post_repair_one_swap_top_candidates.csv", top_candidates)
    write_csv(TABLE_DIR / "paper4_v90_post_repair_one_swap_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v90_post_repair_one_swap_stage_summary.csv", stage_summary)
    write_csv(TABLE_DIR / "paper4_v90_claim_blockers.csv", blockers)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v90_post_repair_one_swap_reprice_executed",
                "allowed": True,
                "artifact": "paper4_v90_post_repair_one_swap_summary.csv",
                "boundary": "post-v89 one-swap screen only",
            },
            {
                "claim_id": "v90_post_repair_one_swap_local_optimality",
                "allowed": bool(
                    summary["post_repair_one_swap_local_optimality_cleared_v90"].iloc[0]
                ),
                "artifact": "paper4_v90_claim_blockers.csv",
                "boundary": "false if feasible improving swaps remain",
            },
            {
                "claim_id": "v90_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v90_claim_blockers.csv",
                "boundary": "multi-swap/global gap certificate missing",
            },
            {
                "claim_id": "v90_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v90_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v90_claim_matrix_delta.csv", claim_matrix)

    row = summary.iloc[0]
    status = {
        "phase": "v90_post_v89_one_swap_reprice",
        "schema_version": "2026-05-15.90",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "summary_rows_v90": int(len(summary)),
        "stage_summary_rows_v90": int(len(stage_summary)),
        "candidate_pair_rows_v90": int(len(pairs)),
        "top_candidate_rows_v90": int(len(top_candidates)),
        "claim_blocker_rows_v90": int(len(blockers)),
        "selected_rows_v90": int(row["selected_rows_v90"]),
        "candidate_add_rows_v90": int(row["candidate_add_rows_v90"]),
        "total_pair_rows_screened_v90": int(row["total_pair_rows_screened_v90"]),
        "return_improving_pair_rows_v90": int(row["return_improving_pair_rows_v90"]),
        "budget_return_feasible_pair_rows_v90": int(row["budget_return_feasible_pair_rows_v90"]),
        "source_prefilter_pair_rows_v90": int(row["source_prefilter_pair_rows_v90"]),
        "source_exact_pair_rows_v90": int(row["source_exact_pair_rows_v90"]),
        "cvar_feasible_pair_rows_v90": int(row["cvar_feasible_pair_rows_v90"]),
        "one_swap_improving_rows_v90": int(row["one_swap_improving_rows_v90"]),
        "best_one_swap_return_delta_v90": float(row["best_one_swap_return_delta_v90"]),
        "best_one_swap_cvar90_after_v90": float(row["best_one_swap_cvar90_after_v90"]),
        "post_repair_one_swap_local_optimality_cleared_v90": bool(
            row["post_repair_one_swap_local_optimality_cleared_v90"]
        ),
        "full_universe_integer_optimality_claim_allowed_v90": False,
        "paper1_promotion_allowed_v90": False,
        "paper4_working_champion_changed_v90": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v90 is post-v89 one-swap pricing only; global integer and promotion "
            "claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v90_status.json", status)
    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v90": build_v90()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
