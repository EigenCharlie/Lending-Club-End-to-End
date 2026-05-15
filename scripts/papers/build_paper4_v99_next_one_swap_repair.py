#!/usr/bin/env python3
"""Build Paper 4 v99 next one-swap repair artifacts.

v99 applies the best feasible post-v97 one-drop/one-add swap identified by v98.
It creates a ninth local repair candidate for follow-up pricing, not a final
champion or full-universe integer certificate.
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
NOTEBOOK = PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md"
FORBIDDEN_FINAL_PROMOTION = STATUS_DIR / "paper4_final_promotion.json"
FAMILIES = ["grade", "score_decile", "income_band", "dti_band", "period", "state_top20"]


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_csv(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_parquet(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def write_csv(path: Path, df: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(df) if isinstance(df, list) else df
    out.to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _best_v98_swap() -> pd.Series:
    top = read_csv("paper4_v98_post_repair_one_swap_top_candidates.csv")
    if top.empty:
        raise RuntimeError("Missing v98 top candidates; run v98 first.")
    return top.sort_values("return_delta_v98", ascending=False).iloc[0]


def _source_summary(
    universe: pd.DataFrame,
    portfolio: pd.DataFrame,
    source_caps: pd.DataFrame,
) -> pd.DataFrame:
    exposure = float(portfolio["loan_amnt"].sum())
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        caps = (
            source_caps.loc[source_caps["source_family"].astype(str).eq(family)]
            .set_index("source_id")["cap_share_v80"]
            .astype(float)
            .to_dict()
        )
        portfolio_by_source = portfolio.groupby(family, dropna=False)["loan_amnt"].sum()
        for source_id in sorted(universe[family].dropna().astype(str).unique()):
            source_exposure = float(portfolio_by_source.get(source_id, 0.0))
            share = source_exposure / max(exposure, 1.0)
            cap = float(caps.get(source_id, 1.0))
            rows.append(
                {
                    "portfolio_label_v99": "next_one_swap_repair_candidate",
                    "source_family": family,
                    "source_id": source_id,
                    "cap_share_v99": cap,
                    "source_exposure_v99": source_exposure,
                    "source_share_v99": share,
                    "source_slack_v99": cap - share,
                    "source_cap_violated_v99": share > cap + 1e-7,
                    "claim_boundary_v99": "ninth post-swap source diagnostic only",
                }
            )
    return pd.DataFrame(rows)


def _build_repair() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    previous = read_parquet("paper4_v97_next_one_swap_repair_allocations.parquet")
    previous_summary = read_csv("paper4_v97_next_one_swap_repair_summary.csv")
    prev_row = previous_summary.loc[
        previous_summary["portfolio_label_v97"].eq("next_one_swap_repair_candidate")
    ].iloc[0]
    source_caps = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_caps = source_caps.loc[
        source_caps["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    best = _best_v98_swap()
    if universe.empty or previous.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    add_id = str(best["added_loan_id_v98"])
    drop_id = str(best["dropped_loan_id_v98"])
    add_row = universe.loc[universe["loan_id"].astype(str).eq(add_id)].head(1).copy()
    if add_row.empty:
        raise RuntimeError(f"Could not find added loan {add_id} in v55 universe.")

    repaired = previous.loc[~previous["loan_id"].astype(str).eq(drop_id)].copy()
    if len(repaired) != len(previous) - 1:
        raise RuntimeError(f"Could not drop selected loan {drop_id} from v97 portfolio.")
    keep_cols = ["loan_id", "loan_amnt", *FAMILIES]
    repaired = pd.concat([repaired[keep_cols], add_row[keep_cols]], ignore_index=True)
    repaired["loan_id"] = repaired["loan_id"].astype(str)

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    repaired_idx = idx_by_id.loc[repaired["loan_id"].astype(str)].to_numpy()
    scenario_losses = losses[:, repaired_idx].sum(axis=1)
    repaired["mean_return_v99"] = mean_returns[repaired_idx]
    repaired["selected_v99"] = 1
    repaired["portfolio_label_v99"] = "next_one_swap_repair_candidate"
    repaired["repair_action_v99"] = np.where(
        repaired["loan_id"].eq(add_id), "added_from_v98_best_swap", "kept_from_v97"
    )
    repaired["claim_boundary_v99"] = (
        "ninth one-swap repair candidate only; requires post-repair repricing"
    )

    source_summary = _source_summary(universe, repaired, source_caps)
    objective_return = float(repaired["mean_return_v99"].sum())
    exposure = float(repaired["loan_amnt"].sum())
    cvar90 = v70._tail_cvar(scenario_losses)
    action = pd.DataFrame(
        [
            {
                "policy_id": str(best["policy_id"]),
                "regime_v99": str(best["regime_v98"]),
                "added_loan_id_v99": add_id,
                "dropped_loan_id_v99": drop_id,
                "added_loan_amount_v99": float(best["added_loan_amount_v98"]),
                "dropped_loan_amount_v99": float(best["dropped_loan_amount_v98"]),
                "added_mean_return_v99": float(best["added_mean_return_v98"]),
                "dropped_mean_return_v99": float(best["dropped_mean_return_v98"]),
                "return_delta_v99": float(best["return_delta_v98"]),
                "cvar90_after_repair_v99": cvar90,
                "exposure_after_repair_v99": exposure,
                "source_cap_violations_after_repair_v99": int(
                    source_summary["source_cap_violated_v99"].sum()
                ),
                "claim_boundary_v99": "best v98 swap applied; not post-repair optimality",
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "portfolio_label_v99": "next_one_swap_repair_candidate",
                "selected_rows_v99": int(len(repaired)),
                "portfolio_exposure_v99": exposure,
                "objective_return_v99": objective_return,
                "scenario_loss_mean_v99": float(scenario_losses.mean()),
                "scenario_loss_cvar90_v99": cvar90,
                "source_cap_violations_v99": int(source_summary["source_cap_violated_v99"].sum()),
                "max_source_share_v99": float(source_summary["source_share_v99"].max()),
                "min_source_slack_v99": float(source_summary["source_slack_v99"].min()),
                "delta_return_vs_v97_v99": objective_return
                - float(prev_row["objective_return_v97"]),
                "delta_cvar90_vs_v97_v99": cvar90 - float(prev_row["scenario_loss_cvar90_v97"]),
                "delta_exposure_vs_v97_v99": exposure - float(prev_row["portfolio_exposure_v97"]),
                "exposure_min_v99": float(prev_row["exposure_min_v97"]),
                "exposure_max_v99": float(prev_row["exposure_max_v97"]),
                "cvar_cap_v99": float(prev_row["cvar_cap_v97"]),
                "budget_feasible_v99": exposure >= float(prev_row["exposure_min_v97"]) - 1e-7
                and exposure <= float(prev_row["exposure_max_v97"]) + 1e-7,
                "cvar_feasible_v99": cvar90 <= float(prev_row["cvar_cap_v97"]) + 1e-7,
                "source_feasible_v99": not source_summary["source_cap_violated_v99"]
                .astype(bool)
                .any(),
                "repair_candidate_feasible_v99": bool(
                    exposure >= float(prev_row["exposure_min_v97"]) - 1e-7
                    and exposure <= float(prev_row["exposure_max_v97"]) + 1e-7
                    and cvar90 <= float(prev_row["cvar_cap_v97"]) + 1e-7
                    and not source_summary["source_cap_violated_v99"].astype(bool).any()
                ),
                "post_repair_one_swap_optimality_claim_allowed_v99": False,
                "full_universe_integer_optimality_claim_allowed_v99": False,
                "claim_boundary_v99": (
                    "ninth one-swap repair candidate; must rerun omitted-universe pricing"
                ),
            }
        ]
    )
    return repaired, summary, action, source_summary


def _claim_blockers(summary: pd.DataFrame) -> pd.DataFrame:
    feasible = (
        bool(summary["repair_candidate_feasible_v99"].iloc[0]) if not summary.empty else False
    )
    return pd.DataFrame(
        [
            {
                "blocker_id_v99": "next_one_swap_repair_candidate_created",
                "blocking_v99": not feasible,
                "evidence_count_v99": int(feasible),
                "required_next_artifact_v99": "paper4_v99_next_one_swap_repair_summary.csv",
                "claim_boundary_v99": "candidate exists only if budget/source/CVaR remain feasible",
            },
            {
                "blocker_id_v99": "post_repair_one_swap_repricing_missing",
                "blocking_v99": True,
                "evidence_count_v99": 1,
                "required_next_artifact_v99": "paper4_v100_post_repair_one_swap_reprice.csv",
                "claim_boundary_v99": "repair must be re-priced after changing the portfolio again",
            },
            {
                "blocker_id_v99": "multi_swap_integer_pricing_missing",
                "blocking_v99": True,
                "evidence_count_v99": 1,
                "required_next_artifact_v99": "paper4_v100_iterated_swap_or_milp_repair.csv",
                "claim_boundary_v99": "nine applied swaps do not cover multi-loan exchanges",
            },
            {
                "blocker_id_v99": "global_integer_gap_certificate_missing",
                "blocking_v99": True,
                "evidence_count_v99": 1,
                "required_next_artifact_v99": "paper4_v100_global_integer_gap_protocol.csv",
                "claim_boundary_v99": "no branch-and-price/global full-universe integer certificate",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v99 ninth one-swap repair candidate.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v99_next_one_swap_repair_summary.csv"
                ),
                "boundary": "Candidate generated from v98 best swap; requires post-repair re-pricing.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v99 repaired portfolio is post-repair locally optimal.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v99_claim_blockers.csv"
                ),
                "boundary": "Post-repair one-swap screen has not been rerun after v99.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v99 replaces Paper Estrella or proves full-universe integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v99_claim_blockers.csv"
                ),
                "boundary": "No promotion, no dynamic gate, no global gap certificate.",
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
                    "v99 applies the best v98 one-swap repair and recomputes portfolio metrics."
                ),
                "status": "post_repair_pricing_required",
                "next_artifact": "paper4_v100_post_repair_one_swap_reprice.csv",
                "success_condition": (
                    "post-v99 one-swap/integer pricing finds no feasible improving swaps"
                ),
                "last_wave": "v99",
                "execution_result": "ninth_one_swap_repair_candidate_created",
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
    start = "<!-- V99_NEXT_ONE_SWAP_REPAIR_START -->"
    end = "<!-- V99_NEXT_ONE_SWAP_REPAIR_END -->"
    block = f"""
{start}

## Wave v99: Ninth One-Swap Repair Candidate

Generated: {status["generated_at_utc"]}

### Objective

Apply the best feasible post-v97 one-drop/one-add swap found by v98 and
recompute portfolio return, budget, source and CVaR metrics. This continues the
local integer repair loop; it is not a final champion or optimality certificate.

### Results

- Added loan: `{status["added_loan_id_v99"]}`.
- Dropped loan: `{status["dropped_loan_id_v99"]}`.
- Selected rows after repair: `{status["selected_rows_v99"]}`.
- Return delta vs v97: `{status["delta_return_vs_v97_v99"]}`.
- CVaR90 delta vs v97: `{status["delta_cvar90_vs_v97_v99"]}`.
- Budget feasible: `{status["budget_feasible_v99"]}`.
- Source feasible: `{status["source_feasible_v99"]}`.
- CVaR feasible: `{status["cvar_feasible_v99"]}`.
- Post-repair local optimality claim allowed:
  `{status["post_repair_one_swap_optimality_claim_allowed_v99"]}`.

### Interpretation

v99 improves the v97 repaired candidate while preserving budget, source and
CVaR feasibility. The next required experiment is v100 post-repair one-swap
pricing because every repair changes the set of possible improving exchanges.

### Claim Impact

- Allowed: ninth one-swap repair candidate created.
- Still prohibited: post-repair local optimality, full-universe integer
  optimality, Paper Estrella replacement, final Paper 4 promotion and live
  deployment.

### Quarto Promotion Decision

Keep v99 in the living notebook. Promote only after the repair/reprice loop
terminates and stronger integer/dynamic/promotion gates pass.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v99() -> dict[str, Any]:
    started = datetime.now(UTC)
    allocations, summary, action, source_summary = _build_repair()
    blockers = _claim_blockers(summary)
    allocations.to_parquet(
        TABLE_DIR / "paper4_v99_next_one_swap_repair_allocations.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v99_next_one_swap_repair_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v99_next_one_swap_repair_action.csv", action)
    write_csv(TABLE_DIR / "paper4_v99_next_one_swap_repair_source_summary.csv", source_summary)
    write_csv(TABLE_DIR / "paper4_v99_claim_blockers.csv", blockers)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v99_next_one_swap_repair_executed",
                "allowed": True,
                "artifact": "paper4_v99_next_one_swap_repair_summary.csv",
                "boundary": "best v98 swap applied",
            },
            {
                "claim_id": "v99_repair_candidate_feasible",
                "allowed": bool(summary["repair_candidate_feasible_v99"].iloc[0]),
                "artifact": "paper4_v99_next_one_swap_repair_summary.csv",
                "boundary": "budget/source/CVaR feasibility only",
            },
            {
                "claim_id": "v99_post_repair_one_swap_optimality",
                "allowed": False,
                "artifact": "paper4_v99_claim_blockers.csv",
                "boundary": "post-repair pricing not rerun after v99",
            },
            {
                "claim_id": "v99_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v99_claim_blockers.csv",
                "boundary": "multi-swap/global gap certificate missing",
            },
            {
                "claim_id": "v99_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v99_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v99_claim_matrix_delta.csv", claim_matrix)

    row = summary.iloc[0]
    action_row = action.iloc[0]
    status = {
        "phase": "v99_next_one_swap_repair",
        "schema_version": "2026-05-15.99",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "allocation_rows_v99": int(len(allocations)),
        "summary_rows_v99": int(len(summary)),
        "action_rows_v99": int(len(action)),
        "source_summary_rows_v99": int(len(source_summary)),
        "claim_blocker_rows_v99": int(len(blockers)),
        "added_loan_id_v99": str(action_row["added_loan_id_v99"]),
        "dropped_loan_id_v99": str(action_row["dropped_loan_id_v99"]),
        "selected_rows_v99": int(row["selected_rows_v99"]),
        "portfolio_exposure_v99": float(row["portfolio_exposure_v99"]),
        "objective_return_v99": float(row["objective_return_v99"]),
        "scenario_loss_cvar90_v99": float(row["scenario_loss_cvar90_v99"]),
        "source_cap_violations_v99": int(row["source_cap_violations_v99"]),
        "delta_return_vs_v97_v99": float(row["delta_return_vs_v97_v99"]),
        "delta_cvar90_vs_v97_v99": float(row["delta_cvar90_vs_v97_v99"]),
        "delta_exposure_vs_v97_v99": float(row["delta_exposure_vs_v97_v99"]),
        "budget_feasible_v99": bool(row["budget_feasible_v99"]),
        "source_feasible_v99": bool(row["source_feasible_v99"]),
        "cvar_feasible_v99": bool(row["cvar_feasible_v99"]),
        "repair_candidate_feasible_v99": bool(row["repair_candidate_feasible_v99"]),
        "post_repair_one_swap_optimality_claim_allowed_v99": False,
        "full_universe_integer_optimality_claim_allowed_v99": False,
        "paper1_promotion_allowed_v99": False,
        "paper4_working_champion_changed_v99": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v99 creates a ninth repaired candidate only; post-repair pricing and global "
            "integer claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v99_status.json", status)
    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v99": build_v99()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
