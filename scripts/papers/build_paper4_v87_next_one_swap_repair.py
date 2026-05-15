#!/usr/bin/env python3
"""Build Paper 4 v87 next one-swap repair artifacts.

v87 applies the best feasible post-repair one-drop/one-add swap identified by
v86 and recomputes portfolio metrics. This is the third local integer repair
in the v82-v86 loop. It is not a final champion, post-repair local optimality
certificate, or full-universe integer proof.
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


def _best_v86_swap() -> pd.Series:
    top = read_csv("paper4_v86_post_repair_one_swap_top_candidates.csv")
    if top.empty:
        raise RuntimeError("Missing v86 top candidates; run v86 first.")
    return top.sort_values("return_delta_v86", ascending=False).iloc[0]


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
                    "portfolio_label_v87": "next_one_swap_repair_candidate",
                    "source_family": family,
                    "source_id": source_id,
                    "cap_share_v87": cap,
                    "source_exposure_v87": source_exposure,
                    "source_share_v87": share,
                    "source_slack_v87": cap - share,
                    "source_cap_violated_v87": share > cap + 1e-7,
                    "claim_boundary_v87": "third post-swap source diagnostic only",
                }
            )
    return pd.DataFrame(rows)


def _build_repair() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    previous = read_parquet("paper4_v85_next_one_swap_repair_allocations.parquet")
    summary_v85 = read_csv("paper4_v85_next_one_swap_repair_summary.csv")
    v85_row = summary_v85.loc[
        summary_v85["portfolio_label_v85"].eq("next_one_swap_repair_candidate")
    ].iloc[0]
    source_caps = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_caps = source_caps.loc[
        source_caps["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    best = _best_v86_swap()
    if universe.empty or previous.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    add_id = str(best["added_loan_id_v86"])
    drop_id = str(best["dropped_loan_id_v86"])
    add_row = universe.loc[universe["loan_id"].astype(str).eq(add_id)].head(1).copy()
    if add_row.empty:
        raise RuntimeError(f"Could not find added loan {add_id} in v55 universe.")

    repaired = previous.loc[~previous["loan_id"].astype(str).eq(drop_id)].copy()
    if len(repaired) != len(previous) - 1:
        raise RuntimeError(f"Could not drop selected loan {drop_id} from v85 portfolio.")
    keep_cols = ["loan_id", "loan_amnt", *FAMILIES]
    repaired = pd.concat([repaired[keep_cols], add_row[keep_cols]], ignore_index=True)
    repaired["loan_id"] = repaired["loan_id"].astype(str)

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    repaired_idx = idx_by_id.loc[repaired["loan_id"].astype(str)].to_numpy()
    scenario_losses = losses[:, repaired_idx].sum(axis=1)
    repaired["mean_return_v87"] = mean_returns[repaired_idx]
    repaired["selected_v87"] = 1
    repaired["portfolio_label_v87"] = "next_one_swap_repair_candidate"
    repaired["repair_action_v87"] = np.where(
        repaired["loan_id"].eq(add_id), "added_from_v86_best_swap", "kept_from_v85"
    )
    repaired["claim_boundary_v87"] = (
        "third one-swap repair candidate only; requires post-repair repricing"
    )

    source_summary = _source_summary(universe, repaired, source_caps)
    objective_return = float(repaired["mean_return_v87"].sum())
    exposure = float(repaired["loan_amnt"].sum())
    cvar90 = v70._tail_cvar(scenario_losses)
    action = pd.DataFrame(
        [
            {
                "policy_id": str(best["policy_id"]),
                "regime_v87": str(best["regime_v86"]),
                "added_loan_id_v87": add_id,
                "dropped_loan_id_v87": drop_id,
                "added_loan_amount_v87": float(best["added_loan_amount_v86"]),
                "dropped_loan_amount_v87": float(best["dropped_loan_amount_v86"]),
                "added_mean_return_v87": float(best["added_mean_return_v86"]),
                "dropped_mean_return_v87": float(best["dropped_mean_return_v86"]),
                "return_delta_v87": float(best["return_delta_v86"]),
                "cvar90_after_repair_v87": cvar90,
                "exposure_after_repair_v87": exposure,
                "source_cap_violations_after_repair_v87": int(
                    source_summary["source_cap_violated_v87"].sum()
                ),
                "claim_boundary_v87": "best v86 swap applied; not post-repair optimality",
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "portfolio_label_v87": "next_one_swap_repair_candidate",
                "selected_rows_v87": int(len(repaired)),
                "portfolio_exposure_v87": exposure,
                "objective_return_v87": objective_return,
                "scenario_loss_mean_v87": float(scenario_losses.mean()),
                "scenario_loss_cvar90_v87": cvar90,
                "source_cap_violations_v87": int(source_summary["source_cap_violated_v87"].sum()),
                "max_source_share_v87": float(source_summary["source_share_v87"].max()),
                "min_source_slack_v87": float(source_summary["source_slack_v87"].min()),
                "delta_return_vs_v85_v87": objective_return
                - float(v85_row["objective_return_v85"]),
                "delta_cvar90_vs_v85_v87": cvar90 - float(v85_row["scenario_loss_cvar90_v85"]),
                "delta_exposure_vs_v85_v87": exposure - float(v85_row["portfolio_exposure_v85"]),
                "exposure_min_v87": float(v85_row["exposure_min_v85"]),
                "exposure_max_v87": float(v85_row["exposure_max_v85"]),
                "cvar_cap_v87": float(v85_row["cvar_cap_v85"]),
                "budget_feasible_v87": exposure >= float(v85_row["exposure_min_v85"]) - 1e-7
                and exposure <= float(v85_row["exposure_max_v85"]) + 1e-7,
                "cvar_feasible_v87": cvar90 <= float(v85_row["cvar_cap_v85"]) + 1e-7,
                "source_feasible_v87": not source_summary["source_cap_violated_v87"]
                .astype(bool)
                .any(),
                "repair_candidate_feasible_v87": bool(
                    exposure >= float(v85_row["exposure_min_v85"]) - 1e-7
                    and exposure <= float(v85_row["exposure_max_v85"]) + 1e-7
                    and cvar90 <= float(v85_row["cvar_cap_v85"]) + 1e-7
                    and not source_summary["source_cap_violated_v87"].astype(bool).any()
                ),
                "post_repair_one_swap_optimality_claim_allowed_v87": False,
                "full_universe_integer_optimality_claim_allowed_v87": False,
                "claim_boundary_v87": (
                    "third one-swap repair candidate; must rerun omitted-universe pricing"
                ),
            }
        ]
    )
    return repaired, summary, action, source_summary


def _claim_blockers(summary: pd.DataFrame) -> pd.DataFrame:
    feasible = (
        bool(summary["repair_candidate_feasible_v87"].iloc[0]) if not summary.empty else False
    )
    return pd.DataFrame(
        [
            {
                "blocker_id_v87": "next_one_swap_repair_candidate_created",
                "blocking_v87": not feasible,
                "evidence_count_v87": int(feasible),
                "required_next_artifact_v87": "paper4_v87_next_one_swap_repair_summary.csv",
                "claim_boundary_v87": "candidate exists only if budget/source/CVaR remain feasible",
            },
            {
                "blocker_id_v87": "post_repair_one_swap_repricing_missing",
                "blocking_v87": True,
                "evidence_count_v87": 1,
                "required_next_artifact_v87": "paper4_v88_post_repair_one_swap_reprice.csv",
                "claim_boundary_v87": "repair must be re-priced after changing the portfolio again",
            },
            {
                "blocker_id_v87": "multi_swap_integer_pricing_missing",
                "blocking_v87": True,
                "evidence_count_v87": 1,
                "required_next_artifact_v87": "paper4_v88_iterated_swap_or_milp_repair.csv",
                "claim_boundary_v87": "three applied swaps do not cover multi-loan exchanges",
            },
            {
                "blocker_id_v87": "global_integer_gap_certificate_missing",
                "blocking_v87": True,
                "evidence_count_v87": 1,
                "required_next_artifact_v87": "paper4_v88_global_integer_gap_protocol.csv",
                "claim_boundary_v87": "no branch-and-price/global full-universe integer certificate",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v87 third one-swap repair candidate.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v87_next_one_swap_repair_summary.csv"
                ),
                "boundary": "Candidate generated from v86 best swap; requires post-repair re-pricing.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v87 repaired portfolio is post-repair locally optimal.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v87_claim_blockers.csv"
                ),
                "boundary": "Post-repair one-swap screen has not been rerun after v87.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v87 replaces Paper Estrella or proves full-universe integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v87_claim_blockers.csv"
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
                    "v87 applies the best v86 one-swap repair and recomputes portfolio metrics."
                ),
                "status": "post_repair_pricing_required",
                "next_artifact": "paper4_v88_post_repair_one_swap_reprice.csv",
                "success_condition": (
                    "post-v87 one-swap/integer pricing finds no feasible improving swaps"
                ),
                "last_wave": "v87",
                "execution_result": "third_one_swap_repair_candidate_created",
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
    start = "<!-- V87_NEXT_ONE_SWAP_REPAIR_START -->"
    end = "<!-- V87_NEXT_ONE_SWAP_REPAIR_END -->"
    block = f"""
{start}

## Wave v87: Third One-Swap Repair Candidate

Generated: {status["generated_at_utc"]}

### Objective

Apply the best feasible post-repair one-drop/one-add swap found by v86 and
recompute portfolio return, budget, source and CVaR metrics. This continues the
local integer repair loop; it is not a final champion or optimality certificate.

### Results

- Added loan: `{status["added_loan_id_v87"]}`.
- Dropped loan: `{status["dropped_loan_id_v87"]}`.
- Selected rows after repair: `{status["selected_rows_v87"]}`.
- Return delta vs v85: `{status["delta_return_vs_v85_v87"]}`.
- CVaR90 delta vs v85: `{status["delta_cvar90_vs_v85_v87"]}`.
- Budget feasible: `{status["budget_feasible_v87"]}`.
- Source feasible: `{status["source_feasible_v87"]}`.
- CVaR feasible: `{status["cvar_feasible_v87"]}`.
- Post-repair local optimality claim allowed:
  `{status["post_repair_one_swap_optimality_claim_allowed_v87"]}`.

### Interpretation

v87 improves the v85 repaired candidate while preserving budget, source and
CVaR feasibility. The next required experiment is v88 post-repair one-swap
pricing because every repair changes the set of possible improving exchanges.

### Claim Impact

- Allowed: third one-swap repair candidate created.
- Still prohibited: post-repair local optimality, full-universe integer
  optimality, Paper Estrella replacement, final Paper 4 promotion and live
  deployment.

### Quarto Promotion Decision

Keep v87 in the living notebook. Promote only after the repair/reprice loop
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


def build_v87() -> dict[str, Any]:
    started = datetime.now(UTC)
    allocations, summary, action, source_summary = _build_repair()
    blockers = _claim_blockers(summary)
    allocations.to_parquet(
        TABLE_DIR / "paper4_v87_next_one_swap_repair_allocations.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v87_next_one_swap_repair_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v87_next_one_swap_repair_action.csv", action)
    write_csv(TABLE_DIR / "paper4_v87_next_one_swap_repair_source_summary.csv", source_summary)
    write_csv(TABLE_DIR / "paper4_v87_claim_blockers.csv", blockers)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v87_next_one_swap_repair_executed",
                "allowed": True,
                "artifact": "paper4_v87_next_one_swap_repair_summary.csv",
                "boundary": "best v86 swap applied",
            },
            {
                "claim_id": "v87_repair_candidate_feasible",
                "allowed": bool(summary["repair_candidate_feasible_v87"].iloc[0]),
                "artifact": "paper4_v87_next_one_swap_repair_summary.csv",
                "boundary": "budget/source/CVaR feasibility only",
            },
            {
                "claim_id": "v87_post_repair_one_swap_optimality",
                "allowed": False,
                "artifact": "paper4_v87_claim_blockers.csv",
                "boundary": "post-repair pricing not rerun after v87",
            },
            {
                "claim_id": "v87_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v87_claim_blockers.csv",
                "boundary": "multi-swap/global gap certificate missing",
            },
            {
                "claim_id": "v87_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v87_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v87_claim_matrix_delta.csv", claim_matrix)

    row = summary.iloc[0]
    action_row = action.iloc[0]
    status = {
        "phase": "v87_next_one_swap_repair",
        "schema_version": "2026-05-15.87",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "allocation_rows_v87": int(len(allocations)),
        "summary_rows_v87": int(len(summary)),
        "action_rows_v87": int(len(action)),
        "source_summary_rows_v87": int(len(source_summary)),
        "claim_blocker_rows_v87": int(len(blockers)),
        "added_loan_id_v87": str(action_row["added_loan_id_v87"]),
        "dropped_loan_id_v87": str(action_row["dropped_loan_id_v87"]),
        "selected_rows_v87": int(row["selected_rows_v87"]),
        "portfolio_exposure_v87": float(row["portfolio_exposure_v87"]),
        "objective_return_v87": float(row["objective_return_v87"]),
        "scenario_loss_cvar90_v87": float(row["scenario_loss_cvar90_v87"]),
        "source_cap_violations_v87": int(row["source_cap_violations_v87"]),
        "delta_return_vs_v85_v87": float(row["delta_return_vs_v85_v87"]),
        "delta_cvar90_vs_v85_v87": float(row["delta_cvar90_vs_v85_v87"]),
        "delta_exposure_vs_v85_v87": float(row["delta_exposure_vs_v85_v87"]),
        "budget_feasible_v87": bool(row["budget_feasible_v87"]),
        "source_feasible_v87": bool(row["source_feasible_v87"]),
        "cvar_feasible_v87": bool(row["cvar_feasible_v87"]),
        "repair_candidate_feasible_v87": bool(row["repair_candidate_feasible_v87"]),
        "post_repair_one_swap_optimality_claim_allowed_v87": False,
        "full_universe_integer_optimality_claim_allowed_v87": False,
        "paper1_promotion_allowed_v87": False,
        "paper4_working_champion_changed_v87": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v87 creates a third repaired candidate only; post-repair pricing and global "
            "integer claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v87_status.json", status)
    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v87": build_v87()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
