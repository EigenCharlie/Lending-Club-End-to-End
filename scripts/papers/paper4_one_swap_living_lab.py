"""Shared builders for Paper 4 one-swap living-lab waves."""

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


def _top_swap(pricing_version: int) -> pd.Series:
    top = read_csv(f"paper4_v{pricing_version}_post_repair_one_swap_top_candidates.csv")
    if top.empty:
        raise RuntimeError(f"Missing v{pricing_version} top candidates.")
    return top.sort_values(f"return_delta_v{pricing_version}", ascending=False).iloc[0]


def _reprice_pair_columns(version: int) -> list[str]:
    columns = [
        "policy_id",
        f"regime_v{version}",
        f"added_loan_id_v{version}",
        f"dropped_loan_id_v{version}",
        f"added_loan_amount_v{version}",
        f"dropped_loan_amount_v{version}",
        f"added_mean_return_v{version}",
        f"dropped_mean_return_v{version}",
        f"return_delta_v{version}",
        f"objective_return_after_swap_v{version}",
        f"exposure_after_swap_v{version}",
        f"budget_swap_feasible_v{version}",
        f"source_swap_feasible_v{version}",
        f"source_min_slack_after_swap_v{version}",
        f"max_source_share_after_swap_v{version}",
        f"source_cap_violations_after_swap_v{version}",
        f"first_source_block_family_v{version}",
        f"first_source_block_id_v{version}",
        f"loss_mean_after_swap_v{version}",
        f"cvar90_after_swap_v{version}",
        f"cvar_swap_feasible_v{version}",
        f"one_swap_improves_return_v{version}",
        f"integer_screen_scope_v{version}",
        f"claim_boundary_v{version}",
    ]
    for family in FAMILIES:
        columns.extend([f"added_{family}_v{version}", f"dropped_{family}_v{version}"])
    return columns


def _source_summary(
    *,
    universe: pd.DataFrame,
    portfolio: pd.DataFrame,
    source_caps: pd.DataFrame,
    version: int,
    ordinal: str,
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
                    f"portfolio_label_v{version}": "next_one_swap_repair_candidate",
                    "source_family": family,
                    "source_id": source_id,
                    f"cap_share_v{version}": cap,
                    f"source_exposure_v{version}": source_exposure,
                    f"source_share_v{version}": share,
                    f"source_slack_v{version}": cap - share,
                    f"source_cap_violated_v{version}": share > cap + 1e-7,
                    f"claim_boundary_v{version}": f"{ordinal} post-swap source diagnostic only",
                }
            )
    return pd.DataFrame(rows)


def _append_or_replace_block(path: Path, start: str, end: str, block: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8")


def _indefinite_article(text: str) -> str:
    return "an" if text[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def build_repair_wave(
    *,
    version: int,
    previous_repair_version: int,
    pricing_version: int,
    ordinal: str,
    next_reprice_version: int,
) -> dict[str, Any]:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    previous = read_parquet(
        f"paper4_v{previous_repair_version}_next_one_swap_repair_allocations.parquet"
    )
    previous_summary = read_csv(
        f"paper4_v{previous_repair_version}_next_one_swap_repair_summary.csv"
    )
    prev_row = previous_summary.loc[
        previous_summary[f"portfolio_label_v{previous_repair_version}"].eq(
            "next_one_swap_repair_candidate"
        )
    ].iloc[0]
    source_caps = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_caps = source_caps.loc[
        source_caps["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    best = _top_swap(pricing_version)
    if universe.empty or previous.empty:
        raise RuntimeError(f"Missing v{previous_repair_version} portfolio or v55 universe.")

    add_id = str(best[f"added_loan_id_v{pricing_version}"])
    drop_id = str(best[f"dropped_loan_id_v{pricing_version}"])
    add_row = universe.loc[universe["loan_id"].astype(str).eq(add_id)].head(1).copy()
    if add_row.empty:
        raise RuntimeError(f"Could not find added loan {add_id} in v55 universe.")

    repaired = previous.loc[~previous["loan_id"].astype(str).eq(drop_id)].copy()
    if len(repaired) != len(previous) - 1:
        raise RuntimeError(
            f"Could not drop selected loan {drop_id} from v{previous_repair_version} portfolio."
        )
    keep_cols = ["loan_id", "loan_amnt", *FAMILIES]
    repaired = pd.concat([repaired[keep_cols], add_row[keep_cols]], ignore_index=True)
    repaired["loan_id"] = repaired["loan_id"].astype(str)

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    repaired_idx = idx_by_id.loc[repaired["loan_id"].astype(str)].to_numpy()
    scenario_losses = losses[:, repaired_idx].sum(axis=1)
    repaired[f"mean_return_v{version}"] = mean_returns[repaired_idx]
    repaired[f"selected_v{version}"] = 1
    repaired[f"portfolio_label_v{version}"] = "next_one_swap_repair_candidate"
    repaired[f"repair_action_v{version}"] = np.where(
        repaired["loan_id"].eq(add_id),
        f"added_from_v{pricing_version}_best_swap",
        f"kept_from_v{previous_repair_version}",
    )
    repaired[f"claim_boundary_v{version}"] = (
        f"{ordinal} one-swap repair candidate only; requires post-repair repricing"
    )

    source_summary = _source_summary(
        universe=universe,
        portfolio=repaired,
        source_caps=source_caps,
        version=version,
        ordinal=ordinal,
    )
    objective_return = float(repaired[f"mean_return_v{version}"].sum())
    exposure = float(repaired["loan_amnt"].sum())
    cvar90 = v70._tail_cvar(scenario_losses)
    action = pd.DataFrame(
        [
            {
                "policy_id": str(best["policy_id"]),
                f"regime_v{version}": str(best[f"regime_v{pricing_version}"]),
                f"added_loan_id_v{version}": add_id,
                f"dropped_loan_id_v{version}": drop_id,
                f"added_loan_amount_v{version}": float(
                    best[f"added_loan_amount_v{pricing_version}"]
                ),
                f"dropped_loan_amount_v{version}": float(
                    best[f"dropped_loan_amount_v{pricing_version}"]
                ),
                f"added_mean_return_v{version}": float(
                    best[f"added_mean_return_v{pricing_version}"]
                ),
                f"dropped_mean_return_v{version}": float(
                    best[f"dropped_mean_return_v{pricing_version}"]
                ),
                f"return_delta_v{version}": float(best[f"return_delta_v{pricing_version}"]),
                f"cvar90_after_repair_v{version}": cvar90,
                f"exposure_after_repair_v{version}": exposure,
                f"source_cap_violations_after_repair_v{version}": int(
                    source_summary[f"source_cap_violated_v{version}"].sum()
                ),
                f"claim_boundary_v{version}": (
                    f"best v{pricing_version} swap applied; not post-repair optimality"
                ),
            }
        ]
    )
    budget_feasible = exposure >= float(prev_row[f"exposure_min_v{previous_repair_version}"]) - 1e-7
    budget_feasible = (
        budget_feasible
        and exposure <= float(prev_row[f"exposure_max_v{previous_repair_version}"]) + 1e-7
    )
    cvar_feasible = cvar90 <= float(prev_row[f"cvar_cap_v{previous_repair_version}"]) + 1e-7
    source_feasible = not source_summary[f"source_cap_violated_v{version}"].astype(bool).any()
    summary = pd.DataFrame(
        [
            {
                f"portfolio_label_v{version}": "next_one_swap_repair_candidate",
                f"selected_rows_v{version}": int(len(repaired)),
                f"portfolio_exposure_v{version}": exposure,
                f"objective_return_v{version}": objective_return,
                f"scenario_loss_mean_v{version}": float(scenario_losses.mean()),
                f"scenario_loss_cvar90_v{version}": cvar90,
                f"source_cap_violations_v{version}": int(
                    source_summary[f"source_cap_violated_v{version}"].sum()
                ),
                f"max_source_share_v{version}": float(
                    source_summary[f"source_share_v{version}"].max()
                ),
                f"min_source_slack_v{version}": float(
                    source_summary[f"source_slack_v{version}"].min()
                ),
                f"delta_return_vs_v{previous_repair_version}_v{version}": objective_return
                - float(prev_row[f"objective_return_v{previous_repair_version}"]),
                f"delta_cvar90_vs_v{previous_repair_version}_v{version}": cvar90
                - float(prev_row[f"scenario_loss_cvar90_v{previous_repair_version}"]),
                f"delta_exposure_vs_v{previous_repair_version}_v{version}": exposure
                - float(prev_row[f"portfolio_exposure_v{previous_repair_version}"]),
                f"exposure_min_v{version}": float(
                    prev_row[f"exposure_min_v{previous_repair_version}"]
                ),
                f"exposure_max_v{version}": float(
                    prev_row[f"exposure_max_v{previous_repair_version}"]
                ),
                f"cvar_cap_v{version}": float(prev_row[f"cvar_cap_v{previous_repair_version}"]),
                f"budget_feasible_v{version}": budget_feasible,
                f"cvar_feasible_v{version}": cvar_feasible,
                f"source_feasible_v{version}": source_feasible,
                f"repair_candidate_feasible_v{version}": bool(
                    budget_feasible and cvar_feasible and source_feasible
                ),
                f"post_repair_one_swap_optimality_claim_allowed_v{version}": False,
                f"full_universe_integer_optimality_claim_allowed_v{version}": False,
                f"claim_boundary_v{version}": (
                    f"{ordinal} one-swap repair candidate; must rerun omitted-universe pricing"
                ),
            }
        ]
    )
    feasible = bool(summary[f"repair_candidate_feasible_v{version}"].iloc[0])
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{version}": "next_one_swap_repair_candidate_created",
                f"blocking_v{version}": not feasible,
                f"evidence_count_v{version}": int(feasible),
                f"required_next_artifact_v{version}": (
                    f"paper4_v{version}_next_one_swap_repair_summary.csv"
                ),
                f"claim_boundary_v{version}": (
                    "candidate exists only if budget/source/CVaR remain feasible"
                ),
            },
            {
                f"blocker_id_v{version}": "post_repair_one_swap_repricing_missing",
                f"blocking_v{version}": True,
                f"evidence_count_v{version}": 1,
                f"required_next_artifact_v{version}": (
                    f"paper4_v{next_reprice_version}_post_repair_one_swap_reprice.csv"
                ),
                f"claim_boundary_v{version}": (
                    "repair must be re-priced after changing the portfolio again"
                ),
            },
            {
                f"blocker_id_v{version}": "multi_swap_integer_pricing_missing",
                f"blocking_v{version}": True,
                f"evidence_count_v{version}": 1,
                f"required_next_artifact_v{version}": (
                    f"paper4_v{next_reprice_version}_iterated_swap_or_milp_repair.csv"
                ),
                f"claim_boundary_v{version}": (
                    f"{ordinal} applied swap sequence does not cover multi-loan exchanges"
                ),
            },
            {
                f"blocker_id_v{version}": "global_integer_gap_certificate_missing",
                f"blocking_v{version}": True,
                f"evidence_count_v{version}": 1,
                f"required_next_artifact_v{version}": (
                    f"paper4_v{next_reprice_version}_global_integer_gap_protocol.csv"
                ),
                f"claim_boundary_v{version}": (
                    "no branch-and-price/global full-universe integer certificate"
                ),
            },
        ]
    )

    repaired.to_parquet(
        TABLE_DIR / f"paper4_v{version}_next_one_swap_repair_allocations.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / f"paper4_v{version}_next_one_swap_repair_summary.csv", summary)
    write_csv(TABLE_DIR / f"paper4_v{version}_next_one_swap_repair_action.csv", action)
    write_csv(
        TABLE_DIR / f"paper4_v{version}_next_one_swap_repair_source_summary.csv",
        source_summary,
    )
    write_csv(TABLE_DIR / f"paper4_v{version}_claim_blockers.csv", blockers)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": f"v{version}_next_one_swap_repair_executed",
                "allowed": True,
                "artifact": f"paper4_v{version}_next_one_swap_repair_summary.csv",
                "boundary": f"best v{pricing_version} swap applied",
            },
            {
                "claim_id": f"v{version}_repair_candidate_feasible",
                "allowed": feasible,
                "artifact": f"paper4_v{version}_next_one_swap_repair_summary.csv",
                "boundary": "budget/source/CVaR feasibility only",
            },
            {
                "claim_id": f"v{version}_post_repair_one_swap_optimality",
                "allowed": False,
                "artifact": f"paper4_v{version}_claim_blockers.csv",
                "boundary": f"post-repair pricing not rerun after v{version}",
            },
            {
                "claim_id": f"v{version}_full_universe_integer_optimality",
                "allowed": False,
                "artifact": f"paper4_v{version}_claim_blockers.csv",
                "boundary": "multi-swap/global gap certificate missing",
            },
            {
                "claim_id": f"v{version}_paper1_or_final_promotion",
                "allowed": False,
                "artifact": f"paper4_v{version}_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / f"paper4_v{version}_claim_matrix_delta.csv", claim_matrix)
    _update_repair_claim_boundaries(version, ordinal)
    _update_repair_backlog(version, pricing_version, next_reprice_version, ordinal)

    row = summary.iloc[0]
    action_row = action.iloc[0]
    status = {
        "phase": f"v{version}_next_one_swap_repair",
        "schema_version": f"2026-05-15.{version}",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        f"allocation_rows_v{version}": int(len(repaired)),
        f"summary_rows_v{version}": int(len(summary)),
        f"action_rows_v{version}": int(len(action)),
        f"source_summary_rows_v{version}": int(len(source_summary)),
        f"claim_blocker_rows_v{version}": int(len(blockers)),
        f"added_loan_id_v{version}": str(action_row[f"added_loan_id_v{version}"]),
        f"dropped_loan_id_v{version}": str(action_row[f"dropped_loan_id_v{version}"]),
        f"selected_rows_v{version}": int(row[f"selected_rows_v{version}"]),
        f"portfolio_exposure_v{version}": float(row[f"portfolio_exposure_v{version}"]),
        f"objective_return_v{version}": float(row[f"objective_return_v{version}"]),
        f"scenario_loss_cvar90_v{version}": float(row[f"scenario_loss_cvar90_v{version}"]),
        f"source_cap_violations_v{version}": int(row[f"source_cap_violations_v{version}"]),
        f"delta_return_vs_v{previous_repair_version}_v{version}": float(
            row[f"delta_return_vs_v{previous_repair_version}_v{version}"]
        ),
        f"delta_cvar90_vs_v{previous_repair_version}_v{version}": float(
            row[f"delta_cvar90_vs_v{previous_repair_version}_v{version}"]
        ),
        f"delta_exposure_vs_v{previous_repair_version}_v{version}": float(
            row[f"delta_exposure_vs_v{previous_repair_version}_v{version}"]
        ),
        f"budget_feasible_v{version}": bool(row[f"budget_feasible_v{version}"]),
        f"source_feasible_v{version}": bool(row[f"source_feasible_v{version}"]),
        f"cvar_feasible_v{version}": bool(row[f"cvar_feasible_v{version}"]),
        f"repair_candidate_feasible_v{version}": bool(row[f"repair_candidate_feasible_v{version}"]),
        f"post_repair_one_swap_optimality_claim_allowed_v{version}": False,
        f"full_universe_integer_optimality_claim_allowed_v{version}": False,
        f"paper1_promotion_allowed_v{version}": False,
        f"paper4_working_champion_changed_v{version}": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            f"v{version} creates {_indefinite_article(ordinal)} {ordinal} repaired "
            "candidate only; post-repair pricing and global integer claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / f"paper4_v{version}_status.json", status)
    _update_repair_notebook(
        status=status,
        version=version,
        previous_repair_version=previous_repair_version,
        pricing_version=pricing_version,
        next_reprice_version=next_reprice_version,
        ordinal=ordinal,
    )
    return status


def _update_repair_claim_boundaries(version: int, ordinal: str) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": f"Paper 4 has a v{version} {ordinal} one-swap repair candidate.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    f"paper4_v{version}_next_one_swap_repair_summary.csv"
                ),
                "boundary": (
                    f"Candidate generated from v{version - 1} best swap; requires "
                    "post-repair re-pricing."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": f"v{version} repaired portfolio is post-repair locally optimal.",
                "allowed": False,
                "evidence_artifact": (
                    f"reports/paper_material/paper4/tables/paper4_v{version}_claim_blockers.csv"
                ),
                "boundary": f"Post-repair one-swap screen has not been rerun after v{version}.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    f"v{version} replaces Paper Estrella or proves full-universe integer optimality."
                ),
                "allowed": False,
                "evidence_artifact": (
                    f"reports/paper_material/paper4/tables/paper4_v{version}_claim_blockers.csv"
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


def _update_repair_backlog(
    version: int,
    pricing_version: int,
    next_reprice_version: int,
    ordinal: str,
) -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "CVaR/OCE",
                "executable_item": (
                    f"v{version} applies the best v{pricing_version} one-swap repair "
                    "and recomputes portfolio metrics."
                ),
                "status": "post_repair_pricing_required",
                "next_artifact": f"paper4_v{next_reprice_version}_post_repair_one_swap_reprice.csv",
                "success_condition": (
                    f"post-v{version} one-swap/integer pricing finds no feasible improving swaps"
                ),
                "last_wave": f"v{version}",
                "execution_result": f"{ordinal}_one_swap_repair_candidate_created",
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


def _update_repair_notebook(
    *,
    status: dict[str, Any],
    version: int,
    previous_repair_version: int,
    pricing_version: int,
    next_reprice_version: int,
    ordinal: str,
) -> None:
    start = f"<!-- V{version}_NEXT_ONE_SWAP_REPAIR_START -->"
    end = f"<!-- V{version}_NEXT_ONE_SWAP_REPAIR_END -->"
    title_ordinal = ordinal.title()
    block = f"""
{start}

## Wave v{version}: {title_ordinal} One-Swap Repair Candidate

Generated: {status["generated_at_utc"]}

### Objective

Apply the best feasible post-v{previous_repair_version} one-drop/one-add swap
found by v{pricing_version} and recompute portfolio return, budget, source and
CVaR metrics. This continues the local integer repair loop; it is not a final
champion or optimality certificate.

### Results

- Added loan: `{status[f"added_loan_id_v{version}"]}`.
- Dropped loan: `{status[f"dropped_loan_id_v{version}"]}`.
- Selected rows after repair: `{status[f"selected_rows_v{version}"]}`.
- Return delta vs v{previous_repair_version}:
  `{status[f"delta_return_vs_v{previous_repair_version}_v{version}"]}`.
- CVaR90 delta vs v{previous_repair_version}:
  `{status[f"delta_cvar90_vs_v{previous_repair_version}_v{version}"]}`.
- Budget feasible: `{status[f"budget_feasible_v{version}"]}`.
- Source feasible: `{status[f"source_feasible_v{version}"]}`.
- CVaR feasible: `{status[f"cvar_feasible_v{version}"]}`.
- Post-repair local optimality claim allowed:
  `{status[f"post_repair_one_swap_optimality_claim_allowed_v{version}"]}`.

### Interpretation

v{version} improves the v{previous_repair_version} repaired candidate while
preserving budget, source and CVaR feasibility. The next required experiment is
v{next_reprice_version} post-repair one-swap pricing because every repair
changes the set of possible improving exchanges.

### Claim Impact

- Allowed: {ordinal} one-swap repair candidate created.
- Still prohibited: post-repair local optimality, full-universe integer
  optimality, Paper Estrella replacement, final Paper 4 promotion and live
  deployment.

### Quarto Promotion Decision

Keep v{version} in the living notebook. Promote only after the repair/reprice
loop terminates and stronger integer/dynamic/promotion gates pass.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def build_reprice_wave(
    *,
    version: int,
    previous_repair_version: int,
    next_repair_version: int,
) -> dict[str, Any]:
    started = datetime.now(UTC)
    pairs, summary, stage_summary = _candidate_pairs_for_reprice(
        version=version,
        previous_repair_version=previous_repair_version,
    )
    top_candidates = pairs.sort_values(
        [f"one_swap_improves_return_v{version}", f"return_delta_v{version}"],
        ascending=[False, False],
    ).head(200)
    blockers = _reprice_claim_blockers(
        summary=summary,
        version=version,
        previous_repair_version=previous_repair_version,
        next_repair_version=next_repair_version,
    )
    write_csv(TABLE_DIR / f"paper4_v{version}_post_repair_one_swap_reprice.csv", pairs)
    write_csv(
        TABLE_DIR / f"paper4_v{version}_post_repair_one_swap_top_candidates.csv",
        top_candidates,
    )
    write_csv(TABLE_DIR / f"paper4_v{version}_post_repair_one_swap_summary.csv", summary)
    write_csv(
        TABLE_DIR / f"paper4_v{version}_post_repair_one_swap_stage_summary.csv",
        stage_summary,
    )
    write_csv(TABLE_DIR / f"paper4_v{version}_claim_blockers.csv", blockers)
    local_optimality_cleared = bool(
        summary[f"post_repair_one_swap_local_optimality_cleared_v{version}"].iloc[0]
    )
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": f"v{version}_post_repair_one_swap_reprice_executed",
                "allowed": True,
                "artifact": f"paper4_v{version}_post_repair_one_swap_summary.csv",
                "boundary": f"post-v{previous_repair_version} one-swap screen only",
            },
            {
                "claim_id": f"v{version}_post_repair_one_swap_local_optimality",
                "allowed": local_optimality_cleared,
                "artifact": f"paper4_v{version}_claim_blockers.csv",
                "boundary": "false if feasible improving swaps remain",
            },
            {
                "claim_id": f"v{version}_full_universe_integer_optimality",
                "allowed": False,
                "artifact": f"paper4_v{version}_claim_blockers.csv",
                "boundary": "multi-swap/global gap certificate missing",
            },
            {
                "claim_id": f"v{version}_paper1_or_final_promotion",
                "allowed": False,
                "artifact": f"paper4_v{version}_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / f"paper4_v{version}_claim_matrix_delta.csv", claim_matrix)
    _update_reprice_claim_boundaries(
        version,
        previous_repair_version,
        local_optimality_cleared=local_optimality_cleared,
    )
    _update_reprice_backlog(
        version,
        previous_repair_version,
        next_repair_version,
        local_optimality_cleared=local_optimality_cleared,
    )

    row = summary.iloc[0]
    best_return_delta = row[f"best_one_swap_return_delta_v{version}"]
    best_cvar90_after = row[f"best_one_swap_cvar90_after_v{version}"]
    status = {
        "phase": f"v{version}_post_v{previous_repair_version}_one_swap_reprice",
        "schema_version": f"2026-05-15.{version}",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        f"summary_rows_v{version}": int(len(summary)),
        f"stage_summary_rows_v{version}": int(len(stage_summary)),
        f"candidate_pair_rows_v{version}": int(len(pairs)),
        f"top_candidate_rows_v{version}": int(len(top_candidates)),
        f"claim_blocker_rows_v{version}": int(len(blockers)),
        f"selected_rows_v{version}": int(row[f"selected_rows_v{version}"]),
        f"candidate_add_rows_v{version}": int(row[f"candidate_add_rows_v{version}"]),
        f"total_pair_rows_screened_v{version}": int(row[f"total_pair_rows_screened_v{version}"]),
        f"return_improving_pair_rows_v{version}": int(
            row[f"return_improving_pair_rows_v{version}"]
        ),
        f"budget_return_feasible_pair_rows_v{version}": int(
            row[f"budget_return_feasible_pair_rows_v{version}"]
        ),
        f"source_prefilter_pair_rows_v{version}": int(
            row[f"source_prefilter_pair_rows_v{version}"]
        ),
        f"source_exact_pair_rows_v{version}": int(row[f"source_exact_pair_rows_v{version}"]),
        f"cvar_feasible_pair_rows_v{version}": int(row[f"cvar_feasible_pair_rows_v{version}"]),
        f"one_swap_improving_rows_v{version}": int(row[f"one_swap_improving_rows_v{version}"]),
        f"best_one_swap_return_delta_v{version}": (
            None if pd.isna(best_return_delta) else float(best_return_delta)
        ),
        f"best_one_swap_cvar90_after_v{version}": (
            None if pd.isna(best_cvar90_after) else float(best_cvar90_after)
        ),
        f"post_repair_one_swap_local_optimality_cleared_v{version}": bool(
            row[f"post_repair_one_swap_local_optimality_cleared_v{version}"]
        ),
        f"full_universe_integer_optimality_claim_allowed_v{version}": False,
        f"paper1_promotion_allowed_v{version}": False,
        f"paper4_working_champion_changed_v{version}": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            f"v{version} is post-v{previous_repair_version} one-swap pricing only; "
            "global integer and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / f"paper4_v{version}_status.json", status)
    _update_reprice_notebook(status, version, previous_repair_version)
    return status


def _candidate_pairs_for_reprice(
    *,
    version: int,
    previous_repair_version: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    selected = read_parquet(
        f"paper4_v{previous_repair_version}_next_one_swap_repair_allocations.parquet"
    )
    source_summary = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_summary = source_summary.loc[
        source_summary["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    repair_summary = read_csv(f"paper4_v{previous_repair_version}_next_one_swap_repair_summary.csv")
    repair_row = repair_summary.loc[
        repair_summary[f"portfolio_label_v{previous_repair_version}"].eq(
            "next_one_swap_repair_candidate"
        )
    ].iloc[0]
    action = read_csv(f"paper4_v{previous_repair_version}_next_one_swap_repair_action.csv")
    if universe.empty or selected.empty or source_summary.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    selected_ids = set(selected["loan_id"].astype(str))
    candidates = universe.loc[~universe["loan_id"].astype(str).isin(selected_ids)].copy()
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy()
    candidate_idx = idx_by_id.loc[candidates["loan_id"].astype(str)].to_numpy()
    selected = selected.reset_index(drop=True)
    candidates = candidates.reset_index(drop=True)
    selected[f"mean_return_if_dropped_v{version}"] = mean_returns[selected_idx]
    candidates[f"mean_return_if_added_v{version}"] = mean_returns[candidate_idx]

    policy_id = (
        str(action["policy_id"].iloc[0])
        if not action.empty
        else f"v{previous_repair_version}_repair_candidate"
    )
    regime = (
        str(action[f"regime_v{previous_repair_version}"].iloc[0])
        if not action.empty
        else f"post_v{previous_repair_version}_repair"
    )
    current_exposure = float(repair_row[f"portfolio_exposure_v{previous_repair_version}"])
    exposure_min = float(repair_row[f"exposure_min_v{previous_repair_version}"])
    exposure_max = float(repair_row[f"exposure_max_v{previous_repair_version}"])
    cvar_cap = float(repair_row[f"cvar_cap_v{previous_repair_version}"])
    current_objective_return = float(repair_row[f"objective_return_v{previous_repair_version}"])
    current_losses = losses[:, selected_idx].sum(axis=1)

    add_amount = candidates["loan_amnt"].to_numpy(float)
    drop_amount = selected["loan_amnt"].to_numpy(float)
    add_return = candidates[f"mean_return_if_added_v{version}"].to_numpy(float)
    drop_return = selected[f"mean_return_if_dropped_v{version}"].to_numpy(float)
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
            add_row[f"mean_return_if_added_v{version}"]
            - drop_row[f"mean_return_if_dropped_v{version}"]
        )
        row: dict[str, Any] = {
            "policy_id": policy_id,
            f"regime_v{version}": regime,
            f"added_loan_id_v{version}": str(add_row["loan_id"]),
            f"dropped_loan_id_v{version}": str(drop_row["loan_id"]),
            f"added_loan_amount_v{version}": float(add_row["loan_amnt"]),
            f"dropped_loan_amount_v{version}": float(drop_row["loan_amnt"]),
            f"added_mean_return_v{version}": float(add_row[f"mean_return_if_added_v{version}"]),
            f"dropped_mean_return_v{version}": float(
                drop_row[f"mean_return_if_dropped_v{version}"]
            ),
            f"return_delta_v{version}": return_delta,
            f"objective_return_after_swap_v{version}": current_objective_return + return_delta,
            f"exposure_after_swap_v{version}": new_total,
            f"budget_swap_feasible_v{version}": True,
            f"source_swap_feasible_v{version}": True,
            f"source_min_slack_after_swap_v{version}": min_slack,
            f"max_source_share_after_swap_v{version}": max_share,
            f"source_cap_violations_after_swap_v{version}": violations,
            f"first_source_block_family_v{version}": first_family,
            f"first_source_block_id_v{version}": first_source,
            f"loss_mean_after_swap_v{version}": float(swapped_losses.mean()),
            f"cvar90_after_swap_v{version}": cvar_after,
            f"cvar_swap_feasible_v{version}": cvar_after <= cvar_cap + 1e-7,
            f"one_swap_improves_return_v{version}": (
                return_delta > 1e-9 and cvar_after <= cvar_cap + 1e-7
            ),
            f"integer_screen_scope_v{version}": (
                f"post_v{previous_repair_version}_one_drop_one_add_whole_loan_swap"
            ),
            f"claim_boundary_v{version}": (
                f"post-v{previous_repair_version} one-swap pricing only; "
                "not multi-swap or global proof"
            ),
        }
        for family in FAMILIES:
            row[f"added_{family}_v{version}"] = str(add_row[family])
            row[f"dropped_{family}_v{version}"] = str(drop_row[family])
        rows.append(row)

    pairs = pd.DataFrame(rows, columns=_reprice_pair_columns(version))
    cvar_feasible_pairs = (
        int(pairs[f"cvar_swap_feasible_v{version}"].sum()) if not pairs.empty else 0
    )
    improving_pairs = (
        int(pairs[f"one_swap_improves_return_v{version}"].sum()) if not pairs.empty else 0
    )
    best = pairs.sort_values(f"return_delta_v{version}", ascending=False).head(1)
    local_claim_boundary = (
        f"post-v{previous_repair_version} one-swap screen cleared; "
        "multi-swap/global proof still missing"
        if improving_pairs == 0
        else (
            f"post-v{previous_repair_version} one-swap screen only; "
            "repeat repair/repricing if improvements remain"
        )
    )
    summary = pd.DataFrame(
        [
            {
                "policy_id": policy_id,
                f"regime_v{version}": regime,
                f"selected_rows_v{version}": int(len(selected)),
                f"candidate_add_rows_v{version}": int(len(candidates)),
                f"total_pair_rows_screened_v{version}": total_pairs,
                f"return_improving_pair_rows_v{version}": return_pairs,
                f"budget_return_feasible_pair_rows_v{version}": budget_return_pairs,
                f"source_prefilter_pair_rows_v{version}": source_prefilter_pairs,
                f"source_exact_pair_rows_v{version}": int(len(pairs)),
                f"cvar_feasible_pair_rows_v{version}": cvar_feasible_pairs,
                f"one_swap_improving_rows_v{version}": improving_pairs,
                f"best_one_swap_return_delta_v{version}": float(
                    best[f"return_delta_v{version}"].iloc[0]
                )
                if not best.empty
                else np.nan,
                f"best_one_swap_cvar90_after_v{version}": float(
                    best[f"cvar90_after_swap_v{version}"].iloc[0]
                )
                if not best.empty
                else np.nan,
                f"current_exposure_v{version}": current_exposure,
                f"exposure_min_v{version}": exposure_min,
                f"exposure_max_v{version}": exposure_max,
                f"cvar_cap_v{version}": cvar_cap,
                f"current_objective_return_v{version}": current_objective_return,
                f"post_repair_one_swap_local_optimality_cleared_v{version}": (improving_pairs == 0),
                f"full_universe_integer_optimality_claim_allowed_v{version}": False,
                f"claim_boundary_v{version}": local_claim_boundary,
            }
        ]
    )
    stage_summary = pd.DataFrame(
        [
            {f"stage_v{version}": "all_pairs", f"pair_rows_v{version}": total_pairs},
            {f"stage_v{version}": "return_improving", f"pair_rows_v{version}": return_pairs},
            {
                f"stage_v{version}": "budget_return_feasible",
                f"pair_rows_v{version}": budget_return_pairs,
            },
            {
                f"stage_v{version}": "source_prefilter_feasible",
                f"pair_rows_v{version}": source_prefilter_pairs,
            },
            {
                f"stage_v{version}": "source_exact_feasible",
                f"pair_rows_v{version}": int(len(pairs)),
            },
            {
                f"stage_v{version}": "cvar_feasible_improving",
                f"pair_rows_v{version}": cvar_feasible_pairs,
            },
        ]
    )
    stage_summary[f"claim_boundary_v{version}"] = (
        f"post-v{previous_repair_version} one-swap screen stage count only"
    )
    return pairs, summary, stage_summary


def _reprice_claim_blockers(
    *,
    summary: pd.DataFrame,
    version: int,
    previous_repair_version: int,
    next_repair_version: int,
) -> pd.DataFrame:
    improving = int(summary[f"one_swap_improving_rows_v{version}"].iloc[0])
    improvement_next_artifact = (
        f"paper4_v{next_repair_version}_apply_next_swap_or_iterate.csv"
        if improving > 0
        else f"paper4_v{version}_one_swap_local_optimality_evidence.csv"
    )
    improvement_boundary = (
        f"feasible improving post-v{previous_repair_version} one-swaps block local optimality"
        if improving > 0
        else f"no feasible improving post-v{previous_repair_version} one-swaps remain"
    )
    return pd.DataFrame(
        [
            {
                f"blocker_id_v{version}": f"post_v{previous_repair_version}_one_swap_improvement_found",
                f"blocking_v{version}": improving > 0,
                f"evidence_count_v{version}": improving,
                f"required_next_artifact_v{version}": improvement_next_artifact,
                f"claim_boundary_v{version}": improvement_boundary,
            },
            {
                f"blocker_id_v{version}": "multi_swap_integer_pricing_missing",
                f"blocking_v{version}": True,
                f"evidence_count_v{version}": 1,
                f"required_next_artifact_v{version}": (
                    f"paper4_v{next_repair_version}_iterated_swap_or_milp_repair.csv"
                ),
                f"claim_boundary_v{version}": "one-swap screen does not cover multi-loan exchanges",
            },
            {
                f"blocker_id_v{version}": "global_integer_gap_certificate_missing",
                f"blocking_v{version}": True,
                f"evidence_count_v{version}": 1,
                f"required_next_artifact_v{version}": (
                    f"paper4_v{next_repair_version}_global_integer_gap_protocol.csv"
                ),
                f"claim_boundary_v{version}": (
                    "no branch-and-price/global full-universe integer certificate"
                ),
            },
        ]
    )


def _update_reprice_claim_boundaries(
    version: int,
    previous_repair_version: int,
    *,
    local_optimality_cleared: bool,
) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": (
                    f"Paper 4 has a v{version} post-v{previous_repair_version} "
                    "one-swap pricing screen."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    f"paper4_v{version}_post_repair_one_swap_reprice.csv"
                ),
                "boundary": (
                    f"One-swap local screen after v{previous_repair_version} repair; "
                    "not global integer proof."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    f"v{version} proves the v{previous_repair_version} repaired "
                    "portfolio is locally optimal."
                ),
                "allowed": local_optimality_cleared,
                "evidence_artifact": (
                    f"reports/paper_material/paper4/tables/paper4_v{version}_claim_blockers.csv"
                ),
                "boundary": (
                    f"Allowed only if no feasible improving post-v{previous_repair_version} "
                    "one-swaps remain."
                ),
                "prohibited_claim_flag": not local_optimality_cleared,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": f"v{version} proves full-universe integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    f"reports/paper_material/paper4/tables/paper4_v{version}_claim_blockers.csv"
                ),
                "boundary": (
                    "Requires iterated repair, multi-swap/branch-and-price or global gap "
                    "certificate."
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


def _update_reprice_backlog(
    version: int,
    previous_repair_version: int,
    next_repair_version: int,
    *,
    local_optimality_cleared: bool,
) -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    next_artifact = (
        f"paper4_v{version}_multi_swap_or_global_gap_protocol.csv"
        if local_optimality_cleared
        else f"paper4_v{next_repair_version}_apply_next_swap_or_iterated_repair.csv"
    )
    status = (
        "one_swap_local_optimality_cleared"
        if local_optimality_cleared
        else "iterate_if_improving_swaps_remain"
    )
    execution_result = (
        f"post_v{previous_repair_version}_one_swap_reprice_cleared"
        if local_optimality_cleared
        else f"post_v{previous_repair_version}_one_swap_reprice_completed"
    )
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "CVaR/OCE",
                "executable_item": (
                    f"v{version} reruns one-swap pricing after the v{previous_repair_version} "
                    "repair over all non-selected comparable loans."
                ),
                "status": status,
                "next_artifact": next_artifact,
                "success_condition": (
                    f"no feasible improving post-v{previous_repair_version} one-swaps remain"
                ),
                "last_wave": f"v{version}",
                "execution_result": execution_result,
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


def _update_reprice_notebook(
    status: dict[str, Any],
    version: int,
    previous_repair_version: int,
) -> None:
    start = f"<!-- V{version}_POST_V{previous_repair_version}_ONE_SWAP_REPRICE_START -->"
    end = f"<!-- V{version}_POST_V{previous_repair_version}_ONE_SWAP_REPRICE_END -->"
    best_delta = status[f"best_one_swap_return_delta_v{version}"]
    best_delta_text = (
        "not applicable; no feasible improving one-swaps" if best_delta is None else str(best_delta)
    )
    block = f"""
{start}

## Wave v{version}: Post-v{previous_repair_version} One-Swap Repricing

Generated: {status["generated_at_utc"]}

### Objective

Rerun one-drop/one-add integer pricing after the v{previous_repair_version}
repair, using all non-selected loans from the comparable v55 universe as
possible additions. This tests whether the v{previous_repair_version} candidate
is one-swap locally optimal.

### Results

- Pair rows screened: `{status[f"total_pair_rows_screened_v{version}"]}`.
- Candidate add rows: `{status[f"candidate_add_rows_v{version}"]}`.
- Return-improving pairs: `{status[f"return_improving_pair_rows_v{version}"]}`.
- Exact source-feasible pairs: `{status[f"source_exact_pair_rows_v{version}"]}`.
- CVaR-feasible improving one-swaps: `{status[f"one_swap_improving_rows_v{version}"]}`.
- Best post-v{previous_repair_version} one-swap return delta:
  `{best_delta_text}`.
- Post-v{previous_repair_version} local optimality cleared:
  `{status[f"post_repair_one_swap_local_optimality_cleared_v{version}"]}`.

### Interpretation

v{version} is the required re-pricing after v{previous_repair_version} changed
the portfolio. If additional feasible improving one-swaps remain, the lab
should continue the repair/reprice loop; if it clears, the next blocker would
still be multi-swap/global integer evidence.

### Claim Impact

- Allowed: post-v{previous_repair_version} one-swap pricing screen completed.
- Still prohibited: full-universe integer optimality, Paper Estrella
  replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v{version} in the living notebook. Promote only after the repair loop
terminates and stronger integer/dynamic/promotion gates pass.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)
