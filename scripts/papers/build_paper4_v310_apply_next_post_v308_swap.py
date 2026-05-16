#!/usr/bin/env python3
"""Build Paper 4 v310 applied post-v308 one-swap repair artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

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

VERSION = 310
SIGNAL_VERSION = 309
BASE_VERSION = 308
BASELINE_VERSION = 295
NEXT_VERSION = 311
TARGET_SELECTED_ROWS = 171


def _best_signal() -> pd.Series:
    top = read_csv("paper4_v309_post_v308_one_swap_top_candidates.csv")
    if top.empty:
        raise RuntimeError("Missing v309 top candidates; run v309 first.")
    feasible = top.loc[top[f"one_swap_improves_return_v{SIGNAL_VERSION}"].astype(bool)].copy()
    if feasible.empty:
        raise RuntimeError("v310 requires a feasible improving v309 one-swap signal.")
    return feasible.sort_values(f"return_delta_v{SIGNAL_VERSION}", ascending=False).iloc[0]


def _source_summary(
    *,
    universe: pd.DataFrame,
    portfolio: pd.DataFrame,
    source_caps: pd.DataFrame,
) -> pd.DataFrame:
    exposure = float(portfolio["loan_amnt"].sum())
    rows: list[dict[str, Any]] = []
    cap_col = f"cap_share_v{BASE_VERSION}"
    for family in FAMILIES:
        caps = (
            source_caps.loc[source_caps["source_family"].astype(str).eq(family)]
            .set_index("source_id")[cap_col]
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
                    f"portfolio_label_v{VERSION}": "post_v308_best_one_swap_repair_candidate",
                    "source_family": family,
                    "source_id": source_id,
                    f"cap_share_v{VERSION}": cap,
                    f"source_exposure_v{VERSION}": source_exposure,
                    f"source_share_v{VERSION}": share,
                    f"source_slack_v{VERSION}": cap - share,
                    f"source_cap_violated_v{VERSION}": share > cap + 1e-7,
                    f"claim_boundary_v{VERSION}": "v310 post-swap source diagnostic only",
                }
            )
    return pd.DataFrame(rows)


def _coverage_delta(
    *,
    before: pd.DataFrame,
    after: pd.DataFrame,
    observed_ids: set[str],
    add_id: str,
    drop_id: str,
) -> pd.DataFrame:
    before_observed = int(before["loan_id"].astype(str).isin(observed_ids).sum())
    after_observed = int(after["loan_id"].astype(str).isin(observed_ids).sum())
    before_missing = int(len(before) - before_observed)
    after_missing = int(len(after) - after_observed)
    return pd.DataFrame(
        [
            {
                f"coverage_id_v{VERSION}": "v47_observed_cashflow_proxy_coverage",
                f"before_observed_rows_v{VERSION}": before_observed,
                f"after_observed_rows_v{VERSION}": after_observed,
                f"before_missing_rows_v{VERSION}": before_missing,
                f"after_missing_rows_v{VERSION}": after_missing,
                f"delta_missing_rows_v{VERSION}": after_missing - before_missing,
                f"added_loan_id_v{VERSION}": add_id,
                f"added_has_observed_proxy_v{VERSION}": add_id in observed_ids,
                f"dropped_loan_id_v{VERSION}": drop_id,
                f"dropped_had_observed_proxy_v{VERSION}": drop_id in observed_ids,
                f"claim_boundary_v{VERSION}": (
                    "proxy coverage audit only; not contractual IFRS9 coverage"
                ),
            }
        ]
    )


def _build_repair() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v308_apply_next_post_v306_swap_allocations.parquet")
    base_summary = read_csv("paper4_v308_apply_next_post_v306_swap_summary.csv")
    signal_summary = read_csv("paper4_v309_post_v308_one_swap_summary.csv")
    v47_panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet")
    source_caps = read_csv("paper4_v308_apply_next_post_v306_swap_source_summary.csv")
    best = _best_signal()
    if any(
        df.empty
        for df in [universe, selected, base_summary, signal_summary, v47_panel, source_caps]
    ):
        raise RuntimeError("Missing v55, v308, v309 or v47 inputs for v310.")

    universe["loan_id"] = universe["loan_id"].astype(str)
    selected["loan_id"] = selected["loan_id"].astype(str)
    source_caps["source_id"] = source_caps["source_id"].astype(str)
    for family in FAMILIES:
        universe[family] = universe[family].astype(str)
        selected[family] = selected[family].astype(str)

    add_id = str(best[f"added_loan_id_v{SIGNAL_VERSION}"])
    drop_id = str(best[f"dropped_loan_id_v{SIGNAL_VERSION}"])
    add_row = universe.loc[
        universe["loan_id"].eq(add_id), ["loan_id", "loan_amnt", *FAMILIES]
    ].copy()
    if add_row.empty:
        raise RuntimeError(f"Could not find added loan {add_id} in v55 universe.")

    keep_cols = ["loan_id", "loan_amnt", *FAMILIES]
    kept = selected.loc[~selected["loan_id"].eq(drop_id), keep_cols].copy()
    if len(kept) != len(selected) - 1:
        raise RuntimeError(f"Could not drop selected loan {drop_id} from v308 portfolio.")
    repaired = pd.concat([kept, add_row], ignore_index=True)
    repaired["loan_id"] = repaired["loan_id"].astype(str)
    for family in FAMILIES:
        repaired[family] = repaired[family].astype(str)

    observed_ids = set(v47_panel["loan_id"].astype(str))
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    repaired_idx = idx_by_id.loc[repaired["loan_id"].astype(str)].to_numpy()
    scenario_losses = losses[:, repaired_idx].sum(axis=1)
    repaired[f"mean_return_v{VERSION}"] = mean_returns[repaired_idx]
    repaired[f"selected_v{VERSION}"] = 1
    repaired[f"portfolio_label_v{VERSION}"] = "post_v308_best_one_swap_repair_candidate"
    repaired[f"observed_v47_proxy_v{VERSION}"] = repaired["loan_id"].isin(observed_ids)
    repaired[f"proxy_coverage_role_v{VERSION}"] = np.where(
        repaired[f"observed_v47_proxy_v{VERSION}"],
        "observed_v47_proxy",
        "unobserved_needs_proxy_or_imputation",
    )
    repaired[f"repair_action_v{VERSION}"] = np.where(
        repaired["loan_id"].eq(add_id),
        f"added_from_v{SIGNAL_VERSION}_best_feasible_swap",
        f"kept_from_v{BASE_VERSION}",
    )
    repaired[f"claim_boundary_v{VERSION}"] = (
        "post-v308 one-swap repair candidate only; requires post-repair repricing"
    )

    source_summary = _source_summary(
        universe=universe,
        portfolio=repaired,
        source_caps=source_caps,
    )
    coverage = _coverage_delta(
        before=selected,
        after=repaired,
        observed_ids=observed_ids,
        add_id=add_id,
        drop_id=drop_id,
    )
    objective_return = float(repaired[f"mean_return_v{VERSION}"].sum())
    exposure = float(repaired["loan_amnt"].sum())
    cvar90 = v70._tail_cvar(scenario_losses)
    source_violations = int(source_summary[f"source_cap_violated_v{VERSION}"].sum())
    base_row = base_summary.iloc[0]
    signal_row = signal_summary.iloc[0]

    action = pd.DataFrame(
        [
            {
                "policy_id": str(best["policy_id"]),
                f"regime_v{VERSION}": str(best[f"regime_v{SIGNAL_VERSION}"]),
                f"signal_version_v{VERSION}": SIGNAL_VERSION,
                f"base_version_v{VERSION}": BASE_VERSION,
                f"added_loan_id_v{VERSION}": add_id,
                f"dropped_loan_id_v{VERSION}": drop_id,
                f"added_loan_amount_v{VERSION}": float(
                    best[f"added_loan_amount_v{SIGNAL_VERSION}"]
                ),
                f"dropped_loan_amount_v{VERSION}": float(
                    best[f"dropped_loan_amount_v{SIGNAL_VERSION}"]
                ),
                f"added_mean_return_v{VERSION}": float(
                    best[f"added_mean_return_v{SIGNAL_VERSION}"]
                ),
                f"dropped_mean_return_v{VERSION}": float(
                    best[f"dropped_mean_return_v{SIGNAL_VERSION}"]
                ),
                f"return_delta_v{VERSION}": float(best[f"return_delta_v{SIGNAL_VERSION}"]),
                f"predicted_cvar90_after_swap_v{VERSION}": float(
                    best[f"cvar90_after_swap_v{SIGNAL_VERSION}"]
                ),
                f"actual_cvar90_after_repair_v{VERSION}": cvar90,
                f"exposure_after_repair_v{VERSION}": exposure,
                f"source_cap_violations_after_repair_v{VERSION}": source_violations,
                f"added_has_observed_proxy_v{VERSION}": add_id in observed_ids,
                f"dropped_had_observed_proxy_v{VERSION}": drop_id in observed_ids,
                f"claim_boundary_v{VERSION}": (
                    "best v309 feasible swap applied; not post-repair optimality"
                ),
            }
        ]
    )

    budget_feasible = (
        exposure >= float(signal_row[f"exposure_min_v{SIGNAL_VERSION}"]) - 1e-7
        and exposure <= float(signal_row[f"exposure_max_v{SIGNAL_VERSION}"]) + 1e-7
    )
    cvar_feasible = cvar90 <= float(signal_row[f"current_cvar90_v{SIGNAL_VERSION}"]) + 1e-7
    source_feasible = source_violations == 0
    summary = pd.DataFrame(
        [
            {
                f"portfolio_label_v{VERSION}": "post_v308_best_one_swap_repair_candidate",
                f"signal_version_v{VERSION}": SIGNAL_VERSION,
                f"base_version_v{VERSION}": BASE_VERSION,
                f"baseline_version_v{VERSION}": BASELINE_VERSION,
                f"selected_rows_v{VERSION}": int(len(repaired)),
                f"base_selected_rows_v{VERSION}": int(base_row[f"selected_rows_v{BASE_VERSION}"]),
                f"cardinality_restored_v{VERSION}": int(len(repaired)) == TARGET_SELECTED_ROWS,
                f"added_rows_v{VERSION}": 1,
                f"dropped_rows_v{VERSION}": 1,
                f"portfolio_exposure_v{VERSION}": exposure,
                f"objective_return_v{VERSION}": objective_return,
                f"scenario_loss_mean_v{VERSION}": float(scenario_losses.mean()),
                f"scenario_loss_cvar90_v{VERSION}": cvar90,
                f"source_cap_violations_v{VERSION}": source_violations,
                f"max_source_share_v{VERSION}": float(
                    source_summary[f"source_share_v{VERSION}"].max()
                ),
                f"min_source_slack_v{VERSION}": float(
                    source_summary[f"source_slack_v{VERSION}"].min()
                ),
                f"observed_v47_proxy_rows_v{VERSION}": int(
                    repaired[f"observed_v47_proxy_v{VERSION}"].sum()
                ),
                f"missing_v47_proxy_rows_v{VERSION}": int(
                    (~repaired[f"observed_v47_proxy_v{VERSION}"]).sum()
                ),
                f"delta_missing_v47_proxy_rows_vs_v{BASE_VERSION}_v{VERSION}": int(
                    coverage[f"delta_missing_rows_v{VERSION}"].iloc[0]
                ),
                f"delta_return_vs_v{BASE_VERSION}_v{VERSION}": objective_return
                - float(base_row[f"objective_return_v{BASE_VERSION}"]),
                f"delta_cvar90_vs_v{BASE_VERSION}_v{VERSION}": cvar90
                - float(base_row[f"scenario_loss_cvar90_v{BASE_VERSION}"]),
                f"delta_exposure_vs_v{BASE_VERSION}_v{VERSION}": exposure
                - float(base_row[f"portfolio_exposure_v{BASE_VERSION}"]),
                f"budget_feasible_v{VERSION}": budget_feasible,
                f"cvar_feasible_v{VERSION}": cvar_feasible,
                f"source_feasible_v{VERSION}": source_feasible,
                f"repair_candidate_feasible_v{VERSION}": (
                    budget_feasible and cvar_feasible and source_feasible
                ),
                f"post_repair_pricing_required_v{VERSION}": True,
                f"post_repair_one_swap_optimality_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                f"paper4_final_promotion_created_v{VERSION}": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": f"paper4_v{NEXT_VERSION}_post_v310_reprice.csv",
                f"claim_boundary_v{VERSION}": (
                    "applied best v309 one-swap repair; repricing and proxy coverage gates remain"
                ),
            }
        ]
    )
    return repaired, summary, action, source_summary, coverage


def _claim_blockers(summary: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    feasible = bool(summary[f"repair_candidate_feasible_v{VERSION}"].iloc[0])
    delta_missing = int(coverage[f"delta_missing_rows_v{VERSION}"].iloc[0])
    return pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "best_v309_swap_repair_candidate_created",
                f"blocking_v{VERSION}": not feasible,
                f"evidence_count_v{VERSION}": int(feasible),
                f"required_next_artifact_v{VERSION}": (
                    "paper4_v310_apply_next_post_v308_swap_summary.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "candidate exists only if budget/source/CVaR remain feasible"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "post_repair_one_swap_repricing_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": f"paper4_v{NEXT_VERSION}_post_v310_reprice.csv",
                f"claim_boundary_v{VERSION}": "repair must be re-priced after changing portfolio",
            },
            {
                f"blocker_id_v{VERSION}": "proxy_coverage_regression_or_gap",
                f"blocking_v{VERSION}": delta_missing >= 0,
                f"evidence_count_v{VERSION}": int(
                    summary[f"missing_v47_proxy_rows_v{VERSION}"].iloc[0]
                ),
                f"required_next_artifact_v{VERSION}": "future_cashflow_proxy_or_ifrs9_coverage_gate",
                f"claim_boundary_v{VERSION}": (
                    "applied return-improving swap increases missing observed proxy rows"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "global_dynamic_online_gates_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_global_dynamic_online_validation",
                f"claim_boundary_v{VERSION}": "no global, dynamic, online or deployment gate created",
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


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v310_best_v309_swap_applied",
                "allowed": True,
                "artifact": "paper4_v310_apply_next_post_v308_swap_summary.csv",
                "boundary": "single feasible v309 repair applied and audited",
            },
            {
                "claim_id": "v310_return_improves_and_cvar_lowers_vs_v308",
                "allowed": True,
                "artifact": "paper4_v310_apply_next_post_v308_swap_summary.csv",
                "boundary": "static scenario proxy metrics only; requires repricing",
            },
            {
                "claim_id": "v310_post_repair_local_optimality",
                "allowed": False,
                "artifact": "paper4_v310_claim_blockers.csv",
                "boundary": "post-v310 repricing has not been executed",
            },
            {
                "claim_id": "v310_working_champion",
                "allowed": False,
                "artifact": "paper4_v310_claim_blockers.csv",
                "boundary": "global, dynamic, online and deployment evidence missing",
            },
            {
                "claim_id": "v310_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v310_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v310 applied post-v308 one-swap repair candidate.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v310_apply_next_post_v308_swap_summary.csv"
                ),
                "boundary": "Best feasible v309 swap applied; requires post-repair repricing.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v310 improves return and lowers CVaR versus v308.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v310_apply_next_post_v308_swap_summary.csv"
                ),
                "boundary": "Static scenario proxy metrics only; not dynamic validation.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v310 repaired portfolio is post-repair locally optimal.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v310_claim_blockers.csv"
                ),
                "boundary": "Post-v310 repricing has not been executed.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v310 authorizes a Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v310_claim_blockers.csv"
                ),
                "boundary": "Global, dynamic, online and deployment gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v310 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v310_claim_blockers.csv"
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
                "lane": "Source Governance/Global",
                "executable_item": (
                    "v310 applies the best CVaR-feasible v309 one-swap signal to the "
                    "v308 repaired candidate and audits return, CVaR, source caps and "
                    "proxy coverage."
                ),
                "status": "post_v308_best_one_swap_applied_requires_repricing",
                "next_artifact": f"paper4_v{NEXT_VERSION}_post_v310_reprice.csv",
                "success_condition": (
                    "post-v310 repricing finds no feasible improving one-swaps or records "
                    "the next repair signal"
                ),
                "last_wave": "v310",
                "execution_result": (
                    "return_improving_cvar_lowering_swap_applied_with_proxy_coverage_gap"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v310")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V310_APPLY_POST_V308_SWAP_START -->"
    end = "<!-- V310_APPLY_POST_V308_SWAP_END -->"
    block = f"""
{start}

## Wave v310: Apply Best Post-v308 One-Swap Repair

Generated: {status["generated_at_utc"]}

### Objective

v309 found feasible improving one-swaps after the v308 repair. v310 applies
the best CVaR-feasible signal and audits the repaired static portfolio before
any dynamic/global gate is attempted.

### Results

- Added loan: `{status["added_loan_id_v310"]}`.
- Dropped loan: `{status["dropped_loan_id_v310"]}`.
- Return delta vs v308: `{status["delta_return_vs_v308_v310"]}`.
- CVaR90 delta vs v308: `{status["delta_cvar90_vs_v308_v310"]}`.
- Objective return: `{status["objective_return_v310"]}`.
- CVaR90 after repair: `{status["scenario_loss_cvar90_v310"]}`.
- Source cap violations: `{status["source_cap_violations_v310"]}`.
- Missing v47 proxy rows: `{status["missing_v47_proxy_rows_v310"]}`.
- Delta missing proxy rows vs v308:
  `{status["delta_missing_v47_proxy_rows_vs_v308_v310"]}`.
- Repair candidate feasible: `{status["repair_candidate_feasible_v310"]}`.
- Post-repair repricing required:
  `{status["post_repair_pricing_required_v310"]}`.

### Interpretation

v310 improves static return and lowers CVaR relative to v308, while preserving
budget, source caps and cardinality. The evidence-quality cost compounds: the
applied swap adds another loan without observed v47/v299 cashflow proxy
coverage and drops one with observed coverage.

### Claim Impact

- Allowed: applied one-swap repair candidate and static metric audit.
- Still prohibited: post-repair local optimality, full-universe/global
  optimality, Paper 4 working champion, Paper Estrella replacement, final
  Paper 4 promotion, contractual IFRS9 and live deployability claims.

### Quarto Promotion Decision

Keep v310 in the living notebook. The next wave must reprice v310.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    allocations, summary, action, source_summary, coverage = _build_repair()
    blockers = _claim_blockers(summary, coverage)
    claim_matrix = _claim_matrix()

    allocations.to_parquet(TABLE_DIR / "paper4_v310_apply_next_post_v308_swap_allocations.parquet")
    write_csv(TABLE_DIR / "paper4_v310_apply_next_post_v308_swap_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v310_apply_next_post_v308_swap_action.csv", action)
    write_csv(
        TABLE_DIR / "paper4_v310_apply_next_post_v308_swap_source_summary.csv", source_summary
    )
    write_csv(TABLE_DIR / "paper4_v310_proxy_coverage_delta.csv", coverage)
    write_csv(TABLE_DIR / "paper4_v310_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v310_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = summary.iloc[0]
    action_row = action.iloc[0]
    status = {
        "phase": "v310_apply_next_post_v308_swap",
        "schema_version": "2026-05-16.310",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "signal_version_v310": SIGNAL_VERSION,
        "base_version_v310": BASE_VERSION,
        "baseline_version_v310": BASELINE_VERSION,
        "summary_rows_v310": int(len(summary)),
        "action_rows_v310": int(len(action)),
        "allocation_rows_v310": int(len(allocations)),
        "source_summary_rows_v310": int(len(source_summary)),
        "coverage_rows_v310": int(len(coverage)),
        "claim_blocker_rows_v310": int(len(blockers)),
        "claim_matrix_rows_v310": int(len(claim_matrix)),
        "added_loan_id_v310": str(action_row[f"added_loan_id_v{VERSION}"]),
        "dropped_loan_id_v310": str(action_row[f"dropped_loan_id_v{VERSION}"]),
        "added_has_observed_proxy_v310": bool(action_row[f"added_has_observed_proxy_v{VERSION}"]),
        "dropped_had_observed_proxy_v310": bool(
            action_row[f"dropped_had_observed_proxy_v{VERSION}"]
        ),
        "selected_rows_v310": int(row[f"selected_rows_v{VERSION}"]),
        "base_selected_rows_v310": int(row[f"base_selected_rows_v{VERSION}"]),
        "cardinality_restored_v310": bool(row[f"cardinality_restored_v{VERSION}"]),
        "portfolio_exposure_v310": float(row[f"portfolio_exposure_v{VERSION}"]),
        "objective_return_v310": float(row[f"objective_return_v{VERSION}"]),
        "scenario_loss_mean_v310": float(row[f"scenario_loss_mean_v{VERSION}"]),
        "scenario_loss_cvar90_v310": float(row[f"scenario_loss_cvar90_v{VERSION}"]),
        "source_cap_violations_v310": int(row[f"source_cap_violations_v{VERSION}"]),
        "max_source_share_v310": float(row[f"max_source_share_v{VERSION}"]),
        "min_source_slack_v310": float(row[f"min_source_slack_v{VERSION}"]),
        "observed_v47_proxy_rows_v310": int(row[f"observed_v47_proxy_rows_v{VERSION}"]),
        "missing_v47_proxy_rows_v310": int(row[f"missing_v47_proxy_rows_v{VERSION}"]),
        "delta_missing_v47_proxy_rows_vs_v308_v310": int(
            row[f"delta_missing_v47_proxy_rows_vs_v{BASE_VERSION}_v{VERSION}"]
        ),
        "delta_return_vs_v308_v310": float(row[f"delta_return_vs_v{BASE_VERSION}_v{VERSION}"]),
        "delta_cvar90_vs_v308_v310": float(row[f"delta_cvar90_vs_v{BASE_VERSION}_v{VERSION}"]),
        "delta_exposure_vs_v308_v310": float(row[f"delta_exposure_vs_v{BASE_VERSION}_v{VERSION}"]),
        "budget_feasible_v310": bool(row[f"budget_feasible_v{VERSION}"]),
        "cvar_feasible_v310": bool(row[f"cvar_feasible_v{VERSION}"]),
        "source_feasible_v310": bool(row[f"source_feasible_v{VERSION}"]),
        "repair_candidate_feasible_v310": bool(row[f"repair_candidate_feasible_v{VERSION}"]),
        "post_repair_pricing_required_v310": bool(row[f"post_repair_pricing_required_v{VERSION}"]),
        "post_repair_one_swap_optimality_claim_allowed_v310": False,
        "working_champion_claim_allowed_v310": False,
        "full_universe_integer_optimality_claim_allowed_v310": False,
        "paper1_promotion_allowed_v310": False,
        "paper4_working_champion_changed_v310": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v310": f"paper4_v{NEXT_VERSION}_post_v310_reprice.csv",
        "claim_boundary": (
            "v310 applies a post-v308 one-swap repair; no working champion or final "
            "promotion is authorized"
        ),
    }
    write_json(STATUS_DIR / "paper4_v310_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v310": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
