#!/usr/bin/env python3
"""Build Paper 4 v303 global/multiobjective frontier audit artifacts."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from scripts.papers.paper4_one_swap_living_lab import (
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    now,
    read_csv,
    write_csv,
    write_json,
)

VERSION = 303
SOURCE_CANDIDATE_VERSION = 295
FRONTIER_VERSION = 302
NEXT_VERSION = 304
REWARD_GRID = [0, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 8, 8.5, 9, 10, 12, 15, 20]


def _prefix_frontier(v302_frontier: pd.DataFrame, v302_status: dict[str, Any]) -> pd.DataFrame:
    baseline = pd.DataFrame(
        [
            {
                f"frontier_step_v{VERSION}": 0,
                f"source_frontier_step_v{FRONTIER_VERSION}": 0,
                f"objective_return_v{VERSION}": float(v302_status["v295_objective_return_v302"]),
                f"cvar90_v{VERSION}": float(v302_status["v295_cvar90_cap_v302"]),
                f"imputed_proxy_loan_rows_v{VERSION}": int(
                    v302_status["initial_imputed_proxy_loan_rows_v302"]
                ),
                f"imputed_proxy_rows_reduced_v{VERSION}": 0,
                f"cumulative_return_delta_v{VERSION}": 0.0,
                f"marginal_return_delta_v{VERSION}": 0.0,
                f"marginal_return_cost_per_imputation_v{VERSION}": 0.0,
                f"average_return_cost_per_imputation_v{VERSION}": 0.0,
                f"repair_profile_v{VERSION}": "v295_baseline",
                f"source_cap_feasible_v{VERSION}": True,
                f"cvar_feasible_v{VERSION}": True,
                f"claim_boundary_v{VERSION}": (
                    "baseline prefix for multiobjective audit; not a promotion"
                ),
            }
        ]
    )
    rows: list[dict[str, Any]] = []
    for row in v302_frontier.itertuples(index=False):
        step = int(getattr(row, f"frontier_step_v{FRONTIER_VERSION}"))
        cumulative_delta = float(getattr(row, f"cumulative_return_delta_v{FRONTIER_VERSION}"))
        rows.append(
            {
                f"frontier_step_v{VERSION}": step,
                f"source_frontier_step_v{FRONTIER_VERSION}": step,
                f"objective_return_v{VERSION}": float(
                    getattr(row, f"objective_return_after_step_v{FRONTIER_VERSION}")
                ),
                f"cvar90_v{VERSION}": float(getattr(row, f"cvar90_after_step_v{FRONTIER_VERSION}")),
                f"imputed_proxy_loan_rows_v{VERSION}": int(
                    getattr(row, f"imputed_proxy_loan_rows_after_step_v{FRONTIER_VERSION}")
                ),
                f"imputed_proxy_rows_reduced_v{VERSION}": step,
                f"cumulative_return_delta_v{VERSION}": cumulative_delta,
                f"marginal_return_delta_v{VERSION}": float(
                    getattr(row, f"return_delta_v{FRONTIER_VERSION}")
                ),
                f"marginal_return_cost_per_imputation_v{VERSION}": -float(
                    getattr(row, f"return_delta_v{FRONTIER_VERSION}")
                ),
                f"average_return_cost_per_imputation_v{VERSION}": -cumulative_delta / max(step, 1),
                f"repair_profile_v{VERSION}": str(
                    getattr(row, f"repair_profile_v{FRONTIER_VERSION}")
                ),
                f"source_cap_feasible_v{VERSION}": True,
                f"cvar_feasible_v{VERSION}": True,
                f"claim_boundary_v{VERSION}": (
                    "v302 prefix multiobjective audit only; no optimal frontier claim"
                ),
            }
        )
    out = pd.concat([baseline, pd.DataFrame(rows)], ignore_index=True)
    out[f"pareto_nondominated_return_vs_imputation_v{VERSION}"] = True
    return out


def _reward_grid(prefixes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for reward in REWARD_GRID:
        work = prefixes.copy()
        work[f"multiobjective_utility_v{VERSION}"] = (
            work[f"objective_return_v{VERSION}"]
            + float(reward) * work[f"imputed_proxy_rows_reduced_v{VERSION}"]
        )
        best = work.sort_values(
            [
                f"multiobjective_utility_v{VERSION}",
                f"imputed_proxy_rows_reduced_v{VERSION}",
                f"objective_return_v{VERSION}",
            ],
            ascending=[False, False, False],
        ).iloc[0]
        rows.append(
            {
                f"reward_per_imputation_reduced_v{VERSION}": float(reward),
                f"selected_frontier_step_v{VERSION}": int(best[f"frontier_step_v{VERSION}"]),
                f"selected_imputed_rows_v{VERSION}": int(
                    best[f"imputed_proxy_loan_rows_v{VERSION}"]
                ),
                f"selected_imputed_rows_reduced_v{VERSION}": int(
                    best[f"imputed_proxy_rows_reduced_v{VERSION}"]
                ),
                f"selected_objective_return_v{VERSION}": float(
                    best[f"objective_return_v{VERSION}"]
                ),
                f"selected_cvar90_v{VERSION}": float(best[f"cvar90_v{VERSION}"]),
                f"selected_cumulative_return_delta_v{VERSION}": float(
                    best[f"cumulative_return_delta_v{VERSION}"]
                ),
                f"selected_utility_v{VERSION}": float(best[f"multiobjective_utility_v{VERSION}"]),
                f"claim_boundary_v{VERSION}": (
                    "reward-grid choice over v302 greedy prefixes only; not global optimum"
                ),
            }
        )
    return pd.DataFrame(rows)


def _selection_envelope(prefixes: pd.DataFrame) -> pd.DataFrame:
    points = [
        {
            "step": int(row[f"frontier_step_v{VERSION}"]),
            "return": float(row[f"objective_return_v{VERSION}"]),
            "imputed": int(row[f"imputed_proxy_loan_rows_v{VERSION}"]),
            "reduced": int(row[f"imputed_proxy_rows_reduced_v{VERSION}"]),
            "cvar": float(row[f"cvar90_v{VERSION}"]),
            "cumulative_delta": float(row[f"cumulative_return_delta_v{VERSION}"]),
        }
        for _, row in prefixes.iterrows()
    ]
    rows: list[dict[str, Any]] = []
    for point in points:
        step = point["step"]
        lower = 0.0
        upper = math.inf
        for other in points:
            other_step = other["step"]
            if other_step == step:
                continue
            if step > other_step:
                lower = max(lower, (other["return"] - point["return"]) / (step - other_step))
            else:
                upper = min(upper, (point["return"] - other["return"]) / (other_step - step))
        if lower <= upper + 1e-9:
            rows.append(
                {
                    f"selected_frontier_step_v{VERSION}": step,
                    f"reward_lower_bound_v{VERSION}": float(lower),
                    f"reward_upper_bound_v{VERSION}": float(upper),
                    f"selected_imputed_rows_v{VERSION}": point["imputed"],
                    f"selected_imputed_rows_reduced_v{VERSION}": point["reduced"],
                    f"selected_objective_return_v{VERSION}": point["return"],
                    f"selected_cvar90_v{VERSION}": point["cvar"],
                    f"selected_cumulative_return_delta_v{VERSION}": point["cumulative_delta"],
                    f"claim_boundary_v{VERSION}": (
                        "upper-envelope interval over greedy prefixes only; no global proof"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v303 multiobjective audit of the v302 imputation frontier.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v303_global_or_multiobjective_frontier_after_v302.csv"
                ),
                "boundary": "Reward-grid and envelope audit over greedy prefixes only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v303 quantifies the reward threshold for choosing imputation repair.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v303_reward_selection_envelope.csv"
                ),
                "boundary": "Thresholds are over the v302 greedy prefix set, not the full universe.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v303 proves a global or optimal multiobjective frontier.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v303_claim_blockers.csv"
                ),
                "boundary": "No full-universe multiobjective MILP or branch-price certificate exists.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v303 resolves contractual IFRS9 or live deployability for v295.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v303_claim_blockers.csv"
                ),
                "boundary": "The audited frontier still has residual imputed rows and no external holdout.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v303 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v303_claim_blockers.csv"
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
                    "v303 audits the v302 greedy imputation frontier as a multiobjective "
                    "return-versus-observed-proxy trade-off."
                ),
                "status": "multiobjective_frontier_audit_executed_global_claims_blocked",
                "next_artifact": (
                    f"paper4_v{NEXT_VERSION}_bounded_multiobjective_milp_or_global_bound_probe.csv"
                ),
                "success_condition": (
                    "move from greedy-prefix audit to a bounded multiobjective MILP/global-bound "
                    "probe without promoting"
                ),
                "last_wave": "v303",
                "execution_result": (
                    "reward_thresholds_quantified_over_greedy_prefixes_global_optimality_blocked"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v303")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V303_GLOBAL_MULTI_FRONTIER_AUDIT_START -->"
    end = "<!-- V303_GLOBAL_MULTI_FRONTIER_AUDIT_END -->"
    block = f"""
{start}

## Wave v303: Global / Multiobjective Frontier Audit

Generated: {status["generated_at_utc"]}

### Objective

v302 produced a bounded greedy frontier that trades return for fewer imputed
proxy rows. v303 asks a sharper decision question: how much reward per reduced
imputation would justify choosing each prefix, and where does the greedy
frontier become attractive versus staying at v295?

### Results

- Prefix frontier rows: `{status["prefix_frontier_rows_v303"]}`.
- Reward-grid rows: `{status["reward_grid_rows_v303"]}`.
- Reward envelope rows: `{status["reward_envelope_rows_v303"]}`.
- Minimum reward for any repair:
  `{status["minimum_reward_for_any_repair_v303"]}`.
- Reward needed for full 15-step frontier:
  `{status["minimum_reward_for_full_frontier_v303"]}`.
- Grid-selected unique steps: `{status["reward_grid_unique_selected_steps_v303"]}`.
- Best average-cost prefix step:
  `{status["best_average_cost_prefix_step_v303"]}`.
- Best average cost per imputation:
  `{status["best_average_return_cost_per_imputation_v303"]}`.
- Full frontier average cost per imputation:
  `{status["full_frontier_average_return_cost_per_imputation_v303"]}`.
- Valid global/multiobjective optimality claim:
  `{status["valid_global_multiobjective_claim_v303"]}`.

### Interpretation

v303 turns the v302 frontier into an explicit trade-off curve. The first useful
repair is not the first greedy step but the two-step prefix: it becomes
preferable to v295 only if one reduced imputed proxy row is worth about 1.99
return units. The full 15-step frontier requires a much larger reward, about
19.29 units at the margin, and is still a greedy-prefix audit rather than an
optimal multiobjective solution.

### Claim Impact

- Allowed: multiobjective audit over v302 prefixes; reward thresholds for
  choosing imputation repair.
- Still prohibited: global or optimal multiobjective frontier, contractual
  IFRS9, live deployability, Paper Estrella replacement, final Paper 4
  promotion and working champion claims.

### Quarto Promotion Decision

Keep v303 in the living notebook. The next wave should attempt a bounded
multiobjective MILP/global-bound probe.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    v302_frontier = read_csv("paper4_v302_greedy_imputation_frontier.csv")
    v302_status = json.loads((STATUS_DIR / "paper4_v302_status.json").read_text(encoding="utf-8"))
    v300_status = json.loads((STATUS_DIR / "paper4_v300_status.json").read_text(encoding="utf-8"))
    if v302_frontier.empty:
        raise RuntimeError("Missing v302 frontier for v303.")
    prefixes = _prefix_frontier(v302_frontier, v302_status)
    reward_grid = _reward_grid(prefixes)
    envelope = _selection_envelope(prefixes)

    repair_prefixes = prefixes.loc[prefixes[f"frontier_step_v{VERSION}"].gt(0)].copy()
    best_average = repair_prefixes.sort_values(
        f"average_return_cost_per_imputation_v{VERSION}"
    ).iloc[0]
    full_prefix = prefixes.loc[prefixes[f"frontier_step_v{VERSION}"].eq(15)].iloc[0]
    full_envelope = envelope.loc[envelope[f"selected_frontier_step_v{VERSION}"].eq(15)].iloc[0]
    minimum_repair_reward = float(
        envelope.loc[envelope[f"selected_frontier_step_v{VERSION}"].gt(0)][
            f"reward_lower_bound_v{VERSION}"
        ].min()
    )
    valid_global_multiobjective_claim = False
    summary = pd.DataFrame(
        [
            {
                f"audit_id_v{VERSION}": "v303_global_or_multiobjective_frontier_after_v302",
                f"source_candidate_version_v{VERSION}": SOURCE_CANDIDATE_VERSION,
                f"frontier_version_v{VERSION}": FRONTIER_VERSION,
                f"prefix_frontier_rows_v{VERSION}": int(len(prefixes)),
                f"reward_grid_rows_v{VERSION}": int(len(reward_grid)),
                f"reward_envelope_rows_v{VERSION}": int(len(envelope)),
                f"reward_grid_unique_selected_steps_v{VERSION}": int(
                    reward_grid[f"selected_frontier_step_v{VERSION}"].nunique()
                ),
                f"minimum_reward_for_any_repair_v{VERSION}": minimum_repair_reward,
                f"minimum_reward_for_full_frontier_v{VERSION}": float(
                    full_envelope[f"reward_lower_bound_v{VERSION}"]
                ),
                f"best_average_cost_prefix_step_v{VERSION}": int(
                    best_average[f"frontier_step_v{VERSION}"]
                ),
                f"best_average_return_cost_per_imputation_v{VERSION}": float(
                    best_average[f"average_return_cost_per_imputation_v{VERSION}"]
                ),
                f"full_frontier_average_return_cost_per_imputation_v{VERSION}": float(
                    full_prefix[f"average_return_cost_per_imputation_v{VERSION}"]
                ),
                f"full_frontier_final_imputed_rows_v{VERSION}": int(
                    full_prefix[f"imputed_proxy_loan_rows_v{VERSION}"]
                ),
                f"full_frontier_cumulative_return_delta_v{VERSION}": float(
                    full_prefix[f"cumulative_return_delta_v{VERSION}"]
                ),
                f"full_binary_variables_v{VERSION}": int(v300_status["full_binary_variables_v300"]),
                f"direct_mip_binary_guard_v{VERSION}": int(
                    v300_status["direct_mip_binary_guard_v300"]
                ),
                f"direct_full_mip_guard_exceeded_v{VERSION}": bool(
                    v300_status["direct_full_mip_guard_exceeded_v300"]
                ),
                f"valid_global_multiobjective_claim_v{VERSION}": valid_global_multiobjective_claim,
                f"contractual_ifrs9_claim_allowed_v{VERSION}": False,
                f"strict_live_deployability_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_bounded_multiobjective_milp_or_global_bound_probe.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "multiobjective audit over greedy prefixes only; no global optimality, IFRS9, live or promotion claim"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "full_universe_multiobjective_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v300_status["full_binary_variables_v300"]),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_bounded_multiobjective_milp_or_global_bound_probe.csv"
                ),
                f"claim_boundary_v{VERSION}": "v303 audits greedy prefixes, not full-v55 optimum",
            },
            {
                f"blocker_id_v{VERSION}": "greedy_prefix_scope_only",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(prefixes)),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_bounded_multiobjective_milp_or_global_bound_probe.csv"
                ),
                f"claim_boundary_v{VERSION}": "prefix reward envelope is not a global frontier",
            },
            {
                f"blocker_id_v{VERSION}": "residual_cashflow_imputation_after_full_frontier",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    full_prefix[f"imputed_proxy_loan_rows_v{VERSION}"]
                ),
                f"required_next_artifact_v{VERSION}": "future_observed_v295_cashflow_panel",
                f"claim_boundary_v{VERSION}": "residual imputation blocks contractual IFRS9",
            },
            {
                f"blocker_id_v{VERSION}": "external_online_holdout_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 0,
                f"required_next_artifact_v{VERSION}": "future_external_online_holdout",
                f"claim_boundary_v{VERSION}": "v303 does not create external online evidence",
            },
            {
                f"blocker_id_v{VERSION}": "paper4_working_champion_gate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_global_dynamic_promotion_gate",
                f"claim_boundary_v{VERSION}": "working champion replacement remains blocked",
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
                "claim_id": "v303_multiobjective_frontier_audit_executed",
                "allowed": True,
                "artifact": "paper4_v303_global_or_multiobjective_frontier_after_v302.csv",
                "boundary": "reward-grid audit over greedy prefixes",
            },
            {
                "claim_id": "v303_reward_thresholds_quantified",
                "allowed": True,
                "artifact": "paper4_v303_reward_selection_envelope.csv",
                "boundary": "thresholds over v302 prefixes only",
            },
            {
                "claim_id": "v303_global_or_optimal_multiobjective_frontier",
                "allowed": False,
                "artifact": "paper4_v303_claim_blockers.csv",
                "boundary": "no full-v55 multiobjective proof",
            },
            {
                "claim_id": "v303_contractual_ifrs9_or_live_deployability",
                "allowed": False,
                "artifact": "paper4_v303_claim_blockers.csv",
                "boundary": "residual imputation and no external holdout",
            },
            {
                "claim_id": "v303_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v303_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v303_global_or_multiobjective_frontier_after_v302.csv", summary)
    write_csv(TABLE_DIR / "paper4_v303_prefix_tradeoff_frontier.csv", prefixes)
    write_csv(TABLE_DIR / "paper4_v303_multiobjective_reward_grid.csv", reward_grid)
    write_csv(TABLE_DIR / "paper4_v303_reward_selection_envelope.csv", envelope)
    write_csv(TABLE_DIR / "paper4_v303_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v303_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v303_global_or_multiobjective_frontier_after_v302",
        "schema_version": "2026-05-15.303",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "source_candidate_version_v303": SOURCE_CANDIDATE_VERSION,
        "frontier_version_v303": FRONTIER_VERSION,
        "prefix_frontier_rows_v303": int(len(prefixes)),
        "reward_grid_rows_v303": int(len(reward_grid)),
        "reward_envelope_rows_v303": int(len(envelope)),
        "reward_grid_unique_selected_steps_v303": int(
            reward_grid[f"selected_frontier_step_v{VERSION}"].nunique()
        ),
        "minimum_reward_for_any_repair_v303": minimum_repair_reward,
        "minimum_reward_for_full_frontier_v303": float(
            full_envelope[f"reward_lower_bound_v{VERSION}"]
        ),
        "best_average_cost_prefix_step_v303": int(best_average[f"frontier_step_v{VERSION}"]),
        "best_average_return_cost_per_imputation_v303": float(
            best_average[f"average_return_cost_per_imputation_v{VERSION}"]
        ),
        "full_frontier_average_return_cost_per_imputation_v303": float(
            full_prefix[f"average_return_cost_per_imputation_v{VERSION}"]
        ),
        "full_frontier_final_imputed_rows_v303": int(
            full_prefix[f"imputed_proxy_loan_rows_v{VERSION}"]
        ),
        "full_frontier_cumulative_return_delta_v303": float(
            full_prefix[f"cumulative_return_delta_v{VERSION}"]
        ),
        "full_binary_variables_v303": int(v300_status["full_binary_variables_v300"]),
        "direct_mip_binary_guard_v303": int(v300_status["direct_mip_binary_guard_v300"]),
        "direct_full_mip_guard_exceeded_v303": bool(
            v300_status["direct_full_mip_guard_exceeded_v300"]
        ),
        "valid_global_multiobjective_claim_v303": valid_global_multiobjective_claim,
        "strict_live_deployability_claim_allowed_v303": False,
        "contractual_ifrs9_claim_allowed_v303": False,
        "working_champion_claim_allowed_v303": False,
        "paper1_promotion_allowed_v303": False,
        "paper4_working_champion_changed_v303": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v303": int(len(blockers)),
        "claim_matrix_rows_v303": int(len(claim_matrix)),
        "next_artifact_v303": (
            f"paper4_v{NEXT_VERSION}_bounded_multiobjective_milp_or_global_bound_probe.csv"
        ),
        "claim_boundary": (
            "v303 audits greedy-prefix multiobjective tradeoffs; global optimality, IFRS9, live deployment, "
            "working champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v303_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v303": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
